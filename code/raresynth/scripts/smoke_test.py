"""
End-to-end smoke test on simulated data.

Runs the whole pipeline -- MoDiT training, mechanism-guided sampling,
imputation, all evaluation metrics, and every baseline -- on a small simulated
dataset with a *known* cross-modal dependence structure.  This does two things:

1. Verifies the code runs before any real data is downloaded.
2. Provides a sanity check with ground truth: the simulation builds pathology
   and radiology latents as a known function of the transcriptomic latent plus
   noise, so a correct implementation should show CMCS near 1 for MoDiT and
   near 0 for the independent-modality baseline.  If that ordering does not
   appear here, it will not appear on real data either, and the problem is the
   code rather than the biology.

Runs in a few minutes on CPU.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raresynth.baselines.multimodal_vae import MultimodalVAE
from raresynth.baselines.simple import GaussianCopula
from raresynth.eval.coherence import cross_modal_coherence
from raresynth.eval.fidelity import frechet_distance, prdc
from raresynth.eval.privacy import privacy_report
from raresynth.eval.tasks import evaluate_imputation, retrieval_at_k
from raresynth.model.diffusion import EMA, GaussianDiffusion
from raresynth.model.dit import MoDiT, ModalitySpec

SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# small spec so the test is fast
SPEC = ModalitySpec(dims={"geno": 16, "rna": 16, "path": 24, "rad": 16, "ehr": 24})
D = SPEC.total_dim
N, N_TISSUE, D_HPO, D_GENE = 4000, 6, 24, 16


def simulate(n):
    """Latents with a known cross-modal dependence: path/rad/geno are linear
    functions of rna plus tissue offset plus noise."""
    tissue = rng.integers(0, N_TISSUE, n)
    tis_off = rng.standard_normal((N_TISSUE, SPEC.dims["rna"])) * 0.8
    z_rna = tis_off[tissue] + rng.standard_normal((n, SPEC.dims["rna"])) * 0.5

    parts, mats = {}, {}
    for m in ("geno", "path", "rad", "ehr"):
        W = rng.standard_normal((SPEC.dims["rna"], SPEC.dims[m])) / np.sqrt(
            SPEC.dims["rna"]
        )
        mats[m] = W
        noise = 0.35 if m != "ehr" else 0.9  # ehr deliberately weakly coupled
        parts[m] = z_rna @ W + rng.standard_normal((n, SPEC.dims[m])) * noise

    X = np.concatenate(
        [parts["geno"], z_rna, parts["path"], parts["rad"], parts["ehr"]], axis=1
    ).astype(np.float32)
    # standardise per coordinate: the real pipeline L2-normalises each modality
    # latent, so the model always sees O(1)-scale inputs
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    c_hpo = np.concatenate(
        [np.eye(N_TISSUE)[tissue], rng.standard_normal((n, D_HPO - N_TISSUE)) * 0.1],
        axis=1,
    ).astype(np.float32)
    c_gene = rng.standard_normal((n, D_GENE)).astype(np.float32)
    return X, tissue.astype(np.int64), c_hpo, c_gene, mats


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  |  D = {D}")

    X, tissue, c_hpo, c_gene, _ = simulate(N)
    n_tr = int(N * 0.6)
    n_va = int(N * 0.2)
    sl_tr, sl_va, sl_te = slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, N)

    Xt = torch.as_tensor(X)
    tt = torch.as_tensor(tissue)
    ht = torch.as_tensor(c_hpo)
    gt = torch.as_tensor(c_gene)
    avail = torch.ones(N, SPEC.n_modalities)
    avail[rng.random(N) < 0.4, 3] = 0.0  # radiology missing for 40%, as in TCGA

    # ---------------- MoDiT ----------------
    model = MoDiT(
        spec=SPEC, d_model=128, depth=4, n_heads=4, tokens_per_modality=1,
        d_hpo=D_HPO, d_gene=D_GENE, n_tissues=N_TISSUE,
    ).to(dev)
    print(f"MoDiT params: {model.n_params()/1e6:.2f}M")
    diff = GaussianDiffusion(400).to(dev)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for ep in range(300):
        model.train()
        perm = torch.randperm(n_tr)
        tot = 0.0
        for i in range(0, n_tr, 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            loss, _ = diff.training_loss(
                model, Xt[idx].to(dev), avail=avail[idx].to(dev),
                c_hpo=ht[idx].to(dev), c_gene=gt[idx].to(dev), tissue=tt[idx].to(dev),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot += loss.item()
        if ep % 100 == 0:
            print(f"  epoch {ep:3d}  loss {tot / max(1, n_tr // 128):.4f}")

    ema.copy_to(model)
    model.eval()

    n_gen = n_va
    idx = torch.arange(n_tr, n_tr + n_gen)
    with torch.no_grad():
        synth = diff.ddim_sample(
            model, (n_gen, D), n_steps=100, cfg_scale=1.5,
            c_hpo=ht[idx].to(dev), c_gene=gt[idx].to(dev), tissue=tt[idx].to(dev),
            device=dev,
        ).cpu().numpy()

    # ---------------- Baselines ----------------
    copula = GaussianCopula().fit(X[sl_tr])
    synth_copula = copula.sample(n_gen, seed=SEED)

    # independent-modality control: shuffle each modality slice independently,
    # which preserves every marginal exactly and destroys the joint
    synth_indep = synth.copy()
    for m, sl in SPEC.slices().items():
        synth_indep[:, sl] = synth_indep[rng.permutation(n_gen), sl]

    mvae = MultimodalVAE(SPEC, d_latent=32, hidden=256, fusion="poe").to(dev)
    optv = torch.optim.AdamW(mvae.parameters(), lr=1e-3)
    for _ in range(200):
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 256):
            j = perm[i : i + 256]
            optv.zero_grad()
            loss, _ = mvae(Xt[j].to(dev), avail[j].to(dev))
            loss.backward()
            optv.step()
    synth_mvae = mvae.sample(n_gen, device=dev).cpu().numpy()

    # ---------------- Evaluation ----------------
    real_val, real_test = X[sl_va], X[sl_te]
    print("\n--- Fréchet distance (joint, lower better) ---")
    for name, S in [
        ("MoDiT", synth), ("MVAE", synth_mvae), ("Copula", synth_copula),
        ("Independent (shuffled)", synth_indep),
    ]:
        print(f"  {name:24s} {frechet_distance(real_test, S):8.3f}")

    print("\n--- PRDC, MoDiT ---")
    for k, v in prdc(real_test[:400], synth[:400]).items():
        print(f"  {k:12s} {v:.3f}")

    print("\n--- Cross-modal coherence (mean CMCS; 1.0 = matches real) ---")
    pairs = [("rna", "path"), ("rna", "rad"), ("rna", "geno"), ("path", "rna")]
    for name, S in [
        ("MoDiT", synth), ("MVAE", synth_mvae), ("Copula", synth_copula),
        ("Independent (shuffled)", synth_indep),
    ]:
        c = cross_modal_coherence(X[sl_tr], real_val, S, SPEC, pairs=pairs)
        print(f"  {name:24s} CMCS {c['_mean_cmcs']:6.3f}   "
              f"shuffled-control r2 {c['_mean_shuffled']:6.3f}")

    print("\n--- Imputation: generate radiology from the other four ---")
    obs_mask = torch.ones(1, D)
    obs_mask[:, SPEC.slices()["rad"]] = 0.0
    ti = torch.arange(n_tr + n_va, N)[:300]
    with torch.no_grad():
        imp = diff.ddim_sample(
            model, (len(ti), D), n_steps=100, cfg_scale=1.5,
            x_obs=Xt[ti].to(dev), obs_mask=obs_mask.expand(len(ti), D).to(dev),
            c_hpo=ht[ti].to(dev), c_gene=gt[ti].to(dev), tissue=tt[ti].to(dev),
            device=dev,
        ).cpu().numpy()
    for m, v in evaluate_imputation(imp, X[ti.numpy()], SPEC, ["rad"]).items():
        print(f"  {m}: R2 {v['r2']:+.3f}  cosine {v['cosine_mean']:.3f}")
    sl = SPEC.slices()["rad"]
    for k, v in retrieval_at_k(imp[:, sl], X[ti.numpy()][:, sl]).items():
        print(f"  {k:12s} {v:.3f}")

    print("\n--- Privacy ---")
    for k, v in privacy_report(synth, X[sl_tr], real_test).items():
        print(f"  {k:40s} {v:.4f}" if isinstance(v, float) else f"  {k:40s} {v}")

    print("\nSmoke test complete.")
    print("Expected ordering if the implementation is correct:")
    print("  CMCS:  MoDiT > MVAE > Copula >> Independent(shuffled) ~ 0")
    print("  MIA AUC near 0.5; no exact duplicates.")


if __name__ == "__main__":
    main()
