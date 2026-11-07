"""Energy guidance toward the mapped direction with the target, coordinate-shuffle and variance-stratified-shuffle directions."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")

import numpy as np
import torch

from raresynth.model.dit import MoDiT, ModalitySpec
from raresynth.model.diffusion import GaussianDiffusion
from raresynth.model.guidance import _cos_energy
from raresynth.eval.fidelity import frechet_distance, prdc

PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
GTEX_BASELINE = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/fibroblast_baseline.npz"
BEST = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/best.pt"
DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
OUT_PATH = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/guidance_redo.json"
SCALES = [10, 30, 100, 300, 1000, 3000]
SEED_BASE = 2000

def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cos_rows(u, dhat):
    return (u @ dhat) / (np.linalg.norm(u, axis=1) + 1e-12)


def shift_samples(x, dhat, alpha):
    return unit(x + alpha * dhat)


def mean_cos_after_shift(x, b, dhat, alpha):
    return float(cos_rows(shift_samples(x, dhat, alpha) - b, dhat).mean())


def match_alpha(x, b, dhat, target, hi=50.0, iters=60):
    if target <= mean_cos_after_shift(x, b, dhat, 0.0):
        return 0.0
    if mean_cos_after_shift(x, b, dhat, hi) < target:
        return float("nan")
    lo, h = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + h)
        if mean_cos_after_shift(x, b, dhat, mid) < target:
            lo = mid
        else:
            h = mid
    return 0.5 * (lo + h)


def sqdist(a, b):
    return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def nn_dist(a, b):
    return np.sqrt(sqdist(a, b).min(axis=1))


def fit_tissue(z, y):
    comps = {}
    for c in np.unique(y):
        zc = z[y == c]
        if len(zc) >= 3:
            comps[int(c)] = (zc.mean(axis=0), np.cov(zc, rowvar=False), len(zc))
    return comps


def stratified_shuffle(d, sd, block=32, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(sd)
    out = d.copy()
    for start in range(0, len(d), block):
        idx = order[start:start + block]
        out[idx] = d[rng.permutation(idx)]
    return out


def allocate(counts, n):
    counts = np.asarray(counts, dtype=float)
    frac = counts / counts.sum() * n
    base = np.floor(frac).astype(int)
    base[np.argsort(-(frac - base))[: int(n - base.sum())]] += 1
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny settings to check the pipeline runs")
    ap.add_argument("--n", type=int, default=206)
    ap.add_argument("--n-steps", type=int, default=200)
    args = ap.parse_args()
    n_gen = 16 if args.smoke else args.n
    n_steps = 20 if args.smoke else args.n_steps
    scales = [100] if args.smoke else SCALES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(BEST, map_location=dev, weights_only=False)
    a, mu, sd, bound = ck["args"], ck["mu"], ck["sd"], ck["bound"]
    d = np.load(DATA)
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"]
    rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"]
    z_tr, z_va = (rna_tr - mu) / sd, (rna_va - mu) / sd

    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    d_shuf = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance_SHUFFLED_CONTROL.npz")["rna_embedding_guidance"].astype(np.float64)
    b = np.load(GTEX_BASELINE)["baseline"].astype(np.float64)
    dh_true, dh_shuf = d_true / np.linalg.norm(d_true), d_shuf / np.linalg.norm(d_shuf)
    print(f"direction norms: true {np.linalg.norm(d_true):.4f}, shuffled {np.linalg.norm(d_shuf):.4f}; "
         f"baseline norm {np.linalg.norm(b):.4f}; cos(true, shuffled) = {float(dh_true @ dh_shuf):.4f}", flush=True)
    d_strat = stratified_shuffle(d_true, sd)
    dh_strat = d_strat / np.linalg.norm(d_strat)
    mu_off = rna_tr.mean(axis=0) - b
    top10, top50 = np.argsort(-sd)[:10], np.argsort(-sd)[:50]
    print("alignment of each direction with the data (a direction concentrated on high-variance coordinates "
         "and on the train-mean-minus-baseline offset starts with a large cosine before any guidance):")
    for nm, dd in (("true", d_true), ("shuffled", d_shuf), ("stratified shuffle", d_strat)):
        print(f"  {nm:>18}: share of |d|^2 on the 10 / 50 highest-std coordinates = "
             f"{(dd[top10]**2).sum()/(dd**2).sum():.3f} / {(dd[top50]**2).sum()/(dd**2).sum():.3f}; "
             f"cos(d, train mean - baseline) = {float(dd @ mu_off / (np.linalg.norm(dd) * np.linalg.norm(mu_off))):+.3f}; "
             f"cos(d, d_true) = {float(dd @ d_true / (np.linalg.norm(dd) * np.linalg.norm(d_true))):+.3f}", flush=True)
    print(f"  (for reference, 10 / 50 of 768 coordinates would hold {10/768:.3f} / {50/768:.3f} of a direction spread evenly)", flush=True)

    spec = ModalitySpec(dims={"rna": 768})
    model = MoDiT(spec=spec, d_model=a["d_model"], depth=a["depth"], n_heads=a["heads"],
                  tokens_per_modality=a["tokens_per_modality"], d_hpo=768, d_gene=256,
                  n_tissues=8, p_uncond=a["p_uncond"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    diffusion = GaussianDiffusion(a["timesteps"]).to(dev)
    B = torch.from_numpy(bound).float().to(dev)
    mu_t, sd_t = torch.from_numpy(mu).float().to(dev), torch.from_numpy(sd).float().to(dev)
    b_t = torch.from_numpy(b).float().to(dev)

    class RnaEnergy:
        def __init__(self, direction):
            self.d = torch.from_numpy(direction).float().to(dev)

        def __call__(self, z):
            delta = (z * sd_t + mu_t) - b_t
            return _cos_energy(delta, self.d.unsqueeze(0).expand_as(delta))

    E_true, E_shuf, E_strat = RnaEnergy(d_true), RnaEnergy(d_shuf), RnaEnergy(d_strat)

    with torch.no_grad():
        z_base = ((torch.from_numpy(b).float().to(dev) - mu_t) / sd_t).unsqueeze(0)
        z_move = (((torch.from_numpy(b + d_true).float().to(dev)) - mu_t) / sd_t).unsqueeze(0)
        e0, e1 = float(E_true(z_base)), float(E_true(z_move))
    print(f"energy sanity: at baseline {e0:.4f}; moved exactly along its direction {e1:.4f}", flush=True)
    assert e1 < e0, "energy did not decrease when moving along its own direction"

    def generate(scale, energy, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn((n_gen, 768), device=dev, generator=g)
        ts = torch.linspace(diffusion.T - 1, 0, n_steps, device=dev).long()
        kw = dict(avail=torch.ones(n_gen, 1, device=dev), c_hpo=None, c_gene=None,
                  tissue=torch.zeros(n_gen, dtype=torch.long, device=dev))
        for i, t_cur in enumerate(ts):
            tb = t_cur.repeat(n_gen)
            ab_t = diffusion.alpha_bar[t_cur]
            t_next = ts[i + 1] if i + 1 < len(ts) else None
            ab_next = diffusion.alpha_bar[t_next] if t_next is not None else torch.tensor(1.0, device=dev)
            with torch.no_grad():
                eps = diffusion._eps_with_cfg(model, x, tb, 0.0, **kw)
                x0 = torch.maximum(torch.minimum((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt(), B), -B)
            if energy is not None and scale != 0.0:
                xg = x0.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    (grad,) = torch.autograd.grad(energy(xg).sum(), xg)
                with torch.no_grad():
                    x0 = torch.maximum(torch.minimum(x0 - scale * (1 - ab_t).sqrt() * grad, B), -B)
            with torch.no_grad():
                eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
                x = x0 if t_next is None else ab_next.sqrt() * x0 + (1 - ab_next).clamp(min=0).sqrt() * eps
        return x.double().cpu().numpy() * sd + mu

    rng = np.random.default_rng(777)
    d_val_nn = nn_dist(z_va, z_tr)
    real_subsets = [rna_va[rng.choice(len(rna_va), min(n_gen, len(rna_va)), replace=False)] for _ in range(3)]

    def realism(x_unit):
        nrm = np.linalg.norm(x_unit, axis=1)
        runs = []
        for real in real_subsets:
            pr = prdc(real, x_unit, k=5)
            runs.append({"fid": float(frechet_distance(real, x_unit)), "precision": float(pr["precision"]),
                         "recall": float(pr["recall"]), "coverage": float(pr["coverage"])})
        m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        m["norm_band"] = float(((nrm >= 0.8) & (nrm <= 1.2)).mean())
        m["nn_ratio"] = float(np.median(nn_dist((x_unit - mu) / sd, z_tr)) / np.median(d_val_nn))
        return m

    comps = fit_tissue(z_tr, y_tr)
    classes, counts = np.unique(y_va, return_counts=True)
    labels = np.repeat(classes, allocate(counts, n_gen))
    gauss_z = np.zeros((n_gen, 768))
    for c in np.unique(labels):
        m_ = labels == c
        mu_c, cov_c, _ = comps[int(c)]
        gauss_z[m_] = rng.multivariate_normal(mu_c, cov_c, size=int(m_.sum()), method="svd", check_valid="ignore")
    gauss_x = gauss_z * sd + mu

    t0 = time.time()
    base_seed = SEED_BASE
    x_null = generate(0.0, None, base_seed)
    print(f"unguided null generated ({time.time()-t0:.0f}s); mean cos(u, d_true) = "
         f"{cos_rows(x_null - b, dh_true).mean():+.4f}, cos(u, d_shuf) = {cos_rows(x_null - b, dh_shuf).mean():+.4f}", flush=True)
    results = {"unguided": {"cos_true": float(cos_rows(x_null - b, dh_true).mean()),
                            "cos_shuf": float(cos_rows(x_null - b, dh_shuf).mean()),
                            "cos_strat": float(cos_rows(x_null - b, dh_strat).mean()), "realism": realism(x_null)},
               "scales": {}}

    for k, scale in enumerate(scales):
        seed = SEED_BASE + 1 + k
        x_t = generate(float(scale), E_true, seed)
        x_s = generate(float(scale), E_shuf, seed)
        x_c = generate(float(scale), E_strat, seed)
        x_0 = x_null if k == 0 else x_null
        c = {"true_on_true": float(cos_rows(x_t - b, dh_true).mean()),
             "true_on_shuf": float(cos_rows(x_t - b, dh_shuf).mean()),
             "shuf_on_shuf": float(cos_rows(x_s - b, dh_shuf).mean()),
             "shuf_on_true": float(cos_rows(x_s - b, dh_true).mean()),
             "strat_on_strat": float(cos_rows(x_c - b, dh_strat).mean()),
             "strat_on_true": float(cos_rows(x_c - b, dh_true).mean())}
        target = c["true_on_true"]
        alpha_a = match_alpha(x_null, b, dh_true, target)
        alpha_b = match_alpha(gauss_x, b, dh_true, target)
        rec = {"cos": c, "alpha_shift_diffusion": alpha_a, "alpha_shift_gaussian": alpha_b,
               "realism_guided_true": realism(x_t), "realism_guided_shuffled": realism(x_s)}
        rec["realism_shift_diffusion"] = realism(shift_samples(x_null, dh_true, alpha_a)) if np.isfinite(alpha_a) else None
        rec["realism_shift_gaussian"] = realism(shift_samples(gauss_x, dh_true, alpha_b)) if np.isfinite(alpha_b) else None
        results["scales"][str(scale)] = rec
        print(f"\n--- guidance scale {scale} ---")
        u0 = results["unguided"]
        print(f"  specificity (levels): true-guided cos(d_true)={c['true_on_true']:+.4f} cos(d_shuf)={c['true_on_shuf']:+.4f} | "
             f"shuffled-guided cos(d_shuf)={c['shuf_on_shuf']:+.4f} cos(d_true)={c['shuf_on_true']:+.4f} | "
             f"strat-guided cos(d_strat)={c['strat_on_strat']:+.4f} cos(d_true)={c['strat_on_true']:+.4f} | "
             f"unguided cos(d_true)={u0['cos_true']:+.4f}", flush=True)
        print(f"  specificity (CHANGE vs unguided): own direction: true {c['true_on_true']-u0['cos_true']:+.4f}, "
             f"shuffled {c['shuf_on_shuf']-u0['cos_shuf']:+.4f}, stratified {c['strat_on_strat']-u0['cos_strat']:+.4f} | "
             f"spill onto d_true: from shuffled {c['shuf_on_true']-u0['cos_true']:+.4f}, from stratified {c['strat_on_true']-u0['cos_true']:+.4f}", flush=True)
        for name, m in (("guided (diffusion)", rec["realism_guided_true"]),
                        ("shift of diffusion samples", rec["realism_shift_diffusion"]),
                        ("shift of Gaussian samples", rec["realism_shift_gaussian"])):
            if m is None:
                print(f"  {name:>28}: target cosine unreachable by a unit-projected shift")
            else:
                print(f"  {name:>28}: prec={m['precision']:.3f} rec={m['recall']:.3f} cov={m['coverage']:.3f} "
                     f"FID={m['fid']:.4f} band={m['norm_band']:.2f} nn_ratio={m['nn_ratio']:.2f}", flush=True)

    print(f"\n{'='*100}\nSUMMARY: realism of true-guided samples vs post-hoc shifts at matched achieved cosine\n{'='*100}")
    print(f"{'scale':>6} | {'cos(d_true)':>11} | {'chg own':>8} | {'spill shuf':>10} | {'spill strat':>11} | {'precision: guided / shift-diff / shift-gauss':>46} | {'nn_ratio: guided / shift-diff':>28}")
    for scale in scales:
        r = results["scales"][str(scale)]
        g, sa, sb = r["realism_guided_true"], r["realism_shift_diffusion"], r["realism_shift_gaussian"]
        f = lambda m: f"{m['precision']:.3f}" if m else "  n/a"
        nn = lambda m: f"{m['nn_ratio']:.2f}" if m else " n/a"
        u_t = results["unguided"]["cos_true"]
        print(f"{scale:>6} | {r['cos']['true_on_true']:>+11.4f} | {r['cos']['true_on_true']-u_t:>+8.4f} | "
             f"{r['cos']['shuf_on_true']-u_t:>+10.4f} | {r['cos']['strat_on_true']-u_t:>+11.4f} | "
             f"{f(g)+' / '+f(sa)+' / '+f(sb):>46} | {nn(g)+' / '+nn(sa):>28}")
    print("\nREADING: direction-specific if the true-guided cosine RISES relative to unguided while the spill of the shuffled and stratified-shuffle controls onto d_true stays near zero. The diffusion model adds value to steering only "
         "if its guided precision/nn_ratio stay better than the shifted comparators at the same achieved cosine.")

    if not args.smoke:
        with open(OUT_PATH, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"saved -> {OUT_PATH}")
    else:
        print("smoke run finished (nothing saved)")


if __name__ == "__main__":
    main()
