"""
The real MoDiT training loop -- supersedes the original train_dit.py,
which was written speculatively before RareSynthTCGADataset,
GenomicSetEncoder, or GatedABMIL had ever been run against real data, and
assumed all five modalities arrived as precomputed flat vectors. That
assumption is wrong for geno/path (see MANUSCRIPT_NOTES.md): both are
UNTRAINED aggregators that must be trained jointly with MoDiT, consuming
RAW per-case material from the Dataset, not a precomputed vector.

Two real numerical-stability bugs were found and fixed BEFORE this script
was written, both confirmed live (not reasoned about in the abstract):

  1. GatedABMIL's masking convention is True=hide (matches PyTorch's
     usual masked_fill convention), the OPPOSITE of the Dataset's
     path_mask (True=real tile). A case with NO real pathology has
     path_mask entirely False, so the naive inverse (~path_mask) is
     entirely True -- every position masked to -inf, and softmax over an
     all -inf row is NaN. Given only ~41% of TCGA cases have real
     pathology, most batches would have hit this and silently corrupted
     the loss via NaN propagation. Fixed by forcing position 0 of the
     attention mask to always stay unmasked (verified live: eliminates
     the NaN entirely), then multiplying the resulting z_path by the
     real avail flag so a fake-but-non-NaN output never leaks into
     training or the loss.
  2. GenomicSetEncoder was checked for the equivalent risk (missing case
     = all-zero features + gene_idx all 0, repeated) and confirmed safe
     WITHOUT needing the same fix -- it is called without an explicit
     pad_mask (matching the current Dataset design, which does not build
     a per-gene mask, only a whole-modality avail flag), so there is no
     internal masked-softmax step to produce NaN. Still multiplied by
     avail afterward for consistency and to prevent gradient signal into
     GenomicSetEncoder from genuinely-missing cases.

Conditioning simplifications, stated plainly rather than hidden: c_hpo
and c_gene are both None for this training pass -- TCGA cases have no
real HPO phenotype annotations (that is rare-disease-specific data this
project does not have for cancer cases), and gene-level conditioning
matters for INFERENCE-time guided generation toward a specific candidate
gene, not for fitting the general training distribution. Tissue
conditioning IS real: built from the 8 TCGA project codes actually used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data.dataset import RareSynthTCGADataset
from .model.dit import MoDiT, ModalitySpec
from .model.diffusion import EMA, GaussianDiffusion
from .encoders.genomic import GenomicSetEncoder
from .encoders.foundation import GatedABMIL
from .train_dit import modality_loss_weights, random_modality_dropout

TISSUE_VOCAB = {
    "TCGA-BRCA": 0, "TCGA-LUAD": 1, "TCGA-KIRC": 2, "TCGA-GBM": 3,
    "TCGA-LGG": 4, "TCGA-COAD": 5, "TCGA-OV": 6, "TCGA-STAD": 7,
}
N_TISSUES = len(TISSUE_VOCAB)


def project_to_tissue_idx(projects, device):
    idx = [TISSUE_VOCAB.get(p, N_TISSUES) for p in projects]  # N_TISSUES = unknown
    return torch.tensor(idx, dtype=torch.long, device=device)


def compute_z_geno_path(batch, genomic_encoder, pathology_encoder, device):
    """Run the two untrained aggregators on raw per-case material, with
    the verified-safe masking design (see module docstring). Returns
    (z_geno, z_path), both already zeroed for genuinely-missing cases.
    """
    geno_feats = batch["geno_features"].to(device)
    geno_idx = batch["geno_gene_idx"].to(device)
    z_geno = genomic_encoder(geno_feats, geno_idx)  # no pad_mask -- confirmed safe
    z_geno = z_geno * batch["avail"][:, 0:1].to(device)

    path_bag = batch["path_bag"].to(device)
    path_mask = batch["path_mask"].to(device)  # True = real tile
    abmil_mask = ~path_mask                    # True = hide (GatedABMIL's convention)
    abmil_mask = abmil_mask.clone()
    abmil_mask[:, 0] = False                   # CRITICAL: always keep >=1 position,
                                               # confirmed live this eliminates the
                                               # NaN an all-padding bag would
                                               # otherwise produce
    z_path, _ = pathology_encoder(path_bag, mask=abmil_mask)
    z_path = z_path * batch["avail"][:, 2:3].to(device)

    return z_geno, z_path


def compute_data_scale(loader, genomic_encoder, pathology_encoder, spec, device, n_batches=10):
    """Measure the REAL training data's actual std (not assumed, not a
    round guess) and return a scale factor bringing it to ~1.0, matching
    the diffusion process's standard unit-variance assumption.

    Found necessary via direct measurement, not theory: real training x
    has overall std ~0.030 (every modality's own encoder L2-normalizes to
    unit NORM, which for a d-dimensional vector gives per-coordinate std
    on the order of 1/sqrt(d) -- d in the hundreds here, hence the small
    real value), while a first full training run (pre-fix) generated
    samples with std ~1.45 -- a ~48x mismatch, confirmed by directly
    comparing real vs generated statistics, not inferred. A smaller,
    separate cross-modal RELATIVE scale bug (RNA never normalized at all,
    sitting ~6.6x larger than every other modality) was found and fixed
    first, but did NOT resolve this ABSOLUTE scale mismatch on its own --
    confirmed by re-checking generated-sample statistics after that fix
    alone, which were essentially unchanged. Both fixes were necessary;
    neither was sufficient alone.
    """
    sq_sum, n_total = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            z_geno, z_path = compute_z_geno_path(batch, genomic_encoder, pathology_encoder, device)
            z_rna = batch["rna"].to(device)
            z_ehr = batch["ehr"].to(device)
            z_rad = batch["rad"].to(device)
            x = spec.join([z_geno, z_rna, z_path, z_rad, z_ehr])
            sq_sum += (x ** 2).sum().item()
            n_total += x.numel()
    real_std = (sq_sum / n_total) ** 0.5
    scale = 1.0 / real_std
    print(f"  measured real data std={real_std:.5f} over {n_batches} batches "
         f"-> data_scale={scale:.3f} (multiplies x before diffusion, so "
         f"scaled data has std~1.0, matching the diffusion process's "
         f"standard assumption)")
    return scale


def get_lr(step, total_steps, peak_lr, warmup, min_lr_frac=0.02):
    """Linear warmup for `warmup` steps, then cosine decay from peak_lr
    down to peak_lr * min_lr_frac over the remaining steps.

    The original schedule warmed up then held peak_lr CONSTANT for the
    rest of training (no decay at all) -- standard practice in
    essentially every published diffusion transformer is a decay
    schedule after warmup, precisely because a constant LR prevents
    fine-grained convergence late in training. Added after observing the
    first full run's loss oscillate in a narrow band (0.026-0.031) for
    its final ~150 epochs rather than continuing to settle -- consistent
    with an LR too high for late-stage convergence, though this specific
    causal claim was not separately isolated (see PROGRESS.md: this fix
    is deliberately the ONE change made before the next training run, not
    bundled with other hyperparameter changes, so any improvement can be
    attributed to it rather than confounded with several simultaneous
    changes).
    """
    import math
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


def train_step(batch, model, genomic_encoder, pathology_encoder, diffusion,
              spec, loss_weights, device, data_scale, dropout_p=0.3):
    z_geno, z_path = compute_z_geno_path(batch, genomic_encoder, pathology_encoder, device)
    z_rna = batch["rna"].to(device)
    z_ehr = batch["ehr"].to(device)
    z_rad = batch["rad"].to(device)  # always zero -- no radiology encoder exists

    x = spec.join([z_geno, z_rna, z_path, z_rad, z_ehr])  # order matches
                                                          # ModalitySpec's
                                                          # dict order AND
                                                          # the Dataset's
                                                          # avail column
                                                          # order -- both
                                                          # confirmed
                                                          # consistent
    x = x * data_scale  # bring real data (std~0.03) to the diffusion
                        # process's expected ~unit-variance scale -- see
                        # compute_data_scale's docstring for why this is
                        # necessary (confirmed via direct measurement,
                        # not assumed)
    avail = batch["avail"].to(device)
    avail_dropped = random_modality_dropout(avail, p=dropout_p)

    tissue = project_to_tissue_idx(batch["project"], device)

    loss, info = diffusion.training_loss(
        model, x, avail=avail_dropped, c_hpo=None, c_gene=None,
        tissue=tissue, loss_weights=loss_weights,
    )
    return loss, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rna-npz", required=True)
    ap.add_argument("--clinical-npz", required=True)
    ap.add_argument("--genomic-npz", required=True)
    ap.add_argument("--gene-vocab-npz", required=True,
                    help="the gene_vocabulary.npz written alongside genomic_embeddings.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-tiles", type=int, default=2000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tokens-per-modality", type=int, default=2,
                    help="recorded into the checkpoint's args so "
                         "sample_modit.py reads the real value used, "
                         "rather than an independently hardcoded guess")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.02,
                    help="cosine decay floor as a fraction of peak lr "
                         "(e.g. 0.02 = decay down to 2% of peak by the end)")
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--dropout-p", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== loading dataset ===")
    ds = RareSynthTCGADataset(
        args.manifest, args.rna_npz, args.clinical_npz, args.genomic_npz,
        split="train", max_tiles=args.max_tiles,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    gene_vocab = np.load(args.gene_vocab_npz, allow_pickle=True)
    gene_embedding = gene_vocab["gene_embedding"]
    print(f"gene vocabulary: {gene_embedding.shape[0]} genes, "
         f"{gene_embedding.shape[1]}-dim (placeholder embedding)")

    spec = ModalitySpec()
    print(f"ModalitySpec dims: {spec.dims} (total {spec.total_dim})")

    print("\n=== building models ===")
    model = MoDiT(
        spec=spec, d_model=args.d_model, depth=args.depth, n_heads=args.heads,
        tokens_per_modality=args.tokens_per_modality, d_hpo=768, d_gene=256,
        n_tissues=N_TISSUES, p_uncond=0.1,
    ).to(dev)
    genomic_encoder = GenomicSetEncoder(
        gene_embedding, d_model=256, d_out=spec.dims["geno"],
    ).to(dev)
    pathology_encoder = GatedABMIL(
        d_in=768, d_out=spec.dims["path"],
    ).to(dev)

    n_params = (model.n_params()
               + sum(p.numel() for p in genomic_encoder.parameters())
               + sum(p.numel() for p in pathology_encoder.parameters()))
    print(f"total trainable parameters: {n_params/1e6:.1f}M "
         f"(MoDiT {model.n_params()/1e6:.1f}M + GenomicSetEncoder "
         f"{sum(p.numel() for p in genomic_encoder.parameters())/1e6:.1f}M + "
         f"GatedABMIL {sum(p.numel() for p in pathology_encoder.parameters())/1e6:.1f}M)")

    diffusion = GaussianDiffusion(args.timesteps).to(dev)
    all_params = (list(model.parameters()) + list(genomic_encoder.parameters())
                 + list(pathology_encoder.parameters()))
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-5)
    ema_model = EMA(model, args.ema_decay)
    lw = modality_loss_weights(spec, dev)

    print("\n=== computing data scale from real training data ===")
    data_scale = compute_data_scale(loader, genomic_encoder, pathology_encoder, spec, dev)

    print(f"\n=== training: {len(ds)} cases, {len(loader)} batches/epoch, "
         f"{args.epochs} epochs ===")
    step = 0
    total_steps = args.epochs * len(loader)
    log = []
    for ep in range(args.epochs):
        model.train()
        genomic_encoder.train()
        pathology_encoder.train()
        ep_loss, nb, t0 = 0.0, 0, time.time()

        for batch in loader:
            lr_now = get_lr(step, total_steps, args.lr, args.warmup, args.min_lr_frac)
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            loss, info = train_step(
                batch, model, genomic_encoder, pathology_encoder, diffusion,
                spec, lw, dev, data_scale, dropout_p=args.dropout_p,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            ema_model.update(model)  # NOTE: EMA tracks MoDiT's own weights
                                     # only, not the two encoders -- see
                                     # PROGRESS.md open items, a reasonable
                                     # scope for a first working version

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {step} -- stopping "
                     f"immediately rather than continuing to train on "
                     f"corrupted gradients. If this triggers, the masking "
                     f"fix in compute_z_geno_path needs to be revisited.")
                return

            ep_loss += loss.item()
            nb += 1
            step += 1

        rec = {"epoch": ep, "loss": ep_loss / max(nb, 1), "sec": time.time() - t0}
        log.append(rec)
        print(f"[{ep:4d}] loss {rec['loss']:.5f}  ({rec['sec']:.1f}s)")

        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.shadow,
                "genomic_encoder": genomic_encoder.state_dict(),
                "pathology_encoder": pathology_encoder.state_dict(),
                "data_scale": data_scale,
                "args": vars(args),
            }, out / "checkpoint.pt")
            (out / "log.json").write_text(json.dumps(log, indent=2))

    print("\ndone ->", out)


if __name__ == "__main__":
    main()
