"""
Generate synthetic multimodal profiles from a trained MoDiT checkpoint.

This is the piece that unblocks every downstream evaluation metric
(FID/PRDC/C2ST, CMCS, mechanism retrieval, privacy) -- none of them can
run without real generated samples to compare against real held-out data,
and none have been run yet anywhere in this project against a real
trained model.

Unconditional generation (no guidance, no x_obs) is the starting point:
generate N samples matching the real training tissue distribution, no
mechanism steering. Guided/conditional generation modes come later, once
this baseline path is confirmed correct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model.dit import MoDiT, ModalitySpec
from .model.diffusion import GaussianDiffusion
from .train_modit import TISSUE_VOCAB, N_TISSUES


def load_trained_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt["args"]
    spec = ModalitySpec()

    model = MoDiT(
        spec=spec, d_model=train_args["d_model"], depth=train_args["depth"],
        n_heads=train_args["heads"],
        tokens_per_modality=train_args.get("tokens_per_modality", 2),  # .get()
        # with default 2 for backward compatibility: the real checkpoint
        # already trained (modit_full) predates tokens_per_modality being
        # recorded into args at all, and was trained with the
        # then-hardcoded value of 2 -- this default matches that exactly,
        # so the existing real checkpoint still loads correctly
        d_hpo=768, d_gene=256, n_tissues=N_TISSUES, p_uncond=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(train_args["timesteps"]).to(device)
    # default 1.0 for backward compatibility with checkpoints trained
    # before data_scale existed at all (the original modit_full_PRENORM_STALE
    # and the RNA-normalization-only intermediate checkpoint both predate
    # this) -- 1.0 means "no rescaling," which is what those runs actually
    # did, so this correctly reproduces their real behavior rather than
    # silently applying a rescaling they were never trained with
    data_scale = ckpt.get("data_scale", 1.0)
    return model, diffusion, spec, train_args, data_scale


def sample_unconditional(model, diffusion, spec, n_samples, tissue_dist,
                         device, data_scale=1.0, n_steps=200, batch_size=64, seed=0):
    """tissue_dist: dict of {project_name: fraction}, e.g. matching the
    real training set's tissue proportions -- so generated samples reflect
    the same tissue mix, not an arbitrary/uniform one.

    data_scale: the SAME factor used to rescale real data up to ~unit
    variance before training (see train_modit.py's compute_data_scale) --
    generated samples come out in that SCALED space, so must be divided
    by data_scale here to return them to the real embedding space
    (matching what the RNA/genomic/clinical/pathology encoders actually
    produce, and what any downstream evaluation metric expects).
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    tissue_names = list(tissue_dist.keys())
    tissue_probs = np.array([tissue_dist[t] for t in tissue_names])
    tissue_probs = tissue_probs / tissue_probs.sum()

    all_x, all_tissue = [], []
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            b = min(batch_size, n_samples - n_done)
            sampled_tissue_names = np.random.choice(tissue_names, size=b, p=tissue_probs)
            tissue_idx = torch.tensor(
                [TISSUE_VOCAB[t] for t in sampled_tissue_names],
                dtype=torch.long, device=device,
            )
            avail = torch.ones(b, spec.n_modalities, device=device)  # full modality generation

            x = diffusion.ddim_sample(
                model, (b, spec.total_dim), n_steps=n_steps,
                avail=avail, tissue=tissue_idx, device=device,
                generator=generator,
            )
            x = x / data_scale  # back to the real embedding space
            all_x.append(x.cpu().numpy())
            all_tissue.extend(sampled_tissue_names.tolist())
            n_done += b
            print(f"  generated {n_done}/{n_samples}")

    return np.concatenate(all_x, axis=0), np.array(all_tissue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True,
                    help="used only to compute the real tissue distribution to match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, spec, train_args, data_scale = load_trained_model(args.checkpoint, dev)
    print(f"loaded checkpoint (trained {train_args['epochs']} epochs, "
         f"d_model={train_args['d_model']}, depth={train_args['depth']}, "
         f"data_scale={data_scale:.3f})")

    import json
    from collections import Counter
    manifest = json.loads(Path(args.manifest).read_text())
    train_projects = [e["project"] for e in manifest.values()
                      if e["split"] == "train" and e["project"] in TISSUE_VOCAB]
    counts = Counter(train_projects)
    tissue_dist = {p: c / len(train_projects) for p, c in counts.items()}
    print(f"real training tissue distribution: {tissue_dist}")

    print(f"\ngenerating {args.n_samples} unconditional samples...")
    x, tissue = sample_unconditional(
        model, diffusion, spec, args.n_samples, tissue_dist, dev,
        data_scale=data_scale, n_steps=args.n_steps,
        batch_size=args.batch_size, seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=x, tissue=tissue, spec_dims=json.dumps(spec.dims))
    print(f"\nsaved -> {out}")
    print(f"  shape: {x.shape}, any NaN: {np.isnan(x).any()}, "
         f"any Inf: {np.isinf(x).any()}")
    print(f"  per-coordinate stats: mean={x.mean():.4f}, std={x.std():.4f}, "
         f"min={x.min():.4f}, max={x.max():.4f}")


if __name__ == "__main__":
    main()
