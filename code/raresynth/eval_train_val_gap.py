"""
Compare train-split vs held-out val-split loss on a trained checkpoint --
the real test of whether MoDiT + GenomicSetEncoder + GatedABMIL learned a
genuinely generalizable denoising objective or memorized the small
(3,836-case) training set. Training loss alone cannot distinguish these;
a model that has memorized its training examples can still show a low,
stable training loss while performing much worse on cases it never saw.

Evaluation is done WITHOUT random modality dropout (dropout_p effectively
0 -- the raw avail mask is used directly, not a further-dropped version)
so the comparison reflects the model's actual denoising skill on the
real available modalities for each split, not an apples-to-oranges
comparison confounded by different dropout randomness between runs.
"""

from __future__ import annotations

import argparse

import torch

from .data.dataset import RareSynthTCGADataset
from .model.dit import MoDiT, ModalitySpec
from .model.diffusion import GaussianDiffusion
from .train_dit import modality_loss_weights
from .train_modit import compute_z_geno_path, project_to_tissue_idx
from .encoders.genomic import GenomicSetEncoder
from .encoders.foundation import GatedABMIL


@torch.no_grad()
def evaluate_split(loader, model, genomic_encoder, pathology_encoder,
                   diffusion, spec, loss_weights, device, data_scale=1.0,
                   n_batches=None, fixed_t=None):
    """Average loss over a split, no gradient, no modality dropout (uses
    the real avail mask directly). If fixed_t is given, all samples use
    that exact diffusion timestep instead of a random one per sample --
    makes repeated evaluations directly comparable to each other (removes
    one further source of noise between separate evaluate_split calls).

    data_scale: MUST match the value used during training (see
    train_modit.py's compute_data_scale) -- x is multiplied by it before
    diffusion, exactly mirroring train_step, so this evaluation is a fair
    comparison against what the model was actually trained on rather than
    an inconsistent, differently-scaled input.
    """
    model.eval()
    genomic_encoder.eval()
    pathology_encoder.eval()

    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        z_geno, z_path = compute_z_geno_path(batch, genomic_encoder, pathology_encoder, device)
        z_rna = batch["rna"].to(device)
        z_ehr = batch["ehr"].to(device)
        z_rad = batch["rad"].to(device)
        x = spec.join([z_geno, z_rna, z_path, z_rad, z_ehr])
        x = x * data_scale
        avail = batch["avail"].to(device)  # real avail, no dropout
        tissue = project_to_tissue_idx(batch["project"], device)

        B = x.shape[0]
        if fixed_t is not None:
            t = torch.full((B,), fixed_t, dtype=torch.long, device=device)
            x_t, noise = diffusion.q_sample(x, t)
            pred = model(x_t, t, avail=avail, c_hpo=None, c_gene=None, tissue=tissue)
            se = (pred - noise) ** 2
            if loss_weights is not None:
                se = se * loss_weights.view(1, -1)
            loss = se.mean()
        else:
            loss, _ = diffusion.training_loss(
                model, x, avail=avail, c_hpo=None, c_gene=None,
                tissue=tissue, loss_weights=loss_weights,
            )
        total += loss.item() * B
        n += B

    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rna-npz", required=True)
    ap.add_argument("--clinical-npz", required=True)
    ap.add_argument("--genomic-npz", required=True)
    ap.add_argument("--gene-vocab-npz", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-batches", type=int, default=None,
                    help="cap on batches per split, for a quick check; "
                         "None evaluates the whole split")
    ap.add_argument("--fixed-t", type=int, default=None,
                    help="fixed diffusion timestep for a directly-comparable "
                         "train-vs-val measurement; default is T//2 "
                         "(computed from the checkpoint's own timestep "
                         "count, not hardcoded, since that varies by run); "
                         "pass -1 to use random timesteps instead")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import numpy as np
    gene_vocab = np.load(args.gene_vocab_npz, allow_pickle=True)
    gene_embedding = gene_vocab["gene_embedding"]

    ckpt = torch.load(args.checkpoint, map_location=dev)
    train_args = ckpt["args"]
    spec = ModalitySpec()

    model = MoDiT(
        spec=spec, d_model=train_args["d_model"], depth=train_args["depth"],
        n_heads=train_args["heads"],
        tokens_per_modality=train_args.get("tokens_per_modality", 2),
        d_hpo=768,
        d_gene=256, n_tissues=8, p_uncond=0.1,
    ).to(dev)
    model.load_state_dict(ckpt["model"])

    genomic_encoder = GenomicSetEncoder(
        gene_embedding, d_model=256, d_out=spec.dims["geno"],
    ).to(dev)
    genomic_encoder.load_state_dict(ckpt["genomic_encoder"])

    pathology_encoder = GatedABMIL(d_in=768, d_out=spec.dims["path"]).to(dev)
    pathology_encoder.load_state_dict(ckpt["pathology_encoder"])

    diffusion = GaussianDiffusion(train_args["timesteps"]).to(dev)
    lw = modality_loss_weights(spec, dev)
    data_scale = ckpt.get("data_scale", 1.0)  # 1.0 default: backward-compatible
                                              # with checkpoints trained
                                              # before this fix existed

    T = train_args["timesteps"]
    if args.fixed_t is None:
        fixed_t = T // 2
    elif args.fixed_t < 0:
        fixed_t = None
    else:
        if not (0 <= args.fixed_t < T):
            raise ValueError(f"--fixed-t {args.fixed_t} out of range for "
                            f"this checkpoint's T={T} (must be in [0, {T}))")
        fixed_t = args.fixed_t

    print(f"loaded checkpoint trained with: {train_args}")
    print(f"data_scale={data_scale:.3f}")
    print(f"evaluating with fixed_t={fixed_t} "
         f"({'random per-sample timesteps' if fixed_t is None else 'same timestep for train and val'})")

    for split in ("train", "val"):
        ds = RareSynthTCGADataset(
            args.manifest, args.rna_npz, args.clinical_npz, args.genomic_npz,
            split=split, max_tiles=train_args["max_tiles"],
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss = evaluate_split(loader, model, genomic_encoder, pathology_encoder,
                              diffusion, spec, lw, dev, data_scale=data_scale,
                              n_batches=args.n_batches, fixed_t=fixed_t)
        print(f"  {split}: n={len(ds)}, loss={loss:.5f}")


if __name__ == "__main__":
    main()
