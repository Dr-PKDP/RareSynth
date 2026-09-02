"""
Train MoDiT on real multimodal embeddings.

Expected input: a single .npz produced by the encoder pipeline, containing
    X          (N, 3328) float32   concatenated L2-normalised modality latents
    avail      (N, 5)    float32   1 = modality observed for this sample
    tissue     (N,)      int64     tissue index
    c_hpo      (N, 768)  float32   pooled HPO/clinical-text embedding
    c_gene     (N, 256)  float32   causal/driver gene embedding (zeros if none)
    split      (N,)      <U8       'train' | 'val' | 'test'
    donor_id   (N,)      <U32      used to make splits donor-disjoint

The split must be donor-disjoint, not sample-disjoint: TCGA and GTEx both
contribute several samples per donor, and a random row split leaks the same
individual across train and test, which inflates every fidelity metric.

Usage
-----
    torchrun --nproc_per_node=4 -m raresynth.train_dit \
        --data embeddings.npz --out runs/modit_base --epochs 400
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model.dit import MoDiT, ModalitySpec
from .model.diffusion import EMA, GaussianDiffusion


def modality_loss_weights(spec, device):
    """1/d_m per coordinate so each modality contributes equally to the loss."""
    w = []
    for d in spec.dims.values():
        w.append(torch.full((d,), 1.0 / d))
    w = torch.cat(w)
    return (w / w.mean()).to(device)


def random_modality_dropout(avail, p=0.3, min_keep=1, generator=None):
    """Randomly hide observed modalities during training.

    This is what gives conditional imputation at sampling time and what makes
    the model tolerant of the ~80% of TCGA cases that have no radiology.
    """
    B, M = avail.shape
    drop = (torch.rand(B, M, device=avail.device, generator=generator) < p).float()
    new = avail * (1 - drop)
    empty = new.sum(1) < min_keep
    if empty.any():
        new[empty] = avail[empty]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tokens-per-modality", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--modality-dropout", type=float, default=0.3)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data, allow_pickle=True)
    tr = d["split"] == "train"
    X = torch.as_tensor(d["X"][tr], dtype=torch.float32)
    avail = torch.as_tensor(d["avail"][tr], dtype=torch.float32)
    tissue = torch.as_tensor(d["tissue"][tr], dtype=torch.long)
    c_hpo = torch.as_tensor(d["c_hpo"][tr], dtype=torch.float32)
    c_gene = torch.as_tensor(d["c_gene"][tr], dtype=torch.float32)

    spec = ModalitySpec()
    assert X.shape[1] == spec.total_dim, (X.shape, spec.total_dim)

    model = MoDiT(
        spec=spec,
        d_model=args.d_model,
        depth=args.depth,
        n_heads=args.heads,
        tokens_per_modality=args.tokens_per_modality,
        d_hpo=c_hpo.shape[1],
        d_gene=c_gene.shape[1],
        n_tissues=int(tissue.max().item()) + 1,
        p_uncond=args.p_uncond,
    ).to(dev)
    print(f"MoDiT parameters: {model.n_params()/1e6:.1f}M")

    diff = GaussianDiffusion(args.timesteps).to(dev)
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lw = modality_loss_weights(spec, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev == "cuda")

    n = len(X)
    step = 0
    log = []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss, nb, t0 = 0.0, 0, time.time()
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            if len(idx) < 2:
                continue
            xb = X[idx].to(dev, non_blocking=True)
            ab = random_modality_dropout(
                avail[idx].to(dev), p=args.modality_dropout
            )
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and dev == "cuda"):
                loss, _ = diff.training_loss(
                    model,
                    xb,
                    avail=ab,
                    c_hpo=c_hpo[idx].to(dev),
                    c_gene=c_gene[idx].to(dev),
                    tissue=tissue[idx].to(dev),
                    loss_weights=lw,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            ep_loss += loss.item()
            nb += 1
            step += 1

        rec = {"epoch": ep, "loss": ep_loss / max(nb, 1), "sec": time.time() - t0}
        log.append(rec)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[{ep:4d}] loss {rec['loss']:.5f}  ({rec['sec']:.1f}s)")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "args": vars(args),
                    "spec_dims": spec.dims,
                },
                out / "checkpoint.pt",
            )
            (out / "log.json").write_text(json.dumps(log, indent=2))

    print("done ->", out)


if __name__ == "__main__":
    main()
