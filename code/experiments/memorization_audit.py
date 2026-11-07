"""Nearest-training-row distances of generated and held-out rows, and the training/validation denoising-loss gap."""

import json
import os
import sys

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")

import numpy as np
import torch

from raresynth.model.dit import MoDiT, ModalitySpec
from raresynth.model.diffusion import GaussianDiffusion

BEST = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/best.pt"
DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
OUT_PATH = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/memorization_audit.json"
N_GEN = 413
N_STEPS = 200
SEED = 777
TS = [500, 250, 100, 50, 10]

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--best", default=BEST, help="checkpoint to audit (default: attempt 1, seed 0)")
_ap.add_argument("--out", default=OUT_PATH, help="where to write the audit JSON")
_args = _ap.parse_args()
BEST, OUT_PATH = _args.best, _args.out

def sqdist(a, b):
    return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def nn_dist(a, b, exclude_self=False):
    d = sqdist(a, b)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return np.sqrt(d.min(axis=1))


def verdict(d_gen, d_val):
    ratio = float(np.median(d_gen) / np.median(d_val))
    frac_low = float((d_gen < np.percentile(d_val, 5)).mean())
    if ratio < 0.7 or frac_low > 0.25:
        v = "memorization suspected"
    elif ratio >= 0.9 and frac_low <= 0.10:
        v = "no sign of memorization"
    else:
        v = "borderline"
    return ratio, frac_low, v


def allocate(counts, n):
    counts = np.asarray(counts, dtype=float)
    frac = counts / counts.sum() * n
    base = np.floor(frac).astype(int)
    base[np.argsort(-(frac - base))[: int(n - base.sum())]] += 1
    return base

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(BEST, map_location=dev, weights_only=False)
args, mu, sd, bound = ck["args"], ck["mu"], ck["sd"], ck["bound"]
print(f"loaded {BEST}: epoch {ck['epoch']}, weights={ck['weights']}", flush=True)

d = np.load(DATA)
rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"]
rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"]
z_tr, z_va = (rna_tr - mu) / sd, (rna_va - mu) / sd

spec = ModalitySpec(dims={"rna": 768})
model = MoDiT(spec=spec, d_model=args["d_model"], depth=args["depth"], n_heads=args["heads"],
              tokens_per_modality=args["tokens_per_modality"], d_hpo=768, d_gene=256,
              n_tissues=8, p_uncond=args["p_uncond"]).to(dev)
model.load_state_dict(ck["model"])
model.eval()
diffusion = GaussianDiffusion(args["timesteps"]).to(dev)
B = torch.from_numpy(bound).float().to(dev)


@torch.no_grad()
def sample(cfg, labels, seed=SEED):
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((len(labels), 768), device=dev, generator=gen)
    ts = torch.linspace(diffusion.T - 1, 0, N_STEPS, device=dev).long()
    kw = dict(avail=torch.ones(len(labels), 1, device=dev), c_hpo=None, c_gene=None,
              tissue=torch.from_numpy(labels).long().to(dev))
    for i, t_cur in enumerate(ts):
        tb = t_cur.repeat(len(labels))
        ab_t = diffusion.alpha_bar[t_cur]
        t_next = ts[i + 1] if i + 1 < len(ts) else None
        ab_next = diffusion.alpha_bar[t_next] if t_next is not None else torch.tensor(1.0, device=dev)
        eps = diffusion._eps_with_cfg(model, x, tb, cfg, **kw)
        x0 = torch.maximum(torch.minimum((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt(), B), -B)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        x = x0 if t_next is None else ab_next.sqrt() * x0 + (1 - ab_next).clamp(min=0).sqrt() * eps
    return x.double().cpu().numpy()


classes, counts = np.unique(y_va, return_counts=True)
labels = np.repeat(classes, allocate(counts, N_GEN))
rng = np.random.default_rng(SEED)

print(f"\n{'='*70}\n1. NOVELTY: distance to the nearest TRAINING row (standardized space)\n{'='*70}", flush=True)
gen_cond = sample(1.0, labels)
gen_null = sample(0.0, np.zeros(N_GEN, dtype=np.int64))

d_val = nn_dist(z_va, z_tr)
loo_idx = rng.choice(len(z_tr), len(z_va), replace=False)
d_loo = nn_dist(z_tr[loo_idx], np.delete(z_tr, loo_idx, axis=0))
ref = np.zeros((N_GEN, z_tr.shape[1]))
lab_ref = labels
for c in np.unique(lab_ref):
    zc = z_tr[y_tr == c]
    m = lab_ref == c
    ref[m] = rng.multivariate_normal(zc.mean(0), np.cov(zc, rowvar=False), size=int(m.sum()),
                                     method="svd", check_valid="ignore")
d_ref = nn_dist(ref, z_tr)

rows = {"validation rows": d_val, "train leave-one-out": d_loo,
        "per-tissue Gaussian samples": d_ref,
        "model samples, conditional": nn_dist(gen_cond, z_tr),
        "model samples, null": nn_dist(gen_null, z_tr)}
print(f"{'set':>30} | {'n':>4} | {'min':>7} | {'5th pct':>7} | {'median':>7} | {'95th pct':>8}")
for k, v in rows.items():
    print(f"{k:>30} | {len(v):>4} | {v.min():>7.2f} | {np.percentile(v, 5):>7.2f} | {np.median(v):>7.2f} | {np.percentile(v, 95):>8.2f}")
res = {"nn_distance": {k: {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                           "min": float(v.min())} for k, v in rows.items()}}
for name, dg in (("conditional", rows["model samples, conditional"]), ("null", rows["model samples, null"]),
                 ("per-tissue Gaussian (reference)", d_ref)):
    ratio, frac_low, v = verdict(dg, d_val)
    print(f"  {name:>34}: median(d_gen)/median(d_val) = {ratio:.2f}; fraction closer than val 5th pct = {frac_low:.3f}  -> {v}")
    res.setdefault("verdict", {})[name] = {"ratio": ratio, "frac_below_val_p5": frac_low, "verdict": v}

print(f"\n{'='*70}\n2. OVERFITTING: teacher-forced epsilon-loss, training rows vs validation rows\n{'='*70}", flush=True)
sub = rng.choice(len(z_tr), len(z_va), replace=False)
Xs = {"train": (torch.from_numpy(z_tr[sub]).float().to(dev), torch.from_numpy(y_tr[sub]).long().to(dev)),
      "val": (torch.from_numpy(z_va).float().to(dev), torch.from_numpy(y_va).long().to(dev))}
print(f"{'t':>5} | {'train cond':>10} | {'val cond':>9} | {'gap x':>6} | {'train null':>10} | {'val null':>9}")
gap = {}
with torch.no_grad():
    for t_val in TS:
        out = {}
        for split, (X, Y) in Xs.items():
            g = torch.Generator(device=dev).manual_seed(1000 + t_val)
            eps = torch.randn(X.shape, device=dev, generator=g)
            ab = diffusion.alpha_bar[t_val]
            xt = ab.sqrt() * X + (1 - ab).sqrt() * eps
            tb = torch.full((len(X),), t_val, dtype=torch.long, device=dev)
            av = torch.ones(len(X), 1, device=dev)
            pc = model(xt, tb, avail=av, c_hpo=None, c_gene=None, tissue=Y, force_uncond=False)
            pn = model(xt, tb, avail=av, c_hpo=None, c_gene=None, tissue=Y, force_uncond=True)
            out[split] = (float(((pc - eps) ** 2).mean()), float(((pn - eps) ** 2).mean()))
        gap[t_val] = {"train_cond": out["train"][0], "val_cond": out["val"][0],
                      "train_null": out["train"][1], "val_null": out["val"][1]}
        print(f"{t_val:>5} | {out['train'][0]:>10.4f} | {out['val'][0]:>9.4f} | {out['val'][0]/max(out['train'][0],1e-9):>6.1f} | "
             f"{out['train'][1]:>10.4f} | {out['val'][1]:>9.4f}")
res["train_val_gap"] = gap

print("\nREADING")
v = res["verdict"]["conditional"]
print(f"  novelty (conditional): {v['verdict']} (ratio {v['ratio']:.2f}, {v['frac_below_val_p5']:.1%} of samples closer than the val 5th percentile)")
g50 = gap[50]["val_cond"] / max(gap[50]["train_cond"], 1e-9)
print(f"  overfitting at t=50: validation loss is {g50:.1f}x the training loss "
     f"({'a large gap: overfitting' if g50 > 2 else 'small gap'})")
with open(OUT_PATH, "w") as f:
    json.dump(res, f, indent=2)
print(f"saved -> {OUT_PATH}")
