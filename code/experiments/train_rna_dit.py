"""Train the RNA diffusion transformer on per-coordinate standardized embeddings with tissue conditioning; evaluate the fidelity criteria at fixed epochs."""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")

import numpy as np
import torch

from raresynth.model.dit import MoDiT, ModalitySpec
from raresynth.model.diffusion import EMA, GaussianDiffusion
from raresynth.schedule import get_lr
from raresynth.eval.fidelity import frechet_distance, prdc, c2st

GATES = {"c2st_max": 0.70, "precision_min": 0.50, "recall_min": 0.50, "norm_band_min": 0.90}
TS_GRID = [900, 500, 250, 100, 50, 10]

def linear_eps(xt, ab, mu, V, lam):
    a = np.sqrt(ab)
    s2 = 1.0 - ab
    s = np.sqrt(s2)
    z = (xt - a * mu) @ V
    return (s * z / (ab * lam + s2)) @ V.T


def fit_global(z):
    lam, V = np.linalg.eigh(np.cov(z, rowvar=False))
    return z.mean(axis=0), np.clip(lam, 0, None), V


def fit_tissue(z, y, shrink=0.2):
    cov_g = np.cov(z, rowvar=False)
    comps = {}
    for c in np.unique(y):
        zc = z[y == c]
        if len(zc) < 3:
            continue
        lam, V = np.linalg.eigh((1 - shrink) * np.cov(zc, rowvar=False) + shrink * cov_g)
        comps[int(c)] = (zc.mean(axis=0), np.clip(lam, 0, None), V)
    return comps


def benchmark_losses(z_val, y_val, ab, noise, glob, comps):
    a, s = np.sqrt(ab), np.sqrt(1.0 - ab)
    xt = a * z_val + s * noise
    mu, lam, V = glob
    e_g = linear_eps(xt, ab, mu, V, lam)
    e_t = e_g.copy()
    for c, (mu_c, lam_c, V_c) in comps.items():
        m = (y_val == c)
        if m.any():
            e_t[m] = linear_eps(xt[m], ab, mu_c, V_c, lam_c)
    return float(((e_g - noise) ** 2).mean()), float(((e_t - noise) ** 2).mean())


def allocate(counts, n):
    counts = np.asarray(counts, dtype=float)
    frac = counts / counts.sum() * n
    base = np.floor(frac).astype(int)
    base[np.argsort(-(frac - base))[: int(n - base.sum())]] += 1
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz")
    ap.add_argument("--out", default="/data/pduttapramanik/raresynth/runs/rna_dit_attempt1")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--p-uncond", type=float, default=0.2)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tokens-per-modality", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--audit-epochs", default="25,50,100,200,300,400,500,600,700,800")
    ap.add_argument("--n-gen", type=int, default=206)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preprocess", choices=["standardize", "global_scale"], default="standardize",
                    help="standardize = per-coordinate centre/scale (attempt 1). global_scale = the OLD preprocessing: "
                         "one scalar, no centring (ablation)")
    ap.add_argument("--global-scale", type=float, default=33.40122774789073,
                    help="the scalar used by the original pipeline (data_scale)")
    ap.add_argument("--fixed-clip4-audit", action="store_true",
                    help="also score conditional samples drawn with the old fixed clip of 4.0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    audit_epochs = {int(e) for e in args.audit_epochs.split(",") if e} | {args.epochs}

    d = np.load(args.data)
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"].astype(np.int64)
    rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"].astype(np.int64)
    assert np.abs(np.linalg.norm(rna_tr, axis=1) - 1).max() < 1e-3, "training rows are not all unit norm"
    if args.preprocess == "standardize":
        mu = rna_tr.mean(axis=0)
        sd = np.maximum(rna_tr.std(axis=0), 1e-4)
    else:
        mu = np.zeros(rna_tr.shape[1])
        sd = np.full(rna_tr.shape[1], 1.0 / args.global_scale)
    z_tr, z_va = (rna_tr - mu) / sd, (rna_va - mu) / sd
    bound = np.maximum(1.05 * np.abs(z_tr).max(axis=0), 3.0)
    print(f"train n={len(z_tr)}, val n={len(z_va)}; std floor hit in {(rna_tr.std(axis=0) < 1e-4).sum()} coords; "
         f"standardized |z| > 4: {(np.abs(z_tr) > 4).mean():.5f} of entries, max |z|={np.abs(z_tr).max():.2f}; "
         f"clip bound range [{bound.min():.2f}, {bound.max():.2f}]", flush=True)

    glob = fit_global(z_tr)
    comps = fit_tissue(z_tr, y_tr)
    classes, counts = np.unique(y_va, return_counts=True)
    audit_labels = np.repeat(classes, allocate(counts, args.n_gen))

    spec = ModalitySpec(dims={"rna": 768})
    model = MoDiT(spec=spec, d_model=args.d_model, depth=args.depth, n_heads=args.heads,
                  tokens_per_modality=args.tokens_per_modality, d_hpo=768, d_gene=256,
                  n_tissues=8, p_uncond=args.p_uncond).to(dev)
    print(f"MoDiT parameters: {model.n_params()/1e6:.1f}M", flush=True)
    diffusion = GaussianDiffusion(args.timesteps).to(dev)
    ema = EMA(model, args.ema_decay)
    eval_model = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    X = torch.from_numpy(z_tr).float().to(dev)
    Y = torch.from_numpy(y_tr).long().to(dev)
    Zv = torch.from_numpy(z_va).float().to(dev)
    Yv = torch.from_numpy(y_va).long().to(dev)
    B = torch.from_numpy(bound).float().to(dev)
    n = len(X)
    steps_per_epoch = n // args.batch_size
    total_steps = args.epochs * steps_per_epoch

    @torch.no_grad()
    def teacher_forced(m):
        out = {}
        for t_val in TS_GRID:
            gen = torch.Generator(device=dev).manual_seed(1000 + t_val)
            eps = torch.randn((len(Zv), 768), device=dev, generator=gen)
            ab = diffusion.alpha_bar[t_val]
            xt = ab.sqrt() * Zv + (1 - ab).sqrt() * eps
            tb = torch.full((len(Zv),), t_val, dtype=torch.long, device=dev)
            av = torch.ones(len(Zv), 1, device=dev)
            pc = m(xt, tb, avail=av, c_hpo=None, c_gene=None, tissue=Yv, force_uncond=False)
            pn = m(xt, tb, avail=av, c_hpo=None, c_gene=None, tissue=Yv, force_uncond=True)
            bg, bt = benchmark_losses(z_va, y_va, float(ab.item()), eps.double().cpu().numpy(), glob, comps)
            out[t_val] = {"model_cond": float(((pc - eps) ** 2).mean().item()),
                          "model_null": float(((pn - eps) ** 2).mean().item()),
                          "lin_global": bg, "lin_tissue": bt}
        return out

    @torch.no_grad()
    def sample(m, cfg, labels, seed=777, Bclip=None):
        Bc = B if Bclip is None else Bclip
        gen = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn((len(labels), 768), device=dev, generator=gen)
        ts = torch.linspace(diffusion.T - 1, 0, args.n_steps, device=dev).long()
        kw = dict(avail=torch.ones(len(labels), 1, device=dev), c_hpo=None, c_gene=None,
                  tissue=torch.from_numpy(labels).long().to(dev))
        for i, t_cur in enumerate(ts):
            tb = t_cur.repeat(len(labels))
            ab_t = diffusion.alpha_bar[t_cur]
            t_next = ts[i + 1] if i + 1 < len(ts) else None
            ab_next = diffusion.alpha_bar[t_next] if t_next is not None else torch.tensor(1.0, device=dev)
            eps = diffusion._eps_with_cfg(m, x, tb, cfg, **kw)
            x0 = torch.maximum(torch.minimum((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt(), Bc), -Bc)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            x = x0 if t_next is None else ab_next.sqrt() * x0 + (1 - ab_next).clamp(min=0).sqrt() * eps
        return x.double().cpu().numpy() * sd + mu

    def score(real_pool, fake, seed0=0):
        nrm = np.linalg.norm(fake, axis=1)
        band = float(((nrm >= 0.8) & (nrm <= 1.2)).mean())
        rng = np.random.default_rng(777)
        runs = []
        for r in range(3):
            real = real_pool[rng.choice(len(real_pool), min(len(fake), len(real_pool)), replace=False)]
            pr = prdc(real, fake, k=5)
            runs.append({"fid": float(frechet_distance(real, fake)), "precision": float(pr["precision"]),
                         "recall": float(pr["recall"]), "coverage": float(pr["coverage"]),
                         "c2st": float(c2st(real, fake, seed=seed0 + r, n_permutations=0)["c2st_accuracy"])})
        m = {k: float(np.mean([x[k] for x in runs])) for k in runs[0]}
        m.update({"norm_median": float(np.median(nrm)), "norm_band": band})
        passed = [m["c2st"] <= GATES["c2st_max"], m["precision"] >= GATES["precision_min"],
                  m["recall"] >= GATES["recall_min"], band >= GATES["norm_band_min"]]
        m["gates_passed"] = int(sum(passed))
        return m

    audit_log, best = [], {"score": (-1, -9.0)}

    def audit(ep, train_loss):
        t0 = time.time()
        rec = {"epoch": ep, "train_loss": train_loss, "weights": {}}
        for wname in ("raw", "ema"):
            if wname == "raw":
                eval_model.load_state_dict(model.state_dict())
            else:
                eval_model.load_state_dict(ema.shadow)
            eval_model.eval()
            tf = teacher_forced(eval_model)
            null = score(rna_va, sample(eval_model, 0.0, np.zeros(args.n_gen, dtype=np.int64)))
            cond = score(rna_va, sample(eval_model, 1.0, audit_labels))
            rec["weights"][wname] = {"teacher_forced": tf, "null": null, "cond": cond}
            if args.fixed_clip4_audit:
                cond4 = score(rna_va, sample(eval_model, 1.0, audit_labels, Bclip=torch.full((768,), 4.0, device=dev)))
                rec["weights"][wname]["cond_clip4"] = cond4
                print(f"  [{wname}] cond with the OLD fixed clip 4.0: norm med={cond4['norm_median']:.3f} "
                     f"band={cond4['norm_band']:.2f} prec={cond4['precision']:.3f} rec={cond4['recall']:.3f} "
                     f"C2ST={cond4['c2st']:.3f}  gates {cond4['gates_passed']}/4", flush=True)
            tfs = "  ".join(f"t={t}: {v['model_cond']:.4f}/{v['lin_global']:.4f}/{v['lin_tissue']:.4f}"
                            for t, v in tf.items() if t in (500, 250, 50))
            print(f"  [{wname}] teacher-forced model_cond/lin_global/lin_tissue  {tfs}")
            for mode, m in (("null", null), ("cond", cond)):
                print(f"  [{wname}] {mode:>4}: norm med={m['norm_median']:.3f} band={m['norm_band']:.2f} "
                     f"FID={m['fid']:.4f} prec={m['precision']:.3f} rec={m['recall']:.3f} "
                     f"cov={m['coverage']:.3f} C2ST={m['c2st']:.3f}  gates {m['gates_passed']}/4", flush=True)
            sc = (cond["gates_passed"], -cond["c2st"])
            if sc > best["score"]:
                best.update({"score": sc, "epoch": ep, "weights": wname})
                torch.save({"model": eval_model.state_dict(), "mu": mu, "sd": sd, "bound": bound,
                            "args": vars(args), "epoch": ep, "weights": wname, "audit": rec["weights"][wname]},
                           os.path.join(args.out, "best.pt"))
        audit_log.append(rec)
        with open(os.path.join(args.out, "audit_log.json"), "w") as f:
            json.dump(audit_log, f, indent=2)
        print(f"  audit at epoch {ep} done in {time.time()-t0:.0f}s; best so far: epoch {best.get('epoch')} "
             f"({best.get('weights')}), gates {best['score'][0]}/4, cond C2ST {-best['score'][1]:.3f}", flush=True)

    print(f"training: {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps", flush=True)
    step, clipped = 0, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        ep_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            for g in opt.param_groups:
                g["lr"] = get_lr(step, total_steps, args.lr, args.warmup, args.min_lr_frac)
            opt.zero_grad(set_to_none=True)
            loss, _ = diffusion.training_loss(model, X[idx], avail=torch.ones(len(idx), 1, device=dev),
                                              c_hpo=None, c_gene=None, tissue=Y[idx], loss_weights=None)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            clipped += int(gn.item() > 1.0)
            opt.step()
            ema.update(model)
            ep_loss += loss.item()
            step += 1
        ep_loss /= steps_per_epoch
        if ep == 1 or ep % 10 == 0:
            print(f"[{ep:4d}] loss {ep_loss:.5f}  grad-clip rate {clipped/step:.2f}  ({time.time()-t_start:.0f}s elapsed)", flush=True)
        if ep in audit_epochs:
            audit(ep, ep_loss)
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "mu": mu, "sd": sd, "bound": bound,
                        "args": vars(args), "epoch": ep}, os.path.join(args.out, "last.pt"))

    print(f"\nDONE. best: epoch {best.get('epoch')} ({best.get('weights')} weights), gates passed {best['score'][0]}/4, "
         f"conditional C2ST {-best['score'][1]:.3f}  -> {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
