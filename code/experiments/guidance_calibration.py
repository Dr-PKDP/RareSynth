"""Paired-bootstrap calibration of the guidance scale against the eligibility criteria."""

import argparse
import json
import sys
import time

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")

import numpy as np
import torch

from raresynth.model.dit import MoDiT, ModalitySpec
from raresynth.model.diffusion import GaussianDiffusion
from raresynth.model.guidance import _cos_energy
from raresynth.eval.fidelity import prdc

PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
GTEX_BASELINE = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/fibroblast_baseline.npz"
BEST = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/best.pt"
DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
OUT_PATH = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/guidance_calibration.json"
SCALES = [100, 150, 200, 300, 400, 600, 800]
SEED = 5000
N_BOOT = 2000
CRIT = {"own_lcb": 0.10, "spill_abs": 0.05, "prec_loss_ucb": 0.10, "nn_growth_ucb": 0.15, "norm_growth_ucb": 0.15}

def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cos_rows(u, dhat):
    return (u @ dhat) / (np.linalg.norm(u, axis=1) + 1e-12)


def stratified_shuffle(d, sd, block=32, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(sd)
    out = d.copy()
    for start in range(0, len(d), block):
        idx = order[start:start + block]
        out[idx] = d[rng.permutation(idx)]
    return out


def sqdist(a, b):
    return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def nn_dist(a, b):
    return np.sqrt(sqdist(a, b).min(axis=1))


def precision_indicators(real, fake, k=5):
    d_rr = np.sqrt(sqdist(real, real))
    np.fill_diagonal(d_rr, np.inf)
    radii = np.sort(d_rr, axis=1)[:, k - 1]
    return (np.sqrt(sqdist(fake, real)) <= radii[None, :]).any(axis=1).astype(float)


def allocate(counts, n):
    counts = np.asarray(counts, dtype=float)
    frac = counts / counts.sum() * n
    base = np.floor(frac).astype(int)
    base[np.argsort(-(frac - base))[: int(n - base.sum())]] += 1
    return base


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bootstrap(guided, controls, unguided, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(unguided["cos_true"])
    idx = rng.integers(0, n, size=(n_boot, n))
    m = lambda a: a[idx].mean(axis=1)
    own = m(guided["cos_true"]) - m(unguided["cos_true"])
    prec_loss = m(unguided["prec"]) - m(guided["prec"])
    nn_growth = np.median(guided["nn"][idx], axis=1) / np.median(unguided["nn"][idx], axis=1) - 1.0
    norm_growth = m(guided["norm"]) / m(unguided["norm"]) - 1.0
    out = {"own": ci(own), "prec_loss": ci(prec_loss), "nn_growth": ci(nn_growth), "norm_growth": ci(norm_growth),
           "own_mean": float(own.mean())}
    for name, c in controls.items():
        sp = m(c["cos_true"]) - m(unguided["cos_true"])
        lo, hi = ci(sp)
        out[f"spill_{name}"] = (lo, hi)
        out[f"spill_{name}_abs"] = max(abs(lo), abs(hi))
    return out


def eligibility(b, crit):
    e1 = b["own"][0] > crit["own_lcb"]
    e2 = all(b[k] < crit["spill_abs"] for k in b if k.startswith("spill_") and k.endswith("_abs"))
    e3 = b["prec_loss"][1] < crit["prec_loss_ucb"]
    e4 = b["nn_growth"][1] < crit["nn_growth_ucb"] and b["norm_growth"][1] < crit["norm_growth_ucb"]
    return {"E1": bool(e1), "E2": bool(e2), "E3": bool(e3), "E4": bool(e4), "all": bool(e1 and e2 and e3 and e4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--best", default=BEST, help="checkpoint to calibrate (default: attempt 1, seed 0)")
    ap.add_argument("--out", default=OUT_PATH, help="where to write the calibration JSON")
    args = ap.parse_args()
    n_gen = 16 if args.smoke else 200
    n_steps = 20 if args.smoke else 200
    n_boot = 100 if args.smoke else N_BOOT
    scales = [100] if args.smoke else SCALES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.best, map_location=dev, weights_only=False)
    a, mu, sd, bound = ck["args"], ck["mu"], ck["sd"], ck["bound"]
    d = np.load(DATA)
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"]
    rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"]
    z_tr = (rna_tr - mu) / sd

    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    d_shuf = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance_SHUFFLED_CONTROL.npz")["rna_embedding_guidance"].astype(np.float64)
    d_strat = stratified_shuffle(d_true, sd)
    b = np.load(GTEX_BASELINE)["baseline"].astype(np.float64)
    dh_true = d_true / np.linalg.norm(d_true)

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

    energies = {"true": RnaEnergy(d_true), "shuf": RnaEnergy(d_shuf), "strat": RnaEnergy(d_strat)}

    def generate(scale, energy, cfg, labels):
        g = torch.Generator(device=dev).manual_seed(SEED)
        x = torch.randn((n_gen, 768), device=dev, generator=g)
        ts = torch.linspace(diffusion.T - 1, 0, n_steps, device=dev).long()
        kw = dict(avail=torch.ones(n_gen, 1, device=dev), c_hpo=None, c_gene=None,
                  tissue=torch.from_numpy(labels).long().to(dev))
        for i, t_cur in enumerate(ts):
            tb = t_cur.repeat(n_gen)
            ab_t = diffusion.alpha_bar[t_cur]
            t_next = ts[i + 1] if i + 1 < len(ts) else None
            ab_next = diffusion.alpha_bar[t_next] if t_next is not None else torch.tensor(1.0, device=dev)
            with torch.no_grad():
                eps = diffusion._eps_with_cfg(model, x, tb, cfg, **kw)
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

    def per_sample(x):
        return {"cos_true": cos_rows(x - b, dh_true), "norm": np.linalg.norm(x, axis=1),
                "nn": nn_dist((x - mu) / sd, z_tr), "prec": precision_indicators(rna_va, x)}

    classes, counts = np.unique(y_va, return_counts=True)
    modes = {"null": (0.0, np.zeros(n_gen, dtype=np.int64)),
             "cond_tissue_mix": (1.0, np.repeat(classes, allocate(counts, n_gen)))}
    all_results = {"criteria": CRIT, "scales": scales, "n": n_gen, "n_boot": n_boot}

    for mode, (cfg, labels) in modes.items():
        print(f"\n{'='*100}\nMODE: {mode} (cfg {cfg}), n={n_gen}, {n_boot} paired bootstrap resamples\n{'='*100}", flush=True)
        t0 = time.time()
        x_u = generate(0.0, None, cfg, labels)
        u = per_sample(x_u)
        chk = float(prdc(rna_va[: min(len(rna_va), 206)], x_u, k=5)["precision"]) \
            - float(precision_indicators(rna_va[: min(len(rna_va), 206)], x_u).mean())
        print(f"unguided: mean cos(d_true) = {u['cos_true'].mean():+.4f}, precision = {u['prec'].mean():.3f}, "
             f"median nn = {np.median(u['nn']):.2f}, mean norm = {u['norm'].mean():.3f} "
             f"[precision indicator vs prdc difference {chk:+.2e}] ({time.time()-t0:.0f}s)", flush=True)
        mode_res = {"unguided": {k: float(v.mean()) for k, v in u.items()}, "scales": {}}
        print(f"{'scale':>6} | {'own change [95% CI]':>26} | {'spill shuf |.|':>14} | {'spill strat |.|':>15} | "
             f"{'prec loss UCB':>13} | {'nn growth UCB':>13} | {'norm growth UCB':>15} | E1 E2 E3 E4 | eligible")
        for scale in scales:
            g = per_sample(generate(float(scale), energies["true"], cfg, labels))
            controls = {"shuf": per_sample(generate(float(scale), energies["shuf"], cfg, labels)),
                        "strat": per_sample(generate(float(scale), energies["strat"], cfg, labels))}
            bs = bootstrap(g, controls, u, n_boot, seed=scale)
            el = eligibility(bs, CRIT)
            mode_res["scales"][str(scale)] = {"bootstrap": bs, "eligible": el, "guided_means":
                                              {k: float(v.mean()) for k, v in g.items()}}
            print(f"{scale:>6} | {bs['own_mean']:>+8.4f} [{bs['own'][0]:+.4f}, {bs['own'][1]:+.4f}] | "
                 f"{bs['spill_shuf_abs']:>14.4f} | {bs['spill_strat_abs']:>15.4f} | {bs['prec_loss'][1]:>+13.4f} | "
                 f"{bs['nn_growth'][1]:>+13.4f} | {bs['norm_growth'][1]:>+15.4f} | "
                 f"{' Y ' if el['E1'] else ' n '} {' Y ' if el['E2'] else ' n '} {' Y ' if el['E3'] else ' n '} "
                 f"{' Y ' if el['E4'] else ' n '} | {'YES' if el['all'] else 'no'}", flush=True)
        ok = [s for s in scales if mode_res["scales"][str(s)]["eligible"]["all"]]
        mode_res["selected_scale"] = ok[0] if ok else None
        print(f"\n  {mode}: eligible scales {ok}; SELECTED (smallest eligible) = {mode_res['selected_scale']}")
        all_results[mode] = mode_res

    print("\nREADING: the null-mode selection is the protocol's guidance scale. If no scale is eligible, the "
         "failing criterion column (E1 efficacy, E2 specificity, E3 realism, E4 geometry) shows why.")
    if not args.smoke:
        with open(args.out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"saved -> {args.out}")
    else:
        print("smoke run finished (nothing saved)")


if __name__ == "__main__":
    main()
