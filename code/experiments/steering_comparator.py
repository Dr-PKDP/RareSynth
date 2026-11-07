"""Guided generators and Gaussian steerers (translation and gradient ascent) matched to the same mean cosine, scored with the fidelity criteria."""

import argparse
import json
import os
import sys
import time

import numpy as np

CODE = "/data/pduttapramanik/raresynth/code"
RUNS = "/data/pduttapramanik/raresynth/runs"
DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
GTEX_BASELINE = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/fibroblast_baseline.npz"
OUT_DIR = f"{RUNS}/steering_comparator"
RESULT_PATH = f"{OUT_DIR}/result.json"
SAMPLES_PATH = f"{OUT_DIR}/samples_seed0.npz"
CKPTS = {0: f"{RUNS}/rna_dit_attempt1/best.pt",
         1: f"{RUNS}/rna_dit_attempt1_seed1/best.pt",
         2: f"{RUNS}/rna_dit_attempt1_seed2/best.pt"}
SCALES = [150, 200, 300]
GATES = {"c2st_max": 0.70, "precision_min": 0.50, "recall_min": 0.50, "norm_band_min": 0.90}
N_GEN, REPEATS, NOISE_SEED0, GAUSS_SEED0 = 206, 3, 6100, 7100
MATCH_TOL_BASE = 0.02
LABEL = "validation rows only; test array not opened"

def allocate(counts, n):
    counts = np.asarray(counts, dtype=float)
    frac = counts / counts.sum() * n
    base = np.floor(frac).astype(int)
    base[np.argsort(-(frac - base))[: int(n - base.sum())]] += 1
    return base


def cos_to(x, b, dh):
    u = x - b
    return (u @ dh) / (np.linalg.norm(u, axis=1) + 1e-12)


def cos_grad(x, b, dh):
    u = x - b
    nu = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    c = (u @ dh)[:, None] / nu
    return (dh[None, :] - c * u / nu) / nu


def steer_translate(x, b, dh, lam):
    return x + lam * dh[None, :]


def steer_ascent(x, b, dh, eta, steps=20):
    y = x.copy()
    for _ in range(steps):
        y = y + eta * cos_grad(y, b, dh)
    return y


def match_mean_cos(steer, x, b, dh, target, hi0=1.0, max_double=40, iters=80):
    f = lambda p: float(cos_to(steer(x, p), b, dh).mean())
    base = f(0.0)
    if base >= target:
        return 0.0, base, False
    hi = hi0
    for _ in range(max_double):
        if f(hi) >= target:
            break
        hi *= 2.0
    else:
        return hi, f(hi), False
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return p, f(p), True


def sample_gaussian(mu, cov, n, rng):
    return rng.multivariate_normal(mu, cov, size=n, method="svd", check_valid="ignore")


def fit_tissue_gaussians(x, labels):
    comps = {}
    for c in np.unique(labels):
        xc = x[labels == c]
        if len(xc) >= 3:
            comps[int(c)] = (xc.mean(axis=0), np.cov(xc, rowvar=False), len(xc))
    return comps


def sample_by_labels(comps, labels, rng):
    out = np.empty((len(labels), next(iter(comps.values()))[0].shape[0]))
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        mu, cov, _ = comps[int(k)]
        out[idx] = sample_gaussian(mu, cov, len(idx), rng)
    return out


def fidelity_once(real, fake, seed, fid):
    pr = fid["prdc"](real, fake, k=5)
    return {"fid": float(fid["frechet"](real, fake)), "precision": float(pr["precision"]),
            "recall": float(pr["recall"]), "coverage": float(pr["coverage"]),
            "c2st": float(fid["c2st"](real, fake, seed=seed, n_permutations=0)["c2st_accuracy"])}


def summarize_runs(runs, fake):
    m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    nrm = np.linalg.norm(fake, axis=1)
    m["norm_median"] = float(np.median(nrm))
    m["norm_band"] = float(((nrm >= 0.8) & (nrm <= 1.2)).mean())
    fails = []
    if not m["c2st"] <= GATES["c2st_max"]:
        fails.append("C2ST")
    if not m["precision"] >= GATES["precision_min"]:
        fails.append("precision")
    if not m["recall"] >= GATES["recall_min"]:
        fails.append("recall")
    if not m["norm_band"] >= GATES["norm_band_min"]:
        fails.append("norm band")
    m["gates_passed"] = 4 - len(fails)
    m["failed_gates"] = fails
    return m


def score_fake(real_pool, fake, fid, rng_seed=777):
    rng = np.random.default_rng(rng_seed)
    runs = []
    for r in range(REPEATS):
        real = real_pool[rng.choice(len(real_pool), min(len(fake), len(real_pool)), replace=False)]
        runs.append(fidelity_once(real, fake, r, fid))
    return summarize_runs(runs, fake)


def c2_reading(deltas):
    m = float(np.mean(deltas))
    verdict = ("generator closer to real data" if m <= -0.05 else
               "steered Gaussian closer to real data" if m >= 0.05 else "no distinguishable difference")
    return {"mean": m, "min": float(np.min(deltas)), "max": float(np.max(deltas)), "reading": verdict}


def get_fidelity():
    sys.path.insert(0, CODE)
    from raresynth.eval.fidelity import frechet_distance, prdc, c2st
    return {"frechet": frechet_distance, "prdc": prdc, "c2st": c2st}


def load_generator(path, dev):
    import torch
    sys.path.insert(0, CODE)
    from raresynth.model.dit import MoDiT, ModalitySpec
    from raresynth.model.diffusion import GaussianDiffusion
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    assert ck["epoch"] == a["epochs"] == 800 and ck["weights"] == "ema", \
        f"{path}: expected the epoch-800 EMA checkpoint, got epoch {ck['epoch']} {ck['weights']}"
    spec = ModalitySpec(dims={"rna": 768})
    model = MoDiT(spec=spec, d_model=a["d_model"], depth=a["depth"], n_heads=a["heads"],
                  tokens_per_modality=a["tokens_per_modality"], d_hpo=768, d_gene=256,
                  n_tissues=8, p_uncond=a["p_uncond"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return {"model": model, "diffusion": GaussianDiffusion(a["timesteps"]).to(dev), "dev": dev,
            "B": torch.from_numpy(ck["bound"]).float().to(dev), "mu": ck["mu"], "sd": ck["sd"],
            "n_steps": a["n_steps"]}


def make_energy(gen, direction, baseline):
    import torch
    sys.path.insert(0, CODE)
    from raresynth.model.guidance import _cos_energy
    dev = gen["dev"]
    d = torch.from_numpy(direction).float().to(dev)
    mu_t = torch.from_numpy(gen["mu"]).float().to(dev)
    sd_t = torch.from_numpy(gen["sd"]).float().to(dev)
    b_t = torch.from_numpy(baseline).float().to(dev)

    def energy(z):
        delta = (z * sd_t + mu_t) - b_t
        return _cos_energy(delta, d.unsqueeze(0).expand_as(delta))
    return energy


def generate(gen, energy, scale, cfg, labels, seed, n_steps=None):
    import torch
    dev, model, diffusion, B = gen["dev"], gen["model"], gen["diffusion"], gen["B"]
    n = len(labels)
    n_steps = n_steps or gen["n_steps"]
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((n, 768), device=dev, generator=g)
    ts = torch.linspace(diffusion.T - 1, 0, n_steps, device=dev).long()
    kw = dict(avail=torch.ones(n, 1, device=dev), c_hpo=None, c_gene=None,
              tissue=torch.from_numpy(labels).long().to(dev))
    for i, t_cur in enumerate(ts):
        tb = t_cur.repeat(n)
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
    return x.double().cpu().numpy() * gen["sd"] + gen["mu"]


def fmt(m):
    return (f"C2ST {m['c2st']:.3f} | prec {m['precision']:.3f} | rec {m['recall']:.3f} | cov {m['coverage']:.3f} | "
            f"band {m['norm_band']:.2f} | FID {m['fid']:.4f} | gates {m['gates_passed']}/4"
            + (f" (fails: {', '.join(m['failed_gates'])})" if m["failed_gates"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fid = get_fidelity()
    print(f"EXPLORATORY-DESIGN COMPARATOR ({LABEL}); device {dev}\n", flush=True)

    d = np.load(DATA)
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"].astype(np.int64)
    rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"].astype(np.int64)
    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    dh = d_true / np.linalg.norm(d_true)
    b = np.load(GTEX_BASELINE)["baseline"].astype(np.float64)

    seeds = [0] if args.smoke else sorted(CKPTS)
    scales = [150] if args.smoke else SCALES
    n_gen = 16 if args.smoke else N_GEN
    n_steps = 20 if args.smoke else None
    classes, counts = np.unique(y_va, return_counts=True)
    labels_cond = np.repeat(classes, allocate(counts, n_gen))
    modes = {"null": (0.0, np.zeros(n_gen, dtype=np.int64)), "cond": (1.0, labels_cond)}

    mu_g, cov_g = rna_tr.mean(axis=0), np.cov(rna_tr, rowvar=False)
    comps = fit_tissue_gaussians(rna_tr, y_tr)
    results, samples = {}, {}

    for s in seeds:
        gen = load_generator(CKPTS[s], dev)
        energy = make_energy(gen, d_true, b)
        for mode, (cfg, labels) in modes.items():
            t0 = time.time()
            x_un = generate(gen, None, 0.0, cfg, labels, NOISE_SEED0 + s, n_steps)
            sc_un = score_fake(rna_va, x_un, fid)
            c_un = float(cos_to(x_un, b, dh).mean())
            rng = np.random.default_rng(GAUSS_SEED0 + s)
            base = (sample_gaussian(mu_g, cov_g, n_gen, rng) if mode == "null"
                    else sample_by_labels(comps, labels, rng))
            sc_base = score_fake(rna_va, base, fid)
            c_base = float(cos_to(base, b, dh).mean())
            print(f"[seed {s} | {mode}] unguided generator: mean cos {c_un:+.4f} | {fmt(sc_un)}\n"
                  f"{'':>16}Gaussian base:      mean cos {c_base:+.4f} | {fmt(sc_base)}  ({time.time()-t0:.0f}s)", flush=True)
            if s == 0:
                samples[f"{mode}_unguided_generator"] = x_un
                samples[f"{mode}_gaussian_base"] = base
                samples[f"{mode}_labels"] = labels
            results[f"{s}|{mode}|unguided"] = {"generator": {**sc_un, "mean_cos": c_un},
                                               "gaussian_base": {**sc_base, "mean_cos": c_base}}
            for scale in scales:
                t0 = time.time()
                x_g = generate(gen, energy, float(scale), cfg, labels, NOISE_SEED0 + s, n_steps)
                sc_g = score_fake(rna_va, x_g, fid)
                target = float(cos_to(x_g, b, dh).mean())
                row = {"generator_guided": {**sc_g, "mean_cos": target}}
                print(f"[seed {s} | {mode} | scale {scale}] guided generator: mean cos {target:+.4f} | {fmt(sc_g)}", flush=True)
                for name, steer in (("T_translate", lambda x, p: steer_translate(x, b, dh, p)),
                                    ("G_ascent", lambda x, p: steer_ascent(x, b, dh, p))):
                    p, got, reached = match_mean_cos(steer, base, b, dh, target)
                    x_s = steer(base, p)
                    sc_s = score_fake(rna_va, x_s, fid)
                    row[name] = {**sc_s, "param": float(p), "mean_cos": float(got), "reached": bool(reached)}
                    print(f"{'':>34}{name:>12}: mean cos {got:+.4f} (param {p:.4g}{'' if reached else ', NOT MATCHED'}) | {fmt(sc_s)}", flush=True)
                    if s == 0:
                        samples[f"{mode}_{scale}_{name}"] = x_s
                if s == 0:
                    samples[f"{mode}_{scale}_generator_guided"] = x_g
                row["base_cos_gap"] = float(c_base - c_un)
                results[f"{s}|{mode}|{scale}"] = row
                print(f"{'':>34}({time.time()-t0:.0f}s)", flush=True)
        del gen
        if args.smoke:
            print("smoke run finished (nothing saved)")
            return

    print(f"\n{'='*100}\nSUMMARY ({LABEL})\n{'='*100}")
    summary = {}
    for mode in modes:
        for scale in scales:
            rows = [results[f"{s}|{mode}|{scale}"] for s in seeds]
            n4 = sum(r["generator_guided"]["gates_passed"] == 4 for r in rows)
            failing = sorted({g for r in rows for g in r["generator_guided"]["failed_gates"]})
            entry = {"C1_seeds_4_of_4": n4, "C1_failed_gates": failing,
                     "mean_cos": float(np.mean([r["generator_guided"]["mean_cos"] for r in rows]))}
            gap = float(np.mean([abs(r["base_cos_gap"]) for r in rows]))
            entry["base_cos_gap_abs_mean"] = gap
            line = (f"{mode:>4} scale {scale}: achieved cos {entry['mean_cos']:+.3f}; C1 {n4}/{len(seeds)} seeds 4/4"
                    + (f" (failing: {', '.join(failing)})" if failing else ""))
            for name in ("T_translate", "G_ascent"):
                matched = all(r[name]["reached"] for r in rows)
                rd = c2_reading([r["generator_guided"]["c2st"] - r[name]["c2st"] for r in rows])
                rd["gate_counts"] = {"generator": [r["generator_guided"]["gates_passed"] for r in rows],
                                     "steered": [r[name]["gates_passed"] for r in rows]}
                rd["matched_all_seeds"] = matched
                rd["flag_C3"] = bool(gap > MATCH_TOL_BASE or not matched)
                entry[name] = rd
                line += (f"\n      C2 vs {name}: dC2ST {rd['mean']:+.3f} [{rd['min']:+.3f}, {rd['max']:+.3f}] -> "
                         + (f"FLAGGED (C3), not interpreted" if rd["flag_C3"] else rd["reading"])
                         + f"; gates generator {rd['gate_counts']['generator']} vs steered {rd['gate_counts']['steered']}")
            summary[f"{mode}|{scale}"] = entry
            print(line)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"label": LABEL, "gates": GATES, "scales": scales, "n": n_gen, "noise_seed0": NOISE_SEED0,
               "gauss_seed0": GAUSS_SEED0, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "summary": summary, "rows": results}, open(RESULT_PATH, "w"), indent=2)
    np.savez_compressed(SAMPLES_PATH, **samples)
    print(f"\nsaved -> {RESULT_PATH}\nsaved -> {SAMPLES_PATH} (seed 0 samples, for independent_metric_check.py --samples)")


if __name__ == "__main__":
    main()
