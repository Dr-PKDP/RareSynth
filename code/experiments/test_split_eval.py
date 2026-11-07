"""Evaluation of the three trained generators on the TCGA test split: a validation-row harness, then a single final run."""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CODE = "/data/pduttapramanik/raresynth/code"
RUNS = "/data/pduttapramanik/raresynth/runs"
DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
TEST_NPZ = "/data/pduttapramanik/raresynth/data/derived/rna_test_array.npz"
OUT_DIR = f"{RUNS}/test_split_eval"
HARNESS_PATH = f"{OUT_DIR}/harness_result.json"
RESULT_PATH = f"{OUT_DIR}/test_split_result.json"
CKPTS = {0: f"{RUNS}/rna_dit_attempt1/best.pt",
         1: f"{RUNS}/rna_dit_attempt1_seed1/best.pt",
         2: f"{RUNS}/rna_dit_attempt1_seed2/best.pt"}
VAL_BASELINE_FILES = {"gaussian": f"{RUNS}/cfg_sweep/gaussian_baselines.json",
                      "real": f"{RUNS}/cfg_sweep/real_data_baselines.json"}
GATES = {"c2st_max": 0.70, "precision_min": 0.50, "recall_min": 0.50, "norm_band_min": 0.90}
VAL_SEED, FINAL_SEED, N_GEN, REPEATS = 777, 9000, 206, 3
TOL = 0.005
FID_REL_TOL = 0.02
LABEL = "EXPLORATORY: TCGA test split not verifiable as untouched; generators never trained or selected on it"

def allocate(counts, n):
    counts = np.asarray(counts, dtype=float)
    frac = counts / counts.sum() * n
    base = np.floor(frac).astype(int)
    base[np.argsort(-(frac - base))[: int(n - base.sum())]] += 1
    return base


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fidelity_once(real, fake, seed, fid):
    pr = fid["prdc"](real, fake, k=5)
    return {"fid": float(fid["frechet"](real, fake)), "precision": float(pr["precision"]),
            "recall": float(pr["recall"]), "coverage": float(pr["coverage"]),
            "c2st": float(fid["c2st"](real, fake, seed=seed, n_permutations=0)["c2st_accuracy"])}


def summarize_runs(runs, fake_list):
    m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    nrm = [np.linalg.norm(f, axis=1) for f in fake_list]
    m["norm_median"] = float(np.mean([np.median(x) for x in nrm]))
    m["norm_band"] = float(np.mean([((x >= 0.8) & (x <= 1.2)).mean() for x in nrm]))
    passed = [m["c2st"] <= GATES["c2st_max"], m["precision"] >= GATES["precision_min"],
              m["recall"] >= GATES["recall_min"], m["norm_band"] >= GATES["norm_band_min"]]
    m["gates_passed"] = int(sum(passed))
    return m


def score_fake(real_pool, fake, fid, rng_seed=777, seed0=0):
    rng = np.random.default_rng(rng_seed)
    runs = []
    for r in range(REPEATS):
        real = real_pool[rng.choice(len(real_pool), min(len(fake), len(real_pool)), replace=False)]
        runs.append(fidelity_once(real, fake, seed0 + r, fid))
    return summarize_runs(runs, [fake] * REPEATS)


def baseline_scores(draw, pool, fid, n=N_GEN, seed=0, disjoint=False):
    rng = np.random.default_rng(seed)
    runs, fakes = [], []
    for r in range(REPEATS):
        if disjoint:
            perm = rng.permutation(len(pool))
            fake, real = pool[perm[:n]], pool[perm[n:2 * n]]
        else:
            fake = draw(rng)
            real = pool[rng.choice(len(pool), n, replace=False)]
        fakes.append(fake)
        runs.append(fidelity_once(real, fake, r, fid))
    return summarize_runs(runs, fakes)


def sample_gaussian(mu, cov, n, rng):
    return rng.multivariate_normal(mu, cov, size=n, method="svd", check_valid="ignore")


def fit_tissue_gaussians(x, labels):
    comps = {}
    for c in np.unique(labels):
        xc = x[labels == c]
        if len(xc) >= 3:
            comps[int(c)] = (xc.mean(axis=0), np.cov(xc, rowvar=False), len(xc))
    return comps


def sample_tissue_mixture(comps, n, rng):
    keys = list(comps.keys())
    w = np.array([comps[k][2] for k in keys], dtype=float)
    w /= w.sum()
    picks = rng.choice(len(keys), size=n, p=w)
    out = np.empty((n, comps[keys[0]][0].shape[0]))
    for j, k in enumerate(keys):
        idx = np.where(picks == j)[0]
        if len(idx):
            mu, cov, _ = comps[k]
            out[idx] = sample_gaussian(mu, cov, len(idx), rng)
    return out


def compare_to_recorded(new, rec):
    diffs, ok = {}, True
    for k in ("c2st", "precision", "recall", "coverage", "norm_band"):
        diffs[k] = float(new[k] - rec[k])
        ok &= abs(diffs[k]) <= TOL
    diffs["fid_rel"] = float(abs(new["fid"] - rec["fid"]) / max(abs(rec["fid"]), 1e-9))
    ok &= diffs["fid_rel"] <= FID_REL_TOL
    return {"diffs": diffs, "pass": bool(ok)}


def read_results(test, val, floor_test, floor_val, tissue_gauss_test):
    per = {}
    for s in test:
        dc = test[s]["cond"]["c2st"] - val[s]["cond"]["c2st"]
        per[str(s)] = {"conditional_gates_passed": int(test[s]["cond"]["gates_passed"]),
                       "delta_c2st_cond": float(dc),
                       "delta_c2st_null": float(test[s]["null"]["c2st"] - val[s]["null"]["c2st"]),
                       "c2st_minus_tissue_gaussian": float(test[s]["cond"]["c2st"] - tissue_gauss_test["c2st"])}
    n_pass = sum(1 for v in per.values() if v["conditional_gates_passed"] == 4)
    mx = max(abs(v["delta_c2st_cond"]) for v in per.values())
    drift = ("no sign of a validation-selection effect" if mx <= 0.05 else
             "modest difference (selection effect or split difference, not separable)" if mx <= 0.10 else
             "substantial difference")
    out = {"R1_seeds_with_4_of_4_conditional": n_pass, "R2_max_abs_delta_c2st": float(mx), "R2_reading": drift,
           "per_seed": per}
    if floor_val is not None:
        out["floor_c2st_change_test_minus_val"] = float(floor_test["c2st"] - floor_val["c2st"])
    return out


def self_sha(path=None):
    return hashlib.sha256(open(path or __file__, "rb").read()).hexdigest()


def final_problems(expect_sha, script_sha, harness_path, result_path, test_npz):
    problems = []
    if expect_sha != script_sha:
        problems.append(f"--expect-sha does not match this script ({script_sha})")
    if os.path.exists(result_path):
        problems.append(f"the result file already exists: {result_path} (single use)")
    if not os.path.exists(test_npz):
        problems.append(f"the test array is missing: {test_npz} (run export_rna_test_array.py)")
    if not os.path.exists(harness_path):
        problems.append(f"no harness result: {harness_path} (run --mode harness first)")
    else:
        h = json.load(open(harness_path))
        if not h.get("pass"):
            problems.append("the harness did not pass; do not run the final evaluation")
        if h.get("script_sha256") != script_sha:
            problems.append("the harness was run by a different version of this script; rerun it")
    return problems


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
            "n_steps": a["n_steps"], "n_gen": a["n_gen"]}


def sample(gen, cfg, labels, seed, n_steps=None):
    import torch
    dev, model, diffusion, B = gen["dev"], gen["model"], gen["diffusion"], gen["B"]
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((len(labels), 768), device=dev, generator=g)
    ts = torch.linspace(diffusion.T - 1, 0, n_steps or gen["n_steps"], device=dev).long()
    kw = dict(avail=torch.ones(len(labels), 1, device=dev), c_hpo=None, c_gene=None,
              tissue=torch.from_numpy(labels).long().to(dev))
    with torch.no_grad():
        for i, t_cur in enumerate(ts):
            tb = t_cur.repeat(len(labels))
            ab_t = diffusion.alpha_bar[t_cur]
            t_next = ts[i + 1] if i + 1 < len(ts) else None
            ab_next = diffusion.alpha_bar[t_next] if t_next is not None else torch.tensor(1.0, device=dev)
            eps = diffusion._eps_with_cfg(model, x, tb, cfg, **kw)
            x0 = torch.maximum(torch.minimum((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt(), B), -B)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            x = x0 if t_next is None else ab_next.sqrt() * x0 + (1 - ab_next).clamp(min=0).sqrt() * eps
    return x.double().cpu().numpy() * gen["sd"] + gen["mu"]


def score_generator(gen, pool, y_pool, fid, seed, n_gen, n_steps=None):
    classes, counts = np.unique(y_pool, return_counts=True)
    labels = np.repeat(classes, allocate(counts, n_gen))
    null = score_fake(pool, sample(gen, 0.0, np.zeros(n_gen, dtype=np.int64), seed, n_steps), fid)
    cond = score_fake(pool, sample(gen, 1.0, labels, seed, n_steps), fid)
    return {"cond": cond, "null": null}


def recorded_validation(path, epoch=800):
    log = json.load(open(os.path.join(os.path.dirname(path), "audit_log.json")))
    rec = next(r for r in log if r["epoch"] == epoch)["weights"]["ema"]
    return {"cond": rec["cond"], "null": rec["null"]}


def fmt(m):
    return (f"C2ST {m['c2st']:.3f} | prec {m['precision']:.3f} | rec {m['recall']:.3f} | cov {m['coverage']:.3f} | "
            f"band {m['norm_band']:.2f} | FID {m['fid']:.4f} | gates {m['gates_passed']}/4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["harness", "final"], default="harness")
    ap.add_argument("--expect-sha", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sha = self_sha()
    print(f"script SHA-256: {sha}\n{LABEL}\n", flush=True)

    if args.mode == "final" and not args.smoke:
        problems = final_problems(args.expect_sha, sha, HARNESS_PATH, RESULT_PATH, TEST_NPZ)
        if problems:
            print("REFUSING to run the final evaluation:")
            for p in problems:
                print("  -", p)
            sys.exit(2)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fid = get_fidelity()
    d = np.load(DATA)
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"].astype(np.int64)
    rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"].astype(np.int64)
    seeds = [0] if args.smoke else sorted(CKPTS)
    n_gen = 16 if args.smoke else N_GEN
    n_steps = 20 if args.smoke else None
    os.makedirs(OUT_DIR, exist_ok=True) if not args.smoke else None

    if args.mode == "harness" or args.smoke:
        print("=== HARNESS: recompute the recorded validation audits (validation rows only) ===", flush=True)
        hres = {}
        for s in seeds:
            gen = load_generator(CKPTS[s], dev)
            new = score_generator(gen, rna_va, y_va, fid, VAL_SEED, n_gen, n_steps)
            if args.smoke:
                print(f"  seed {s} (smoke, tiny n, values not comparable): cond {fmt(new['cond'])}\n"
                      f"                null {fmt(new['null'])}", flush=True)
                continue
            rec = recorded_validation(CKPTS[s])
            hres[str(s)] = {m: compare_to_recorded(new[m], rec[m]) for m in ("cond", "null")}
            for m in ("cond", "null"):
                r = hres[str(s)][m]
                print(f"  seed {s} {m:>4}: recomputed C2ST {new[m]['c2st']:.3f} vs recorded {rec[m]['c2st']:.3f} "
                      f"(max |diff| over C2ST/prec/rec/cov/band = "
                      f"{max(abs(r['diffs'][k]) for k in ('c2st','precision','recall','coverage','norm_band')):.4f}, "
                      f"FID rel {r['diffs']['fid_rel']:.4f})  {'OK' if r['pass'] else 'FAIL'}", flush=True)
            del gen
        if not args.smoke:
            ok = all(hres[s][m]["pass"] for s in hres for m in hres[s])
            json.dump({"pass": bool(ok), "script_sha256": sha, "tolerance": TOL, "fid_rel_tolerance": FID_REL_TOL,
                       "seeds": hres, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(HARNESS_PATH, "w"), indent=2)
            print(f"\nHARNESS {'PASSED' if ok else 'FAILED'}; saved -> {HARNESS_PATH}")
            if not ok:
                print("Do not run the final evaluation. Investigate the difference first.")
                sys.exit(1)
            print("Next: python3 test_split_eval.py --mode final --expect-sha " + sha)
            return

    if args.smoke:
        print("\n=== SMOKE of the final path, validation rows standing in for the test rows ===", flush=True)
        pool, y_pool = rna_va, y_va
    else:
        t = np.load(TEST_NPZ)
        pool, y_pool = t["rna_test"].astype(np.float64), t["y_test"].astype(np.int64)
        print(f"test rows: n={len(pool)}; tissue counts={dict(zip(*[a.tolist() for a in np.unique(y_pool, return_counts=True)]))}")
        print("\n=== FINAL: three seeds on the test rows ===", flush=True)
    test, val = {}, {}
    for s in seeds:
        gen = load_generator(CKPTS[s], dev)
        test[s] = score_generator(gen, pool, y_pool, fid, FINAL_SEED, n_gen, n_steps)
        val[s] = test[s] if args.smoke else recorded_validation(CKPTS[s])
        print(f"  seed {s} cond: {fmt(test[s]['cond'])}\n          null: {fmt(test[s]['null'])}", flush=True)
        del gen
    if args.smoke:
        n_b = min(n_gen, len(pool) // 2)
        comps = fit_tissue_gaussians(rna_tr, y_tr)
        b = baseline_scores(lambda rng: sample_tissue_mixture(comps, n_b, rng), pool, fid, n=n_b)
        f = baseline_scores(None, pool, fid, n=n_b, disjoint=True)
        print(f"  baselines run (smoke): tissue Gaussian {fmt(b)}\n                         floor {fmt(f)}\nsmoke run finished (nothing saved)")
        return

    print("\n=== BASELINES on the test rows ===", flush=True)
    mu_t, cov_t = rna_tr.mean(axis=0), np.cov(rna_tr, rowvar=False)
    comps = fit_tissue_gaussians(rna_tr, y_tr)
    base = {
        "real_vs_real_floor": baseline_scores(None, pool, fid, disjoint=True),
        "train_resample": baseline_scores(lambda rng: rna_tr[rng.choice(len(rna_tr), N_GEN, replace=False)], pool, fid),
        "gaussian": baseline_scores(lambda rng: sample_gaussian(mu_t, cov_t, N_GEN, rng), pool, fid),
        "tissue_gaussian": baseline_scores(lambda rng: sample_tissue_mixture(comps, N_GEN, rng), pool, fid),
    }
    for name, m in base.items():
        print(f"  {name:>18}: {fmt(m)}", flush=True)

    val_ref = {}
    try:
        val_ref["real_vs_real_floor"] = json.load(open(VAL_BASELINE_FILES["real"]))["real_vs_real_floor"]["mean"]
        g = json.load(open(VAL_BASELINE_FILES["gaussian"]))
        for k in ("train_resample", "gaussian", "tissue_gaussian"):
            val_ref[k] = g[k]["mean"]
    except Exception as e:
        print(f"  (validation baseline files not read: {e})")

    reading = read_results(test, val, base["real_vs_real_floor"], val_ref.get("real_vs_real_floor"), base["tissue_gaussian"])
    print(f"\n{'='*84}\n{LABEL}\n{'='*84}")
    print(f"{'seed':>4} | {'cond gates':>10} | {'cond C2ST test':>14} | {'validation':>10} | {'delta':>7} | {'null C2ST test':>14} | {'validation':>10}")
    for s in seeds:
        p = reading["per_seed"][str(s)]
        print(f"{s:>4} | {p['conditional_gates_passed']:>7}/4  | {test[s]['cond']['c2st']:>14.3f} | {val[s]['cond']['c2st']:>10.3f} | "
              f"{p['delta_c2st_cond']:>+7.3f} | {test[s]['null']['c2st']:>14.3f} | {val[s]['null']['c2st']:>10.3f}")
    print(f"R1: {reading['R1_seeds_with_4_of_4_conditional']} of 3 seeds pass 4/4 gates (conditional)")
    print(f"R2: largest |delta C2ST| = {reading['R2_max_abs_delta_c2st']:.3f}: {reading['R2_reading']}"
          + (f"; real-vs-real floor C2ST changed by {reading['floor_c2st_change_test_minus_val']:+.3f} between the splits"
             if "floor_c2st_change_test_minus_val" in reading else ""))
    print("R3: conditional C2ST minus tissue-wise Gaussian on the test rows: "
          + ", ".join(f"seed {s} {reading['per_seed'][str(s)]['c2st_minus_tissue_gaussian']:+.3f}" for s in seeds))
    out = {"label": LABEL, "script_sha256": sha, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_test": int(len(pool)),
           "tissue_counts": {str(k): int(v) for k, v in zip(*np.unique(y_pool, return_counts=True))},
           "noise_seed": FINAL_SEED, "test": {str(s): test[s] for s in seeds},
           "validation_recorded": {str(s): val[s] for s in seeds}, "baselines_test": base,
           "baselines_validation_recorded": val_ref, "reading": reading}
    json.dump(out, open(RESULT_PATH, "w"), indent=2)
    print(f"\nsaved -> {RESULT_PATH}  (single use; the script now refuses to run again)")


if __name__ == "__main__":
    main()
