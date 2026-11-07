"""
Noise-draw replicates for the steering comparator (decision 36/37 follow-up; preregistered below).  Run on the server
next to steering_comparator.py and steering_comparator_armP.py.  TRAIN and VALIDATION rows only; test array not opened.

WHY.  The recorded comparator has 3 checkpoints x ONE noise draw each, and one Gaussian-base draw each.  An earlier audit
found that fresh noise moved the generator's conditional C2ST by 0.033-0.060, the size of the 0.05 reading threshold.
The three-seed range is therefore not an uncertainty estimate.  This script adds K independent draws per checkpoint on
BOTH sides.

DRAW r of checkpoint s, mode m (all rematched, nothing reused from the recorded run):
  generator noise seed = 910000 + 1000*s + r        Gaussian-base seed = 920000 + 1000*s + r
  unguided generator + guided generator at every scale; fresh Gaussian base; T and G (and P with --with-P) steered to the
  guided generator's achieved mean cosine IN THAT DRAW; every sample set scored by the unchanged score_fake (same three
  real subsets, rng seed 777, C2ST seeds 0-2, frozen gates).  Draw loop is r-outer, so after each r every cell has the
  same number of draws and a partial run is usable.  Output: <OUT>/noiserep/draws.jsonl, one line per (s, mode, r, scale),
  appended as it goes; rerunning resumes and skips finished lines.

PREREGISTERED ANALYSIS (fixed before any replicate exists; run analyze_noiserep.py):
  primary quantity   D = C2ST(generator guided) - C2ST(steered Gaussian) per (checkpoint, draw), levels, for T and G.
  estimate           equal-weight mean over the 3 checkpoints of the mean over draws.
  interval           95% percentile bootstrap, 10000 replicates, resampling draws WITHIN each checkpoint (conditional on
                     the 3 observed checkpoints).  Sensitivity: also resample checkpoints (3 only, so wide).
  classification    interval wholly <= -0.05: advantage beyond the 0.05 descriptive margin; wholly inside (-0.05, +0.05):
                     within the margin (NOT an equivalence claim); wholly >= +0.05: steered Gaussian closer; otherwise
                     inconclusive.  The 0.05 margin is the original descriptive heuristic and is not changed.
  exploratory        arm P, and the increase contrast, reported separately and labelled post hoc.

Usage (server):
    python3 steering_comparator_noiserep.py --smoke            K=1, one checkpoint, one scale, tiny; writes nothing final
    python3 steering_comparator_noiserep.py --reps 10          full (hours; GPU)
    python3 steering_comparator_noiserep.py --selftest         wiring test with fake generators, no GPU, no server paths
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steering_comparator as sc  # noqa: E402

NOISE_BASE, GAUSS_BASE = 910000, 920000
ARMS = ("T_translate", "G_ascent")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["seed"], r["mode"], r["rep"], str(r["scale"])))
    return done


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def draw_base(mode, mu_g, cov_g, comps, labels, n_gen, seed):
    rng = np.random.default_rng(seed)
    if mode == "null":
        return sc.sample_gaussian(mu_g, cov_g, n_gen, rng)
    return sc.sample_by_labels(comps, labels, rng)


def run_draw(s, mode, r, cfg, labels, scales, n_gen, n_steps, gen, energy, b, dh, fid, rna_va, mu_g, cov_g, comps,
             out_path, done, with_p, ps):
    nseed, gseed = NOISE_BASE + 1000 * s + r, GAUSS_BASE + 1000 * s + r
    key_un = (s, mode, r, "unguided")
    base = draw_base(mode, mu_g, cov_g, comps, labels, n_gen, gseed)
    c_base = float(sc.cos_to(base, b, dh).mean())
    sc_base = sc.score_fake(rna_va, base, fid)
    x_un = None
    if key_un not in done:
        x_un = sc.generate(gen, None, 0.0, cfg, labels, nseed, n_steps)
        sc_un = sc.score_fake(rna_va, x_un, fid)
        append(out_path, {"seed": s, "mode": mode, "rep": r, "scale": "unguided", "noise_seed": nseed, "gauss_seed": gseed,
                          "generator": {**sc_un, "mean_cos": float(sc.cos_to(x_un, b, dh).mean())},
                          "gaussian_base": {**sc_base, "mean_cos": c_base}})
    for scale in scales:
        if (s, mode, r, str(scale)) in done:
            continue
        x_g = sc.generate(gen, energy, float(scale), cfg, labels, nseed, n_steps)
        sc_g = sc.score_fake(rna_va, x_g, fid)
        target = float(sc.cos_to(x_g, b, dh).mean())
        rec = {"seed": s, "mode": mode, "rep": r, "scale": scale, "noise_seed": nseed, "gauss_seed": gseed,
               "generator": {**sc_g, "mean_cos": target}}
        steerers = [("T_translate", lambda x, p: sc.steer_translate(x, b, dh, p)),
                    ("G_ascent", lambda x, p: sc.steer_ascent(x, b, dh, p))]
        if with_p:
            import steering_comparator_armP as ap
            mu, sd, bound = ps
            steerers.append(("P_zascent", lambda x, p: ap.steer_z_ascent(x, b, dh, p, mu, sd, bound)))
        for name, steer in steerers:
            p, got, reached = sc.match_mean_cos(steer, base, b, dh, target)
            rec[name] = {**sc.score_fake(rna_va, steer(base, p), fid), "param": float(p), "mean_cos": float(got),
                         "reached": bool(reached)}
        append(out_path, rec)
        print(f"[s{s} {mode} r{r} scale {scale}] gen C2ST {sc_g['c2st']:.3f} | "
              + " | ".join(f"{n[0]} {rec[n]['c2st']:.3f}{'' if rec[n]['reached'] else '!'}" for n, _ in steerers), flush=True)


def main_run(args, loaders=None):
    """loaders: optional dict of replacement callables for the self-test."""
    L = loaders or {}
    if loaders is None:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = "cpu"
    load_gen, make_energy = L.get("load_generator", sc.load_generator), L.get("make_energy", sc.make_energy)
    fid = L.get("fid") or sc.get_fidelity()
    d = np.load(L.get("data", sc.DATA))
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"].astype(np.int64)
    rna_va, y_va = d["rna_val"].astype(np.float64), d["y_val"].astype(np.int64)
    d_true = L["d_true"] if "d_true" in L else np.load(f"{sc.PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    dh = d_true / np.linalg.norm(d_true)
    b = L["b"] if "b" in L else np.load(sc.GTEX_BASELINE)["baseline"].astype(np.float64)
    seeds = [0] if args.smoke else sorted(L.get("ckpts", sc.CKPTS))
    scales = [150] if args.smoke else sc.SCALES
    n_gen = 16 if args.smoke else sc.N_GEN
    n_steps = 20 if args.smoke else None
    classes, counts = np.unique(y_va, return_counts=True)
    labels_cond = np.repeat(classes, sc.allocate(counts, n_gen))
    modes = {"null": (0.0, np.zeros(n_gen, dtype=np.int64)), "cond": (1.0, labels_cond)}
    mu_g, cov_g = rna_tr.mean(axis=0), np.cov(rna_tr, rowvar=False)
    comps = sc.fit_tissue_gaussians(rna_tr, y_tr)
    ps = None
    if args.with_P:
        import steering_comparator_armP as ap
        ps = ap.preprocessing_from_train(rna_tr)
    out_dir = args.out_dir or os.path.join(sc.OUT_DIR, "noiserep")
    if args.smoke:
        out_dir = tempfile.mkdtemp(prefix="noiserep_smoke_")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "draws.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        json.dump({"script_sha256": sha256(__file__), "comparator_sha256": sha256(sc.__file__), "reps": args.reps,
                   "scales": scales, "seeds": seeds, "noise_base": NOISE_BASE, "gauss_base": GAUSS_BASE,
                   "n_gen": n_gen, "with_P": bool(args.with_P), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "label": "validation rows only; test array not opened"}, open(meta_path, "w"), indent=2)
    done = load_done(out_path)
    print(f"noise-draw replicates; device {dev}; K={args.reps}; output {out_path}; {len(done)} lines already present", flush=True)
    t0 = time.time()
    for r in range(args.reps):
        for s in seeds:
            gen = load_gen(sc.CKPTS[s] if loaders is None else s, dev)
            energy = make_energy(gen, d_true, b)
            for mode, (cfg, labels) in modes.items():
                run_draw(s, mode, r, cfg, labels, scales, n_gen, n_steps, gen, energy, b, dh, fid, rna_va, mu_g, cov_g,
                         comps, out_path, done, args.with_P, ps)
            del gen
        print(f"== draw {r + 1}/{args.reps} complete for all checkpoints and modes ({(time.time() - t0) / 60:.1f} min) ==", flush=True)
    print(f"finished -> {out_path}")
    return out_path


def selftest():
    """Wiring test with fake generators and synthetic data; no GPU, no server paths."""
    tmp = tempfile.mkdtemp(prefix="noiserep_selftest_")
    rng = np.random.default_rng(0)
    d, ntr, nva = 24, 400, 200
    sdv = np.exp(rng.normal(size=d) * 0.5)
    ytr, yva = rng.integers(0, 3, ntr), rng.integers(0, 3, nva)
    sh = rng.normal(size=(3, d)) * 0.3
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    tr, va = unit(rng.normal(size=(ntr, d)) * sdv + sh[ytr]), unit(rng.normal(size=(nva, d)) * sdv + sh[yva])
    np.savez(os.path.join(tmp, "arr.npz"), rna_train=tr, y_train=ytr, rna_val=va, y_val=yva)
    dtrue = rng.normal(size=d); b = np.zeros(d); dh = dtrue / np.linalg.norm(dtrue)

    def fake_generate(gen, energy, scale, cfg, labels, seed, n_steps=None):
        r = np.random.default_rng(seed)
        x = unit(r.normal(size=(len(labels), d)) * sdv + sh[labels % 3])
        return unit(x + (scale / 1500.0) * dh) if energy is not None and scale else x

    sc.generate = fake_generate
    sc.N_GEN, sc.SCALES = 40, [150, 200]
    sc.CKPTS = {0: "a", 1: "b"}
    loaders = {"load_generator": lambda p, dev: {"seed": p}, "make_energy": lambda g, dd, bb: (lambda z: z),
               "fid": {"prdc": lambda a, c, k=5: __import__("raresynth.eval.fidelity", fromlist=["prdc"]).prdc(a, c, k=k),
                       "c2st": lambda a, c, seed=0, n_permutations=0: __import__("raresynth.eval.fidelity", fromlist=["c2st"]).c2st(a, c, seed=seed, n_permutations=0),
                       "frechet": lambda a, c: 0.0},
               "data": os.path.join(tmp, "arr.npz"), "d_true": dtrue, "b": b, "ckpts": sc.CKPTS}
    sys.path.insert(0, os.path.join(HERE, "..", "ver", "P5_ai_audit", "code"))
    sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
    ns = argparse.Namespace(smoke=False, reps=2, with_P=False, out_dir=os.path.join(tmp, "out"))
    path = main_run(ns, loaders)
    lines = [json.loads(x) for x in open(path)]
    assert len(lines) == 2 * 2 * 2 * (1 + 2), len(lines)                  # reps x seeds x modes x (unguided + 2 scales)
    keys = {(r["seed"], r["mode"], r["rep"], str(r["scale"])) for r in lines}
    assert len(keys) == len(lines)
    guided = [r for r in lines if r["scale"] != "unguided"]
    assert all(abs(r[a]["mean_cos"] - r["generator"]["mean_cos"]) < 1e-6 and r[a]["reached"] for r in guided for a in ARMS)
    assert len({r["noise_seed"] for r in lines}) == 2 * 2 and len({r["gauss_seed"] for r in lines}) == 2 * 2
    # resume: a second invocation adds nothing
    main_run(ns, loaders)
    assert len(open(path).read().strip().splitlines()) == len(lines)
    # draws differ between reps (fresh noise, fresh Gaussians)
    g = {(r["seed"], r["mode"], r["rep"], r["scale"]): r for r in lines}
    assert g[(0, "null", 0, 150)]["generator"]["c2st"] != g[(0, "null", 1, 150)]["generator"]["c2st"]
    print("SELFTEST PASSED:", len(lines), "lines; resume adds none; cosine matched in every draw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--with-P", action="store_true", help="also steer with arm P (exploratory, adds CPU time)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.smoke:
        args.reps = 1
    main_run(args)


if __name__ == "__main__":
    main()
