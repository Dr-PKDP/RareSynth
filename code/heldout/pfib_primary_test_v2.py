"""
Primary held-out test for the Pfib_423 cohort (423 patients, unit-scale RNA embeddings).

Hypothesis H1 (one-sided): the offset of the UFM1 case from the rest of the cohort is aligned with
the mechanism-derived direction d_true more than the offsets of other patients are.

Statistic for patient i:  s_i = cos(e_i - mean_{j != i} e_j, d_hat_true)
(the leave-one-out cohort mean is the reference, so platform and batch differences between cohorts
cannot contribute). With one UFM1 case the p-value is its rank: p = #{rows with s >= s_case} / N
(minimum 1/N). With k > 1 cases the statistic is the mean s over the cases and the null is a
Monte-Carlo draw of random k-subsets of all rows.

Reported but NOT tested: the same statistic under the shuffled and variance-stratified-shuffle
directions (specificity), the rank of each other diagnosed patient under d_true, and a sensitivity
analysis against the GTEx fibroblast baseline (cross-platform, descriptive only).

Modes
  dry    stand-in cohort of 423 TCGA breast rows; checks that the p-value is calibrated under the
         null and measures power for a planted shift. Never opens the Pfib embeddings.
  final  opens the Pfib embeddings. One shot: refuses to run if the result file exists, and refuses
         unless --expect-sha equals this file's SHA-256 (recorded in the decision log when the
         protocol is frozen), so the exact frozen code is what runs.

AMENDMENT v2 (2026-10-21, before any cohort result exists; decision 40). The CORE block (statistic, p-value, rank,
controls) is byte-identical to v1. Only run_final changed, to remove four procedural hazards found by AI code audits:
  1. non-finite or zero-norm cohort rows now stop the run (one NaN made every statistic NaN and the p-value 0.0);
  2. rows are L2-normalized (as the TCGA rows are at load; a no-op for unit rows) and the pre-normalization norms are
     recorded in the result file;
  3. every input (identifiers, GTEx baseline, output folder) is validated before anything is computed, everything is
     computed before anything is printed, and the result file is written before the primary line is shown, so the one
     shot cannot be spent by looking at a partial run;
  4. the result records the SHA-256 of the cohort file; --expect-data-sha optionally pins it.

Pass/fail rule (pre-registered): primary success = one-sided p <= 0.05. A p-value above 0.05 is
reported as a negative result. No other statistic may be promoted to primary after the fact.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
GTEX_BASELINE = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/fibroblast_baseline.npz"
TCGA_ARRAYS = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
PFIB = "/data/pduttapramanik/raresynth/data/embeddings/rna_pfib423_full.npz"
OUT_DIR = "/data/pduttapramanik/raresynth/runs/pfib_test"
RESULT = os.path.join(OUT_DIR, "pfib_primary_result.json")

# --- BEGIN CORE ---
def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def loo_cos(E, dhat):
    """cos(e_i - mean_{j != i} e_j, dhat) for every row i, using offset_i = (N e_i - S) / (N - 1)."""
    N = len(E)
    off = (N * E - E.sum(axis=0)) / (N - 1)
    return (off @ dhat) / (np.linalg.norm(off, axis=1) + 1e-12)


def perm_p(s, idx, n_draws=200000, seed=0):
    """One-sided permutation p-value that the case rows have a higher statistic than a random row set."""
    idx = list(idx)
    N = len(s)
    if len(idx) == 1:
        return float((s >= s[idx[0]]).sum() / N)               # exact rank test, minimum 1/N
    rng = np.random.default_rng(seed)
    obs = s[idx].mean()
    draws = np.array([s[rng.choice(N, size=len(idx), replace=False)].mean() for _ in range(n_draws)])
    return float((1 + (draws >= obs).sum()) / (1 + n_draws))


def rank_of(s, i):
    """1 = highest statistic."""
    return int((s > s[i]).sum()) + 1


def plant(E, i, dhat, a):
    E2 = E.copy()
    E2[i] = unit(E[i] + a * dhat)
    return E2


def stratified_shuffle(d, sd, block=32, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(sd)
    out = d.copy()
    for start in range(0, len(d), block):
        idx = order[start:start + block]
        out[idx] = d[rng.permutation(idx)]
    return out
# --- END CORE ---


def load_directions(sd):
    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    d_shuf = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance_SHUFFLED_CONTROL.npz")["rna_embedding_guidance"].astype(np.float64)
    d_strat = stratified_shuffle(d_true, sd)
    return {"true": unit(d_true), "shuffled": unit(d_shuf), "stratified": unit(d_strat)}


def sha256_of_this_file():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_dry(args):
    d = np.load(TCGA_ARRAYS)
    rna_tr, y_tr = d["rna_train"].astype(np.float64), d["y_train"]
    sd = np.maximum(rna_tr.std(axis=0), 1e-4)
    dirs = load_directions(sd)
    rng = np.random.default_rng(args.seed)
    pool = np.where(y_tr == 0)[0]                               # the largest tissue: a single-tissue stand-in cohort
    E = rna_tr[rng.choice(pool, size=423, replace=False)]
    print(f"stand-in cohort: 423 TCGA breast rows (a tumour tissue, NOT a fibroblast cohort); real power will differ.")
    s0 = loo_cos(E, dirs["true"])
    print(f"stand-in statistic s_i under d_true: mean {s0.mean():+.4f}, sd {s0.std():.4f}, range [{s0.min():+.3f}, {s0.max():+.3f}]")
    print(f"\n{'planted shift a':>16} | {'mean s_case':>11} | {'detect p<=0.05':>14} | {'detect p<=0.01':>14} | {'median rank':>11}")
    table = {}
    for a in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        hits05 = hits01 = 0
        ss, rks = [], []
        for _ in range(args.n_sim):
            i = int(rng.integers(0, len(E)))
            E2 = plant(E, i, dirs["true"], a)
            s = loo_cos(E2, dirs["true"])
            p = perm_p(s, [i])
            hits05 += p <= 0.05
            hits01 += p <= 0.01
            ss.append(s[i]); rks.append(rank_of(s, i))
        table[str(a)] = {"mean_s": float(np.mean(ss)), "det05": hits05 / args.n_sim, "det01": hits01 / args.n_sim,
                         "median_rank": float(np.median(rks))}
        r = table[str(a)]
        print(f"{a:>16.2f} | {r['mean_s']:>+11.4f} | {r['det05']:>14.3f} | {r['det01']:>14.3f} | {r['median_rank']:>11.0f}")
    print(f"\ncalibration: at a = 0 the detection rate at p <= 0.05 should be about 0.05 "
         f"(observed {table['0.0']['det05']:.3f}, binomial 95% band +/-{1.96*np.sqrt(0.05*0.95/args.n_sim):.3f}).")
    print("The smallest planted shift with detection >= 0.8 at p <= 0.05 is the stand-in's minimum detectable shift.")
    return {"table": table}


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_final(args):
    # ---- 0. refusals that need no data -------------------------------------------------------------------------
    if os.path.exists(RESULT):
        sys.exit(f"REFUSING: {RESULT} already exists. The held-out test is one shot.")
    sha = sha256_of_this_file()
    if args.expect_sha != sha:
        sys.exit(f"REFUSING: --expect-sha does not match this file ({sha}). The frozen code must be the code that runs.")
    if not args.ufm1_ids:
        sys.exit("REFUSING: --ufm1-ids is required.")
    if len(set(args.ufm1_ids)) != len(args.ufm1_ids) or set(args.ufm1_ids) & set(args.other_ids):
        sys.exit("REFUSING: --ufm1-ids must be unique and must not overlap --other-ids.")
    try:                                                     # the output folder must be writable before anything is shown
        os.makedirs(OUT_DIR, exist_ok=True)
        probe = os.path.join(OUT_DIR, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        sys.exit(f"REFUSING: output folder {OUT_DIR} is not writable ({e}).")

    # ---- 1. load and validate every input; print nothing about the result -------------------------------------
    d = np.load(TCGA_ARRAYS)
    sd = np.maximum(d["rna_train"].astype(np.float64).std(axis=0), 1e-4)       # used only to build the stratified control
    dirs = load_directions(sd)
    for k, v in dirs.items():
        if v.shape != (768,) or not np.isfinite(v).all():
            sys.exit(f"REFUSING: direction '{k}' is not a finite 768-vector.")
    data_sha = sha256_of_file(PFIB)
    if args.expect_data_sha and args.expect_data_sha != data_sha:
        sys.exit(f"REFUSING: cohort file SHA-256 {data_sha} does not match --expect-data-sha.")
    z = np.load(PFIB, allow_pickle=True)
    ids = [str(c) for c in z["case_ids"]]
    E_raw = z["embeddings"].astype(np.float64)
    if E_raw.shape != (423, 768) or len(ids) != 423 or len(set(ids)) != 423:
        sys.exit(f"REFUSING: expected 423 unique ids and a (423, 768) matrix, got {len(ids)} ids ({len(set(ids))} unique), shape {E_raw.shape}.")
    if not np.isfinite(E_raw).all():
        bad = np.where(~np.isfinite(E_raw).all(axis=1))[0]
        sys.exit(f"REFUSING: {len(bad)} cohort row(s) contain non-finite values (row indices {bad[:10].tolist()}).")
    norms_raw = np.linalg.norm(E_raw, axis=1)
    if norms_raw.min() < 1e-6:
        sys.exit(f"REFUSING: a cohort row has (near) zero norm (min {norms_raw.min():.3e}).")
    missing = [c for c in list(args.ufm1_ids) + list(args.other_ids) if c not in ids]
    if missing:
        sys.exit(f"REFUSING: identifier(s) not in the cohort: {missing}.")
    b = np.load(GTEX_BASELINE)["baseline"].astype(np.float64)
    if b.shape != (768,) or not np.isfinite(b).all():
        sys.exit("REFUSING: GTEx baseline is not a finite 768-vector.")
    E = unit(E_raw)                                           # rows enter the statistic at unit scale (no-op if already unit)
    if np.linalg.norm(E - b, axis=1).min() < 1e-9:
        sys.exit("REFUSING: a cohort row equals the GTEx baseline (zero offset).")

    # ---- 2. compute everything; nothing is printed yet ---------------------------------------------------------
    case = [ids.index(c) for c in args.ufm1_ids]
    out = {"sha256": sha, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ufm1_ids": args.ufm1_ids,
           "n_patients": len(ids), "cohort_file_sha256": data_sha,
           "input_row_norms": {"min": float(norms_raw.min()), "max": float(norms_raw.max()), "mean": float(norms_raw.mean()),
                               "max_abs_dev_from_1": float(np.abs(norms_raw - 1).max()),
                               "normalized_before_statistic": bool(np.abs(norms_raw - 1).max() > 1e-6)},
           "primary": {}, "controls": {}, "other_diagnosed": {}, "sensitivity": {}}
    s = loo_cos(E, dirs["true"])
    p = perm_p(s, case)
    out["primary"] = {"statistic": float(s[case].mean()), "p_one_sided": p,
                      "ranks": {c: rank_of(s, ids.index(c)) for c in args.ufm1_ids},
                      "min_possible_p": 1.0 / len(ids), "success": bool(p <= 0.05)}
    out["primary"]["cohort_statistic_mean"], out["primary"]["cohort_statistic_sd"] = float(s.mean()), float(s.std())
    lines = [f"embedding norms before normalization: min {norms_raw.min():.4f}, max {norms_raw.max():.4f}; "
             f"cohort file SHA-256 {data_sha}",
             f"\nPRIMARY: s_case = {s[case].mean():+.4f}; rank(s) = {out['primary']['ranks']}; one-sided p = {p:.4f} "
             f"(minimum possible {1/len(ids):.4f}) -> {'SUCCESS (p <= 0.05)' if p <= 0.05 else 'NEGATIVE (p > 0.05)'}",
             f"  cohort statistic under d_true: mean {s.mean():+.4f}, sd {s.std():.4f}"]
    for name in ("shuffled", "stratified"):
        sc = loo_cos(E, dirs[name])
        out["controls"][name] = {"statistic": float(sc[case].mean()), "p_one_sided": perm_p(sc, case),
                                 "ranks": {c: rank_of(sc, ids.index(c)) for c in args.ufm1_ids}}
        lines.append(f"  control direction '{name}': s_case = {sc[case].mean():+.4f}, p = {out['controls'][name]['p_one_sided']:.4f} (descriptive)")
    for c in args.other_ids:
        out["other_diagnosed"][c] = {"rank_under_d_true": rank_of(s, ids.index(c)), "statistic": float(s[ids.index(c)])}
        lines.append(f"  other diagnosed patient {c}: rank {out['other_diagnosed'][c]['rank_under_d_true']} of {len(ids)} (descriptive)")
    sb = ((E - b) @ dirs["true"]) / np.linalg.norm(E - b, axis=1)
    out["sensitivity"]["gtex_baseline_cos_case"] = float(sb[case].mean())
    out["sensitivity"]["gtex_baseline_rank"] = {c: rank_of(sb, ids.index(c)) for c in args.ufm1_ids}
    lines.append(f"  sensitivity (cross-platform, descriptive): cos(e - GTEx fibroblast baseline, d_true) for the case = {sb[case].mean():+.4f}, "
                 f"rank {out['sensitivity']['gtex_baseline_rank']}")

    # ---- 3. write the result (atomic) BEFORE anything about it is shown ----------------------------------------
    tmp = RESULT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, RESULT)
    for line in lines:
        print(line)
    print(f"\nsaved -> {RESULT}  (one shot: this file now blocks any rerun)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "final"], default="dry")
    ap.add_argument("--n-sim", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ufm1-ids", nargs="*", default=[])
    ap.add_argument("--other-ids", nargs="*", default=[])
    ap.add_argument("--expect-sha", default="")
    ap.add_argument("--expect-data-sha", default="", help="optional SHA-256 of the cohort file")
    args = ap.parse_args()
    print(f"script SHA-256: {sha256_of_this_file()}")
    if args.mode == "dry":
        run_dry(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
