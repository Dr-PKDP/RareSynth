"""
Pre-specified SECONDARY analysis for the fibroblast held-out test: one sample per donor.
Written 2026-10-21, before any cohort result exists.  Does not modify the locked primary script.

Rule (fixed here, in advance):
  * rows come from the same cohort file, unit-normalized like the primary;
  * donors from the manifest column donor_id; the label 'NHDF' is a shared control-line label and every NHDF
    row is kept as its own donor (the manifest does not identify distinct people behind it);
  * for each other donor with several rows, keep the case row if the donor has one, otherwise the first row in
    manifest order; the case's own donor siblings are therefore dropped;
  * same CORE statistic (leave-one-out cosine to the same direction) and the same exact rank p-value on the kept rows.
Refuses to run unless the primary result file exists (it is a sensitivity analysis, never a way to choose).
Reads CORE from the primary script file after checking that file's SHA-256 against --primary-sha.
"""
import argparse, csv, hashlib, json, os, sys, time
import numpy as np

PRIMARY_RESULT = "/data/pduttapramanik/raresynth/runs/pfib_test/pfib_primary_result.json"
OUT = "/data/pduttapramanik/raresynth/runs/pfib_test/pfib_secondary_one_per_donor.json"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_core(path, expect):
    if sha(path) != expect:
        sys.exit("REFUSING: primary script hash differs from --primary-sha.")
    src = open(path).read().split("# --- BEGIN CORE ---")[1].split("# --- END CORE ---")[0]
    ns = {"np": np}; exec(src, ns); return ns

def select_rows(ids, donors, case_ids, keep_label="NHDF"):
    first = {}
    keep = []
    case_donors = {donors[ids.index(c)] for c in case_ids}
    for i, (cid, dn) in enumerate(zip(ids, donors)):
        if dn == keep_label:
            keep.append(i); continue
        if dn in case_donors:
            if cid in case_ids: keep.append(i)
            continue
        if dn not in first:
            first[dn] = i; keep.append(i)
    return keep

def main():
    ap = argparse.ArgumentParser()
    D = "/data/pduttapramanik/raresynth/"
    ap.add_argument("--primary-script", default=D + "pfib_primary_test_v2.py")
    ap.add_argument("--primary-sha", required=True)
    ap.add_argument("--cohort", default=D + "data/embeddings/rna_pfib423_full.npz")
    ap.add_argument("--manifest", default=D + "data/manifests/case_index_pfib423.csv")
    ap.add_argument("--direction", default=D + "data/raw/mechanism/ppn_extracted/ufm1_rna_embedding_guidance.npz")
    ap.add_argument("--ufm1-ids", nargs="+", required=True)
    ap.add_argument("--primary-result", default=PRIMARY_RESULT)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if not os.path.exists(a.primary_result):
        sys.exit("REFUSING: the primary result does not exist yet. Run the secondary only after the primary.")
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists.")
    core = load_core(a.primary_script, a.primary_sha)
    z = np.load(a.cohort, allow_pickle=True)
    ids = [str(c) for c in z["case_ids"]]; E = core["unit"](z["embeddings"].astype(np.float64))
    if not np.isfinite(E).all(): sys.exit("REFUSING: non-finite rows.")
    with open(a.manifest) as fh:
        dmap = {r["case_id"]: r["donor_id"] for r in csv.DictReader(fh)}
    if set(dmap) != set(ids): sys.exit("REFUSING: manifest ids differ from cohort ids.")
    donors = [dmap[c] for c in ids]
    d = np.load(a.direction)["rna_embedding_guidance"].astype(np.float64)
    if d.shape != (768,) or not np.isfinite(d).all(): sys.exit("REFUSING: bad direction.")
    d = d / np.linalg.norm(d)
    keep = select_rows(ids, donors, set(a.ufm1_ids))
    sub = E[keep]; sids = [ids[i] for i in keep]
    s = core["loo_cos"](sub, d)
    case = [sids.index(c) for c in a.ufm1_ids]
    p = core["perm_p"](s, case)
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_rows": len(keep), "n_dropped": len(ids) - len(keep),
           "statistic": float(s[case].mean()), "p_one_sided": p, "ranks": {c: core["rank_of"](s, sids.index(c)) for c in a.ufm1_ids},
           "script_sha256": sha(__file__), "primary_script_sha256": a.primary_sha, "ufm1_ids": a.ufm1_ids,
           "rule": "one row per donor; NHDF rows all kept; case row kept; case donor siblings dropped"}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f: json.dump(res, f, indent=2)
    os.replace(tmp, a.out)
    print(f"SECONDARY (one per donor, {len(keep)} rows, {len(ids)-len(keep)} dropped): s_case = {res['statistic']:+.4f}, "
          f"rank {res['ranks']}, one-sided p = {p:.4f}\nsaved -> {a.out}")

if __name__ == "__main__":
    main()
