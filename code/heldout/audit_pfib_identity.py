"""Read-only identity-alignment check of the Pfib_423 cohort file against its case index."""
import argparse, csv, sys
import numpy as np

def ids(path):
    z = np.load(path, allow_pickle=True)
    return [str(x) for x in z["case_ids"].tolist()]

def main():
    ap = argparse.ArgumentParser()
    D = "/data/pduttapramanik/raresynth/data/"
    ap.add_argument("--ns", default=D + "embeddings/rna_pfib423_ns/rna_embeddings.npz")
    ap.add_argument("--ss", default=D + "embeddings/rna_pfib423_ss/rna_embeddings.npz")
    ap.add_argument("--full", default=D + "embeddings/rna_pfib423_full.npz")
    ap.add_argument("--manifest", default=D + "manifests/case_index_pfib423.csv")
    ap.add_argument("--values", action="store_true")
    ap.add_argument("--case-id", default=None)
    a = ap.parse_args()
    ok = True
    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" -- " + msg if msg else ""))
    ns, ss, full = ids(a.ns), ids(a.ss), ids(a.full)
    with open(a.manifest) as fh:
        man = [r["case_id"] for r in csv.DictReader(fh)]
    print(f"rows: ns {len(ns)}  ss {len(ss)}  full {len(full)}  manifest {len(man)}")
    rep("1a ids unique inside ns", len(set(ns)) == len(ns))
    rep("1b ids unique inside ss", len(set(ss)) == len(ss))
    rep("1c no overlap between ns and ss", not (set(ns) & set(ss)), f"overlap {sorted(set(ns)&set(ss))[:5]}")
    rep("2a full ids unique", len(set(full)) == len(full))
    order = "ns+ss" if full == ns + ss else ("ss+ns" if full == ss + ns else None)
    rep("2b full ids equal the parts as a sequence", order is not None,
        f"order {order}" if order else "neither ns+ss nor ss+ns; the rows were reordered or edited")
    rep("3a full ids == manifest ids (set)", set(full) == set(man),
        f"missing from full {sorted(set(man)-set(full))[:5]}, extra {sorted(set(full)-set(man))[:5]}")
    rep("3b manifest ids unique", len(set(man)) == len(man))
    if a.values:
        zf = np.load(a.full, allow_pickle=True)["embeddings"]
        parts = {"ns": np.load(a.ns, allow_pickle=True)["embeddings"], "ss": np.load(a.ss, allow_pickle=True)["embeddings"]}
        for o in ("ns+ss", "ss+ns"):
            first, second = o.split("+")
            st = np.vstack([parts[first], parts[second]])
            same = st.shape == zf.shape and np.array_equal(st, zf)
            print(f"     values equal stacked {o}: {same}")
            if o == (order or "") : rep("4 full embeddings equal stacked parts in id order", same)
        if order is None: rep("4 full embeddings equal stacked parts in id order", False, "ids not in a part order")
    if a.case_id:
        print(f"case {a.case_id}: in ns {a.case_id in set(ns)}, in ss {a.case_id in set(ss)}")
    print("ALL PASS" if ok else "FAILED"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
