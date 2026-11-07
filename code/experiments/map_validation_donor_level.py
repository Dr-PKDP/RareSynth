"""Donor-level evaluation of the ridge map from LINCS-gene expression to RNA embeddings."""

import csv
import json
import sys

import numpy as np

TCGA_H5AD = "/data/pduttapramanik/raresynth/data/embeddings/rna_tcga_full/h5ad_input/cohort.h5ad"
TCGA_EMB_PATH = "/data/pduttapramanik/raresynth/data/embeddings/rna_tcga_full/rna_embeddings.npz"
PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
SYMBOL_TO_ENSEMBL_PATH = f"{PPN_DIR}/lincs_measured_gene_symbol_to_ensembl.json"
CASE_INDEX_PATH = "/data/pduttapramanik/raresynth/data/manifests/case_index_tcga.csv"
OUT_PATH = "/data/pduttapramanik/raresynth/runs/cfg_sweep/map_validation_donor_level.json"
RECORDED_R2 = 0.8206

def l2_normalize(x, axis=1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def r2_both(Y, P):
    sse = ((Y - P) ** 2).sum(axis=0)
    sst = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    ok = sst > 0
    uniform = float(np.mean(1.0 - sse[ok] / sst[ok]))
    weighted = float(1.0 - sse.sum() / sst.sum())
    return uniform, weighted


def within_tissue(Y, P, tissue, min_n=20):
    per, num, den = {}, 0.0, 0.0
    for t in np.unique(tissue):
        m = tissue == t
        if m.sum() < min_n:
            continue
        yc, pc = Y[m] - Y[m].mean(axis=0), P[m] - P[m].mean(axis=0)
        e, v = float(((yc - pc) ** 2).sum()), float((yc ** 2).sum())
        cs = (yc * pc).sum(axis=1) / (np.linalg.norm(yc, axis=1) * np.linalg.norm(pc, axis=1) + 1e-12)
        per[str(t)] = {"n": int(m.sum()), "r2_centered": 1.0 - e / v, "mean_cos": float(cs.mean())}
        num += e
        den += v
    return per, (1.0 - num / den if den > 0 else float("nan"))


def pc_subspace_r2(Y_train, Y_val, P_val, k=10):
    mu = Y_train.mean(axis=0)
    _, _, Vt = np.linalg.svd(Y_train - mu, full_matrices=False)
    V = Vt[:k].T
    out = {}
    for name, proj in (("top", lambda Z: (Z - mu) @ V), ("rest", lambda Z: (Z - mu) - ((Z - mu) @ V) @ V.T)):
        a, b = proj(Y_val), proj(P_val)
        sse, sst = float(((a - b) ** 2).sum()), float(((a - a.mean(axis=0)) ** 2).sum())
        out[name] = 1.0 - sse / sst
    return out, Vt


def pc_share(d, Vt, ks=(5, 10, 20, 50)):
    d = d / np.linalg.norm(d)
    return {str(k): float(((Vt[:k] @ d) ** 2).sum()) for k in ks}


def main():
    import anndata as ad
    from scipy import sparse
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    with open(CASE_INDEX_PATH) as fh:
        manifest = {r["case_id"]: r for r in csv.DictReader(fh)}
    a = ad.read_h5ad(TCGA_H5AD)
    case_ids = np.array(a.obs["case_id"].values, dtype=str)
    genes = np.array(a.var.index, dtype=str)
    symbol_to_ensembl = json.loads(open(SYMBOL_TO_ENSEMBL_PATH).read())
    overlap = sorted(set(symbol_to_ensembl.values()) & set(genes))
    col = {g: i for i, g in enumerate(genes)}
    cols = np.array([col[g] for g in overlap])
    Xs = a.X[:, cols]
    X = np.log1p(Xs.toarray() if sparse.issparse(Xs) else np.asarray(Xs))
    print(f"expression: {X.shape[0]} cases x {len(cols)} LINCS-overlap genes", flush=True)

    emb = np.load(TCGA_EMB_PATH, allow_pickle=True)
    pos = {str(c): i for i, c in enumerate(emb["case_ids"])}
    assert all(c in pos for c in case_ids), "case_id mismatch between the expression file and the embeddings"
    Y = l2_normalize(emb["embeddings"].astype(np.float64)[np.array([pos[c] for c in case_ids])])

    info = [manifest.get(c) for c in case_ids]
    found = np.array([i is not None for i in info])
    print(f"cases found in the case index: {int(found.sum())} of {len(case_ids)}", flush=True)
    split = np.array([i["split"] if i else "" for i in info])
    tissue = np.array([i.get("project", "") if i else "" for i in info])

    def fit_score(train_idx, val_idx, label):
        mu, sd = X[train_idx].mean(axis=0), X[train_idx].std(axis=0) + 1e-8
        Z = (X - mu) / sd
        model = RidgeCV(alphas=np.logspace(1, 6, 12)).fit(Z[train_idx], Y[train_idx])
        P = model.predict(Z[val_idx])
        print(f"  [{label}] n_train={len(train_idx)}, n_val={len(val_idx)}, selected alpha {model.alpha_:.1f}", flush=True)
        return P

    results = {}
    print("\n=== PART 0: harness, the original random-split protocol over ALL cases ===", flush=True)
    tr, va = train_test_split(np.arange(len(X)), test_size=0.2, random_state=0)
    P0 = fit_score(tr, va, "random 80/20")
    u0, w0 = r2_both(Y[va], P0)
    print(f"  R^2 uniform = {u0:.4f}, variance-weighted = {w0:.4f}; recorded value {RECORDED_R2}")
    harness_ok = min(abs(u0 - RECORDED_R2), abs(w0 - RECORDED_R2)) <= 0.005
    print(f"  harness {'OK: reproduces the recorded value' if harness_ok else 'FAILED: does not reproduce it, do not interpret the rest'}")
    results["part0"] = {"uniform": u0, "weighted": w0, "harness_ok": bool(harness_ok)}

    print("\n=== PART 1: donor-level fit on TRAIN rows, scored on VALIDATION rows (test rows untouched) ===", flush=True)
    tr_i = np.where(found & (split == "train"))[0]
    va_i = np.where(found & (split == "val"))[0]
    P1 = fit_score(tr_i, va_i, "donor-level")
    u1, w1 = r2_both(Y[va_i], P1)
    print(f"  overall R^2: uniform = {u1:.4f}, variance-weighted = {w1:.4f} "
          f"(change from the random-split value: {u1 - u0:+.4f} / {w1 - w0:+.4f})")
    per, pooled = within_tissue(Y[va_i], P1, tissue[va_i])
    print(f"  pooled WITHIN-TISSUE centered R^2 = {pooled:.4f}")
    for t, r in per.items():
        print(f"    {t:>10}: n={r['n']:>3}, centered R^2 = {r['r2_centered']:+.3f}, mean cosine of deviations = {r['mean_cos']:+.3f}")
    sub, Vt = pc_subspace_r2(Y[tr_i], Y[va_i], P1, k=10)
    print(f"  R^2 inside the top-10 principal-component subspace = {sub['top']:.4f}; in the remaining subspace = {sub['rest']:.4f}")
    verdict = ("not computable (no tissue has enough validation rows)" if not np.isfinite(pooled) else
               "reproduces tissue identity but carries little within-tissue structure" if pooled < 0.10 else
               "partial within-tissue structure" if pooled < 0.50 else "substantial within-tissue structure")
    print(f"  reading (pre-stated cutoffs): {verdict}")
    results["part1"] = {"uniform": u1, "weighted": w1, "pooled_within_tissue": pooled, "per_tissue": per,
                        "top10_r2": sub["top"], "rest_r2": sub["rest"], "verdict": verdict}

    print("\n=== PART 2: where the guidance direction lives ===", flush=True)
    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    d_shuf = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance_SHUFFLED_CONTROL.npz")["rna_embedding_guidance"].astype(np.float64)
    shares = {"true": pc_share(d_true, Vt), "shuffled": pc_share(d_shuf, Vt)}
    print(f"  share of squared norm in the top-k principal components of the training embeddings "
          f"(expected for a direction with no preference: k/768)")
    for k in ("5", "10", "20", "50"):
        print(f"    k={k:>2}: true {shares['true'][k]:.3f} | shuffled {shares['shuffled'][k]:.3f} | no preference {int(k)/768:.3f}")
    results["part2"] = shares

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
