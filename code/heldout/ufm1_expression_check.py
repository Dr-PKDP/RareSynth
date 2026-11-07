"""
Expression-level check of the case (POST HOC, DESCRIPTIVE; written 2026-10-21 after the primary result existed).
Question: in the cohort's own raw counts, is UFM1 itself underexpressed in the case relative to the other samples of the
same library preparation?  This is the biology the LINCS knockdown direction stands for; the embedding test cannot tell
us whether the signal is visible in the data at all.  Reads raw counts only (kremer.h5ad); no embeddings.

Method (fixed here): within the case's own library-preparation part (ns or ss), library-size normalize (counts per
million over the retained genes), log1p; z-score each gene against the OTHER samples of that part (case excluded);
report UFM1's z for the case, its rank among the part's samples (1 = lowest), and its rank among all genes with mean
log-CPM >= 1 ordered from most negative z (1 = most underexpressed in the case).  The reference is one part only,
so no batch correction is involved.
"""
import argparse, json, os, sys, time
import numpy as np

UFM1_ENSEMBL = "ENSG00000120686"


def logcpm(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e6)


def case_gene_z(L, case_row, min_mean=1.0):
    ref = np.delete(L, case_row, axis=0)
    mu, sd = ref.mean(axis=0), ref.std(axis=0, ddof=1)
    ok = (L.mean(axis=0) >= min_mean) & (sd > 1e-8)
    z = np.full(L.shape[1], np.nan)
    z[ok] = (L[case_row, ok] - mu[ok]) / sd[ok]
    return z, ok


def summarize(L, genes, case_row, gene_id):
    j = list(genes).index(gene_id)
    z, ok = case_gene_z(L, case_row)
    if not ok[j]:
        return {"error": "gene fails the expression filter in this part"}
    col = L[:, j]
    rank_in_part = int((col < col[case_row]).sum()) + 1          # 1 = lowest among samples
    zz = z[ok]
    rank_among_genes = int((zz < z[j]).sum()) + 1                 # 1 = most underexpressed gene in the case
    return {"z_case_vs_others": float(z[j]), "rank_among_part_samples_low_is_1": rank_in_part, "n_part_samples": int(L.shape[0]),
            "rank_among_genes_most_negative_is_1": rank_among_genes, "n_genes_ranked": int(ok.sum()),
            "case_logcpm": float(col[case_row]), "others_mean_logcpm": float(np.delete(col, case_row).mean())}


def selftest():
    rng = np.random.default_rng(0)
    n, g = 60, 300
    mu = rng.uniform(2, 8, g)
    f = rng.uniform(0.8, 1.2, (n, 1))
    X = rng.poisson(np.exp(mu)[None, :] * f)
    genes = [f"G{i}" for i in range(g)]; genes[5] = UFM1_ENSEMBL
    X[7, 5] = rng.poisson(np.exp(mu[5]) * f[7, 0] * 0.2)                    # the case is low in gene 5
    r = summarize(logcpm(X), genes, 7, UFM1_ENSEMBL)
    assert r["z_case_vs_others"] < -2 and r["rank_among_part_samples_low_is_1"] <= 2 and r["rank_among_genes_most_negative_is_1"] <= 3, r
    X2 = X.copy(); X2[7, 5] = rng.poisson(np.exp(mu[5]) * f[7, 0])         # a typical case is not an outlier
    r2 = summarize(logcpm(X2), genes, 7, UFM1_ENSEMBL)
    assert abs(r2["z_case_vs_others"]) < 3 and r2["rank_among_genes_most_negative_is_1"] > 3, r2
    print("selftest OK"); 


def main():
    ap = argparse.ArgumentParser()
    D = "/data/pduttapramanik/raresynth/data/embeddings/"
    ap.add_argument("--ns", default=D + "rna_pfib423_ns/h5ad_input/kremer.h5ad")
    ap.add_argument("--ss", default=D + "rna_pfib423_ss/h5ad_input/kremer.h5ad")
    ap.add_argument("--case", default="R20754")
    ap.add_argument("--gene", default=UFM1_ENSEMBL)
    ap.add_argument("--out", default="/data/pduttapramanik/raresynth/runs/pfib_test/ufm1_expression_check.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists.")
    import anndata as ad
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "case": a.case, "gene": a.gene, "post_hoc_descriptive": True}
    found = False
    for name, path in (("ns", a.ns), ("ss", a.ss)):
        ad_ = ad.read_h5ad(path)
        ids = [str(x) for x in ad_.obs["case_id"]]
        if a.case not in ids:
            continue
        found = True
        X = ad_.X.toarray() if hasattr(ad_.X, "toarray") else np.asarray(ad_.X)
        genes = [str(x) for x in ad_.var.index]
        if a.gene not in genes:
            sys.exit(f"REFUSING: {a.gene} is not in the {name} gene set (vocabulary filter).")
        res["part"] = name
        res.update(summarize(logcpm(X), genes, ids.index(a.case), a.gene))
    if not found:
        sys.exit(f"REFUSING: {a.case} not found in either part.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, a.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
