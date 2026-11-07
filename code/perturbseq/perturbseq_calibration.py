"""Calibration of the K562 CRISPRi vs LINCS shRNA gene-space comparison.
For each perturbed gene g with >= MINSIG LINCS shRNA signatures, rank the matching LINCS gene-g signature among all LINCS gene signatures
by Spearman correlation with the K562 CRISPRi t-statistic profile of g. Chance rank fraction: uniform on (0,1]."""
import sys, glob
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, json
from scipy.stats import rankdata
H5, SHNPZ, SHMETA, GENEINFO = sys.argv[1:5]
OUT = sys.argv[5] if len(sys.argv) > 5 else "calibration.json"
import os; MINC, MINSIG = int(os.environ.get("MINC", 50)), 3

gi = pd.read_csv(GENEINFO, sep="\t", dtype=str, compression="gzip" if GENEINFO.endswith(".gz") else None)
e2s = dict(zip(gi["pr_gene_id"].astype(str), gi["pr_gene_symbol"].astype(str)))
z = np.load(SHNPZ, allow_pickle=True)
gids = [str(x) for x in z["gene_ids"]]; sigids = [str(x) for x in z["sig_ids"]]
sym = np.array([e2s.get(i, "") for i in gids])
print("loading shRNA matrix ...", flush=True)
Mx_full = z["matrix"]
print("matrix", Mx_full.shape, flush=True)
meta = pd.read_csv(SHMETA, usecols=["sig_id", "pert_iname"], dtype=str)
col = {s: i for i, s in enumerate(sigids)}
meta = meta[meta.sig_id.isin(col)]
by = {g: [col[s] for s in sub.sig_id] for g, sub in meta.groupby("pert_iname")}

a = ad.read_h5ad(H5)
X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float32)
lab = a.obs["perturbation"].astype(str).values; genes = np.array(a.var_names.astype(str))
tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
X = sp.diags((1e4 / tot).astype(np.float32)) @ X; X.data = np.log1p(X.data); X = X.tocsr()
gx = {g: i for i, g in enumerate(genes)}; gd = {}
for i, s in enumerate(sym):
    if s and s not in gd: gd[s] = i
sg = [g for g in genes if g in gd]
ix = np.array([gx[g] for g in sg]); idd = np.array([gd[g] for g in sg])
X = X[:, ix].tocsr()
ctrl = np.where(lab == "control")[0]; Xc = X[ctrl]
m_c = np.asarray(Xc.mean(0)).ravel()
s_c = np.sqrt(np.maximum(np.asarray(Xc.multiply(Xc).mean(0)).ravel() - m_c**2, 0)); ok = s_c > 0
perts = [g for g in sorted(set(lab)) if g != "control" and (lab == g).sum() >= MINC and len(by.get(g, [])) >= MINSIG]
print("shared genes", int(ok.sum()), "perturbations with >=%d LINCS sigs: %d" % (MINSIG, len(perts)), flush=True)
rows, cols, npert = [], [], []
for k, g in enumerate(perts):
    ii = np.where(lab == g)[0]; rows += [k] * len(ii); cols += list(ii); npert.append(len(ii))
Mp = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(perts), X.shape[0])); npert = np.array(npert)
Mean = np.asarray((Mp @ X).todense()) / npert[:, None]
T = (Mean - m_c) / np.sqrt(s_c**2 / npert[:, None] + s_c**2 / len(ctrl) + 1e-12)
T = T[:, ok]
strength = np.sqrt((T**2).mean(1))
# LINCS: mean z-signature for every gene with >= MINSIG sigs (the comparison panel)
panel = [g for g, v in by.items() if len(v) >= MINSIG]
Mx = Mx_full[idd]; del Mx_full
L = np.array([Mx[:, by[g]].mean(1)[ok] for g in panel])
print("LINCS panel genes", len(panel), flush=True)
rT = rankdata(T, axis=1); rL = rankdata(L, axis=1)
rT = (rT - rT.mean(1, keepdims=True)) / np.linalg.norm(rT - rT.mean(1, keepdims=True), axis=1, keepdims=True)
rL = (rL - rL.mean(1, keepdims=True)) / np.linalg.norm(rL - rL.mean(1, keepdims=True), axis=1, keepdims=True)
C = rT @ rL.T
pidx = {g: i for i, g in enumerate(panel)}
fr, cm = [], []
for k, g in enumerate(perts):
    v = C[k]; c = v[pidx[g]]; fr.append(float((v >= c).sum() / len(v))); cm.append(float(c))
fr, cm = np.array(fr), np.array(cm)
res = {"n_genes_tested": len(perts), "n_panel": len(panel), "median_rank_fraction": float(np.median(fr)),
       "fraction_rank_le_0.05": float((fr <= 0.05).mean()), "fraction_rank_le_0.01": float((fr <= 0.01).mean()),
       "chance_le_0.05": 0.05, "median_matched_spearman": float(np.median(cm))}
q = np.quantile(strength, [0, .25, .5, .75, 1.0]); strata = {}
for i in range(4):
    s = (strength >= q[i]) & (strength <= q[i + 1]) if i == 3 else (strength >= q[i]) & (strength < q[i + 1])
    strata["Q%d(strength %.2f-%.2f)" % (i + 1, q[i], q[i + 1])] = {"n": int(s.sum()), "median_rank_fraction": float(np.median(fr[s])), "fraction_le_0.05": float((fr[s] <= 0.05).mean())}
res["by_knockdown_strength_quartile"] = strata
for g in ("UFM1", "UBA5", "UFL1"):
    if g in perts:
        k = perts.index(g); res[g] = {"rank_fraction": fr[k], "matched_spearman": cm[k], "strength": float(strength[k]), "strength_percentile": float((strength <= strength[k]).mean())}
    else: res[g] = "not testable (fewer than %d LINCS sigs or <%d cells)" % (MINSIG, MINC)
best = np.argsort(fr)[:15]; res["best_matched_genes"] = [(perts[i], round(float(fr[i]), 4)) for i in best]
json.dump(res, open(OUT, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
