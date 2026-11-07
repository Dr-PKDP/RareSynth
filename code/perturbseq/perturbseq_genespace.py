"""Gene-space check: does the LINCS UFM1 knockdown signature match the measured K562 UFM1 knockdown signature?
Decision rule (fixed before running): UFM1 upper-tail rank fraction <= 0.05 among all perturbations with >= MINC cells."""
import sys, glob, gzip, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy.stats import rankdata

H5, DIRNPZ, LINCSDIR = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else "genespace_result.json"
import os; MINC = int(os.environ.get("MINC", 50))
rng = np.random.default_rng(0)

cand = sorted(glob.glob(LINCSDIR + "/**/*gene_info*", recursive=True))
if not cand:
    sys.exit("no *gene_info* file under %s ; contents: %s" % (LINCSDIR, glob.glob(LINCSDIR + "/*")[:30]))
gi = pd.read_csv(cand[0], sep="\t", compression="gzip" if cand[0].endswith(".gz") else None, dtype=str)
e2s = dict(zip(gi["pr_gene_id"].astype(str), gi["pr_gene_symbol"].astype(str)))

d = np.load(DIRNPZ, allow_pickle=True)
ids = [str(x) for x in d["measured_gene_ids"]]
dirs = {"knockdown": d["knockdown_direction"].astype(np.float64)}
cons = DIRNPZ.replace("ufm1_guidance_direction", "ufm1_consensus_direction")
try: dirs["consensus"] = np.load(cons, allow_pickle=True)["consensus_direction"].astype(np.float64)
except Exception as e: print("consensus direction not loaded:", e)
sym = np.array([e2s.get(i, "") for i in ids])

a = ad.read_h5ad(H5)
X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float32)
lab = a.obs["perturbation"].astype(str).values
genes = np.array(a.var_names.astype(str))
tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
X = sp.diags((1e4 / tot).astype(np.float32)) @ X
X.data = np.log1p(X.data)
X = X.tocsr()
shared_g = [g for g in genes if g in set(sym) and g != ""]
gi_x = {g: i for i, g in enumerate(genes)}; gi_d = {}
for i, s in enumerate(sym):
    if s and s not in gi_d: gi_d[s] = i
sg = [g for g in genes if g in gi_d]
ix = np.array([gi_x[g] for g in sg]); idd = np.array([gi_d[g] for g in sg])
print("genes shared between LINCS signature and Perturb-seq:", len(sg))
X = X[:, ix].tocsr()

ctrl = np.where(lab == "control")[0]
Xc = X[ctrl]
m_c = np.asarray(Xc.mean(0)).ravel()
s_c = np.sqrt(np.maximum(np.asarray(Xc.multiply(Xc).mean(0)).ravel() - m_c**2, 0))
ok = s_c > 0
print("controls", len(ctrl), "usable genes", int(ok.sum()))

def tstat(mean_p, n_p, mean_c, n_c):
    return (mean_p - mean_c) / np.sqrt(s_c**2 / n_p + s_c**2 / n_c + 1e-12)

def spear(a_, b_):
    return float(np.corrcoef(rankdata(a_), rankdata(b_))[0, 1])

perts = [g for g in sorted(set(lab)) if g != "control" and (lab == g).sum() >= MINC]
rows, cols, npert = [], [], []
for k, g in enumerate(perts):
    ii = np.where(lab == g)[0]; rows += [k] * len(ii); cols += list(ii); npert.append(len(ii))
M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(perts), X.shape[0]))
npert = np.array(npert)
Mean = (M @ X) / npert[:, None]
Mean = np.asarray(Mean.todense()) if sp.issparse(Mean) else np.asarray(Mean)
T = np.array([tstat(Mean[k], npert[k], m_c, len(ctrl))[ok] for k in range(len(perts))])
print("perturbations scored", len(perts))

res = {"n_shared_genes": int(len(sg)), "n_usable_genes": int(ok.sum()), "n_perturbations": len(perts), "n_controls": int(len(ctrl))}
for name, dv in dirs.items():
    v = dv[idd][ok]
    rs = np.array([spear(T[k], v) for k in range(len(perts))])
    r = {}
    for g in ("UFM1", "UBA5", "UFL1"):
        if g in perts:
            k = perts.index(g)
            r[g] = {"n_cells": int(npert[k]), "spearman": float(rs[k]), "rank": int((rs >= rs[k]).sum()),
                    "rank_fraction": float((rs >= rs[k]).sum() / len(rs))}
    # control null: random control groups sized like UFM1, against remaining controls
    n_u = int(npert[perts.index("UFM1")]) if "UFM1" in perts else 130
    nulls = []
    for _ in range(200):
        pick = rng.permutation(len(ctrl)); a_i, b_i = pick[:n_u], pick[n_u:]
        ma, mb = np.asarray(Xc[a_i].mean(0)).ravel(), np.asarray(Xc[b_i].mean(0)).ravel()
        nulls.append(spear(tstat(ma, n_u, mb, len(b_i))[ok], v))
    nulls = np.array(nulls)
    r["control_null_spearman_mean_sd"] = [float(nulls.mean()), float(nulls.std())]
    r["rs_distribution_perturbations_mean_sd"] = [float(rs.mean()), float(rs.std())]
    r["top10_perturbations"] = [(perts[i], round(float(rs[i]), 3)) for i in np.argsort(-rs)[:10]]
    res[name] = r
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
