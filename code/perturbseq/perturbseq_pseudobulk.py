"""Build equal-size pseudobulk profiles (raw counts) per perturbation and for control cell groups, as a Geneformer h5ad input."""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

SRC, OUT = sys.argv[1], sys.argv[2]
NCELL = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = np.random.default_rng(0)

a = ad.read_h5ad(SRC)
X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
assert np.allclose(X.data, np.round(X.data)), "X is not raw counts"
pert = a.obs["perturbation"].astype(str).values
ens = a.var["ensembl_id"].astype(str).str.split(".").str[0].values
first = ~pd.Series(ens).duplicated().values
X, ens = X[:, first], ens[first]
print("cells", X.shape[0], "genes", X.shape[1])

rows, cols, case_ids, kinds, genes = [], [], [], [], []
n_groups = 0
for g in sorted(set(pert)):
    idx = np.where(pert == g)[0]
    if g == "control":
        idx = rng.permutation(idx)
        for k in range(len(idx) // NCELL):
            members = idx[k * NCELL:(k + 1) * NCELL]
            kind, name = "control", f"control|control|{k}"
            rows += [n_groups] * NCELL; cols += list(members)
            case_ids.append(name); kinds.append(kind); genes.append("control"); n_groups += 1
    elif len(idx) >= NCELL:
        members = rng.choice(idx, NCELL, replace=False)
        rows += [n_groups] * NCELL; cols += list(members)
        case_ids.append(f"pert|{g}|0"); kinds.append("pert"); genes.append(g); n_groups += 1
M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_groups, X.shape[0]))
P = (M @ X).astype(np.float32).tocsr()

var = pd.DataFrame(index=ens); var["ensembl_id"] = ens
obs = pd.DataFrame(index=case_ids)
obs["case_id"] = case_ids; obs["kind"] = kinds; obs["gene"] = genes
obs["n_counts"] = np.asarray(P.sum(axis=1)).ravel()
out = ad.AnnData(X=P, var=var, obs=obs)
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.write_h5ad(OUT)
print("pseudobulks", n_groups, "controls", int((obs.kind == "control").sum()), "perturbations", int((obs.kind == "pert").sum()))
print("UFM1/UBA5/UFL1 present:", [g for g in ("UFM1", "UBA5", "UFL1") if g in set(genes)])
print("saved", OUT)
