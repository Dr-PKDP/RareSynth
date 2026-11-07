import sys, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
SRC, DST = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(SRC)
X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
assert np.allclose(X.data[:100000], np.round(X.data[:100000])), "not raw counts"
sym = a.var["gene_name"].astype(str).values
keep = ~pd.Series(sym).duplicated().values
var = pd.DataFrame(index=sym[keep]); var["ensembl_id"] = np.array(a.var_names.astype(str))[keep]
pert = a.obs["gene"].astype(str).replace({"non-targeting": "control"}).values
obs = pd.DataFrame({"perturbation": pert}, index=a.obs_names.astype(str))
b = ad.AnnData(X=X[:, keep].astype(np.float32), obs=obs, var=var)
b.write_h5ad(DST)
print(b.shape, "controls", int((pert == "control").sum()), "UFM1", int((pert == "UFM1").sum()))
