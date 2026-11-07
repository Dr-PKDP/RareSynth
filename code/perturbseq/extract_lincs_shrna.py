"""Extract the LINCS L1000 (GSE92742) Level 5 shRNA signatures used in Supplementary Information S5.

Writes to OUT_DIR:
  trt_sh_matrix.npz         matrix (genes x signatures, float32), gene_ids, sig_ids   (all trt_sh signatures)
  trt_sh_metadata.csv       sig_info rows of those signatures
  ufm1_guidance_direction.npz  knockdown_direction = mean of all trt_sh signatures of UFM1, measured_gene_ids, n_knockdown_sigs

Usage: python extract_lincs_shrna.py LINCS_DIR OUT_DIR [--compare EXISTING_ufm1_guidance_direction.npz]
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

ap = argparse.ArgumentParser()
ap.add_argument("lincs_dir"); ap.add_argument("out_dir"); ap.add_argument("--compare", default=None)
ap.add_argument("--gene", default="UFM1")
a = ap.parse_args()

GCTX = f"{a.lincs_dir}/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
with gzip.open(f"{a.lincs_dir}/GSE92742_Broad_LINCS_sig_info.txt.gz", "rt") as f:
    info = pd.read_csv(f, sep="\t", dtype=str)
sh = info[info["pert_type"] == "trt_sh"].reset_index(drop=True)
print(len(sh), "trt_sh signatures")
df = parse(GCTX, cid=sh["sig_id"].tolist()).data_df
sh = sh.set_index("sig_id").loc[df.columns].reset_index().rename(columns={"cid": "sig_id"})
gene_ids = np.array([str(x) for x in df.index], dtype=str)
sig_ids = np.array([str(x) for x in df.columns], dtype=str)
os.makedirs(a.out_dir, exist_ok=True)
np.savez(f"{a.out_dir}/trt_sh_matrix.npz", matrix=df.values.astype(np.float32),
         gene_ids=gene_ids, sig_ids=sig_ids)
sh.to_csv(f"{a.out_dir}/trt_sh_metadata.csv", index=False)
sel = (sh["pert_iname"] == a.gene).values
kd = df.values[:, sel].astype(np.float64).mean(1).astype(np.float32)
np.savez(f"{a.out_dir}/ufm1_guidance_direction.npz", knockdown_direction=kd,
         measured_gene_ids=gene_ids, n_knockdown_sigs=int(sel.sum()))
print(a.gene, "signatures:", int(sel.sum()), "cell lines:", sorted(sh.loc[sel, "cell_id"].unique()))
if a.compare:
    ref = np.load(a.compare, allow_pickle=True)["knockdown_direction"].astype(np.float64)
    k = kd.astype(np.float64)
    print("cosine with", a.compare, float(ref @ k / np.linalg.norm(ref) / np.linalg.norm(k)))
