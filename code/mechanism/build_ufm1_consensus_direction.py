"""Mean of the UFM1 trt_sh.cgs consensus signatures (LINCS L1000 Level 5, GSE92742) in LINCS gene space."""

import gzip

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

LINCS = "/data/pduttapramanik/raresynth/data/raw/mechanism/lincs"
GCTX = f"{LINCS}/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIG_INFO = f"{LINCS}/GSE92742_Broad_LINCS_sig_info.txt.gz"
OUT_PATH = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted/ufm1_consensus_direction.npz"
GENE = "UFM1"

with gzip.open(SIG_INFO, "rt") as f:
    sig_info = pd.read_csv(f, sep="\t")
subset = sig_info[(sig_info["pert_type"] == "trt_sh.cgs") & (sig_info["pert_iname"] == GENE)].copy()
print(f"{len(subset)} signatures; cell lines: {sorted(subset['cell_id'].unique())}")

df = parse(GCTX, cid=subset["sig_id"].tolist()).data_df
subset_indexed = subset.set_index("sig_id")
assert not [s for s in df.columns if s not in subset_indexed.index], "unmatched signature ids"
aligned = subset_indexed.loc[df.columns].reset_index()
assert list(aligned["sig_id"]) == list(df.columns)

matrix = df.values
measured_gene_ids = np.array(df.index, dtype=str)
unit = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-12)
cos = unit.T @ unit
off_diag = cos[~np.eye(cos.shape[0], dtype=bool)]

direction = matrix.mean(axis=1)
print(f"direction: {direction.shape}, norm {np.linalg.norm(direction):.3f}; "
      f"mean pairwise cosine between signatures {off_diag.mean():.4f}")

np.savez_compressed(
    OUT_PATH,
    consensus_direction=direction.astype(np.float32),
    measured_gene_ids=measured_gene_ids,
    n_consensus_sigs=matrix.shape[1],
    cell_lines=np.array(aligned["cell_id"], dtype=str),
    mean_pairwise_cosine=off_diag.mean(),
)
print(f"saved -> {OUT_PATH}")
