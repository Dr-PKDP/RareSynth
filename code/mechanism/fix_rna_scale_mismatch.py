"""Fit the ridge map from LINCS-gene expression to L2-normalized RNA embeddings, map the UFM1 direction into embedding space and compute the mean GTEx fibroblast embedding."""

import sys
import json

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")

import anndata as ad
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

TCGA_H5AD = "/data/pduttapramanik/raresynth/data/embeddings/rna_tcga_full/h5ad_input/cohort.h5ad"
TCGA_EMB_PATH = "/data/pduttapramanik/raresynth/data/embeddings/rna_tcga_full/rna_embeddings.npz"
GTEX_EMB_PATH = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/rna_embeddings.npz"
SYMBOL_TO_ENSEMBL_PATH = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted/lincs_measured_gene_symbol_to_ensembl.json"
GENE_INFO_PATH_UFM1 = "/data/pduttapramanik/raresynth/data/raw/mechanism/lincs/GSE92742_Broad_LINCS_gene_info.txt.gz"
UFM1_CONSENSUS_PATH = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted/ufm1_consensus_direction.npz"

PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
GTEX_DIR = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast"

def l2_normalize(x, axis=1):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norms + 1e-8)


print("################ STEP 1: refit the linear map on L2-normalized TCGA RNA ################")

a = ad.read_h5ad(TCGA_H5AD)
X_full_case_ids = a.obs["case_id"].values
X_full = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
tcga_genes = np.array(a.var.index, dtype=str)

emb_data = np.load(TCGA_EMB_PATH, allow_pickle=True)
Y_case_ids = emb_data["case_ids"]
Y_full_raw = emb_data["embeddings"]
Y_full_normalized = l2_normalize(Y_full_raw)
print(f"  Y before normalization: mean norm {np.linalg.norm(Y_full_raw, axis=1).mean():.4f}")
print(f"  Y after normalization: mean norm {np.linalg.norm(Y_full_normalized, axis=1).mean():.4f} "
     f"(expect ~1.0)")

case_id_to_y_idx = {cid: i for i, cid in enumerate(Y_case_ids)}
assert all(cid in case_id_to_y_idx for cid in X_full_case_ids), "case_id mismatch"
y_reorder_idx = np.array([case_id_to_y_idx[cid] for cid in X_full_case_ids])
Y_full = Y_full_normalized[y_reorder_idx]
assert list(Y_case_ids[y_reorder_idx]) == list(X_full_case_ids), "pairing failed"
print(f"  {len(X_full_case_ids)} cases, explicitly paired, verified exact match")

symbol_to_ensembl = json.loads(open(SYMBOL_TO_ENSEMBL_PATH).read())
lincs_ensembl_ids = set(symbol_to_ensembl.values())
tcga_gene_set = set(tcga_genes)
overlap_ensembl = sorted(lincs_ensembl_ids & tcga_gene_set)
tcga_gene_to_col = {g: i for i, g in enumerate(tcga_genes)}
overlap_cols = np.array([tcga_gene_to_col[g] for g in overlap_ensembl])
X_overlap = X_full[:, overlap_cols]
print(f"  {len(overlap_ensembl)} overlap genes")

X_log = np.log1p(X_overlap)
train_idx, val_idx = train_test_split(np.arange(len(X_log)), test_size=0.2, random_state=0)
gene_mean = X_log[train_idx].mean(axis=0)
gene_std = X_log[train_idx].std(axis=0) + 1e-8
X_z = (X_log - gene_mean) / gene_std

X_train, X_val = X_z[train_idx], X_z[val_idx]
Y_train, Y_val = Y_full[train_idx], Y_full[val_idx]

alphas = np.logspace(1, 6, 12)
model = RidgeCV(alphas=alphas)
model.fit(X_train, Y_train)
print(f"  selected alpha: {model.alpha_:.2f}")

Y_pred_val = model.predict(X_val)
r2_overall = r2_score(Y_val, Y_pred_val)
r2_per_dim = r2_score(Y_val, Y_pred_val, multioutput="raw_values")
print(f"  held-out R^2 (on L2-normalized targets): {r2_overall:.4f}")
print(f"  R^2 per-dim: mean={r2_per_dim.mean():.4f}, min={r2_per_dim.min():.4f}, "
     f"max={r2_per_dim.max():.4f}")

MAP_OUT_PATH = f"{PPN_DIR}/lincs_to_rna_embedding_map.npz"
np.savez_compressed(
    MAP_OUT_PATH,
    coef=model.coef_.astype(np.float32),
    intercept=model.intercept_.astype(np.float32),
    overlap_ensembl_genes=np.array(overlap_ensembl, dtype=str),
    gene_mean=gene_mean.astype(np.float32),
    gene_std=gene_std.astype(np.float32),
    alpha=model.alpha_,
    held_out_r2_overall=r2_overall,
    held_out_r2_per_dim=r2_per_dim.astype(np.float32),
    fit_on_l2_normalized_targets=True,
)
print(f"  saved corrected map -> {MAP_OUT_PATH}")


print("\n################ STEP 2: reapply the corrected map to the UFM1 direction ################")

import gzip
import pandas as pd

ufm1_data = np.load(UFM1_CONSENSUS_PATH, allow_pickle=True)
ufm1_direction = ufm1_data["consensus_direction"]
ufm1_measured_gene_ids = ufm1_data["measured_gene_ids"]

with gzip.open(GENE_INFO_PATH_UFM1, "rt") as f:
    gene_info = pd.read_csv(f, sep="\t")
pr_gene_id_to_symbol = dict(zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_gene_symbol"]))

ufm1_row_ensembl = []
for pr_id in ufm1_measured_gene_ids:
    symbol = pr_gene_id_to_symbol.get(str(pr_id))
    ens_id = symbol_to_ensembl.get(symbol) if symbol else None
    ufm1_row_ensembl.append(ens_id)
ensembl_to_ufm1_row = {ens: i for i, ens in enumerate(ufm1_row_ensembl) if ens is not None}

aligned_direction = np.zeros(len(overlap_ensembl), dtype=np.float32)
n_missing = 0
for i, ens in enumerate(overlap_ensembl):
    row = ensembl_to_ufm1_row.get(ens)
    if row is None:
        n_missing += 1
        continue
    aligned_direction[i] = ufm1_direction[row]
assert n_missing == 0, f"{n_missing} genes missing -- investigate"

coef = model.coef_
rna_embedding_guidance = coef @ aligned_direction
print(f"  corrected guidance vector: norm {np.linalg.norm(rna_embedding_guidance):.4f} ")

GUIDANCE_OUT_PATH = f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz"
np.savez_compressed(
    GUIDANCE_OUT_PATH,
    rna_embedding_guidance=rna_embedding_guidance.astype(np.float32),
    source_map_held_out_r2=r2_overall,
    n_genes_used=len(overlap_ensembl) - n_missing,
    fit_on_l2_normalized_targets=True,
)
print(f"  saved corrected guidance vector -> {GUIDANCE_OUT_PATH}")


print("\n################ STEP 3: recompute the GTEx fibroblast baseline (L2-normalized) ################")

gtex_emb_data = np.load(GTEX_EMB_PATH, allow_pickle=True)
gtex_embeddings_raw = gtex_emb_data["embeddings"]
gtex_embeddings_normalized = l2_normalize(gtex_embeddings_raw)
fibroblast_baseline = gtex_embeddings_normalized.mean(axis=0)
print(f"  fibroblast baseline norm: {np.linalg.norm(fibroblast_baseline):.4f} ")

FIB_BASELINE_OUT_PATH = f"{GTEX_DIR}/fibroblast_baseline.npz"
np.savez_compressed(
    FIB_BASELINE_OUT_PATH,
    baseline=fibroblast_baseline.astype(np.float32),
    n_samples=len(gtex_embeddings_normalized),
    fit_on_l2_normalized=True,
)
print(f"  saved corrected fibroblast baseline -> {FIB_BASELINE_OUT_PATH}")
