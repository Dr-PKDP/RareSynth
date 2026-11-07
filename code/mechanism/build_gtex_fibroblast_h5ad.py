"""GTEx v8 cultured-fibroblast raw counts as an AnnData input for Geneformer."""

import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")
from raresynth.encoders.rna_to_anndata import load_geneformer_vocab

GCT_PATH = "/data/pduttapramanik/raresynth/data/raw/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
ATTRIBUTES_PATH = "/data/pduttapramanik/raresynth/data/raw/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
TOKEN_DICT_PATH = "/data/pduttapramanik/raresynth/tools/Geneformer/geneformer/token_dictionary_gc104M.pkl"
OUT_PATH = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/h5ad_input/cohort.h5ad"

attributes = pd.read_csv(ATTRIBUTES_PATH, sep="\t", usecols=["SAMPID", "SMTSD"])
sampids = attributes.loc[attributes["SMTSD"] == "Cells - Cultured fibroblasts", "SAMPID"].tolist()
print(f"{len(sampids)} cultured-fibroblast sample ids")

print("\n=== loading Geneformer vocabulary ===")
vocab = load_geneformer_vocab(TOKEN_DICT_PATH)
print(f"  {len(vocab)} genes in vocab")

print("\n=== checking which fibroblast SAMPIDs are actually present as columns "
     "in the GCT file (some annotated samples don't survive GTEx's own QC "
     "into the final expression release) ===")
import gzip
with gzip.open(GCT_PATH, "rt") as f:
    f.readline()
    f.readline()
    header_line = f.readline()
available_columns = set(header_line.rstrip("\n").split("\t"))
sampids_available = [s for s in sampids if s in available_columns]
sampids_missing = [s for s in sampids if s not in available_columns]
print(f"  {len(sampids_available)}/{len(sampids)} fibroblast SAMPIDs present in "
     f"the GCT file's actual columns")
if sampids_missing:
    print(f"  {len(sampids_missing)} missing (not in the final expression "
         f"release, despite having an attribute record) -- proceeding with "
         f"the {len(sampids_available)} available ones")
sampids = sampids_available

print("\n=== reading GCT file, restricted to Name + the available fibroblast columns ===")
print("  (this scans the full 56,200 x 17,382 file even with usecols, so it "
     "may take a few minutes -- run in tmux)")
usecols = ["Name"] + sampids
df = pd.read_csv(GCT_PATH, sep="\t", skiprows=2, usecols=usecols)
print(f"  loaded shape: {df.shape}")

print("\n=== stripping Ensembl version suffixes and restricting to Geneformer vocab ===")
df["ensembl_id"] = df["Name"].str.split(".").str[0]
before = len(df)
df = df[df["ensembl_id"].isin(vocab)].reset_index(drop=True)
print(f"  {len(df)}/{before} genes retained (in Geneformer's vocabulary)")

dup_count = df["ensembl_id"].duplicated().sum()
if dup_count > 0:
    print(f"  WARNING: {dup_count} duplicate unversioned ensembl_id values after "
         f"version-stripping -- keeping first occurrence only")
    df = df.drop_duplicates(subset="ensembl_id", keep="first").reset_index(drop=True)

print("\n=== building AnnData (genes x samples in the file -> samples x genes for AnnData) ===")
counts = df[sampids].values.T.astype(np.float32)
gene_ids = df["ensembl_id"].values

var = pd.DataFrame(index=gene_ids)
var["ensembl_id"] = gene_ids
obs = pd.DataFrame(index=sampids)
obs["case_id"] = sampids
obs["n_counts"] = counts.sum(axis=1)

a = ad.AnnData(X=sp.csr_matrix(counts), var=var, obs=obs)
print(f"  AnnData shape: {a.shape}")
print(f"  n_counts range: [{a.obs['n_counts'].min():.0f}, {a.obs['n_counts'].max():.0f}]")
print(f"  any zero-count samples: {(a.obs['n_counts'] == 0).any()}")

import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
a.write_h5ad(OUT_PATH)
print(f"\nsaved -> {OUT_PATH}")
