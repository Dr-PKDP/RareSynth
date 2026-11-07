"""Map the 12,328 LINCS measured gene symbols to Ensembl ids (MyGene.info, symbol and alias queries)."""

import gzip
import json
from pathlib import Path

import anndata as ad
import mygene
import pandas as pd

LINCS_GENE_INFO = "/data/pduttapramanik/raresynth/data/raw/mechanism/lincs/GSE92742_Broad_LINCS_gene_info.txt.gz"
TCGA_H5AD = "/data/pduttapramanik/raresynth/data/embeddings/rna_tcga_full/h5ad_input/cohort.h5ad"
OUT_PATH = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted/lincs_measured_gene_symbol_to_ensembl.json"

with gzip.open(LINCS_GENE_INFO, "rt") as f:
    gene_info = pd.read_csv(f, sep="\t")
lincs_symbols = sorted(gene_info["pr_gene_symbol"].unique())

result = mygene.MyGeneInfo().querymany(lincs_symbols, scopes="symbol,alias", fields="ensembl.gene",
                                       species="human", verbose=False)
symbol_to_ensembl = {}
for r in result:
    if r.get("notfound") or not r.get("ensembl"):
        continue
    ens = r["ensembl"]
    symbol_to_ensembl[r["query"]] = ens[0]["gene"] if isinstance(ens, list) else ens["gene"]

Path(OUT_PATH).write_text(json.dumps(symbol_to_ensembl, indent=2))
print(f"{len(symbol_to_ensembl)}/{len(lincs_symbols)} symbols resolved; saved -> {OUT_PATH}")

tcga_genes = set(ad.read_h5ad(TCGA_H5AD).var.index)
print(f"overlap with the TCGA gene set: {len(set(symbol_to_ensembl.values()) & tcga_genes)} genes")
