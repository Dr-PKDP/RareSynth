"""
Parse the Kremer et al. combined gene-count matrix into the same AnnData
shape build_anndata_from_star_files produces, so it can feed the identical
downstream tokenize/embed pipeline.

Confirmed file structure (checked directly, not assumed):

    geneID          MUC1365  MUC1347  76622  ...   (120 columns: 1 id + 119 samples)
    ENSG00000000003.15_5   899      815      844   ...
    ENSG00000000005.6_4    0        0        0     ...
    ...                                              (62,492 gene rows)

Two things this format needs that the GDC parser did not:

1. Genes are ROWS and samples are COLUMNS -- the transpose of GDC's
   per-case files. The matrix needs transposing to the samples x genes
   convention used everywhere else in this pipeline.

2. Gene ids carry a SECOND suffix beyond the Ensembl version
   (e.g. "ENSG00000000003.15_5") that does not follow the "_2" / "_3"
   collision-disambiguation pattern one might guess (values 5, 4, 6, 7, 7,
   6, 4, 5, 10 seen across consecutive rows, no relation to duplicate
   count). What it encodes was not determined and does not need to be --
   only the base ENSG id is needed to match Geneformer's vocabulary, so it
   is stripped along with the version rather than interpreted.

Sample column headers (e.g. "MUC1365", "76622") match sampleAnnotation.tsv's
RNA_ID field directly -- confirmed against a live head of both files.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_ENSG_PREFIX = re.compile(r"^(ENSG\d+)")


def strip_kremer_gene_suffix(gene_id: str) -> str:
    """'ENSG00000000003.15_5' -> 'ENSG00000000003'. Returns None if the id
    does not start with ENSG followed by digits (should not happen for this
    file, but a malformed row should be dropped, not silently mismatched).
    """
    m = _ENSG_PREFIX.match(gene_id)
    return m.group(1) if m else None


def build_kremer_anndata(gene_counts_path, out_h5ad_path, gene_vocab=None,
                         min_genes_detected=200):
    """Same output contract as build_anndata_from_star_files: samples x
    genes AnnData with obs['case_id'], obs['n_counts'], var['ensembl_id'].
    """
    import anndata as ad
    import scipy.sparse as sp
    from pathlib import Path

    df = pd.read_csv(gene_counts_path, sep="\t", index_col=0)
    sample_ids = list(df.columns)

    base_ids = df.index.map(strip_kremer_gene_suffix)
    n_unparseable = base_ids.isna().sum()
    if n_unparseable:
        print(f"  {n_unparseable} gene id(s) did not match the expected "
             f"ENSG prefix pattern and will be dropped")
    df = df.loc[base_ids.notna()]
    base_ids = base_ids[base_ids.notna()]

    # collapse any duplicate base ids (two versioned/suffixed rows mapping
    # to the same gene after stripping) by summing counts -- same policy
    # as the GDC parser's duplicate handling, reported rather than silent
    df = df.groupby(base_ids.values).sum()
    if gene_vocab is not None:
        before = len(df)
        df = df.loc[df.index.isin(gene_vocab)]
        print(f"  vocabulary filter: {before} -> {len(df)} genes")

    # transpose to samples x genes -- the convention used everywhere else
    X_dense = df.values.T.astype(np.float32)
    gene_list = list(df.index)

    n_detected = (X_dense > 0).sum(axis=1)
    keep_mask = n_detected >= min_genes_detected
    dropped = [(sample_ids[i], f"only {n_detected[i]} genes detected "
                                f"(< {min_genes_detected})")
              for i in range(len(sample_ids)) if not keep_mask[i]]

    X_kept = X_dense[keep_mask]
    samples_kept = [s for s, k in zip(sample_ids, keep_mask) if k]
    if not samples_kept:
        raise RuntimeError(f"no samples survived filtering -- {len(dropped)} dropped")

    obs = pd.DataFrame(index=samples_kept)
    obs["case_id"] = samples_kept
    obs["n_counts"] = X_kept.sum(axis=1)

    var = pd.DataFrame(index=gene_list)
    var["ensembl_id"] = gene_list

    adata = ad.AnnData(X=sp.csr_matrix(X_kept), obs=obs, var=var)

    out = Path(out_h5ad_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)

    print(f"wrote {out} : {adata.shape[0]} samples x {adata.shape[1]} genes")
    print(f"dropped {len(dropped)} sample(s)")
    for sid, reason in dropped:
        print(f"  {sid}: {reason}")

    return adata, dropped
