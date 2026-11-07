"""Parse the Pfib_423 gene-count matrices into the AnnData format used for TCGA."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_ENSG_PREFIX = re.compile(r"^(ENSG\d+)")


def strip_kremer_gene_suffix(gene_id: str) -> str:
    m = _ENSG_PREFIX.match(gene_id)
    return m.group(1) if m else None


def build_kremer_anndata(gene_counts_path, out_h5ad_path, gene_vocab=None,
                         min_genes_detected=200):
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

    df = df.groupby(base_ids.values).sum()
    if gene_vocab is not None:
        before = len(df)
        df = df.loc[df.index.isin(gene_vocab)]
        print(f"  vocabulary filter: {before} -> {len(df)} genes")

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
