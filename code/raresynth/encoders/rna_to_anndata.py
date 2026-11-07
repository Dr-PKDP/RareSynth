"""Assemble an AnnData object of raw STAR counts, indexed by unversioned Ensembl id, from per-case GDC files."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .rna_parser import parse_star_counts_tsv


def load_geneformer_vocab(token_dictionary_path):
    with open(token_dictionary_path, "rb") as fh:
        token_dict = pickle.load(fh)
    return {k for k in token_dict.keys() if isinstance(k, str) and k.startswith("ENSG")}


def build_anndata_from_star_files(case_paths, out_h5ad_path, gene_vocab=None,
                                  obs_extra=None, min_genes_detected=200):
    import anndata as ad

    parsed = {}
    dropped = []
    all_genes = set()

    for case_id, path in case_paths.items():
        try:
            expr = parse_star_counts_tsv(path, value_col="unstranded")
        except Exception as e:
            dropped.append((case_id, f"parse failed: {type(e).__name__}: {e}"))
            continue
        if gene_vocab is not None:
            expr = {g: v for g, v in expr.items() if g in gene_vocab}
        n_detected = sum(1 for v in expr.values() if v > 0)
        if n_detected < min_genes_detected:
            dropped.append((case_id,
                           f"only {n_detected} genes detected (< {min_genes_detected})"))
            continue
        parsed[case_id] = expr
        all_genes.update(expr.keys())

    if not parsed:
        raise RuntimeError(
            f"no samples survived parsing -- {len(dropped)} dropped, 0 kept. "
            f"First few reasons: {dropped[:5]}"
        )

    gene_list = sorted(all_genes)
    gene_idx = {g: i for i, g in enumerate(gene_list)}
    case_list = sorted(parsed.keys())

    rows, cols, vals = [], [], []
    for i, case_id in enumerate(case_list):
        for g, v in parsed[case_id].items():
            if v > 0:
                rows.append(i)
                cols.append(gene_idx[g])
                vals.append(v)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(case_list), len(gene_list)),
                      dtype=np.float32)

    obs = pd.DataFrame(index=case_list)
    obs["case_id"] = case_list
    obs["n_counts"] = np.asarray(X.sum(axis=1)).flatten()
    if obs_extra:
        for col in sorted({k for d in obs_extra.values() for k in d.keys()}):
            obs[col] = [obs_extra.get(cid, {}).get(col, "") for cid in case_list]

    var = pd.DataFrame(index=gene_list)
    var["ensembl_id"] = gene_list

    adata = ad.AnnData(X=X, obs=obs, var=var)

    out = Path(out_h5ad_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)

    print(f"wrote {out} : {adata.shape[0]} samples x {adata.shape[1]} genes")
    print(f"dropped {len(dropped)} sample(s)")
    if dropped:
        for cid, reason in dropped[:10]:
            print(f"  {cid}: {reason}")
        if len(dropped) > 10:
            print(f"  ... and {len(dropped) - 10} more")

    return adata, dropped
