"""
Build the AnnData (.h5ad) input Geneformer's TranscriptomeTokenizer requires,
from GDC STAR-counts files.

Confirmed against the actual installed tokenizer.py source (not a tutorial,
not a guess) before writing this:

    Required format: raw counts, .h5ad/.loom/.zarr
    Required var (gene) attribute:  "ensembl_id"
    Required obs (cell/sample) attribute: "n_counts" (total counts)

This means the correct GDC column is "unstranded" (raw STAR counts), NOT
"tpm_unstranded" -- an earlier draft of the RNA encoder (GeneformerEncoder in
encoders/foundation.py) assumed TPM input with a hand-rolled median-division
step. That does not match how Geneformer's normalization actually works
(rank value encoding scaled by n_counts and the gene median dictionary
together) and has been superseded by this module plus the real
TranscriptomeTokenizer / EmbExtractor classes. foundation.py's
GeneformerEncoder should not be used.

The gene set is restricted to Geneformer's own token vocabulary up front.
Any gene not in that vocabulary is dropped by the tokenizer regardless. This
keeps the matrix a consistent, known shape across every sample and cohort
without silently including genes that will never be used.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .rna_parser import parse_star_counts_tsv


def load_geneformer_vocab(token_dictionary_path):
    """Return the set of unversioned Ensembl gene ids Geneformer's tokenizer
    can actually use. Any gene outside this set is dropped by the tokenizer
    regardless of what we feed in, so filtering here keeps the AnnData
    matrix meaningfully sized rather than carrying dead columns.
    """
    with open(token_dictionary_path, "rb") as fh:
        token_dict = pickle.load(fh)
    # keys are gene ids (Ensembl or the small set of special tokens like
    # <pad>/<cls>/<eos> depending on model version) -- keep only ones that
    # look like Ensembl gene ids
    return {k for k in token_dict.keys() if isinstance(k, str) and k.startswith("ENSG")}


def build_anndata_from_star_files(case_paths, out_h5ad_path, gene_vocab=None,
                                  obs_extra=None, min_genes_detected=200):
    """
    case_paths : dict case_id -> path to a GDC STAR-counts .tsv(.gz)
    gene_vocab : optional set of unversioned Ensembl ids to restrict to
                (pass Geneformer's own vocabulary here -- see
                load_geneformer_vocab)
    obs_extra  : optional dict case_id -> dict of extra obs columns
                (e.g. {"organ": "kidney", "cell_type": "tumor"}) to carry
                through as the TranscriptomeTokenizer's custom_attr_name_dict
                source
    min_genes_detected : samples with fewer nonzero genes than this are
                dropped and reported -- a near-empty row is very likely a
                parsing or QC failure, not real biology, and would corrupt
                the per-sample rank encoding silently otherwise.

    Returns (AnnData, list of dropped case_ids with reasons) so the caller
    can log what got excluded rather than it disappearing invisibly.
    """
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

    # build the counts matrix sparsely -- most gene x sample entries are 0,
    # and TCGA-scale sample counts x ~20k genes as a dense float64 array
    # would be wasteful for no benefit (the tokenizer masks zeros anyway)
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
    obs["case_id"] = case_list  # explicit column, not just the index --
    # TranscriptomeTokenizer's custom_attr_name_dict carries named obs
    # COLUMNS through tokenization, not the DataFrame index. Without this,
    # there is no way to recover which output row belongs to which sample:
    # confirmed by a live run where EmbExtractor's returned embeddings had
    # only bare numeric columns (0..767) and a plain RangeIndex, with
    # tokenization run under nproc=16 multiprocessing giving no guarantee
    # that row order was preserved from the AnnData input.
    obs["n_counts"] = np.asarray(X.sum(axis=1)).flatten()
    if obs_extra:
        for col in sorted({k for d in obs_extra.values() for k in d.keys()}):
            obs[col] = [obs_extra.get(cid, {}).get(col, "") for cid in case_list]

    var = pd.DataFrame(index=gene_list)
    var["ensembl_id"] = gene_list  # tokenizer reads this as a column, not just the index

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
