"""
Run the Kremer external-validation-2 cohort through Geneformer.

Deliberately a thin wrapper reusing run_rna_encoder's already-verified
tokenize/embed/identity-check logic rather than a parallel reimplementation
-- only the upstream AnnData construction differs (kremer_rna.py handles the
transposed, double-suffixed, single-combined-matrix format; everything after
that point is identical to the TCGA/CPTAC path and carries the same
case_id-tracking guarantees verified there).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def run(gene_counts_path, geneformer_repo, out_dir, model_version="V2",
       model_size="104M", forward_batch_size=200, nproc=16):
    from .kremer_rna import build_kremer_anndata
    from .rna_to_anndata import load_geneformer_vocab

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(geneformer_repo)

    if model_version == "V2":
        token_dict_path = repo / "geneformer" / f"token_dictionary_gc{model_size}.pkl"
        model_dir = repo / f"Geneformer-V2-{model_size}"
    else:
        token_dict_path = (repo / "geneformer" / "gene_dictionaries_30m"
                          / "token_dictionary_gc30M.pkl")
        model_dir = repo / "Geneformer-V1-10M"

    print(f"loading vocabulary from {token_dict_path}")
    vocab = load_geneformer_vocab(token_dict_path)
    print(f"  {len(vocab)} genes in vocabulary")

    h5ad_dir = out / "h5ad_input"
    h5ad_dir.mkdir(exist_ok=True)
    h5ad_path = h5ad_dir / "kremer.h5ad"
    print(f"\nbuilding AnnData from {gene_counts_path}")
    adata, dropped_build = build_kremer_anndata(gene_counts_path, h5ad_path, gene_vocab=vocab)
    n_input = adata.shape[0]

    from geneformer import TranscriptomeTokenizer
    tokenized_dir = out / "tokenized"
    tokenized_dir.mkdir(exist_ok=True)
    print(f"\ntokenizing -> {tokenized_dir}")
    custom_attrs = {"case_id": "case_id"}
    tk = TranscriptomeTokenizer(custom_attrs, model_version=model_version, nproc=nproc)
    tk.tokenize_data(str(h5ad_dir), str(tokenized_dir), "kremer", file_format="h5ad")

    from geneformer import EmbExtractor
    print(f"\nextracting CLS embeddings from {model_dir}")
    embex = EmbExtractor(
        model_type="Pretrained", num_classes=0, emb_mode="cls",
        max_ncells=None, emb_label=["case_id"],
        forward_batch_size=forward_batch_size, model_version=model_version, nproc=nproc,
    )
    emb_out_dir = out / "embeddings"
    emb_out_dir.mkdir(exist_ok=True)
    embs = embex.extract_embs(str(model_dir), str(tokenized_dir / "kremer.dataset"),
                              str(emb_out_dir), "kremer_embs")

    # identical identity-verification logic to run_rna_encoder.py -- see
    # PROGRESS.md bugs #6/#7 for why this is not optional
    if "case_id" not in embs.columns:
        raise RuntimeError("embs has no case_id column -- stop, do not guess alignment")
    returned_ids_list = list(embs["case_id"])
    returned_ids = set(returned_ids_list)
    if len(returned_ids) != len(returned_ids_list):
        from collections import Counter
        dupes = [c for c, n in Counter(returned_ids_list).items() if n > 1]
        raise RuntimeError(f"DUPLICATE case_id in output: {dupes}")
    input_ids = set(adata.obs["case_id"])
    unexpected = returned_ids - input_ids
    if unexpected:
        raise RuntimeError(f"UNEXPECTED case_id in output: {unexpected}")
    missing = input_ids - returned_ids
    if missing:
        frac = len(missing) / len(input_ids)
        print(f"\n{len(missing)}/{len(input_ids)} ({frac:.1%}) input samples "
             f"did not come back. Missing: {sorted(missing)}")
        (out / "dropped_case_ids.txt").write_text("\n".join(sorted(missing)) + "\n")
        if frac > 0.05:
            raise RuntimeError(f"{frac:.1%} missing -- too large, investigate")

    case_ids_out = embs["case_id"].values
    emb_cols = [c for c in embs.columns if c != "case_id"]
    emb_matrix = embs[emb_cols].values.astype(np.float32)

    np.savez(out / "rna_embeddings.npz", case_ids=case_ids_out, embeddings=emb_matrix)
    print(f"\nsaved -> {out / 'rna_embeddings.npz'}")
    print(f"  n={len(case_ids_out)}, unique_ids={len(set(case_ids_out.tolist()))}, "
         f"unique_rows={len(set(tuple(r) for r in emb_matrix))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-counts", required=True)
    ap.add_argument("--geneformer-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-version", default="V2", choices=["V1", "V2"])
    ap.add_argument("--model-size", default="104M")
    ap.add_argument("--forward-batch-size", type=int, default=200)
    ap.add_argument("--nproc", type=int, default=16)
    args = ap.parse_args()
    run(args.gene_counts, args.geneformer_repo, args.out,
       model_version=args.model_version, model_size=args.model_size,
       forward_batch_size=args.forward_batch_size, nproc=args.nproc)


if __name__ == "__main__":
    main()
