"""TCGA RNA embeddings: case index, AnnData, Geneformer tokenization and CLS embedding extraction (.npz)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def find_star_file(case_id, tcga_manifest, raw_dir):
    rec = tcga_manifest.get(case_id)
    if not rec:
        return None
    rna_files = rec.get("files", {}).get("rna", [])
    if not rna_files:
        return None
    rna_files = sorted(rna_files, key=lambda f: f["file_id"])
    f = rna_files[0]
    p = Path(raw_dir) / f["file_id"] / f["file_name"]
    return str(p) if p.exists() else None


def load_case_paths(case_index_csv, tcga_manifest_path, raw_dir, split_filter=None):
    manifest = json.loads(Path(tcga_manifest_path).read_text())
    out = {}
    missing_on_disk = []
    with open(case_index_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("has_rna") not in ("True", "1", "true"):
                continue
            if split_filter and row.get("split") not in split_filter:
                continue
            case_id = row["case_id"]
            path = find_star_file(case_id, manifest, raw_dir)
            if path is None:
                missing_on_disk.append(case_id)
                continue
            out[case_id] = path

    if missing_on_disk:
        print(f"  {len(missing_on_disk)} case(s) marked has_rna=True in the "
             f"index but no file found on disk (e.g. {missing_on_disk[:5]}) "
             f"-- check the raw_dir path is correct")
    return out


def run_pipeline(case_index_csv, tcga_manifest_path, raw_dir, geneformer_repo,
                 out_dir, split_filter=None, model_version="V2",
                 model_size="104M", forward_batch_size=200, nproc=16):
    from ..encoders.rna_to_anndata import build_anndata_from_star_files, load_geneformer_vocab

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

    if not token_dict_path.exists():
        raise FileNotFoundError(f"token dictionary not found at {token_dict_path} "
                               f"-- verify the actual filename with `find` first")
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found at {model_dir}")

    print(f"loading Geneformer {model_version} vocabulary from {token_dict_path}")
    vocab = load_geneformer_vocab(token_dict_path)
    print(f"  {len(vocab)} genes in vocabulary")

    print("\nlocating STAR-counts files for cases in the index...")
    case_paths = load_case_paths(case_index_csv, tcga_manifest_path, raw_dir,
                                 split_filter=split_filter)
    print(f"  {len(case_paths)} cases with an RNA file found on disk")
    if not case_paths:
        raise RuntimeError("no cases to process -- check paths above")

    h5ad_dir = out / "h5ad_input"
    h5ad_dir.mkdir(exist_ok=True)
    h5ad_path = h5ad_dir / "cohort.h5ad"
    print(f"\nbuilding AnnData -> {h5ad_path}")
    adata, dropped = build_anndata_from_star_files(case_paths, h5ad_path, gene_vocab=vocab)

    from geneformer import TranscriptomeTokenizer
    tokenized_dir = out / "tokenized"
    tokenized_dir.mkdir(exist_ok=True)
    print(f"\ntokenizing -> {tokenized_dir}")
    tk_kwargs = dict(nproc=nproc)
    custom_attrs = {"case_id": "case_id"}
    try:
        tk = TranscriptomeTokenizer(custom_attrs, model_version=model_version, **tk_kwargs)
    except TypeError:
        print("  WARNING: TranscriptomeTokenizer did not accept model_version "
             "as a keyword -- verify manually which dictionary it defaulted "
             "to before trusting the output embeddings")
        tk = TranscriptomeTokenizer(custom_attrs, **tk_kwargs)
    tk.tokenize_data(str(h5ad_dir), str(tokenized_dir), "cohort", file_format="h5ad")

    from geneformer import EmbExtractor
    print(f"\nextracting CLS embeddings from {model_dir}")
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        max_ncells=None,
        emb_label=["case_id"],
        forward_batch_size=forward_batch_size,
        model_version=model_version,
        nproc=nproc,
    )
    emb_out_dir = out / "embeddings"
    emb_out_dir.mkdir(exist_ok=True)
    embs = embex.extract_embs(
        str(model_dir),
        str(tokenized_dir / "cohort.dataset"),
        str(emb_out_dir),
        "cohort_embs",
    )

    if "case_id" not in embs.columns:
        raise RuntimeError(
            "embs has no 'case_id' column even after requesting emb_label="
            "['case_id'] -- the custom attribute did not survive the "
            "tokenize -> extract pipeline. Do not guess row alignment; stop "
            "and inspect embs.columns / a fresh tokenized dataset by hand "
            "before proceeding."
        )

    returned_ids_list = list(embs["case_id"])
    returned_ids = set(returned_ids_list)
    if len(returned_ids) != len(returned_ids_list):
        from collections import Counter
        dupes = [cid for cid, n in Counter(returned_ids_list).items() if n > 1]
        raise RuntimeError(
            f"embs contains DUPLICATE case_id values ({len(dupes)} ids "
            f"appear more than once, e.g. {dupes[:5]}) -- this indicates "
            f"real corruption, not a benign quality exclusion. Stop and "
            f"investigate before trusting anything from this run."
        )

    input_ids = set(case_paths.keys())
    unexpected_extra = returned_ids - input_ids
    if unexpected_extra:
        raise RuntimeError(
            f"embs contains {len(unexpected_extra)} case_id(s) that were "
            f"never in the input (e.g. {list(unexpected_extra)[:5]}) -- "
            f"this indicates real corruption, not a benign quality "
            f"exclusion. Stop and investigate before trusting this run."
        )

    missing = input_ids - returned_ids
    if missing:
        frac = len(missing) / len(input_ids)
        print(f"\n{len(missing)}/{len(input_ids)} ({frac:.1%}) input cases "
             f"did not come back from tokenization/embedding extraction. "
             f"This is most often TranscriptomeTokenizer's internal "
             f"coding+miRNA gene-subset filter excluding a low-quality "
             f"sample, not a pipeline bug -- but the missing ids are listed "
             f"below so this can be checked, and they are recorded in "
             f"dropped_case_ids.txt alongside the output.")
        for cid in sorted(missing)[:20]:
            print(f"  MISSING: {cid}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more (full list in "
                 f"dropped_case_ids.txt)")
        (out / "dropped_case_ids.txt").write_text("\n".join(sorted(missing)) + "\n")
        if frac > 0.05:
            raise RuntimeError(
                f"{frac:.1%} of input cases were dropped -- this is too "
                f"large a fraction to treat as routine quality exclusion. "
                f"Stop and investigate rather than silently proceeding with "
                f"a shrunken cohort."
            )

    case_ids_out = embs["case_id"].values
    emb_cols = [c for c in embs.columns if c != "case_id"]
    emb_matrix = embs[emb_cols].values.astype(np.float32)

    np.savez(
        out / "rna_embeddings.npz",
        case_ids=case_ids_out,
        embeddings=emb_matrix,
    )
    print(f"\nsaved -> {out / 'rna_embeddings.npz'}")
    print(f"dropped during AnnData build: {len(dropped)} (see log above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", required=True)
    ap.add_argument("--tcga-manifest", required=True)
    ap.add_argument("--raw-dir", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/data/raw/tcga")
    ap.add_argument("--geneformer-repo", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/tools/Geneformer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs="+", default=None,
                    help="restrict to these split values, e.g. --split test "
                         "for a quick smoke test on the smallest split")
    ap.add_argument("--model-version", default="V2", choices=["V1", "V2"])
    ap.add_argument("--model-size", default="104M")
    ap.add_argument("--forward-batch-size", type=int, default=200)
    ap.add_argument("--nproc", type=int, default=16)
    args = ap.parse_args()

    run_pipeline(
        args.case_index, args.tcga_manifest, args.raw_dir, args.geneformer_repo,
        args.out, split_filter=args.split, model_version=args.model_version,
        model_size=args.model_size, forward_batch_size=args.forward_batch_size,
        nproc=args.nproc,
    )


if __name__ == "__main__":
    main()
