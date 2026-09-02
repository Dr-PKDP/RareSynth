"""
Genomic encoding: case index -> real MAF file -> build_gene_annotation_matrix
(using all four real annotation sources + the three external gene-level
resources parsed today) -> case_id-tracked combined .npz.

TCGA only for today. CPTAC deferred deliberately: CPTAC's GDC "project"
field is just "CPTAC-2"/"CPTAC-3" (not tissue-specific), unlike TCGA's
per-cancer-type project codes, so it needs its own tissue-mapping logic
(likely from the CPTAC-TCIA collection names, e.g. CPTAC-CCRCC -> Kidney)
that has not been built yet -- consistent with CPTAC pathology also being
deferred earlier in this project.

Storage: ONE combined .npz (not one file per case, unlike pathology).
build_gene_annotation_matrix always returns a FIXED-SHAPE (top_k, 14)
array (zero-padded if a case has fewer than top_k genes with any variant),
unlike pathology's genuinely variable-length tile bags -- so a simple
stacked array, matching the RNA/clinical storage pattern, is the natural
and sufficient format here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# TCGA project code -> GTEx SMTS tissue name, confirmed against GTEx's
# real 30-tissue vocabulary before hardcoding (Breast, Lung, Kidney,
# Brain, Colon, Ovary, Stomach all confirmed present under these exact
# simple names -- no surprises, unlike some GTEx tissues which use more
# specific naming)
PROJECT_TO_TISSUE = {
    "TCGA-BRCA": "Breast",
    "TCGA-LUAD": "Lung",
    "TCGA-KIRC": "Kidney",
    "TCGA-GBM": "Brain",
    "TCGA-LGG": "Brain",
    "TCGA-COAD": "Colon",
    "TCGA-OV": "Ovary",
    "TCGA-STAD": "Stomach",
}


def find_maf_file(case_id, tcga_manifest, raw_dir):
    """Same file-location pattern as run_rna_encoder.find_star_file,
    applied to the 'maf' file type."""
    rec = tcga_manifest.get(case_id)
    if not rec:
        return None
    maf_files = rec.get("files", {}).get("maf", [])
    if not maf_files:
        return None
    maf_files = sorted(maf_files, key=lambda f: f["file_id"])
    f = maf_files[0]
    p = Path(raw_dir) / f["file_id"] / f["file_name"]
    return str(p) if p.exists() else None


def load_case_maf_paths(case_index_csv, tcga_manifest_path, raw_dir):
    import csv

    manifest = json.loads(Path(tcga_manifest_path).read_text())
    out = {}
    project_by_case = {}
    missing = []
    with open(case_index_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("has_maf") not in ("True", "1", "true"):
                continue
            project = row.get("project", "")
            if project not in PROJECT_TO_TISSUE:
                continue  # not one of our 8 mapped projects -- skip cleanly
            case_id = row["case_id"]
            path = find_maf_file(case_id, manifest, raw_dir)
            if path is None:
                missing.append(case_id)
                continue
            out[case_id] = path
            project_by_case[case_id] = project
    if missing:
        print(f"  {len(missing)} case(s) marked has_maf=True but no file "
             f"found on disk (e.g. {missing[:5]})")
    return out, project_by_case


def run_pipeline(case_index_csv, tcga_manifest_path, raw_dir, annotations_dir,
                 priors_dir, gtex_dir, out_dir, top_k=512, limit=None):
    from ..encoders.genomic import build_gene_annotation_matrix, D_GENE_FEAT
    from ..encoders.annotation_lookup import BigWigLookup
    from .clinvar_parser import parse_clinvar_plp_genes
    from .gnomad_constraint import parse_gnomad_constraint
    from .gtex_expression import build_gtex_median_tpm
    from .gene_vocabulary import build_gene_vocabulary, build_placeholder_gene_embedding

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ann = Path(annotations_dir)
    priors = Path(priors_dir)

    print("=== loading external resources (once, not per case) ===")
    t0 = time.time()
    clinvar_plp = parse_clinvar_plp_genes(priors / "clinvar.vcf.gz")
    print(f"  clinvar: {time.time()-t0:.1f}s")

    t0 = time.time()
    gnomad_constraint = parse_gnomad_constraint(
        priors / "gnomad.v4.1.constraint_metrics.tsv"
    )
    print(f"  gnomad constraint: {time.time()-t0:.1f}s")

    t0 = time.time()
    gtex_median_tpm = build_gtex_median_tpm(
        Path(gtex_dir) / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
        Path(gtex_dir) / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    )
    print(f"  gtex tissue expression: {time.time()-t0:.1f}s")

    gene_index = build_gene_vocabulary(gnomad_constraint)
    gene_embedding = build_placeholder_gene_embedding(gene_index)
    print(f"  gene vocabulary: {len(gene_index)} genes (placeholder "
         f"embedding, see gene_vocabulary.py docstring)")

    print("\n=== loading annotation lookups ===")
    am_path = ann / "AlphaMissense_hg38.tsv.gz"
    cadd_lookup = BigWigLookup(ann / "CADD_GRCh38-v1.7.bw")
    gerp_lookup = BigWigLookup(ann / "gerp_conservation_scores.homo_sapiens.GRCh38.bw")
    phylop_lookup = BigWigLookup(ann / "hg38.phyloP100way.bw")
    print("  all four annotation sources loaded")

    print("\n=== locating MAF files ===")
    case_paths, project_by_case = load_case_maf_paths(
        case_index_csv, tcga_manifest_path, raw_dir
    )
    print(f"  {len(case_paths)} cases with a MAF file found on disk")
    if limit is not None:
        case_paths = dict(list(case_paths.items())[:limit])
        print(f"  --limit {limit}: restricting to {len(case_paths)} cases")
    if not case_paths:
        raise RuntimeError("no cases to process -- check paths above")

    all_feats, all_gene_idx, all_n_valid, case_ids_out = [], [], [], []
    n_parse_fail = 0

    print(f"\n=== processing {len(case_paths)} cases ===")
    for i, (case_id, maf_path) in enumerate(case_paths.items()):
        try:
            maf_df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
        except Exception as e:
            print(f"  [{i+1}/{len(case_paths)}] {case_id}: MAF parse FAILED "
                 f"({type(e).__name__}: {e}), skipping")
            n_parse_fail += 1
            continue

        tissue = PROJECT_TO_TISSUE[project_by_case[case_id]]
        feats, gene_idx, n_valid = build_gene_annotation_matrix(
            maf_df, gene_index, clinvar_plp, gnomad_constraint,
            gtex_median_tpm=gtex_median_tpm, tissue=tissue, top_k=top_k,
            am_path=am_path, cadd_lookup=cadd_lookup,
            gerp_lookup=gerp_lookup, phylop_lookup=phylop_lookup,
        )
        all_feats.append(feats)
        all_gene_idx.append(gene_idx)
        all_n_valid.append(n_valid)
        case_ids_out.append(case_id)

        if (i + 1) % 50 == 0 or (i + 1) == len(case_paths):
            print(f"  [{i+1}/{len(case_paths)}] {case_id}: {n_valid} "
                 f"genes with variant data")

    if not case_ids_out:
        raise RuntimeError(f"no cases succeeded ({n_parse_fail} parse failures)")

    case_ids_arr = np.array(case_ids_out)
    feats_arr = np.stack(all_feats).astype(np.float32)   # (N, top_k, D_GENE_FEAT)
    gene_idx_arr = np.stack(all_gene_idx).astype(np.int64)  # (N, top_k)
    n_valid_arr = np.array(all_n_valid, dtype=np.int64)

    assert len(set(case_ids_out)) == len(case_ids_out), \
        "duplicate case_ids in output -- this should be structurally impossible " \
        "given dict-keyed input, investigate immediately if it ever triggers"

    np.savez(out / "genomic_embeddings.npz",
             case_ids=case_ids_arr, gene_features=feats_arr,
             gene_indices=gene_idx_arr, n_valid_genes=n_valid_arr)
    np.savez(out / "gene_vocabulary.npz",
             gene_symbols=np.array(list(gene_index.keys())),
             gene_embedding=gene_embedding)

    print(f"\nsaved -> {out / 'genomic_embeddings.npz'}")
    print(f"  n={len(case_ids_out)}, unique_ids={len(set(case_ids_out))}, "
         f"shape={feats_arr.shape}, parse failures={n_parse_fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", required=True)
    ap.add_argument("--tcga-manifest", required=True)
    ap.add_argument("--raw-dir", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/data/raw/tcga")
    ap.add_argument("--annotations-dir", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/data/raw/annotations")
    ap.add_argument("--priors-dir", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/data/raw/priors")
    ap.add_argument("--gtex-dir", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/data/raw/gtex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_pipeline(args.case_index, args.tcga_manifest, args.raw_dir,
                args.annotations_dir, args.priors_dir, args.gtex_dir,
                args.out, top_k=args.top_k, limit=args.limit)


if __name__ == "__main__":
    main()
