"""
Build the unified per-case manifest joining all four modalities' real
output, for the training Dataset to read from.

This is the single source of truth: case_id -> which modalities are
genuinely available (verified from what actually exists on disk, not
what a case_index CSV claims), plus where to find each modality's data
(an index position into a combined .npz for RNA/clinical/genomic, or a
file path for pathology's per-case .npz).

Scope, matching what has actually been built and verified in this
project so far: TCGA only. RNA also exists for CPTAC/Pfib_423, but
genomic/clinical/pathology do not (all deliberately scoped to TCGA only
during their own build-out -- see PROGRESS.md), so a genuinely joined
multi-modal manifest is only meaningful for TCGA right now. CPTAC and
Pfib_423 remain usable as RNA-only external validation cohorts via their
own separate embedding files, not through this joined manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def build_tcga_manifest(case_index_csv, rna_npz_path, clinical_npz_path,
                        genomic_npz_path, pathology_dir, out_path):
    # load each modality's real case_id list (ground truth: what actually
    # got saved, not what a case_index CSV predicted would be available)
    rna = np.load(rna_npz_path, allow_pickle=True)
    rna_ids = {cid: i for i, cid in enumerate(rna["case_ids"].tolist())}

    clinical = np.load(clinical_npz_path, allow_pickle=True)
    clinical_ids = {cid: i for i, cid in enumerate(clinical["case_ids"].tolist())}

    genomic = np.load(genomic_npz_path, allow_pickle=True)
    genomic_ids = {cid: i for i, cid in enumerate(genomic["case_ids"].tolist())}
    genomic_n_valid = genomic["n_valid_genes"]

    pathology_dir = Path(pathology_dir)
    pathology_files = {
        p.stem: str(p) for p in pathology_dir.glob("*.npz")
    }

    print(f"real per-modality case counts: rna={len(rna_ids)}, "
         f"clinical={len(clinical_ids)}, genomic={len(genomic_ids)}, "
         f"pathology={len(pathology_files)}")

    # base case universe + split/donor info comes from the case index
    manifest = {}
    with open(case_index_csv) as fh:
        for row in csv.DictReader(fh):
            case_id = row["case_id"]
            entry = {
                "donor_id": row.get("donor_id", case_id),
                "split": row.get("split", "unknown"),
                "project": row.get("project", ""),
                "has_rna": case_id in rna_ids,
                "has_clinical": case_id in clinical_ids,
                "has_genomic": case_id in genomic_ids,
                "has_pathology": case_id in pathology_files,
                "has_radiology": False,  # no radiology encoder exists yet
                                        # in this project -- explicit, not
                                        # an oversight (see PROGRESS.md)
            }
            if entry["has_rna"]:
                entry["rna_index"] = rna_ids[case_id]
            if entry["has_clinical"]:
                entry["clinical_index"] = clinical_ids[case_id]
            if entry["has_genomic"]:
                entry["genomic_index"] = genomic_ids[case_id]
                entry["genomic_n_valid"] = int(genomic_n_valid[genomic_ids[case_id]])
            if entry["has_pathology"]:
                entry["pathology_path"] = pathology_files[case_id]

            n_modalities = sum([entry["has_rna"], entry["has_clinical"],
                               entry["has_genomic"], entry["has_pathology"]])
            entry["n_modalities_available"] = n_modalities
            manifest[case_id] = entry

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    # summary
    n = len(manifest)
    print(f"\n{n} total cases in manifest")
    for mod in ("rna", "clinical", "genomic", "pathology", "radiology"):
        c = sum(1 for e in manifest.values() if e[f"has_{mod}"])
        print(f"  has_{mod:10s} {c:5d} ({100*c/n:.1f}%)")

    from collections import Counter
    dist = Counter(e["n_modalities_available"] for e in manifest.values())
    print("\nmodalities-available distribution (of rna/clinical/genomic/pathology):")
    for k in sorted(dist):
        print(f"  {k} modalities: {dist[k]} cases ({100*dist[k]/n:.1f}%)")

    split_dist = Counter(e["split"] for e in manifest.values())
    print("\nsplit distribution:")
    for k, v in sorted(split_dist.items()):
        print(f"  {k}: {v}")

    print(f"\nsaved -> {out}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", required=True)
    ap.add_argument("--rna-npz", required=True)
    ap.add_argument("--clinical-npz", required=True)
    ap.add_argument("--genomic-npz", required=True)
    ap.add_argument("--pathology-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_tcga_manifest(args.case_index, args.rna_npz, args.clinical_npz,
                        args.genomic_npz, args.pathology_dir, args.out)


if __name__ == "__main__":
    main()
