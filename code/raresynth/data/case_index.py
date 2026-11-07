"""Case index: one row per case with donor, project, RNA availability and the donor-level split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _hash_split(donor_id, val_frac=0.1, test_frac=0.1):
    h = int(hashlib.md5(donor_id.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000.0
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def build_tcga_index(manifest_path, out_path, val_frac=0.1, test_frac=0.1):
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for case_id, rec in manifest.items():
        files = rec.get("files", {})
        rows.append({
            "case_id": case_id,
            "donor_id": case_id,
            "project": rec.get("project"),
            "cohort": "tcga",
            "has_rna": bool(files.get("rna")),
            "split": _hash_split(case_id, val_frac, test_frac),
        })
    _write_index(rows, out_path)
    _print_summary(rows, "TCGA")
    return rows


def build_kremer_index(sample_annotation_paths, out_path, project_name="Pfib_423"):
    if isinstance(sample_annotation_paths, (str, Path)):
        sample_annotation_paths = [sample_annotation_paths]

    rows = []
    seen_case_ids = set()
    for path in sample_annotation_paths:
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for r in reader:
                sample_id = r.get("RNA_ID") or r.get("SAMPLE_ID") or r.get("sampleID")
                individual_id = r.get("INDIVIDUAL_ID") or sample_id
                if not sample_id:
                    continue
                if sample_id in seen_case_ids:
                    print(f"  duplicate case_id '{sample_id}' across annotation files; keeping the first occurrence")
                    continue
                seen_case_ids.add(sample_id)
                rows.append({
                    "case_id": sample_id,
                    "donor_id": individual_id,
                    "project": project_name,
                    "cohort": project_name.lower(),
                    "has_rna": True,
                    "split": "external_val_2",
                    "disease_code": r.get("ICD_10", ""),
                })

    _write_index(rows, out_path)
    _print_summary(rows, "Pfib_423")
    return rows


def _write_index(rows, out_path):
    if not rows:
        raise RuntimeError(f"no rows to write for {out_path}")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_summary(rows, label):
    n = len(rows)
    print(f"\n{label}: {n} cases")
    for split in ("train", "val", "test", "external_val_2"):
        n_split = sum(1 for r in rows if r["split"] == split)
        if n_split:
            print(f"  {split:16s} {n_split:6d}")
    n_rna = sum(1 for r in rows if r.get("has_rna"))
    print(f"  has_rna          {n_rna:6d} ({100 * n_rna / n:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tcga")
    t.add_argument("--manifest", required=True)
    t.add_argument("--out", required=True)

    k = sub.add_parser("kremer")
    k.add_argument("--sample-annotation", required=True, nargs="+")
    k.add_argument("--project-name", default="Pfib_423")
    k.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "tcga":
        build_tcga_index(args.manifest, args.out)
    else:
        build_kremer_index(args.sample_annotation, args.out, args.project_name)


if __name__ == "__main__":
    main()
