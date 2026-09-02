"""
Case-level index -- the foundation everything in the encoder pipeline reads
from. Every encoder script below takes this index as input rather than
re-deriving case/modality availability from raw manifests each time.

Split policy
------------
TCGA is split train/val/test at the DONOR level (not sample level -- a donor
can contribute more than one RNA/MAF/clinical record, e.g. tumor plus
normal-adjacent, and letting the same donor appear in both train and test
would leak identity and inflate every fidelity metric). The split is a
deterministic hash of the donor id, not a random draw, so it reproduces
identically across runs without needing to persist a random seed state
anywhere.

CPTAC and Kremer are NOT split train/val/test at all -- they are held out
in full as external validation cohorts 1 and 2 respectively, per the paper
design. Training on any part of an external validation set would invalidate
the validation.

Modality availability is read from what is actually ON DISK, not from what a
manifest claims should exist -- the slide subset, for instance, is 2,000 of
4,255 eligible cases, and the manifest alone cannot tell you which 2,000
without cross-referencing the case list actually downloaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _hash_split(donor_id, val_frac=0.1, test_frac=0.1):
    """Deterministic donor -> {'train','val','test'} assignment.

    md5 rather than Python's built-in hash(): hash() is salted per-process
    in modern Python for security, so it gives a DIFFERENT split every time
    the script runs unless PYTHONHASHSEED is fixed externally. md5 is stable
    across runs, machines, and Python versions with no configuration needed.
    """
    h = int(hashlib.md5(donor_id.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000.0
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def build_tcga_index(manifest_path, slide_subset_path, tcia_dir, out_path,
                     val_frac=0.1, test_frac=0.1):
    manifest = json.loads(Path(manifest_path).read_text())
    slide_cases = set(Path(slide_subset_path).read_text().split())

    # Radiology availability read directly from disk. NBIA nests patient
    # directories under an intermediate folder named after the manifest's
    # databasketId (e.g. tcia_manifest/<Collection>/<PatientID>/), and that
    # intermediate name is not fixed -- an earlier version of this function
    # assumed exactly two levels of nesting (glob("*/*")) and matched
    # COLLECTION directories (e.g. "TCGA-KIRC") instead of patient
    # directories, which of course never matched any real case id and
    # produced 0% radiology coverage despite the data being on disk.
    # Searching recursively for directories whose name is a known case id
    # is robust to whatever depth NBIA actually used.
    rad_cases = _find_case_dirs(tcia_dir, set(manifest.keys()))

    rows = []
    for case_id, rec in manifest.items():
        files = rec.get("files", {})
        rows.append({
            "case_id": case_id,
            "donor_id": case_id,  # TCGA submitter_id IS the donor for our purposes
            "project": rec.get("project"),
            "cohort": "tcga",
            "has_rna": bool(files.get("rna")),
            "has_maf": bool(files.get("maf")),
            "has_clinical": bool(files.get("clinical")),
            "has_slide": case_id in slide_cases,
            "has_radiology": case_id in rad_cases,
            "split": _hash_split(case_id, val_frac, test_frac),
        })

    _write_index(rows, out_path)
    _print_summary(rows, "TCGA")
    return rows


def _find_case_dirs(root_dir, known_case_ids):
    """Recursively find directories under root_dir whose name matches a
    known case id, regardless of nesting depth. Used for on-disk modality
    detection where the exact intermediate folder structure (which depends
    on the manifest's databasketId, an implementation detail of the NBIA
    client) should not be assumed.
    """
    found = set()
    root = Path(root_dir)
    if not root.exists():
        return found
    for p in root.rglob("*"):
        if p.is_dir() and p.name in known_case_ids:
            found.add(p.name)
    return found


def build_cptac_index(manifest_path, cptac_tcia_dir, out_path):
    manifest = json.loads(Path(manifest_path).read_text())
    rad_cases = _find_case_dirs(cptac_tcia_dir, set(manifest.keys()))

    rows = []
    for case_id, rec in manifest.items():
        files = rec.get("files", {})
        rows.append({
            "case_id": case_id,
            "donor_id": case_id,
            "project": rec.get("project"),
            "cohort": "cptac",
            "has_rna": bool(files.get("rna")),
            "has_maf": bool(files.get("maf")),
            "has_clinical": bool(files.get("clinical")),
            "has_slide": False,  # CPTAC pathology comes via TCIA (DICOM), not GDC
            "has_radiology": case_id in rad_cases,
            "split": "external_val_1",  # entire cohort held out, no train/val/test
        })

    _write_index(rows, out_path)
    _print_summary(rows, "CPTAC (external validation 1)")
    return rows


def build_kremer_index(sample_annotation_paths, out_path, project_name="Pfib_423"):
    """Real Mendelian/mitochondrial disease, RNA-seq only. Entirely held out
    as external validation 2.

    sample_annotation_paths: one path, or a list of paths to combine into a
    single cohort (Pfib_423 ships as two separate files -- fib_ns and
    fib_ss, non/strand-specific -- with the SAME schema, confirmed against
    real downloaded headers rather than assumed).

    Column names fixed to match the REAL confirmed header (RNA_ID,
    INDIVIDUAL_ID, TISSUE, SEX, AFFECTED, ICD_10, PAIRED_END,
    STRAND_SPECIFIC) -- an earlier version looked for DISEASE/
    KNOWN_MUTATION columns that do not exist in this file format at all,
    which silently left has_clinical=False and disease_code="" for every
    row since the original Kremer-only index was first built. That bug was
    never caught because _print_summary omits any modality at exactly 0%
    rather than printing it, making a real absence and a silent lookup bug
    visually identical -- fixed alongside this.
    """
    if isinstance(sample_annotation_paths, (str, Path)):
        sample_annotation_paths = [sample_annotation_paths]

    rows = []
    seen_case_ids = set()
    for path in sample_annotation_paths:
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            fieldnames = reader.fieldnames or []
            if "ICD_10" not in fieldnames:
                print(f"  WARNING: {path} has no ICD_10 column (fields: "
                     f"{fieldnames}) -- disease_code/has_clinical will be "
                     f"empty for every row in this file, verify the header "
                     f"before trusting this index")
            for r in reader:
                sample_id = r.get("RNA_ID") or r.get("SAMPLE_ID") or r.get("sampleID")
                individual_id = r.get("INDIVIDUAL_ID") or sample_id
                if not sample_id:
                    continue
                if sample_id in seen_case_ids:
                    print(f"  WARNING: duplicate case_id '{sample_id}' across "
                         f"combined annotation files -- keeping first "
                         f"occurrence only, check for real sample overlap "
                         f"between the input files")
                    continue
                seen_case_ids.add(sample_id)
                icd10 = r.get("ICD_10", "")
                rows.append({
                    "case_id": sample_id,
                    "donor_id": individual_id,
                    "project": project_name,
                    "cohort": project_name.lower(),
                    "has_rna": True,
                    "has_maf": False,
                    "has_clinical": bool(icd10),
                    "has_slide": False,
                    "has_radiology": False,
                    "split": "external_val_2",
                    "disease_code": icd10,
                })

    _write_index(rows, out_path)
    _print_summary(rows, "Kremer (external validation 2)")
    return rows


def _write_index(rows, out_path):
    if not rows:
        raise RuntimeError(f"no rows to write for {out_path} -- check input paths")
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
    for split in ("train", "val", "test", "external_val_1", "external_val_2"):
        n_split = sum(1 for r in rows if r["split"] == split)
        if n_split:
            print(f"  {split:16s} {n_split:6d}")
    for modality in ("rna", "maf", "clinical", "slide", "radiology"):
        n_mod = sum(1 for r in rows if r.get(f"has_{modality}"))
        # always print, even at 0% -- a real absence and a silent lookup
        # bug (e.g. checking for a column name that does not exist in the
        # file) look identical if zero-count rows are omitted, which is
        # exactly how the has_clinical bug in build_kremer_index went
        # unnoticed for as long as it did
        print(f"  has_{modality:12s} {n_mod:6d} ({100*n_mod/n:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tcga")
    t.add_argument("--manifest", required=True)
    t.add_argument("--slide-subset", required=True)
    t.add_argument("--tcia-dir", required=True)
    t.add_argument("--out", required=True)

    c = sub.add_parser("cptac")
    c.add_argument("--manifest", required=True)
    c.add_argument("--tcia-dir", required=True)
    c.add_argument("--out", required=True)

    k = sub.add_parser("kremer")
    k.add_argument("--sample-annotation", required=True, nargs="+",
                   help="one or more sample_annotation.tsv paths to combine "
                        "into a single cohort")
    k.add_argument("--project-name", default="Pfib_423")
    k.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "tcga":
        build_tcga_index(args.manifest, args.slide_subset, args.tcia_dir, args.out)
    elif args.cmd == "cptac":
        build_cptac_index(args.manifest, args.tcia_dir, args.out)
    elif args.cmd == "kremer":
        build_kremer_index(args.sample_annotation, args.out, args.project_name)


if __name__ == "__main__":
    main()
