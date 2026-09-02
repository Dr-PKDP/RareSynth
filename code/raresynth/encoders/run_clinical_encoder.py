"""
Run TCGA clinical BCR XML through ClinicalBERT: case index -> extracted
fields -> templated text -> embedding -> case_id-aligned .npz.

TCGA only, for now. CPTAC's clinical data lives on PDC (Proteomic Data
Commons), not GDC, and is not this BCR-XML format at all -- confirmed
earlier that case_index_cptac.csv shows has_clinical=0% for exactly this
reason (a known, documented gap, not pursued yet). Kremer/Pfib_423 have no
clinical XML either; their only real clinical signal is the ICD_10 code
already captured in case_index_kremer.csv, which is far too sparse (one
ICD-10 code) to be worth a full text-embedding pipeline of its own.

Identity tracking: unlike run_rna_encoder.py, ClinicalEncoder.embed() is a
plain sequential batch loop over a Python list (no multiprocessing, no
internal dataset reordering), so the specific case_id-misalignment risk
that produced bug #6 does not apply here in the same form. Case ids are
still tracked explicitly alongside each text rather than relied on via
positional order, and the same final n/unique_ids/unique_rows check is
still run -- consistency with the rest of this pipeline, not because this
exact bug is expected to recur here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def find_clinical_file(case_id, tcga_manifest, raw_dir):
    """Locate the real patient-level clinical XML among GDC's clinical
    file bundle for one case.

    GDC's "Clinical Supplement" data type bundles EIGHT files per case, not
    one -- confirmed live: nationwidechildrens.org_clinical.<case>.xml (the
    one we want, patient-level demographics/diagnosis/vital status) plus
    seven others with completely different schemas or formats:
    nationwidechildrens.org_omf.<case>.xml (other malignancy form, valid
    XML but no vital_status/gender/age at all), and five .txt TSV tables
    (clinical_follow_up, clinical_patient, clinical_radiation, clinical_nte,
    clinical_drug -- note the underscore, "clinical_X", vs the wanted file's
    period, "clinical."). An earlier version of this function picked among
    all eight by sorting on the GDC file_id (a UUID unrelated to file type)
    and taking the first -- effectively a random pick per case, which
    explained a real, serious result: only 426 of 4,759 real on-disk
    clinical files were successfully used, not because the other ~4,300
    cases lacked clinical data, but because most of them had the wrong file
    silently selected (a .txt caught as a parse failure, or the omf.xml
    parsing successfully but extracting near-zero fields) -- both correctly
    dropped by downstream checks, but for the wrong underlying reason, and
    the true cause invisible without directly cross-referencing the real
    on-disk file count against the pipeline's output count.

    The distinguishing pattern: the wanted filename starts with
    "nationwidechildrens.org_clinical." (a period right after "clinical",
    not an underscore) and ends with ".xml".
    """
    rec = tcga_manifest.get(case_id)
    if not rec:
        return None
    clinical_files = rec.get("files", {}).get("clinical", [])
    if not clinical_files:
        return None

    matching = [
        f for f in clinical_files
        if "org_clinical." in f["file_name"] and f["file_name"].endswith(".xml")
    ]
    if not matching:
        return None
    matching = sorted(matching, key=lambda f: f["file_id"])  # deterministic
    # if genuinely more than one file matches this pattern (unexpected, but
    # do not silently guess), still only use the first and note it
    if len(matching) > 1:
        print(f"  WARNING: {case_id} has {len(matching)} files matching the "
             f"patient-clinical pattern (expected 1): "
             f"{[f['file_name'] for f in matching]} -- using the first, "
             f"deterministically, but this is worth checking by hand")
    f = matching[0]
    p = Path(raw_dir) / f["file_id"] / f["file_name"]
    return str(p) if p.exists() else None


def load_case_paths(case_index_csv, tcga_manifest_path, raw_dir):
    manifest = json.loads(Path(tcga_manifest_path).read_text())
    out = {}
    missing_on_disk = []
    project_by_case = {}
    with open(case_index_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("has_clinical") not in ("True", "1", "true"):
                continue
            case_id = row["case_id"]
            path = find_clinical_file(case_id, manifest, raw_dir)
            if path is None:
                missing_on_disk.append(case_id)
                continue
            out[case_id] = path
            project_by_case[case_id] = row.get("project", "")

    if missing_on_disk:
        print(f"  {len(missing_on_disk)} case(s) marked has_clinical=True "
             f"but no file found on disk (e.g. {missing_on_disk[:5]})")
    return out, project_by_case


def run_pipeline(case_index_csv, tcga_manifest_path, raw_dir, out_dir,
                 model_name="emilyalsentzer/Bio_ClinicalBERT",
                 device="cuda", batch_size=32, min_fields=2):
    from .clinical_xml import parse_tcga_clinical_xml, format_tcga_clinical_text
    from .foundation import ClinicalEncoder

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("locating clinical XML files for cases in the index...")
    case_paths, project_by_case = load_case_paths(
        case_index_csv, tcga_manifest_path, raw_dir
    )
    print(f"  {len(case_paths)} cases with a clinical file found on disk")

    texts, case_ids, dropped = [], [], []
    n_parse_fail = 0
    for case_id, path in case_paths.items():
        try:
            fields = parse_tcga_clinical_xml(path)
        except Exception as e:
            n_parse_fail += 1
            dropped.append((case_id, f"parse failed: {type(e).__name__}: {e}"))
            continue
        if len(fields) < min_fields:
            dropped.append((case_id, f"only {len(fields)} field(s) extracted "
                                    f"(< {min_fields}), likely a near-empty record"))
            continue
        text = format_tcga_clinical_text(fields, project=project_by_case.get(case_id))
        texts.append(text)
        case_ids.append(case_id)

    print(f"  {len(texts)} cases with usable clinical text "
         f"({len(dropped)} dropped, {n_parse_fail} of those from parse "
         f"failures)")
    if dropped[:10]:
        for cid, reason in dropped[:10]:
            print(f"    {cid}: {reason}")

    if not texts:
        raise RuntimeError("no usable clinical text -- check paths above")

    print(f"\nembedding {len(texts)} clinical summaries with {model_name} ...")
    encoder = ClinicalEncoder(model_name=model_name, device=device)
    embeddings = encoder.embed(texts, batch_size=batch_size)

    # identity check -- see module docstring for why this risk differs from
    # the RNA pipeline's, but the check is kept for consistency regardless
    if len(embeddings) != len(case_ids):
        raise RuntimeError(
            f"embedding count ({len(embeddings)}) != case_id count "
            f"({len(case_ids)}) -- stop, do not assume alignment"
        )
    case_ids_arr = np.array(case_ids)
    n_unique_ids = len(set(case_ids))
    n_unique_rows = len(set(tuple(row) for row in embeddings))
    if n_unique_ids != len(case_ids):
        raise RuntimeError(f"duplicate case_ids in output: "
                          f"{len(case_ids) - n_unique_ids} duplicates")

    np.savez(out / "clinical_embeddings.npz",
             case_ids=case_ids_arr, embeddings=embeddings.astype(np.float32))
    print(f"\nsaved -> {out / 'clinical_embeddings.npz'}")
    print(f"  n={len(case_ids)}, unique_ids={n_unique_ids}, "
         f"unique_rows={n_unique_rows}, shape={embeddings.shape}")

    (out / "dropped_case_ids.txt").write_text(
        "\n".join(f"{cid}\t{reason}" for cid, reason in dropped) + "\n"
        if dropped else ""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", required=True)
    ap.add_argument("--tcga-manifest", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    run_pipeline(args.case_index, args.tcga_manifest, args.raw_dir, args.out,
                model_name=args.model_name, device=args.device,
                batch_size=args.batch_size)


if __name__ == "__main__":
    main()
