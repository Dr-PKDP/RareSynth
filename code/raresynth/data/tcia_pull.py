"""
TCIA radiology acquisition -- Phase 6.

NBIA Data Retriever always needs a .tcia manifest file; there is no flag to
just hand it a collection name. The GUI path for building one is the TCIA
web "cart", which does not scale to matching thousands of already-selected
TCGA cases by hand. Instead this queries TCIA's public REST API directly for
each collection, filters server-side results down to the patient IDs already
selected in the slide subset (TCIA PatientIDs are the same TCGA barcodes GDC
uses, e.g. "TCGA-A2-A0D2"), and writes a manifest in the documented format:

    downloadServerUrl=https://services.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet
    includeAnnotation=false
    noOfrRetry=4
    databasketId=<name>.tcia
    manifestVersion=3.0
    ListOfSeriesToDownload=
    <SeriesInstanceUID>
    <SeriesInstanceUID>
    ...

No API key or login is required for TCIA's public collections (which is all
of TCGA); the ``-l <credential file>`` flag documented for the CLI is only
needed for restricted collections and is not used here.

Usage
-----
    # 1. Build the manifest (network call, queries TCIA per collection)
    python -m raresynth.data.tcia_pull manifest \
        --collections TCGA-BRCA TCGA-LUAD TCGA-KIRC TCGA-GBM TCGA-LGG TCGA-COAD TCGA-OV TCGA-STAD \
        --case-list /data/pduttapramanik/raresynth/data/manifests/slide_subset_cases.txt \
        --out /data/pduttapramanik/raresynth/data/manifests/tcia_manifest.tcia

    # 2. Run the NBIA Data Retriever CLI against it
    python -m raresynth.data.tcia_pull download \
        --manifest /data/pduttapramanik/raresynth/data/manifests/tcia_manifest.tcia \
        --out /data/pduttapramanik/raresynth/data/raw/tcia \
        --retriever-bin /data/pduttapramanik/raresynth/tools/nbia-extracted/opt/nbia-data-retriever/bin/nbia-data-retriever
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import requests

TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DOWNLOAD_SERVER = "https://services.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet"
USER_AGENT = "raresynth-research-script/1.0 (contact: pduttapramanik@uth.tmh.edu)"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def get_series(collection, session=None, patient_id=None, timeout=180):
    """Query TCIA's public getSeries endpoint for one collection.

    Returns a list of series dicts (SeriesInstanceUID, PatientID, Modality,
    BodyPartExamined, ...). Unknown/empty collections return an empty list
    rather than raising, since not every TCGA project has a matched TCIA
    collection and the caller iterates over a candidate list.
    """
    s = session or _session()
    params = {"Collection": collection}
    if patient_id:
        params["PatientID"] = patient_id
    r = s.get(f"{TCIA_API}/getSeries", params=params, timeout=timeout)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def build_manifest(collections, case_ids, out_path, session=None,
                   databasket_id=None, include_annotation=False):
    """Query each collection, keep only series whose PatientID is in
    case_ids, write a .tcia manifest.

    case_ids: iterable of TCGA submitter_ids, e.g. from slide_subset_cases.txt
    """
    s = session or _session()
    keep = set(case_ids)
    series_uids = []
    per_collection_counts = {}

    for coll in collections:
        all_series = get_series(coll, session=s)
        matched = [x for x in all_series
                  if x.get("PatientID") in keep and x.get("SeriesInstanceUID")]
        series_uids.extend(x["SeriesInstanceUID"] for x in matched)
        per_collection_counts[coll] = (len(all_series), len(matched))
        print(f"  {coll:16s} {len(all_series):5d} series total, "
             f"{len(matched):5d} matched to our case list", flush=True)
        time.sleep(0.5)

    series_uids = sorted(set(series_uids))  # de-dupe, stable order
    basket = databasket_id or f"raresynth-manifest.tcia"

    lines = [
        f"downloadServerUrl={DOWNLOAD_SERVER}",
        f"includeAnnotation={'true' if include_annotation else 'false'}",
        "noOfrRetry=4",
        f"databasketId={basket}",
        "manifestVersion=3.0",
        "ListOfSeriesToDownload=",
    ] + series_uids

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"\n{len(series_uids)} unique series written -> {out_path}")
    return series_uids, per_collection_counts


def run_retriever(manifest_path, out_dir, retriever_bin, credential_file=None,
                  verbose=True, force=True, max_retries=3, retry_wait=60):
    """Invoke the NBIA Data Retriever CLI.

    The CLI's own retry count (--noOfrRetry, set inside the manifest) covers
    per-series retries during one run; this wraps the whole invocation with
    an outer retry too, since a run that dies partway (killed session, node
    reboot) should not require starting the entire collection over -- rerun
    with -f (force/skip-existing) and it picks up what is missing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [retriever_bin, "--cli", str(manifest_path), "-d", str(out)]
    if credential_file:
        cmd += ["-l", str(credential_file)]
    if verbose:
        cmd.append("-v")
    if force:
        cmd.append("-f")

    print("running:", " ".join(cmd), flush=True)
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("NBIA Data Retriever finished successfully.")
            return True
        print(f"NBIA Data Retriever exited with code {result.returncode} "
             f"(attempt {attempt}/{max_retries})")
        if attempt < max_retries:
            print(f"retrying in {retry_wait}s ...", flush=True)
            time.sleep(retry_wait)
    print("NBIA Data Retriever failed after all retries. Files already "
         "downloaded are kept; rerun this same command later to resume "
         "(the -f flag skips files already present).")
    return False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest")
    m.add_argument("--collections", nargs="+", required=True)
    m.add_argument("--case-list", required=True,
                   help="text file, one TCGA submitter_id per line")
    m.add_argument("--out", required=True)

    d = sub.add_parser("download")
    d.add_argument("--manifest", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--retriever-bin", required=True)
    d.add_argument("--credential-file", default=None)
    d.add_argument("--max-retries", type=int, default=3)
    d.add_argument("--retry-wait", type=float, default=60.0)

    args = ap.parse_args()
    if args.cmd == "manifest":
        case_ids = set(Path(args.case_list).read_text().split())
        build_manifest(args.collections, case_ids, args.out)
    elif args.cmd == "download":
        run_retriever(args.manifest, args.out, args.retriever_bin,
                      credential_file=args.credential_file,
                      max_retries=args.max_retries, retry_wait=args.retry_wait)


if __name__ == "__main__":
    main()
