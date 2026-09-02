"""
GDC (TCGA) download script — Phase 2 and part of Phase 5.

Etiquette
---------
GDC is a shared NCI resource used by the whole cancer-genomics community. This
script is deliberately conservative:
  - sequential batches, not parallel connections (--workers defaults to 1)
  - a fixed delay between batches
  - small batch size (default 50 files) so a network hiccup loses minutes,
    not hours
  - full resumability: every completed file ID is logged, and reruns skip
    them, so an interrupted overnight job just picks up where it left off
  - a standard descriptive User-Agent identifying the script, not a browser
    spoof

Usage
-----
    # 1. Build the manifest (fast, just metadata)
    python -m raresynth.data.gdc_pull manifest --projects TCGA-BRCA TCGA-LUAD \
        --out /data/pduttapramanik/raresynth/data/manifests/tcga_manifest.json

    # 2. Download RNA-seq + MAF + clinical (small, run first)
    python -m raresynth.data.gdc_pull download --manifest tcga_manifest.json \
        --types rna maf clinical \
        --out /data/pduttapramanik/raresynth/data/raw/tcga

    # 3. Download a stratified slide subset (large, run overnight)
    python -m raresynth.data.gdc_pull download --manifest tcga_manifest.json \
        --types slide --case-list slide_subset_cases.txt \
        --out /data/pduttapramanik/raresynth/data/raw/tcga/slides
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

GDC_API = "https://api.gdc.cancer.gov"
USER_AGENT = "raresynth-research-script/1.0 (contact: pduttapramanik@uth.tmh.edu)"

DATA_TYPE_MAP = {
    "rna": "Gene Expression Quantification",
    "maf": "Masked Somatic Mutation",
    "clinical": "Clinical Supplement",
    "slide": "Slide Image",
}


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def build_manifest(projects, out_path, session=None, page_size=2000):
    """Index open-access files across the requested TCGA projects.

    Metadata-only: this is cheap and does not need the same throttling as the
    bulk data endpoint, but it still uses one session with a descriptive UA.

    GDC paginates every /files response. A single request capped at
    page_size silently truncates to whatever the cap is -- confirmed on a
    real run, where every one of four data types independently returned
    exactly page_size=2000 files, which is what a truncation looks like, not
    what a coincidence looks like. This loops using GDC's own
    pagination.total field until every page is retrieved.
    """
    s = session or _session()
    manifest = {}
    for data_type, gdc_type in DATA_TYPE_MAP.items():
        content = [
            {"op": "in", "content": {"field": "files.data_type", "value": [gdc_type]}},
            {"op": "in", "content": {"field": "files.access", "value": ["open"]}},
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": list(projects)}},
        ]
        filters = json.dumps({"op": "and", "content": content})
        fields = ",".join([
            "file_id", "file_name", "file_size", "md5sum",
            "cases.submitter_id", "cases.project.project_id",
            "cases.samples.sample_type",
        ])

        all_hits, frm, total = [], 0, None
        while total is None or frm < total:
            params = {
                "filters": filters,
                "fields": fields,
                "format": "JSON",
                "size": str(page_size),
                "from": str(frm),
            }
            r = s.get(f"{GDC_API}/files", params=params, timeout=180)
            r.raise_for_status()
            body = r.json()["data"]
            hits = body["hits"]
            total = body["pagination"]["total"]
            all_hits.extend(hits)
            frm += len(hits)
            if not hits:
                break  # safety valve against an infinite loop on an API change
            time.sleep(0.3)  # be polite between pages, not just between data types

        for h in all_hits:
            for c in h.get("cases", []):
                sid = c.get("submitter_id")
                if not sid:
                    continue
                rec = manifest.setdefault(sid, {
                    "project": c.get("project", {}).get("project_id"),
                    "files": {},
                })
                rec["files"].setdefault(data_type, []).append({
                    "file_id": h["file_id"],
                    "file_name": h["file_name"],
                    "size": h.get("file_size"),
                    "md5sum": h.get("md5sum"),
                })
        print(f"  {data_type:10s} ({gdc_type}): {len(all_hits)} files indexed "
             f"(GDC reports {total} total)", flush=True)
        time.sleep(1.0)  # be polite between the four metadata queries too

    Path(out_path).write_text(json.dumps(manifest, indent=2))
    n_rna_slide = sum(1 for v in manifest.values()
                      if "rna" in v["files"] and "slide" in v["files"])
    print(f"\n{len(manifest)} cases indexed; "
         f"{n_rna_slide} have both RNA-seq and a slide image")
    return manifest


def _load_done(log_path):
    if not Path(log_path).exists():
        return set()
    return set(Path(log_path).read_text().splitlines())


def _mark_done(log_path, file_id):
    with open(log_path, "a") as fh:
        fh.write(file_id + "\n")


def _load_failed(log_path):
    p = Path(log_path)
    if not p.exists():
        return []
    return [line.split("\t")[0] for line in p.read_text().splitlines() if line.strip()]


def _mark_failed(log_path, batch_idx, error):
    with open(log_path, "a") as fh:
        fh.write(f"{batch_idx}\t{type(error).__name__}: {error}\n")


def size_aware_batches(records, max_batch_bytes=1_500_000_000, max_batch_count=50,
                       default_size=50_000_000):
    """Group records so cumulative size stays under max_batch_bytes.

    RNA-seq/MAF/clinical files are KB-MB; whole-slide images are commonly
    hundreds of MB to several GB EACH. A fixed file-count batch size that
    works fine for small files can put 10-20+ GB into a single POST for
    slides, and something on the network path (proxy, load balancer, WAF)
    can then cut the response short deterministically -- every retry hits
    the same cap and fails the same way, which is what happened here: 13 of
    the first 33 slide batches died identically on tar decompression despite
    full retry-with-backoff. Batching by byte budget instead of item count
    fixes this for both data types without needing to special-case slides.

    A single file larger than max_batch_bytes still goes out alone rather
    than being skipped, since GDC will serve it as its own request either way.
    """
    batch, batch_bytes = [], 0
    for r in records:
        sz = r.get("size") or default_size
        if batch and (batch_bytes + sz > max_batch_bytes
                      or len(batch) >= max_batch_count):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(r)
        batch_bytes += sz
    if batch:
        yield batch


def _is_gzip(path):
    """GDC's /data endpoint returns a tar.gz only when multiple files are
    requested. Request exactly one file and it streams the raw file back
    directly -- confirmed by hand: a single slide came back as a bare TIFF
    (SVS is TIFF-based), not an archive, with Content-Type
    application/octet-stream in both cases so the header does not
    distinguish them. Detecting by the gzip magic bytes (1f 8b) is robust to
    this regardless of how a batch ended up at size 1, whether from the
    size-aware batching or from bisection landing on a single oversized file.
    """
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def download_files(file_records, out_dir, log_path, batch_size=50,
                   delay_sec=2.0, session=None, verify_md5=True,
                   max_retries=5, base_backoff=10.0,
                   max_batch_bytes=1_500_000_000):
    """Sequential, resumable, rate-limited download via the GDC bulk endpoint.

    file_records: list of dicts with file_id, file_name, md5sum, size.

    Batches are formed by size_aware_batches(), so ``batch_size`` now acts
    only as an upper bound on item count per batch (max_batch_count); the
    byte budget (max_batch_bytes) is what actually keeps any single request
    to a size the network path can reliably deliver in one shot.

    A single connection drop used to crash the whole run and leave nothing
    marked done for that batch. That is a real failure mode on a link this
    long (laptop -> jump host -> destination -> internet -> GDC) and cannot
    be tolerated across ~14,000 files. Each batch now gets up to
    ``max_retries`` attempts with exponential backoff before being logged as
    failed and skipped, so one bad batch costs minutes, not the whole job.
    A partially written .tar.gz from a failed attempt is deleted before
    retrying so it can never be fed to tar as if it were complete.
    """
    s = session or _session()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fail_log = Path(out_dir) / "_failed_batches.log"

    done = _load_done(log_path)
    todo = [r for r in file_records if r["file_id"] not in done]
    print(f"{len(file_records)} files requested, {len(done)} already done, "
         f"{len(todo)} remaining")

    state = {"n_fail": 0, "n_done": 0, "counter": 0}

    def attempt(batch, tag, depth=0):
        """Try one batch; on exhausted retries, bisect and recurse rather
        than giving up outright. This converges to whatever request size the
        network path actually tolerates without needing that size guessed
        correctly up front -- if even a byte-budgeted batch keeps failing
        identically, splitting it is the general fix.
        """
        state["counter"] += 1
        raw_path = out / f"_batch_{state['counter']:06d}_{tag}.download"
        ids = [r["file_id"] for r in batch]
        total_bytes = sum(r.get("size") or 0 for r in batch)

        for attempt_n in range(1, max_retries + 1):
            try:
                r = s.post(
                    f"{GDC_API}/data",
                    data=json.dumps({"ids": ids}),
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=3600,
                )
                r.raise_for_status()
                with open(raw_path, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)

                if _is_gzip(raw_path):
                    subprocess.run(["tar", "-xzf", str(raw_path), "-C", str(out)],
                                   check=True)
                    raw_path.unlink()
                elif len(batch) == 1:
                    # GDC streams a single requested file back raw, not as a
                    # tar.gz. Recreate the same on-disk layout tar would have
                    # produced (a per-file-id directory) so downstream code
                    # does not need to care which path a file took here.
                    rec = batch[0]
                    dest_dir = out / rec["file_id"]
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / (rec.get("file_name") or f"{rec['file_id']}.bin")
                    raw_path.rename(dest)
                else:
                    # more than one file requested but the response is not
                    # gzip: not a case GDC's documented behavior predicts.
                    # Treat as a failure so retry/bisection handles it rather
                    # than silently mis-saving multi-file content.
                    raw_path.unlink()
                    raise ValueError(
                        f"batch of {len(batch)} files returned non-gzip content "
                        f"(expected a tar.gz for >1 file)"
                    )

                for rec in batch:
                    _mark_done(log_path, rec["file_id"])
                state["n_done"] += len(batch)
                print(f"  [{tag}] {len(batch)} files "
                     f"({total_bytes/1e9:.2f} GB) done "
                     f"[{state['n_done']}/{len(todo)}]"
                     + (f"  (retry {attempt_n})" if attempt_n > 1 else "")
                     + (f"  (depth {depth})" if depth > 0 else ""),
                     flush=True)
                return True

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    subprocess.CalledProcessError,
                    ValueError) as e:
                if raw_path.exists():
                    raw_path.unlink()  # never leave a partial/mismatched file behind
                if attempt_n == max_retries:
                    if len(batch) > 1:
                        print(f"  [{tag}] {len(batch)} files "
                             f"({total_bytes/1e9:.2f} GB) failed after "
                             f"{max_retries} attempts -- splitting and "
                             f"retrying as two smaller batches", flush=True)
                        mid = len(batch) // 2
                        ok1 = attempt(batch[:mid], tag + "a", depth + 1)
                        ok2 = attempt(batch[mid:], tag + "b", depth + 1)
                        return ok1 and ok2
                    state["n_fail"] += 1
                    _mark_failed(fail_log, tag, e)
                    print(f"  [{tag}] single file {ids[0]} FAILED after "
                         f"{max_retries} attempts even alone "
                         f"({type(e).__name__}: {e}) -- logged to {fail_log}, "
                         f"continuing", flush=True)
                    return False
                else:
                    wait = base_backoff * (2 ** (attempt_n - 1))
                    print(f"  [{tag}] attempt {attempt_n}/{max_retries} failed "
                         f"({type(e).__name__}); retrying in {wait:.0f}s",
                         flush=True)
                    time.sleep(wait)
        return False

    for bi, batch in enumerate(
        size_aware_batches(todo, max_batch_bytes=max_batch_bytes,
                           max_batch_count=batch_size)
    ):
        attempt(batch, f"b{bi:05d}")
        time.sleep(delay_sec)

    if state["n_fail"]:
        print(f"\n{state['n_fail']} file(s) failed even after retry+bisection "
             f"-- see {fail_log}. Rerun this exact command to retry only the "
             f"files still missing from the completion log; already-done "
             f"files are skipped automatically.")

    print("download complete ->", out)


def collect_records(manifest, types, case_list=None):
    cases = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
    keep = set(Path(case_list).read_text().split()) if case_list else None
    records = []
    for sid, rec in cases.items():
        if keep is not None and sid not in keep:
            continue
        for t in types:
            records.extend(rec["files"].get(t, []))
    return records


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest")
    m.add_argument("--projects", nargs="+", required=True)
    m.add_argument("--out", required=True)

    d = sub.add_parser("download")
    d.add_argument("--manifest", required=True)
    d.add_argument("--types", nargs="+", required=True,
                   choices=list(DATA_TYPE_MAP.keys()))
    d.add_argument("--out", required=True)
    d.add_argument("--case-list", default=None,
                   help="optional file, one case submitter_id per line, to "
                        "restrict the download to a subset (e.g. the "
                        "stratified slide sample)")
    d.add_argument("--batch-size", type=int, default=50)
    d.add_argument("--delay", type=float, default=2.0)

    args = ap.parse_args()
    if args.cmd == "manifest":
        build_manifest(args.projects, args.out)
    elif args.cmd == "download":
        records = collect_records(args.manifest, args.types, args.case_list)
        log = Path(args.out) / "_completed_file_ids.log"
        download_files(records, args.out, log, batch_size=args.batch_size,
                       delay_sec=args.delay)


if __name__ == "__main__":
    main()
