"""Build a GDC manifest for TCGA projects and download the selected file types in sequential, resumable batches."""

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
                break
            time.sleep(0.3)

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
        time.sleep(1.0)

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
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def download_files(file_records, out_dir, log_path, batch_size=50,
                   delay_sec=2.0, session=None, verify_md5=True,
                   max_retries=5, base_backoff=10.0,
                   max_batch_bytes=1_500_000_000):
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
                    rec = batch[0]
                    dest_dir = out / rec["file_id"]
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / (rec.get("file_name") or f"{rec['file_id']}.bin")
                    raw_path.rename(dest)
                else:
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
                    raw_path.unlink()
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
