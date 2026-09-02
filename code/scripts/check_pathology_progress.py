"""
Check aggregate progress across all parallel pathology-encoding workers.

All shards write to the SAME out_dir (case_id.npz filenames don't collide
across shards, since each case belongs to exactly one shard by
construction) but each shard writes its OWN manifest file
(_manifest_shard<i>.json, not a shared _manifest.json), so combining all
of them gives a complete, non-clobbered picture across every worker.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def check_progress(out_dir, expected_total=None):
    out = Path(out_dir)
    npz_files = list(out.glob("*.npz"))
    print(f"{len(npz_files)} case(s) completed (.npz files present) in {out}")

    if expected_total is not None:
        pct = 100 * len(npz_files) / expected_total
        print(f"  {pct:.1f}% of expected {expected_total} total cases")

    manifests = sorted(out.glob("_manifest_shard*.json")) or list(out.glob("_manifest.json"))
    if manifests:
        combined = {}
        for m in manifests:
            try:
                combined.update(json.loads(m.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        by_status = {}
        for case_id, info in combined.items():
            status = info.get("status", "unknown")
            by_status.setdefault(status, []).append(case_id)
        print(f"\nstatus breakdown (combined from {len(manifests)} shard "
             f"manifest(s)):")
        for status, cases in sorted(by_status.items()):
            print(f"  {status}: {len(cases)}")
            if status == "crashed":
                for cid in cases[:10]:
                    print(f"    {cid}: {combined[cid].get('error', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--expected-total", type=int, default=None)
    args = ap.parse_args()
    check_progress(args.out_dir, args.expected_total)


if __name__ == "__main__":
    main()
