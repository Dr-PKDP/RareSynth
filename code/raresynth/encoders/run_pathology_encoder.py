"""
Pathology encoding: case index -> slide file(s) -> tile -> CTransPath embed
-> per-case bag of tile embeddings.

Output shape differs from the other three encoders deliberately. RNA/
genomic/clinical each produce one fixed-size vector per case; pathology
produces a variable-length BAG of tile embeddings per case, because
GatedABMIL (the aggregator) is untrained by construction and must be
learned jointly with MoDiT, not run now as if it were a frozen feature
extractor -- see MANUSCRIPT_NOTES.md. Running an untrained ABMIL now would
produce a meaningless random-projection "embedding," not a real feature.

A case's tiles are pooled across ALL of its slides into one bag (TCGA
cases commonly have more than one diagnostic slide -- 6,209 slides across
2,000 cases in our subset, confirmed) rather than kept separate per slide,
since ABMIL will attend over a case's whole tissue regardless of which
physical slide a tile came from.

Follows stream-and-discard: raw tiles are never written to disk. Each
slide is tiled, embedded, and the tile arrays discarded before moving to
the next slide -- only the (small) embedding vectors persist.

One .npz per case (not one big file) -- deliberate, for three reasons:
resumability (skip cases whose output already exists), natural support
for the variable-length bags (no ragged-array storage complexity), and
straightforward parallelism (each case is independent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_slide_files(case_id, tcga_manifest, raw_dir):
    """Return ALL existing slide file paths for a case (a case commonly has
    more than one diagnostic slide, unlike the single-file-per-case
    modalities), using the same <file_id>/<file_name> layout gdc_pull.py
    produces for every other data type.
    """
    rec = tcga_manifest.get(case_id)
    if not rec:
        return []
    slide_files = rec.get("files", {}).get("slide", [])
    paths = []
    for f in slide_files:
        p = Path(raw_dir) / f["file_id"] / f["file_name"]
        if p.exists():
            paths.append(str(p))
    return paths


def load_case_slide_paths(case_index_csv, tcga_manifest_path, raw_dir,
                          split_filter=None):
    manifest = json.loads(Path(tcga_manifest_path).read_text())
    out = {}
    missing = []
    import csv
    with open(case_index_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("has_slide") not in ("True", "1", "true"):
                continue
            if split_filter and row.get("split") not in split_filter:
                continue
            case_id = row["case_id"]
            paths = find_slide_files(case_id, manifest, raw_dir)
            if not paths:
                missing.append(case_id)
                continue
            out[case_id] = paths
    if missing:
        print(f"  {len(missing)} case(s) marked has_slide=True but no "
             f"slide file found on disk (e.g. {missing[:5]})")
    return out


def encode_case(case_id, slide_paths, encoder, tile_kwargs, max_tiles_per_case=8000):
    """Tile every slide for one case, embed, pool into one bag. Returns
    (embeddings (K, D) float32 array, n_tiles_per_slide dict) or (None, {})
    if no usable tiles were found on any of the case's slides.
    """
    from .foundation import tile_slide

    all_embs = []
    per_slide_counts = {}
    for slide_path in slide_paths:
        try:
            tiles = tile_slide(slide_path, **tile_kwargs)
        except Exception as e:
            print(f"    {Path(slide_path).name}: tiling FAILED "
                 f"({type(e).__name__}: {e}), skipping this slide")
            continue
        per_slide_counts[Path(slide_path).name] = len(tiles)
        if not tiles:
            continue
        try:
            embs = encoder.embed_tiles(tiles)  # (n_tiles, D) torch tensor
        except Exception as e:
            print(f"    {Path(slide_path).name}: encoding FAILED "
                 f"({type(e).__name__}: {e}), skipping this slide's tiles "
                 f"(e.g. a transient CUDA error) -- other slides for this "
                 f"case still proceed")
            del tiles
            continue
        all_embs.append(embs.numpy())
        del tiles  # stream-and-discard -- do not accumulate raw tile arrays

    if not all_embs:
        return None, per_slide_counts

    pooled = np.concatenate(all_embs, axis=0).astype(np.float32)
    if len(pooled) > max_tiles_per_case:
        rng = np.random.default_rng(hash(case_id) % (2**32))
        idx = rng.choice(len(pooled), max_tiles_per_case, replace=False)
        pooled = pooled[idx]
    return pooled, per_slide_counts


def _limit_internal_threads(n_threads=None):
    """Cap cv2/torch's internal thread pools explicitly, not just via
    environment variables. Confirmed necessary on this project's own
    server: even with OMP_NUM_THREADS etc. set by the launching shell
    script, cv2 in particular maintains its own internal thread pool that
    does not reliably respect those environment variables -- a live
    incident showed a single worker process spawn 429 OS threads and drive
    system load to 755 on a 344-core machine, with zero real throughput,
    despite the launcher already setting the standard BLAS/OpenMP env
    vars. Called explicitly here so this protection exists regardless of
    how the script is invoked (via the parallel launcher, directly for a
    smoke test, etc.) rather than depending on the caller's environment.
    """
    import os
    import cv2
    import torch

    if n_threads is None:
        env_val = os.environ.get("OMP_NUM_THREADS")
        n_threads = int(env_val) if env_val else 4
    cv2.setNumThreads(n_threads)
    torch.set_num_threads(n_threads)
    print(f"  internal thread pools capped at {n_threads} (cv2 + torch)")


def run_pipeline(case_index_csv, tcga_manifest_path, raw_dir, out_dir,
                 encoder_name="ctranspath", device="cuda",
                 split_filter=None, max_tiles_per_case=8000,
                 tile_px=256, level_mpp=0.5, max_tiles_per_slide=4000,
                 limit=None, shard_index=None, n_shards=None):
    _limit_internal_threads()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("locating slide files for cases in the index...")
    case_slides = load_case_slide_paths(case_index_csv, tcga_manifest_path,
                                        raw_dir, split_filter=split_filter)
    print(f"  {len(case_slides)} cases with at least one slide found on disk")
    if not case_slides:
        raise RuntimeError("no cases to process -- check paths above")

    if n_shards is not None:
        if shard_index is None or not (0 <= shard_index < n_shards):
            raise ValueError(f"shard_index must be in [0, {n_shards}) when "
                            f"n_shards is set, got {shard_index}")
        # deterministic: every worker independently computes the SAME full
        # ordered case list (from the same case_index.csv + manifest, both
        # static inputs), then takes every n_shards-th case starting at its
        # own shard_index -- no separate partition file needed, and any
        # worker can be re-run alone (e.g. to retry a shard that crashed)
        # without needing to know what any other shard did
        all_items = list(case_slides.items())
        case_slides = dict(all_items[shard_index::n_shards])
        print(f"  shard {shard_index}/{n_shards}: this worker handles "
             f"{len(case_slides)} of {len(all_items)} total cases")

    if limit is not None:
        # deterministic (not random) truncation: dict insertion order from
        # load_case_slide_paths follows the case_index.csv row order, which
        # is itself deterministic (built from the manifest), so the same
        # --limit value always picks the same cases across reruns -- makes
        # a smoke test result reproducible rather than a different random
        # subset each time
        case_slides = dict(list(case_slides.items())[:limit])
        print(f"  --limit {limit}: restricting to first {len(case_slides)} "
             f"cases for a smoke test")

    if encoder_name == "ctranspath":
        from .foundation import CTransPathEncoder
        encoder = CTransPathEncoder(device=device)
    elif encoder_name == "uni":
        from .foundation import UNIEncoder
        encoder = UNIEncoder(device=device)
    else:
        raise ValueError(f"unknown encoder_name: {encoder_name}")

    tile_kwargs = dict(tile_px=tile_px, level_mpp=level_mpp,
                       max_tiles=max_tiles_per_slide)

    manifest_path = out / (
        f"_manifest_shard{shard_index}.json" if n_shards is not None
        else "_manifest.json"
    )
    manifest = {}
    n_done, n_skipped, n_failed, n_crashed = 0, 0, 0, 0
    for i, (case_id, slide_paths) in enumerate(case_slides.items()):
        dest = out / f"{case_id}.npz"
        if dest.exists():
            n_skipped += 1
            continue

        print(f"[{i+1}/{len(case_slides)}] {case_id} "
             f"({len(slide_paths)} slide(s))")
        try:
            embs, per_slide_counts = encode_case(
                case_id, slide_paths, encoder, tile_kwargs, max_tiles_per_case
            )
        except Exception as e:
            # last-resort guard for an unattended overnight run: nothing
            # inside encode_case should reach here (tiling and encoding are
            # already guarded per-slide above), but if something genuinely
            # unexpected does (e.g. a fatal CUDA error that corrupts the
            # process's CUDA context, an out-of-memory during pooling/
            # concatenation), losing ONE case is vastly preferable to
            # silently losing every remaining case in this shard for the
            # rest of the night. Logged to the manifest, not swallowed
            # silently, so it is visible and fixable in the morning.
            print(f"    UNEXPECTED FAILURE processing {case_id}: "
                 f"{type(e).__name__}: {e} -- logging and continuing to "
                 f"the next case")
            manifest[case_id] = {"status": "crashed",
                                 "error": f"{type(e).__name__}: {e}"}
            n_crashed += 1
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue

        if embs is None:
            print(f"    no usable tiles found on any slide -- skipping case")
            n_failed += 1
            manifest[case_id] = {"status": "no_tiles", "slides": per_slide_counts}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue

        np.savez(dest, case_id=case_id, tile_embeddings=embs, n_tiles=len(embs))
        manifest[case_id] = {"status": "ok", "n_tiles": len(embs),
                            "slides": per_slide_counts}
        n_done += 1

        # write incrementally, not just once at the end -- for an
        # unattended overnight run, a process that dies for any reason
        # outside the per-case try/except above (e.g. during encoder
        # setup on a later retry, or a hard process kill) should still
        # leave a diagnostic trail of what happened up to that point,
        # not silently lose all manifest info
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"    -> {len(embs)} pooled tile embeddings saved")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\ndone: {n_done} succeeded, {n_skipped} already existed "
         f"(skipped), {n_failed} had no usable tiles, {n_crashed} hit an "
         f"unexpected failure (see manifest for details)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", required=True)
    ap.add_argument("--tcga-manifest", required=True)
    ap.add_argument("--raw-dir", required=True,
                    help="e.g. /data/pduttapramanik/raresynth/data/raw/tcga/slides")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="ctranspath", choices=["ctranspath", "uni"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", nargs="+", default=None)
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--level-mpp", type=float, default=0.5)
    ap.add_argument("--max-tiles-per-slide", type=int, default=4000)
    ap.add_argument("--max-tiles-per-case", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict to the first N cases, for a smoke test")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="this worker's shard, 0-indexed (requires --n-shards)")
    ap.add_argument("--n-shards", type=int, default=None,
                    help="total number of parallel workers splitting the cohort")
    args = ap.parse_args()
    run_pipeline(args.case_index, args.tcga_manifest, args.raw_dir, args.out,
                encoder_name=args.encoder, device=args.device,
                split_filter=args.split, max_tiles_per_case=args.max_tiles_per_case,
                tile_px=args.tile_px, level_mpp=args.level_mpp,
                max_tiles_per_slide=args.max_tiles_per_slide, limit=args.limit,
                shard_index=args.shard_index, n_shards=args.n_shards)


if __name__ == "__main__":
    main()
