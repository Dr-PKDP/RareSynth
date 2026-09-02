"""
Stratified case selection for the whole-slide-image subset.

Downloading every TCGA diagnostic slide is a ~12 TB pull that nobody needs:
the model trains on a slide-level embedding, not the raw image, and a few
thousand well-chosen cases carry the same statistical information as the full
archive for the purpose of fitting an encoder and a diffusion model.

Selection rule
--------------
From the manifest built by gdc_pull.py, keep only cases that have *both*
RNA-seq and a slide (otherwise the case cannot be used as a paired training
example), then sample proportionally to each project's share of that paired
population, capped so no single cancer type dominates the subset. Radiology
and clinical availability are recorded but not required, since modality
dropout is designed to handle a partially observed case.

Usage
-----
    python -m raresynth.data.select_slide_subset \
        --manifest /data/pduttapramanik/raresynth/data/manifests/tcga_manifest.json \
        --target-n 2000 \
        --out /data/pduttapramanik/raresynth/data/manifests/slide_subset_cases.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def select_subset(manifest_path, target_n=2000, max_frac_per_project=0.20,
                  seed=0, require_rna=True):
    cases = json.loads(Path(manifest_path).read_text())

    eligible = {}
    for sid, rec in cases.items():
        has_rna = "rna" in rec["files"]
        has_slide = "slide" in rec["files"]
        if has_slide and (has_rna or not require_rna):
            eligible[sid] = rec

    by_project = defaultdict(list)
    for sid, rec in eligible.items():
        by_project[rec["project"]].append(sid)

    n_eligible = len(eligible)
    print(f"{n_eligible} cases eligible (slide"
         f"{' + RNA-seq' if require_rna else ''}) across {len(by_project)} projects")

    rng = np.random.default_rng(seed)
    cap = int(target_n * max_frac_per_project)
    selected = []
    remaining_budget = target_n

    # proportional allocation with a per-project cap, largest project first
    # so smaller projects are not crowded out by rounding
    order = sorted(by_project, key=lambda p: -len(by_project[p]))
    for i, proj in enumerate(order):
        pool = by_project[proj]
        share = len(pool) / n_eligible
        n_remaining_projects = len(order) - i
        alloc = min(cap, len(pool),
                   max(1, round(target_n * share)),
                   remaining_budget - (n_remaining_projects - 1))
        alloc = max(alloc, 0)
        chosen = rng.choice(pool, size=min(alloc, len(pool)), replace=False)
        selected.extend(chosen.tolist())
        remaining_budget -= len(chosen)
        print(f"  {proj:16s} pool={len(pool):5d}  selected={len(chosen):4d}")

    # top-up: caps and small pools often leave budget unused; fill it from
    # whichever eligible cases were not already picked, so the final count
    # tracks the target rather than silently falling short
    if remaining_budget > 0:
        picked = set(selected)
        leftover = [sid for sid in eligible if sid not in picked]
        top_up = rng.choice(leftover, size=min(remaining_budget, len(leftover)),
                            replace=False)
        if len(top_up):
            selected.extend(top_up.tolist())
            print(f"  {'(top-up)':16s} pool={len(leftover):5d}  "
                 f"selected={len(top_up):4d}")

    print(f"\ntotal selected: {len(selected)} / target {target_n}")

    has_rad = sum(1 for sid in selected if "rad" in cases[sid]["files"])
    print(f"  of which have radiology on file: {has_rad} "
         f"({100*has_rad/max(len(selected),1):.0f}%)")

    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target-n", type=int, default=2000)
    ap.add_argument("--max-frac-per-project", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selected = select_subset(args.manifest, args.target_n,
                             args.max_frac_per_project, args.seed)
    Path(args.out).write_text("\n".join(sorted(selected)) + "\n")
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
