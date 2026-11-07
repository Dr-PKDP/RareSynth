"""Statistics of the training array relevant to the clip bound: largest coordinate, top-coordinate share of the squared norm and rows beyond the clip."""

import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
DATA_SCALE = 33.40122774789073
CLIP = 4.0

d = np.load(PATH)
x = d["rna_train"].astype(np.float64)
norms = np.linalg.norm(x, axis=1)
print(f"rows: {len(x)}, dimension: {x.shape[1]}; unscaled norm min/median/max = "
      f"{norms.min():.5f} / {np.median(norms):.5f} / {norms.max():.5f} (all rows should be unit norm)")
assert np.abs(norms - 1).max() < 1e-3, "rows are not unit norm: zero placeholder rows may be present"

xs = x * DATA_SCALE
mean_sq_norm = (xs ** 2).sum(axis=1).mean()
share = (xs ** 2).mean(axis=0) / mean_sq_norm
top = int(np.argmax(share))
print(f"largest |value| in scaled space: {np.abs(xs).max():.2f}")
print(f"coordinate {top}: carries {share[top]:.3f} of the mean squared norm; mean value {xs[:, top].mean():+.2f}")
print(f"top-3 / top-10 coordinates carry {np.sort(share)[::-1][:3].sum():.3f} / {np.sort(share)[::-1][:10].sum():.3f} of the squared norm")
beyond = np.abs(xs) > CLIP
print(f"fraction of coordinates with |value| > {CLIP}: {beyond.mean():.5f}; "
      f"fraction of rows with at least one such coordinate: {beyond.any(axis=1).mean():.4f}")
ratio = np.linalg.norm(np.clip(xs, -CLIP, CLIP), axis=1) / np.linalg.norm(xs, axis=1)
print(f"norm of REAL rows after clamping to +/-{CLIP}, as a fraction of the original: "
      f"median {np.median(ratio):.3f} (5th-95th percentile {np.percentile(ratio, 5):.3f}-{np.percentile(ratio, 95):.3f})")



def report(label, arr):
    n_entries, n_rows = int((np.abs(arr) > CLIP).sum()), int((np.abs(arr) > CLIP).any(axis=1).sum())
    sq = (arr ** 2).mean(axis=0) / (arr ** 2).sum(axis=1).mean()
    ratio = np.linalg.norm(np.clip(arr, -CLIP, CLIP), axis=1) / np.linalg.norm(arr, axis=1)
    print(f"{label:<34} max|v|={np.abs(arr).max():7.2f} | top-coordinate share={sq.max():.4f} | "
          f"entries beyond {CLIP}: {n_entries} ({n_entries / arr.size:.2e}) | rows with any: {n_rows} of {len(arr)} | "
          f"median clamped/original norm={np.median(ratio):.3f}")

print("\nROBUSTNESS (scale changes and mean-centering):")
report("scaled x1", xs)
report("scaled x0.5", xs * 0.5)
report("scaled x2", xs * 2.0)
report("centered by training mean, x1", (x - x.mean(axis=0)) * DATA_SCALE)
