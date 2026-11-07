"""Write the unit-norm RNA embedding arrays of the TCGA train/val splits, or of the test split with --split test."""

import argparse
import csv
import os
import sys

import numpy as np

ROOT = "/data/pduttapramanik/raresynth"
CASE_INDEX = f"{ROOT}/data/manifests/case_index_tcga.csv"
RNA_NPZ = f"{ROOT}/data/embeddings/rna_tcga_full/rna_embeddings.npz"
OUT_TRAIN_VAL = f"{ROOT}/data/derived/rna_clean_arrays.npz"
OUT_TEST = f"{ROOT}/data/derived/rna_test_array.npz"

TISSUE_VOCAB = {
    "TCGA-BRCA": 0, "TCGA-LUAD": 1, "TCGA-KIRC": 2, "TCGA-GBM": 3,
    "TCGA-LGG": 4, "TCGA-COAD": 5, "TCGA-OV": 6, "TCGA-STAD": 7,
}
EXPECTED = {"train": 3389, "val": 413, "test": 453}


def load_rows(split, index_rows, rna_row, embeddings):
    ids = sorted(r["case_id"] for r in index_rows
                 if r["split"] == split and r["project"] in TISSUE_VOCAB and r["case_id"] in rna_row)
    project = {r["case_id"]: r["project"] for r in index_rows}
    x = embeddings[np.array([rna_row[c] for c in ids])]
    x = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    y = np.array([TISSUE_VOCAB[project[c]] for c in ids], dtype=np.int64)
    return x, y


def check(name, x, y):
    norms = np.linalg.norm(x, axis=1)
    assert np.isfinite(x).all(), f"{name}: non-finite values"
    assert np.abs(norms - 1.0).max() < 1e-3, f"{name}: a row is not unit norm"
    assert y.min() >= 0 and y.max() < len(TISSUE_VOCAB), f"{name}: tissue index out of range"
    assert len(x) == EXPECTED[name], f"{name}: {len(x)} rows, expected {EXPECTED[name]}"
    print(f"{name}: n={len(x)}, norm range=({norms.min():.5f}, {norms.max():.5f}), "
          f"tissue counts={dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train_val", "test"], default="train_val")
    args = ap.parse_args()

    with open(CASE_INDEX) as fh:
        index_rows = list(csv.DictReader(fh))
    rna = np.load(RNA_NPZ, allow_pickle=True)
    rna_row = {str(c): i for i, c in enumerate(rna["case_ids"])}
    emb = rna["embeddings"]

    if args.split == "train_val":
        x_tr, y_tr = load_rows("train", index_rows, rna_row, emb)
        x_va, y_va = load_rows("val", index_rows, rna_row, emb)
        check("train", x_tr, y_tr)
        check("val", x_va, y_va)
        os.makedirs(os.path.dirname(OUT_TRAIN_VAL), exist_ok=True)
        np.savez(OUT_TRAIN_VAL, rna_train=x_tr, y_train=y_tr, rna_val=x_va, y_val=y_va)
        print(f"saved -> {OUT_TRAIN_VAL}")
    else:
        if os.path.exists(OUT_TEST):
            sys.exit(f"{OUT_TEST} already exists")
        x_te, y_te = load_rows("test", index_rows, rna_row, emb)
        check("test", x_te, y_te)
        np.savez(OUT_TEST, rna_test=x_te, y_test=y_te)
        print(f"saved -> {OUT_TEST}")


if __name__ == "__main__":
    main()
