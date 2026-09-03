"""
Real fidelity evaluation: real held-out data vs the trained MoDiT's
generated samples, with proper feature-space dimensionality reduction so
FID/PRDC/C2ST are numerically meaningful rather than a small-n-large-p
covariance-estimation artifact (see eval/tissue_classifier.py's docstring
for the live demonstration of why this matters -- raw-space FID between
two samples of the IDENTICAL distribution came out as 4939, not ~0).

Feature-space design: a SEPARATE tissue-of-origin classifier is trained
for the joint representation AND for each individual modality slice,
using real data only. An earlier version used PCA (fit on real data) for
the per-modality slices instead of a classifier, reasoning that training
5 separate classifiers was more engineering than necessary -- this was
WRONG, confirmed by a live investigation: PCA is unsupervised and finds
directions of maximum variance in real data with no awareness of tissue
identity at all. When a CFG-scale experiment produced a large, measured
change in the raw generated RNA/EHR values (mean_abs_diff ratio 0.9-1.4x
the signal's own scale) but recall in the PCA feature space stayed
BIT-IDENTICAL across three separate sampling runs, the conclusion was
that whatever direction CFG was actually improving lived outside the
handful of top-variance components PCA kept -- the metric was
structurally blind to exactly the kind of improvement being measured,
not because the improvement wasn't real. A classifier, being trained
specifically to separate tissues, does not have this blind spot by
construction. Fixed to use a per-modality classifier for every slice.

  - "rad" is EXCLUDED entirely: real radiology is always exactly zero by
    construction (no radiology encoder exists in this project), so there
    is no genuine real distribution to compare generated samples against
    for this modality -- including it would produce a meaningless,
    trivially-either-perfect-or-meaningless number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data.dataset import RareSynthTCGADataset
from .model.dit import ModalitySpec
from .train_modit import compute_z_geno_path, TISSUE_VOCAB
from .encoders.genomic import GenomicSetEncoder
from .encoders.foundation import GatedABMIL
from .eval.fidelity import per_modality_report, frechet_distance, prdc, c2st
from .eval.tissue_classifier import train_tissue_classifier


def get_real_x_and_tissue(manifest_path, rna_npz, clinical_npz, genomic_npz,
                          genomic_encoder, pathology_encoder, split, device,
                          max_tiles=2000, batch_size=64):
    """Real x vectors (UNSCALED -- real embedding space, not the
    diffusion-training data_scale-multiplied space) plus integer tissue
    labels, for a given split.
    """
    spec = ModalitySpec()
    ds = RareSynthTCGADataset(manifest_path, rna_npz, clinical_npz, genomic_npz,
                              split=split, max_tiles=max_tiles)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_x, all_tissue = [], []
    with torch.no_grad():
        for batch in loader:
            z_geno, z_path = compute_z_geno_path(batch, genomic_encoder, pathology_encoder, device)
            x = spec.join([z_geno, batch["rna"].to(device), z_path,
                          batch["rad"].to(device), batch["ehr"].to(device)])
            all_x.append(x.cpu().numpy())
            all_tissue.extend(batch["project"])
    x = np.concatenate(all_x, axis=0)
    tissue_idx = np.array([TISSUE_VOCAB.get(p, -1) for p in all_tissue])
    keep = tissue_idx >= 0  # drop any case whose project isn't in our 8-tissue vocab
    return x[keep], tissue_idx[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rna-npz", required=True)
    ap.add_argument("--clinical-npz", required=True)
    ap.add_argument("--genomic-npz", required=True)
    ap.add_argument("--gene-vocab-npz", required=True)
    ap.add_argument("--synthetic-npz", required=True,
                    help="output of sample_modit.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classifier-epochs", type=int, default=100)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ModalitySpec()

    gene_vocab = np.load(args.gene_vocab_npz, allow_pickle=True)
    ckpt = torch.load(args.checkpoint, map_location=dev)
    genomic_encoder = GenomicSetEncoder(gene_vocab["gene_embedding"], d_model=256,
                                        d_out=spec.dims["geno"]).to(dev)
    genomic_encoder.load_state_dict(ckpt["genomic_encoder"])
    pathology_encoder = GatedABMIL(d_in=768, d_out=spec.dims["path"]).to(dev)
    pathology_encoder.load_state_dict(ckpt["pathology_encoder"])

    print("=== extracting real x vectors (train + val) ===")
    x_train, y_train = get_real_x_and_tissue(
        args.manifest, args.rna_npz, args.clinical_npz, args.genomic_npz,
        genomic_encoder, pathology_encoder, "train", dev,
    )
    x_val, y_val = get_real_x_and_tissue(
        args.manifest, args.rna_npz, args.clinical_npz, args.genomic_npz,
        genomic_encoder, pathology_encoder, "val", dev,
    )
    print(f"  train: {x_train.shape}, val: {x_val.shape}")

    print("\n=== training tissue-of-origin classifier on REAL data ===")
    classifier, val_acc = train_tissue_classifier(
        x_train, y_train, x_val, y_val, n_tissues=len(TISSUE_VOCAB),
        device=dev, epochs=args.classifier_epochs,
    )
    if val_acc < 0.3:  # chance is 0.125 for 8 classes
        print(f"  WARNING: classifier val accuracy ({val_acc:.3f}) is not "
             f"much above chance -- FID in this feature space may not be "
             f"very informative. Proceeding, but flag this in any report.")

    print("\n=== loading generated samples ===")
    synth = np.load(args.synthetic_npz, allow_pickle=True)
    x_fake_all = synth["x"]
    print(f"  generated: {x_fake_all.shape}")

    # use REAL VAL split (never seen during training) as the comparison
    # set -- comparing against train data would not test generalization
    real = x_val
    fake = x_fake_all[:len(real)] if len(x_fake_all) >= len(real) else x_fake_all
    real = real[:len(fake)]
    print(f"  comparing {len(real)} real (val) vs {len(fake)} generated samples")

    print("\n=== computing fidelity metrics ===")
    slices = {"joint": slice(0, real.shape[1])}
    slices.update({k: v for k, v in spec.slices().items() if k != "rad"})
    # rad excluded -- see module docstring

    # train a SEPARATE tissue classifier per modality slice, not just
    # joint -- PCA was tried first and found structurally blind to
    # CFG-driven improvements (see module docstring); a classifier,
    # trained specifically to separate tissues, does not have that blind
    # spot for any modality, including ones with weaker signal than
    # pathology/RNA (a classifier that ends up near chance accuracy for a
    # genuinely low-signal modality is itself an honest, informative
    # result, not a reason to fall back to an uninformed feature space)
    modality_classifiers = {"joint": classifier}
    modality_accs = {"joint": val_acc}
    for name, sl in slices.items():
        if name == "joint":
            continue
        print(f"\n  training {name}-specific tissue classifier...")
        mod_clf, mod_acc = train_tissue_classifier(
            x_train[:, sl], y_train, x_val[:, sl], y_val,
            n_tissues=len(TISSUE_VOCAB), device=dev,
            epochs=args.classifier_epochs,
        )
        modality_classifiers[name] = mod_clf
        modality_accs[name] = mod_acc
        if mod_acc < 0.3:
            print(f"    NOTE: {name} classifier accuracy ({mod_acc:.3f}) "
                 f"is close to chance (0.125) -- this modality may "
                 f"genuinely carry weak tissue signal on its own; FID/PRDC "
                 f"in this feature space should be interpreted with that "
                 f"in mind, not treated as equally informative to a "
                 f"high-accuracy modality's numbers")

    results = {}
    for name, sl in slices.items():
        r, f = real[:, sl], fake[:, sl]
        clf = modality_classifiers[name]
        r_feat, f_feat = clf.embed(r, device=dev), clf.embed(f, device=dev)

        row = {"fid": frechet_distance(r_feat, f_feat)}
        row.update(prdc(r_feat, f_feat, k=5))
        row.update(c2st(r_feat, f_feat, seed=0))
        results[name] = row
        print(f"  {name}: FID={row['fid']:.4f}, precision={row['precision']:.3f}, "
             f"recall={row['recall']:.3f}, c2st_acc={row['c2st_accuracy']:.3f} "
             f"(p={row['p_value']:.4f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "results": results,
        "modality_classifier_val_acc": modality_accs,
        "n_real": len(real), "n_fake": len(fake),
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
