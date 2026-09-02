"""
Real fidelity evaluation: real held-out data vs the trained MoDiT's
generated samples, with proper feature-space dimensionality reduction so
FID/PRDC/C2ST are numerically meaningful rather than a small-n-large-p
covariance-estimation artifact (see eval/tissue_classifier.py's docstring
for the live demonstration of why this matters -- raw-space FID between
two samples of the IDENTICAL distribution came out as 4939, not ~0).

Feature-space design:
  - "joint" (the full concatenated vector, the case that matters most for
    the paper's central cross-modal claim): the tissue-of-origin
    classifier's penultimate features (eval/tissue_classifier.py),
    trained fresh on REAL data for this evaluation.
  - each individual modality slice: PCA (fit on real data only, applied
    to both real and generated), reducing to a well-conditioned dimension
    -- simpler than training a separate classifier per modality, and does
    not assume every modality carries equally strong tissue-discriminative
    signal (clinical/genomic are lower-information-content by nature and
    might not support a well-trained per-modality classifier the way
    pathology or RNA would).
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
from sklearn.decomposition import PCA

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
    ap.add_argument("--pca-dim", type=int, default=32)
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

    results = {}
    for name, sl in slices.items():
        r, f = real[:, sl], fake[:, sl]
        if name == "joint":
            r_feat, f_feat = classifier.embed(r, device=dev), classifier.embed(f, device=dev)
        else:
            n_comp = min(args.pca_dim, r.shape[1], len(r) - 1)
            pca = PCA(n_components=n_comp)
            pca.fit(r)  # fit on REAL only
            r_feat, f_feat = pca.transform(r), pca.transform(f)

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
        "tissue_classifier_val_acc": val_acc,
        "n_real": len(real), "n_fake": len(fake),
        "pca_dim": args.pca_dim,
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
