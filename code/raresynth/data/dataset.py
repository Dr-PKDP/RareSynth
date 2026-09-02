"""
PyTorch Dataset reading the joined case manifest (build_case_manifest.py)
and serving real per-case training examples.

Genomic and pathology are returned as RAW material (gene features + gene
indices; a padded tile-embedding bag + mask), not precomputed fixed
vectors -- GenomicSetEncoder and GatedABMIL are untrained by design and
must run as part of the model's forward pass during training, with their
parameters in the optimizer, not as a frozen preprocessing step (see
MANUSCRIPT_NOTES.md). RNA and clinical are returned as their real
precomputed fixed vectors directly.

Radiology is always zero-filled and always marked unavailable -- no
radiology encoder exists anywhere in this project yet (TCIA/CPTAC imaging
was downloaded but never processed into any embedding). This is an
explicit placeholder, not an oversight: ModalitySpec still reserves a
"rad" slot (512-dim) for when/if that encoder is built, and every case's
avail mask correctly marks it False in the meantime, which is exactly the
condition MoDiT's modality-dropout training is designed to handle.

Cases with ZERO available modalities are excluded by default (confirmed
real, not a bug: 46/4865 TCGA cases, all sharing one submission-batch
barcode pattern with no molecular data of any kind ever deposited) --
such a case contributes nothing to any training objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RareSynthTCGADataset(Dataset):
    def __init__(self, manifest_path, rna_npz_path, clinical_npz_path,
                genomic_npz_path, split=None, max_tiles=2000,
                path_embed_dim=768, exclude_zero_modality=True):
        """
        split: None (all splits) or one of "train"/"val"/"test", or a list
              of split names to include.
        max_tiles: pathology bags are subsampled/padded to exactly this
              many tiles per case for batching (a bag with more tiles than
              this is randomly subsampled -- DETERMINISTICALLY per epoch
              is NOT enforced here; a different random subsample each time
              __getitem__ is called is standard practice for MIL-style
              training and typically improves regularization, not a bug).
        """
        manifest = json.loads(Path(manifest_path).read_text())

        if split is not None:
            if isinstance(split, str):
                split = [split]
            manifest = {cid: e for cid, e in manifest.items() if e["split"] in split}

        if exclude_zero_modality:
            before = len(manifest)
            manifest = {cid: e for cid, e in manifest.items()
                       if e["n_modalities_available"] > 0}
            excluded = before - len(manifest)
            if excluded:
                print(f"  excluded {excluded} case(s) with zero available "
                     f"modalities")

        self.case_ids = sorted(manifest.keys())  # sorted: reproducible
                                                  # __getitem__ index order
                                                  # across runs
        self.manifest = manifest
        self.max_tiles = max_tiles
        self.path_embed_dim = path_embed_dim

        # RNA/clinical/genomic are modest in size -- load fully into memory
        # once. Pathology (up to ~8000 tiles x 768-dim per case x 2000
        # cases) is NOT preloaded -- read lazily per case in __getitem__,
        # which is also the natural pattern for a DataLoader with
        # num_workers>0 (each worker reads what it needs, no giant shared
        # upfront load).
        rna = np.load(rna_npz_path, allow_pickle=True)
        # RNA embeddings come directly from Geneformer's own EmbExtractor,
        # unlike every other modality's encoder (ClinicalEncoder,
        # GenomicSetEncoder, GatedABMIL), all of which explicitly
        # L2-normalize their own output before returning. Confirmed a
        # real scale mismatch on real data: RNA's mean row norm is 6.58
        # (max |value| 4.15, already exceeding the diffusion process's
        # x0_clip=4.0 default even in genuine training data) while
        # clinical's is exactly 1.00000. Normalized here so RNA enters
        # the shared diffusion process on the same footing as every other
        # modality -- without this, RNA's much larger natural scale would
        # dominate the combined representation's effective signal-to-noise
        # ratio at any given diffusion timestep, directly undermining
        # cross-modal coherence between modalities that are NOT
        # comparably scaled to it.
        rna_raw = rna["embeddings"]
        rna_norms = np.linalg.norm(rna_raw, axis=1, keepdims=True)
        self.rna_embeddings = (rna_raw / (rna_norms + 1e-8)).astype(np.float32)

        clinical = np.load(clinical_npz_path, allow_pickle=True)
        self.clinical_embeddings = clinical["embeddings"]
        genomic = np.load(genomic_npz_path, allow_pickle=True)
        self.genomic_features = genomic["gene_features"]
        self.genomic_gene_indices = genomic["gene_indices"]

        print(f"RareSynthTCGADataset: {len(self.case_ids)} cases "
             f"(split={split or 'all'})")

    def __len__(self):
        return len(self.case_ids)

    def _load_pathology_bag(self, path):
        d = np.load(path, allow_pickle=True)
        embs = d["tile_embeddings"]  # (K, path_embed_dim), K varies
        k = embs.shape[0]

        if k >= self.max_tiles:
            idx = np.random.choice(k, self.max_tiles, replace=False)
            bag = embs[idx]
            mask = np.ones(self.max_tiles, dtype=bool)
        else:
            bag = np.zeros((self.max_tiles, self.path_embed_dim), dtype=np.float32)
            bag[:k] = embs
            mask = np.zeros(self.max_tiles, dtype=bool)
            mask[:k] = True
        return bag.astype(np.float32), mask

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        e = self.manifest[case_id]

        if e["has_rna"]:
            rna = self.rna_embeddings[e["rna_index"]].astype(np.float32)
        else:
            rna = np.zeros(self.rna_embeddings.shape[1], dtype=np.float32)

        if e["has_clinical"]:
            ehr = self.clinical_embeddings[e["clinical_index"]].astype(np.float32)
        else:
            ehr = np.zeros(self.clinical_embeddings.shape[1], dtype=np.float32)

        if e["has_genomic"]:
            gi = e["genomic_index"]
            geno_feats = self.genomic_features[gi]
            geno_idx = self.genomic_gene_indices[gi]
        else:
            geno_feats = np.zeros(self.genomic_features.shape[1:], dtype=np.float32)
            geno_idx = np.zeros(self.genomic_gene_indices.shape[1:], dtype=np.int64)

        if e["has_pathology"]:
            path_bag, path_mask = self._load_pathology_bag(e["pathology_path"])
        else:
            path_bag = np.zeros((self.max_tiles, self.path_embed_dim), dtype=np.float32)
            path_mask = np.zeros(self.max_tiles, dtype=bool)

        # radiology: always zero, always unavailable -- see module docstring
        rad = np.zeros(512, dtype=np.float32)

        avail = np.array([
            e["has_genomic"], e["has_rna"], e["has_pathology"],
            False,  # rad -- never available
            e["has_clinical"],
        ], dtype=np.float32)  # order matches ModalitySpec: geno, rna, path, rad, ehr

        return {
            "case_id": case_id,
            "rna": torch.from_numpy(rna),
            "ehr": torch.from_numpy(ehr),
            "geno_features": torch.from_numpy(geno_feats),
            "geno_gene_idx": torch.from_numpy(geno_idx),
            "path_bag": torch.from_numpy(path_bag),
            "path_mask": torch.from_numpy(path_mask),
            "rad": torch.from_numpy(rad),
            "avail": torch.from_numpy(avail),
            "project": e.get("project", ""),
            "split": e["split"],
        }
