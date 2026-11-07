"""Coordinate-shuffled control: a fixed permutation (seed 0) of the mapped UFM1 direction."""

import numpy as np

PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
SEED = 0

true_direction = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"]
perm = np.random.default_rng(SEED).permutation(len(true_direction))
shuffled = true_direction[perm]
assert np.isclose(np.linalg.norm(true_direction), np.linalg.norm(shuffled), atol=1e-6)

out = f"{PPN_DIR}/ufm1_rna_embedding_guidance_SHUFFLED_CONTROL.npz"
np.savez_compressed(out, rna_embedding_guidance=shuffled.astype(np.float32), permutation_seed=SEED,
                    source="coordinate permutation of the true UFM1 direction")
cos = float(true_direction @ shuffled / (np.linalg.norm(true_direction) * np.linalg.norm(shuffled)))
print(f"cosine(true, shuffled) = {cos:.4f}; saved -> {out}")
