"""Stack the non-strand-specific and strand-specific Pfib_423 embeddings into one cohort file (case_ids, embeddings)."""

import numpy as np

D = "/data/pduttapramanik/raresynth/data/embeddings/"
ns = np.load(D + "rna_pfib423_ns/rna_embeddings.npz", allow_pickle=True)
ss = np.load(D + "rna_pfib423_ss/rna_embeddings.npz", allow_pickle=True)

case_ids = np.concatenate([ns["case_ids"], ss["case_ids"]])
embeddings = np.vstack([ns["embeddings"], ss["embeddings"]])
assert len(set(case_ids.tolist())) == len(case_ids)

np.savez(D + "rna_pfib423_full.npz", case_ids=case_ids, embeddings=embeddings)
print(f"{len(case_ids)} samples -> {D}rna_pfib423_full.npz")
