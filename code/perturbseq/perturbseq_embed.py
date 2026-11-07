"""Geneformer V2 (104M) CLS embeddings of the Perturb-seq pseudobulk h5ad.
Usage: python perturbseq_embed.py GENEFORMER_REPO OUT_DIR   (OUT_DIR/h5ad_input/cohort.h5ad is read; OUT_DIR/rna_embeddings.npz is written)"""
import sys
from pathlib import Path
import numpy as np, anndata as ad
REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
H5AD_DIR = OUT / "h5ad_input"
MV, MS = "V2", "104M"
model_dir = REPO / f"Geneformer-V2-{MS}"
a = ad.read_h5ad(H5AD_DIR / "cohort.h5ad")
input_ids = set(a.obs["case_id"])
print(a.shape[0], "samples,", a.shape[1], "genes")
from geneformer import TranscriptomeTokenizer, EmbExtractor
tok = OUT / "tokenized"; tok.mkdir(parents=True, exist_ok=True)
TranscriptomeTokenizer({"case_id": "case_id"}, model_version=MV, nproc=16).tokenize_data(str(H5AD_DIR), str(tok), "cohort", file_format="h5ad")
emb_dir = OUT / "embeddings"; emb_dir.mkdir(parents=True, exist_ok=True)
embs = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cls", max_ncells=None, emb_label=["case_id"],
                    forward_batch_size=16, model_version=MV, nproc=16).extract_embs(str(model_dir), str(tok / "cohort.dataset"), str(emb_dir), "cohort_embs")
ids = list(embs["case_id"])
assert len(ids) == len(set(ids)) and set(ids) <= input_ids
missing = input_ids - set(ids)
print(len(missing), "of", len(input_ids), "dropped by tokenizer")
assert len(missing) / len(input_ids) < 0.05
cols = [c for c in embs.columns if c != "case_id"]
np.savez(OUT / "rna_embeddings.npz", case_ids=np.array(ids), embeddings=embs[cols].values.astype(np.float32))
print("saved", OUT / "rna_embeddings.npz", embs[cols].shape)
