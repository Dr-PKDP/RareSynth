"""Geneformer tokenization and CLS embedding extraction for the GTEx fibroblast AnnData."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/pduttapramanik/raresynth/tools/Geneformer")
H5AD_DIR = Path("/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/h5ad_input")
OUT = Path("/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast")
MODEL_VERSION = "V2"
MODEL_SIZE = "104M"
FORWARD_BATCH_SIZE = 200
NPROC = 16

token_dict_path = REPO / "geneformer" / f"token_dictionary_gc{MODEL_SIZE}.pkl"
model_dir = REPO / f"Geneformer-V2-{MODEL_SIZE}"
if not token_dict_path.exists():
    raise FileNotFoundError(f"token dictionary not found at {token_dict_path}")
if not model_dir.exists():
    raise FileNotFoundError(f"model directory not found at {model_dir}")

print("=== confirming the input h5ad and its case_ids ===")
import anndata as ad
a = ad.read_h5ad(H5AD_DIR / "cohort.h5ad")
input_ids = set(a.obs["case_id"])
print(f"  {a.shape[0]} samples, {a.shape[1]} genes, {len(input_ids)} distinct case_ids")

print("\n=== tokenizing ===")
from geneformer import TranscriptomeTokenizer
tokenized_dir = OUT / "tokenized"
tokenized_dir.mkdir(parents=True, exist_ok=True)
custom_attrs = {"case_id": "case_id"}
tk_kwargs = dict(nproc=NPROC)
try:
    tk = TranscriptomeTokenizer(custom_attrs, model_version=MODEL_VERSION, **tk_kwargs)
except TypeError:
    print("  WARNING: TranscriptomeTokenizer did not accept model_version as a "
         "keyword -- verify manually which dictionary it defaulted to before "
         "trusting the output embeddings")
    tk = TranscriptomeTokenizer(custom_attrs, **tk_kwargs)
tk.tokenize_data(str(H5AD_DIR), str(tokenized_dir), "cohort", file_format="h5ad")

print("\n=== extracting CLS embeddings ===")
from geneformer import EmbExtractor
embex = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls",
    max_ncells=None,
    emb_label=["case_id"],
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_VERSION,
    nproc=NPROC,
)
emb_out_dir = OUT / "embeddings"
emb_out_dir.mkdir(parents=True, exist_ok=True)
embs = embex.extract_embs(
    str(model_dir),
    str(tokenized_dir / "cohort.dataset"),
    str(emb_out_dir),
    "cohort_embs",
)

print("\n=== validating identity survived the pipeline ===")
if "case_id" not in embs.columns:
    raise RuntimeError(
        "embs has no 'case_id' column even after requesting emb_label=['case_id'] "
        "-- stop and inspect embs.columns / a fresh tokenized dataset by hand "
        "before proceeding."
    )

returned_ids_list = list(embs["case_id"])
returned_ids = set(returned_ids_list)
if len(returned_ids) != len(returned_ids_list):
    from collections import Counter
    dupes = [cid for cid, n in Counter(returned_ids_list).items() if n > 1]
    raise RuntimeError(
        f"embs contains DUPLICATE case_id values ({len(dupes)} ids appear more "
        f"than once, e.g. {dupes[:5]}) -- real corruption, stop and investigate."
    )

unexpected_extra = returned_ids - input_ids
if unexpected_extra:
    raise RuntimeError(
        f"embs contains {len(unexpected_extra)} case_id(s) never in the input "
        f"(e.g. {list(unexpected_extra)[:5]}) -- real corruption, stop and "
        f"investigate."
    )

missing = input_ids - returned_ids
if missing:
    frac = len(missing) / len(input_ids)
    print(f"  {len(missing)}/{len(input_ids)} ({frac:.1%}) input samples did not "
         f"come back -- most often TranscriptomeTokenizer's internal "
         f"coding+miRNA gene-subset filter, not a pipeline bug. Listed below "
         f"and saved to dropped_case_ids.txt.")
    for cid in sorted(missing)[:20]:
        print(f"    MISSING: {cid}")
    (OUT / "dropped_case_ids.txt").write_text("\n".join(sorted(missing)) + "\n")
    if frac > 0.05:
        raise RuntimeError(
            f"{frac:.1%} of input samples were dropped -- too large to treat "
            f"as routine quality exclusion. Stop and investigate."
        )
else:
    print(f"  all {len(input_ids)} samples returned, no drops")

print("\n=== saving ===")
case_ids_out = embs["case_id"].values
emb_cols = [c for c in embs.columns if c != "case_id"]
emb_matrix = embs[emb_cols].values.astype(np.float32)
print(f"  final embeddings shape: {emb_matrix.shape}")

np.savez(
    OUT / "rna_embeddings.npz",
    case_ids=case_ids_out,
    embeddings=emb_matrix,
)
print(f"\nsaved -> {OUT / 'rna_embeddings.npz'}")
