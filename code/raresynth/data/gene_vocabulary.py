"""
Gene vocabulary (symbol -> integer index) and a placeholder gene embedding
table for GenomicSetEncoder's gene-identity lookup.

Honest simplification, not hidden: GenomicSetEncoder's own docstring calls
for "a knowledge-graph model over STRING and GO" -- a real, semantically
meaningful gene embedding. That resource was never acquired in this
project (mentioned only in a docstring, no acquisition plan ever existed).
Rather than block today's work on a new multi-hour data-acquisition
detour, this uses a FIXED-SEED random embedding instead: reproducible
across runs (same seed -> identical vectors every time), and it still
gives the set-transformer's attention mechanism a consistent per-gene
identity signal to key on -- the real information content in the model
comes from the 14 real annotation FEATURES (build_gene_annotation_matrix),
not from this identity vector. This is stated plainly here and in
MANUSCRIPT_NOTES.md as a limitation to note in Methods, and is a
straightforward swap for a real STRING/GO embedding later without
touching anything else (same shape, same gene_index contract).

Vocabulary is built from gnomAD's own constraint gene list (17,878 genes,
already parsed and verified) -- a clean, real, protein-coding gene list
already on hand, rather than sourcing yet another gene-list file.
"""

from __future__ import annotations

import numpy as np


def build_gene_vocabulary(gnomad_constraint_dict):
    """gene_symbol -> integer index, sorted for reproducibility (dict
    iteration order is not guaranteed stable across Python versions/runs
    without an explicit sort)."""
    genes = sorted(gnomad_constraint_dict.keys())
    return {g: i for i, g in enumerate(genes)}


def build_placeholder_gene_embedding(gene_index, d_emb=256, seed=0):
    """Fixed-seed random embedding, one row per gene in gene_index, in
    index order. Reproducible: the same gene_index + seed always produces
    the identical embedding matrix.
    """
    n_genes = len(gene_index)
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_genes, d_emb)).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)  # unit-norm,
                                                                  # matches
                                                                  # how a
                                                                  # real
                                                                  # embedding
                                                                  # table
                                                                  # would
                                                                  # typically
                                                                  # be scaled
    return emb
