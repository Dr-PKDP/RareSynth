"""
Genomic encoder.

The draft specified two mutually inconsistent genomic representations: a
12-dimensional per-gene annotation vector (Section 2.1) and a 5,313-track
Enformer delta per rare variant (Section 3.1).  The second is not computable
at cohort scale.  A 196,608 bp Enformer forward pass per allele, per variant,
per gene, across ~19,200 genes and ~10,000 patients is on the order of 10^8 to
10^9 forward passes; on 8x A100 that is measured in GPU-years, not GPU-hours.

What is computable, and what this module implements:

  * A per-gene annotation vector built from open resources (ClinVar, gnomAD
    v4 constraint, and the sample's own variant calls) plus, since 2026-08-31,
    AlphaMissense, CADD, GERP, and phyloP -- all four were assumed present as
    MAF columns in the original design and are NOT (a real downloaded GDC MAF
    has no such columns; confirmed by direct inspection). They are joined
    separately from data/annotation_sources.py + encoders/annotation_lookup.py
    instead. SIFT and PolyPhen, which genuinely ARE present in the real MAF
    (as VEP-style "label(score)" compound strings, not plain floats), are
    used as an additional deleteriousness signal the original design did not
    plan for.

  * A permutation-invariant set encoder (Set Transformer with inducing points)
    over the top-k most-perturbed genes for a given sample.  Attention over a
    variable-length gene set is the right inductive bias here: a genome is a
    set of perturbed genes, not a fixed-order sequence, and summing a linear
    projection over all 19,223 genes as the draft proposed discards which
    genes carry the signal.

  * An optional Enformer path retained honestly at the scale where it is
    affordable: regulatory deltas for the ~200 curated disease genes only
    (~400 forward passes), used as a supporting analysis and as a source of
    the geno direction vector for mechanism guidance, not as the cohort-wide
    encoder.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn

_VEP_SCORE_PATTERN = re.compile(r"\(([\d.]+)\)")


def parse_vep_score(value):
    """Extract the numeric score from a VEP-style 'label(score)' string, as
    found in the real MAF's SIFT and PolyPhen columns (e.g.
    "deleterious(0.02)", "possibly_damaging(0.907)"). Returns None for
    empty/unparseable values -- NOT 0.0, since 0.0 is itself a real,
    meaningful SIFT score (maximally deleterious) and must not be confused
    with missing data. Verified against real confirmed MAF values before
    being used here (see PROGRESS.md).
    """
    if not value or not isinstance(value, str):
        return None
    m = _VEP_SCORE_PATTERN.search(value)
    return float(m.group(1)) if m else None

# per-gene annotation features, in fixed order.
#
# Direction of aggregation matters and is easy to get backwards:
#   - AlphaMissense, CADD, PolyPhen, GERP, phyloP: HIGHER = more damaging /
#     more conserved -> aggregate with MAX (worst observed variant in the
#     gene drives the feature)
#   - SIFT: LOWER = more damaging (0 = maximally deleterious, 1 = tolerated)
#     -> aggregate with MIN, the opposite of every other score here. Getting
#     this backwards would silently make the most-tolerated SIFT variant in
#     a gene look like the most damaging one.
#   - gnomAD_AF: LOWER = rarer = more likely pathogenic -> aggregate with MIN
GENE_FEATURES = [
    "alphamissense_max",    # AlphaMissense pathogenicity, external tabix join
    "cadd_max",              # CADD, external bigWig join (position-max approximation)
    "sift_min",               # SIFT, from MAF directly -- LOWER = worse, use MIN
    "polyphen_max",          # PolyPhen, from MAF directly -- HIGHER = worse, use MAX
    "gerp_max",               # GERP conservation, external bigWig join
    "phylop_max",            # phyloP conservation, external bigWig join
    "n_hc_lof",                # count of high-confidence LoF variants, from MAF
    "n_missense",             # count of missense variants, from MAF
    "clinvar_plp",             # 0/1 ClinVar P/LP variant present, external join
    "pli",                       # gnomAD constraint, external join
    "loeuf",                    # gnomAD constraint, external join
    "gnomad_af_min",          # rarest gnomAD_AF observed, from MAF directly
    "max_af_min",              # rarest MAX_AF (broader pop. coverage than gnomAD alone), from MAF
    "expressed_in_tissue",   # 0/1 from GTEx median TPM > 1 for the tissue, external join
]
D_GENE_FEAT = len(GENE_FEATURES)


class MAB(nn.Module):
    """Multihead attention block (Lee et al., Set Transformer)."""

    def __init__(self, d, n_heads, ln=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.n1 = nn.LayerNorm(d) if ln else nn.Identity()
        self.n2 = nn.LayerNorm(d) if ln else nn.Identity()

    def forward(self, q, kv, key_padding_mask=None):
        h, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                         need_weights=False)
        q = self.n1(q + h)
        return self.n2(q + self.ff(q))


class ISAB(nn.Module):
    """Induced set attention block: O(n*m) instead of O(n^2)."""

    def __init__(self, d, n_heads, n_inducing=32):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, n_inducing, d) * 0.02)
        self.mab0 = MAB(d, n_heads)
        self.mab1 = MAB(d, n_heads)

    def forward(self, x, key_padding_mask=None):
        h = self.mab0(self.I.expand(x.shape[0], -1, -1), x,
                      key_padding_mask=key_padding_mask)
        return self.mab1(x, h)


class GenomicSetEncoder(nn.Module):
    """(B, K, D_GENE_FEAT) gene annotations + gene identity -> (B, 512).

    Parameters
    ----------
    gene_embedding
        (n_genes, d_gene_emb) matrix from a knowledge-graph model over STRING
        and GO.  Frozen.  Gives the encoder access to functional context that
        the annotation features alone do not carry.
    """

    def __init__(self, gene_embedding: np.ndarray, d_model=256, d_out=512,
                 n_heads=8, depth=3, n_inducing=32, n_seeds=4):
        super().__init__()
        E = torch.as_tensor(gene_embedding, dtype=torch.float32)
        self.register_buffer("gene_emb", E)
        self.feat_proj = nn.Sequential(
            nn.Linear(D_GENE_FEAT, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.gene_proj = nn.Linear(E.shape[1], d_model)
        self.blocks = nn.ModuleList(
            [ISAB(d_model, n_heads, n_inducing) for _ in range(depth)]
        )
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.pool = MAB(d_model, n_heads)
        self.out = nn.Sequential(
            nn.Linear(d_model * n_seeds, d_out), nn.GELU(), nn.Linear(d_out, d_out)
        )

    def forward(self, feats, gene_idx, pad_mask=None):
        """feats: (B, K, F); gene_idx: (B, K) long; pad_mask: (B, K) bool, True=pad."""
        h = self.feat_proj(feats) + self.gene_proj(self.gene_emb[gene_idx])
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask)
        z = self.pool(self.seeds.expand(h.shape[0], -1, -1), h,
                      key_padding_mask=pad_mask)
        z = self.out(z.flatten(1))
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)


def build_gene_annotation_matrix(
    maf_df, gene_index, clinvar, gnomad_constraint, gtex_median_tpm=None,
    tissue=None, top_k=512, am_path=None, cadd_lookup=None,
    gerp_lookup=None, phylop_lookup=None, tabix_bin="tabix",
):
    """Assemble the per-sample top-k perturbed-gene set.

    maf_df
        Open-tier GDC Masked Somatic Mutation table for one case. Real
        confirmed columns used here: Hugo_Symbol, Variant_Classification,
        SIFT, PolyPhen, gnomAD_AF, MAX_AF, Chromosome, Start_Position,
        Reference_Allele, Tumor_Seq_Allele2 -- NOT AlphaMissense/CADD_phred/
        GERP/phyloP, which do not exist in a real downloaded MAF (an
        earlier version of this function assumed they did and silently
        produced all-zero values for them; see PROGRESS.md).
    am_path
        Path to the local AlphaMissense tabix-indexed file, or None to skip
        (alphamissense_max will be 0 for every gene).
    cadd_lookup, gerp_lookup, phylop_lookup
        encoders.annotation_lookup.BigWigLookup instances, or None to skip
        that feature (will be 0 for every gene).
    Returns (feats (K,F) float32, gene_idx (K,) int64, n_valid int).
    """
    import pandas as pd

    lof = {"Frame_Shift_Del", "Frame_Shift_Ins", "Nonsense_Mutation",
           "Splice_Site", "Splice_Region", "Translation_Start_Site",
           "Nonstop_Mutation"}

    # ---- batched external lookups, once per sample, not once per gene ----
    am_scores = {}
    if am_path is not None and len(maf_df):
        from .annotation_lookup import batch_query_alphamissense
        variants = list(zip(
            maf_df["Chromosome"], maf_df["Start_Position"].astype(int),
            maf_df["Reference_Allele"], maf_df["Tumor_Seq_Allele2"],
        ))
        am_scores = batch_query_alphamissense(am_path, variants, tabix_bin=tabix_bin)

    def bw_lookup(lookup_obj, chrom, pos):
        if lookup_obj is None:
            return None
        return lookup_obj.query(chrom, int(pos))

    rows = {}
    for g, sub in maf_df.groupby("Hugo_Symbol"):
        if g not in gene_index:
            continue

        sift_scores = [parse_vep_score(v) for v in sub.get("SIFT", pd.Series(dtype=str))]
        sift_scores = [v for v in sift_scores if v is not None]
        polyphen_scores = [parse_vep_score(v) for v in sub.get("PolyPhen", pd.Series(dtype=str))]
        polyphen_scores = [v for v in polyphen_scores if v is not None]

        am_vals = [am_scores.get((c, int(p), r, a)) for c, p, r, a in
                  zip(sub["Chromosome"], sub["Start_Position"],
                     sub["Reference_Allele"], sub["Tumor_Seq_Allele2"])]
        am_vals = [v for v in am_vals if v is not None]

        cadd_vals = [bw_lookup(cadd_lookup, c, p) for c, p in
                    zip(sub["Chromosome"], sub["Start_Position"])]
        cadd_vals = [v for v in cadd_vals if v is not None]
        gerp_vals = [bw_lookup(gerp_lookup, c, p) for c, p in
                    zip(sub["Chromosome"], sub["Start_Position"])]
        gerp_vals = [v for v in gerp_vals if v is not None]
        phylop_vals = [bw_lookup(phylop_lookup, c, p) for c, p in
                      zip(sub["Chromosome"], sub["Start_Position"])]
        phylop_vals = [v for v in phylop_vals if v is not None]

        gnomad_af = sub.get("gnomAD_AF", pd.Series(dtype=float)).dropna()
        max_af = sub.get("MAX_AF", pd.Series(dtype=float)).dropna()

        rows[g] = {
            "alphamissense_max": max(am_vals) if am_vals else 0.0,
            "cadd_max": max(cadd_vals) if cadd_vals else 0.0,
            # SIFT: LOWER = more damaging -- MIN, not max. Getting this
            # backwards silently rewards tolerated variants as if damaging.
            "sift_min": min(sift_scores) if sift_scores else 1.0,  # 1.0 = tolerated default, not 0 (which would mean "maximally damaging by default")
            "polyphen_max": max(polyphen_scores) if polyphen_scores else 0.0,
            "gerp_max": max(gerp_vals) if gerp_vals else 0.0,
            "phylop_max": max(phylop_vals) if phylop_vals else 0.0,
            "n_hc_lof": float(sub["Variant_Classification"].isin(lof).sum()),
            "n_missense": float((sub["Variant_Classification"] == "Missense_Mutation").sum()),
            "clinvar_plp": float(g in clinvar),
            "pli": float(gnomad_constraint.get(g, {}).get("pLI", 0.0)),
            "loeuf": float(gnomad_constraint.get(g, {}).get("LOEUF", 2.0)),
            "gnomad_af_min": float(gnomad_af.min()) if len(gnomad_af) else 1.0,
            "max_af_min": float(max_af.min()) if len(max_af) else 1.0,
            "expressed_in_tissue": float(
                gtex_median_tpm is None
                or gtex_median_tpm.get((g, tissue), 0.0) > 1.0
            ),
        }

    if not rows:
        return (np.zeros((top_k, D_GENE_FEAT), np.float32),
                np.zeros(top_k, np.int64), 0)

    # rank genes by a simple deleteriousness proxy so the top-k truncation
    # keeps the informative ones
    def priority(r):
        return (2.0 * r["n_hc_lof"] + r["cadd_max"] / 10.0
                + 3.0 * r["clinvar_plp"] + 2.0 * r["pli"]
                + (1.0 - r["sift_min"]))  # low sift_min (damaging) -> high priority

    ordered = sorted(rows.items(), key=lambda kv: -priority(kv[1]))[:top_k]
    feats = np.zeros((top_k, D_GENE_FEAT), np.float32)
    idx = np.zeros(top_k, np.int64)
    for i, (g, r) in enumerate(ordered):
        feats[i] = [r[f] for f in GENE_FEATURES]
        idx[i] = gene_index[g]
    return feats, idx, len(ordered)


def clinvar_pathogenicity_direction(gene, gene_index, encoder, clinvar_variants,
                                    device="cpu"):
    """Direction vector in genomic latent space for a gene's known P/LP variants.

    Used as the ``geno`` term of the mechanism-consistency energy: it is the
    one modality whose expected shift can be read off a curated resource
    rather than inferred through the cross-modal map.
    """
    feats = np.zeros((1, 1, D_GENE_FEAT), np.float32)
    feats[0, 0, GENE_FEATURES.index("clinvar_plp")] = 1.0
    feats[0, 0, GENE_FEATURES.index("cadd_max")] = float(
        clinvar_variants.get(gene, {}).get("max_cadd", 25.0)
    )
    feats[0, 0, GENE_FEATURES.index("n_hc_lof")] = float(
        clinvar_variants.get(gene, {}).get("n_lof", 1.0)
    )
    gi = torch.tensor([[gene_index[gene]]], dtype=torch.long, device=device)
    with torch.no_grad():
        z = encoder(torch.as_tensor(feats, device=device), gi)
        z0 = encoder(torch.zeros_like(torch.as_tensor(feats, device=device)), gi)
    d = (z - z0).squeeze(0)
    return (d / (d.norm() + 1e-8)).cpu().numpy()
