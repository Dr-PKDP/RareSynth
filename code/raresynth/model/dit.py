"""
MoDiT: modality-token diffusion transformer for multimodal biomedical latents.

Design rationale
----------------
The original RareSynth denoiser was a 1-D U-Net applied to the concatenation
[z_geno ; z_rna ; z_path ; z_rad ; z_ehr] in R^3328.  A 1-D convolution over
that vector treats the coordinate index as a spatial axis, which it is not:
coordinate 47 and coordinate 48 of a Geneformer [CLS] embedding have no
adjacency relationship, and the receptive field of a stride-2 stack crosses
modality boundaries as though they were neighbouring pixels.

MoDiT instead projects each modality latent to a shared width d_model and
treats the five modalities as five tokens.  Cross-modal structure is then
modelled by self-attention between tokens, which is permutation-aware,
respects modality boundaries, and handles a missing modality by masking a
token rather than by zero-filling a slice of a convolution input.

Conditioning uses adaLN-Zero (Peebles & Xie, DiT): the timestep and condition
embedding produce per-block scale/shift/gate parameters, with the gate
initialised at zero so every block starts as an identity map.

Missing modalities are first-class.  Radiology is available for only a
minority of TCGA cases, so the model is trained with random modality dropout
and an explicit availability mask; the same mechanism gives conditional
imputation at sampling time for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Modality specification
# --------------------------------------------------------------------------

@dataclass
class ModalitySpec:
    """Ordered specification of the modality latents the model operates on.

    ``dims`` maps a modality name to the dimensionality of its frozen-encoder
    latent.  Order is fixed and defines token order and slice offsets.
    """

    dims: dict = field(
        default_factory=lambda: {
            "geno": 512,
            "rna": 768,   # Geneformer's real confirmed output dim -- the
                          # original spec assumed 512 before any real
                          # encoder had been run; fixed to match reality
                          # (verified: rna_tcga_full.npz etc. all show
                          # shape (N, 768))
            "path": 1024,
            "rad": 512,
            "ehr": 768,
        }
    )

    @property
    def names(self):
        return list(self.dims.keys())

    @property
    def n_modalities(self):
        return len(self.dims)

    @property
    def total_dim(self):
        return sum(self.dims.values())

    def slices(self):
        out, off = {}, 0
        for k, d in self.dims.items():
            out[k] = slice(off, off + d)
            off += d
        return out

    def split(self, x: torch.Tensor):
        """(B, total_dim) -> list of (B, d_m) in modality order."""
        return list(torch.split(x, list(self.dims.values()), dim=-1))

    def join(self, parts):
        return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal embedding of a (B,) integer timestep tensor."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ConditionEncoder(nn.Module):
    """Builds the global conditioning vector from timestep and side information.

    Condition components
      - t              diffusion timestep
      - c_hpo          pooled HPO-term embedding (from a frozen biomedical LM)
      - c_gene         causal-gene embedding (STRING/GO knowledge-graph vector)
      - c_tissue       tissue-of-origin index

    Classifier-free guidance is supported by dropping the *whole* semantic
    condition (hpo + gene + tissue) with probability ``p_uncond`` during
    training and substituting a learned null embedding.
    """

    def __init__(
        self,
        d_model: int,
        d_hpo: int = 768,
        d_gene: int = 256,
        n_tissues: int = 64,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.p_uncond = p_uncond
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model)
        )
        self.hpo_proj = nn.Linear(d_hpo, d_model)
        self.gene_proj = nn.Linear(d_gene, d_model)
        self.tissue_emb = nn.Embedding(n_tissues + 1, d_model)  # last index = unknown
        self.null_sem = nn.Parameter(torch.zeros(d_model))
        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.d_model = d_model
        self.n_tissues = n_tissues

    def forward(self, t, c_hpo=None, c_gene=None, tissue=None, force_uncond=False):
        B = t.shape[0]
        dev = t.device
        t_emb = self.time_mlp(timestep_embedding(t, self.d_model))

        sem = torch.zeros(B, self.d_model, device=dev)
        if c_hpo is not None:
            sem = sem + self.hpo_proj(c_hpo)
        if c_gene is not None:
            sem = sem + self.gene_proj(c_gene)
        tissue = (
            torch.full((B,), self.n_tissues, dtype=torch.long, device=dev)
            if tissue is None
            else tissue
        )
        sem = sem + self.tissue_emb(tissue)
        sem = self.merge(sem)

        if force_uncond:
            drop = torch.ones(B, 1, device=dev)
        elif self.training and self.p_uncond > 0:
            drop = (torch.rand(B, 1, device=dev) < self.p_uncond).float()
        else:
            drop = torch.zeros(B, 1, device=dev)
        sem = drop * self.null_sem.expand(B, -1) + (1.0 - drop) * sem
        return t_emb + sem


# --------------------------------------------------------------------------
# Transformer block with adaLN-Zero
# --------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MoDiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, d_model)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond, key_padding_mask=None):
        s1, sc1, g1, s2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, sc1)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = modulate(self.norm2(x), s2, sc2)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


# --------------------------------------------------------------------------
# MoDiT
# --------------------------------------------------------------------------

class MoDiT(nn.Module):
    """Noise-prediction network over modality tokens.

    Parameters
    ----------
    spec        modality specification (dimensions and order)
    d_model     token width
    depth       number of transformer blocks
    n_heads     attention heads
    tokens_per_modality
        A modality latent may be split into several tokens so that wider
        modalities (path: 1024, ehr: 768) get proportionally more capacity.
        Set to 1 for the minimal five-token model.
    """

    def __init__(
        self,
        spec: ModalitySpec | None = None,
        d_model: int = 512,
        depth: int = 12,
        n_heads: int = 8,
        tokens_per_modality: int = 2,
        d_hpo: int = 768,
        d_gene: int = 256,
        n_tissues: int = 64,
        p_uncond: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spec = spec or ModalitySpec()
        self.tpm = tokens_per_modality
        self.d_model = d_model
        M = self.spec.n_modalities

        # per-modality in/out projections (each modality has its own head:
        # the latents come from different frozen encoders and are not
        # commensurable, so weight sharing across modalities is not assumed)
        self.in_proj = nn.ModuleList(
            [nn.Linear(d, d_model * self.tpm) for d in self.spec.dims.values()]
        )
        self.out_proj = nn.ModuleList(
            [nn.Linear(d_model * self.tpm, d) for d in self.spec.dims.values()]
        )
        for lin in self.out_proj:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        # learned modality-type and intra-modality position embeddings
        self.mod_emb = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(self.tpm, d_model) * 0.02)
        # embedding added to tokens of a modality that is absent for this sample
        self.missing_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        self.cond = ConditionEncoder(d_model, d_hpo, d_gene, n_tissues, p_uncond)
        self.blocks = nn.ModuleList(
            [MoDiTBlock(d_model, n_heads, dropout=dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)

    # -- helpers ---------------------------------------------------------
    def tokenize(self, x, avail):
        """(B, total_dim) -> (B, M*tpm, d_model)."""
        parts = self.spec.split(x)
        toks = []
        for m, (p, proj) in enumerate(zip(parts, self.in_proj)):
            t = proj(p).view(p.shape[0], self.tpm, self.d_model)
            t = t + self.mod_emb[m][None, None] + self.pos_emb[None]
            miss = (1.0 - avail[:, m]).view(-1, 1, 1)
            t = t + miss * self.missing_emb[None, None]
            toks.append(t)
        return torch.cat(toks, dim=1)

    def detokenize(self, h):
        B = h.shape[0]
        h = h.view(B, self.spec.n_modalities, self.tpm * self.d_model)
        return self.spec.join(
            [proj(h[:, m]) for m, proj in enumerate(self.out_proj)]
        )

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        x_t,
        t,
        avail=None,
        c_hpo=None,
        c_gene=None,
        tissue=None,
        force_uncond=False,
    ):
        """Predict the noise added to x_0.

        avail : (B, M) float tensor, 1 = modality observed, 0 = missing.
                Missing modalities are still denoised (that is how imputation
                works); the mask tells the network which tokens carry real
                conditioning information.
        """
        B = x_t.shape[0]
        if avail is None:
            avail = torch.ones(B, self.spec.n_modalities, device=x_t.device)
        cond = self.cond(t, c_hpo, c_gene, tissue, force_uncond=force_uncond)
        h = self.tokenize(x_t, avail)
        for blk in self.blocks:
            h = blk(h, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return self.detokenize(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
