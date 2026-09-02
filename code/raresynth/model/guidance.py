"""
Mechanism-consistency energy E(x; g, t) and its use as sampling-time guidance.

This is the component that carries the paper's central claim: that a rare
disease is a *correlated* perturbation across modalities originating at a
causal gene, and that a generator which does not enforce that correlation
produces incoherent patients.

Rather than baking the mechanism into the training corpus (which made the
diffusion model a distillation of the perturbation MLPs), the mechanism is
expressed as a differentiable energy over a candidate multimodal vector:

    E(x; g, t) = w_rna  * [1 - cos( x_rna  - zbar_rna(t),  d_rna(g, t) )]
               + w_cm   * sum_m [1 - cos( x_m - zbar_m(t), d_m(g, t) )]
               + w_path * [1 - cos( x_geno - zbar_geno(t), d_geno(g) )]
               + w_anch * || P_t(x) - zbar(t) ||^2 / D

where
  zbar(t)     tissue-specific healthy baseline (GTEx / normal-adjacent TCGA)
  d_rna(g,t)  PPN-predicted transcriptomic shift direction for gene g
  d_m(g,t)    direction prior for modality m, obtained from the reduced-rank
              cross-modal map applied to d_rna
  d_geno(g)   pathogenicity direction for gene g, from ClinVar P/LP variants
  P_t         projection onto the tissue subspace; the anchor term keeps the
              sample from drifting off the real-data manifold under strong
              guidance

Every term is a cosine on a *direction*, not a match to a magnitude.  The
paired evidence available (tens of tumour/normal cancer types) supports a
claim about direction; it does not support a claim about how large the shift
should be in an unseen monogenic disorder.  Constraining only what the data
supports is also what keeps the guidance from collapsing diversity.

The relative weights, and the global guidance scale s, are the paper's main
ablation axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cos_energy(delta, direction, eps=1e-8):
    """1 - cos(delta, direction), averaged over the batch."""
    return 1.0 - F.cosine_similarity(delta, direction, dim=-1, eps=eps)


class MechanismEnergy:
    """Differentiable mechanism-consistency energy over the full latent vector.

    Parameters
    ----------
    spec        ModalitySpec (defines slice offsets)
    baseline    (B, D) or (D,) tensor: tissue-matched healthy baseline
    directions  dict modality -> (B, d_m) or (d_m,) expected shift direction.
                Modalities absent from the dict contribute nothing.
    weights     dict modality -> float
    anchor_w    weight on the manifold-anchor term
    max_norm    optional cap on ||x - baseline|| per modality; beyond this the
                anchor term grows quadratically.  Prevents strong guidance from
                pushing samples arbitrarily far off-manifold.
    """

    def __init__(
        self,
        spec,
        baseline,
        directions: dict,
        weights: dict | None = None,
        anchor_w: float = 0.1,
        max_norm: float | None = 3.0,
    ):
        self.spec = spec
        self.slices = spec.slices()
        self.baseline = baseline
        self.directions = directions
        self.weights = weights or {k: 1.0 for k in directions}
        self.anchor_w = anchor_w
        self.max_norm = max_norm

    def __call__(self, x, **_):
        base = self.baseline
        if base.dim() == 1:
            base = base.unsqueeze(0).expand_as(x)
        delta_all = x - base
        total = x.new_zeros(x.shape[0])

        for m, sl in self.slices.items():
            d = self.directions.get(m)
            if d is None:
                continue
            dm = delta_all[:, sl]
            if d.dim() == 1:
                d = d.unsqueeze(0).expand_as(dm)
            total = total + self.weights.get(m, 1.0) * _cos_energy(dm, d)

        if self.anchor_w > 0:
            if self.max_norm is None:
                total = total + self.anchor_w * (delta_all**2).mean(dim=-1)
            else:
                excess = (delta_all.norm(dim=-1) - self.max_norm).clamp(min=0)
                total = total + self.anchor_w * excess**2
        return total


def build_directions(
    spec,
    ppn,
    baseline_rna,
    gene_emb,
    cm_priors: dict,
    geno_direction=None,
    device="cpu",
):
    """Assemble the per-modality direction dict for one causal gene.

    cm_priors : dict modality -> CrossModalDirectionPrior (numpy-fitted).
                Applied to the PPN shift to obtain the expected direction in
                pathology / radiology / EHR space.
    """
    import numpy as np

    with torch.no_grad():
        d_rna = ppn(baseline_rna.to(device), gene_emb.to(device))
    dirs = {"rna": d_rna}

    d_rna_np = d_rna.detach().cpu().numpy()
    for m, prior in cm_priors.items():
        if prior is None:
            continue
        d_m = prior.predict(d_rna_np)
        dirs[m] = torch.as_tensor(d_m, dtype=torch.float32, device=device)

    if geno_direction is not None:
        dirs["geno"] = torch.as_tensor(
            geno_direction, dtype=torch.float32, device=device
        )
    return dirs


def guidance_scale_schedule(step, n_steps, s_max, warmup=0.1, decay="cosine"):
    """Optional time-varying guidance scale.

    Applying full guidance at t ~ T (pure noise) steers a vector that carries
    no signal yet; applying it at t ~ 0 fights the model's own high-frequency
    detail.  A ramp that peaks in the middle of the trajectory is what worked
    in preliminary runs and is exposed here as an ablation knob.
    """
    import math

    frac = step / max(n_steps - 1, 1)
    if frac < warmup:
        return s_max * frac / warmup
    if decay == "constant":
        return s_max
    u = (frac - warmup) / (1 - warmup)
    return s_max * 0.5 * (1 + math.cos(math.pi * u))
