"""
Perturbation Prediction Network (PPN) and the cross-modal direction prior.

PPN
---
Maps (baseline transcriptomic latent for tissue t, causal gene embedding e_g)
to a predicted shift in the RNA latent.  Trained on LINCS L1000 consensus
signatures and DepMap CRISPR knockouts, both of which give paired
(perturbed, control) expression from which a latent shift is computed.

Two changes from the draft:

* The evaluation split is by **gene**, not by sample, and the held-out-gene
  set is reported separately.  A split that lets the same gene appear in train
  and test measures memorisation of the gene embedding, not generalisation to
  a new disease gene, and 40% of the target diseases have no direct
  perturbation data at all.

* Direction is scored as well as magnitude.  Cosine similarity between
  predicted and observed shift is the quantity the guidance term actually
  uses; L2 error alone is dominated by shift magnitude, which varies by orders
  of magnitude across cell lines and is not what we need to be right about.

CrossModalDirectionPrior
------------------------
The draft's CPM fitted three MLPs (~1.5M parameters) on 50 paired tumour/normal
shift observations.  That is not estimable.  What is estimable from 33-50
paired observations is a *low-rank linear* map with strong shrinkage: we fit
partial least squares / reduced-rank regression with rank r <= 8 and ridge
penalty, chosen by leave-one-cancer-type-out cross-validation.  The result is
used only as a direction prior inside the guidance energy, never as a
generative pathway, so the burden it carries matches the evidence available.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PPN(nn.Module):
    def __init__(self, d_rna: int = 512, d_gene: int = 256, hidden: int = 1024,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(d_rna + d_gene, hidden), nn.SiLU(), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(n_blocks)
            ]
        )
        self.out = nn.Linear(hidden, d_rna)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_rna_baseline, e_gene):
        h = self.inp(torch.cat([z_rna_baseline, e_gene], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def ppn_loss(pred, obs, lam_l2: float = 1e-3, lam_dir: float = 1.0):
    """Magnitude term + direction term.

    The direction term is what the mechanism guidance consumes downstream, so
    it is trained for explicitly rather than hoped for as a by-product of MSE.
    """
    mse = ((pred - obs) ** 2).mean()
    cos = torch.nn.functional.cosine_similarity(pred, obs, dim=-1).mean()
    reg = (pred**2).mean()
    return mse + lam_dir * (1 - cos) + lam_l2 * reg, {
        "mse": mse.item(),
        "cos": cos.item(),
    }


class GeneNeighbourImputer:
    """k-NN imputation of shifts for genes with no perturbation measurement.

    Kept from the original design, but with the neighbourhood restricted to
    genes whose *measured* shift passed a reproducibility filter, and with the
    similarity floor exposed so that a gene with no functionally close
    measured neighbour returns ``None`` rather than a meaningless average.
    """

    def __init__(self, gene_emb: np.ndarray, gene_ids, measured_shifts: dict,
                 k: int = 5, min_sim: float = 0.3):
        self.E = gene_emb / (np.linalg.norm(gene_emb, axis=1, keepdims=True) + 1e-8)
        self.gene_ids = list(gene_ids)
        self.index = {g: i for i, g in enumerate(self.gene_ids)}
        self.measured = measured_shifts
        self.k, self.min_sim = k, min_sim
        self.measured_idx = np.array(
            [self.index[g] for g in measured_shifts if g in self.index]
        )
        self.measured_names = [g for g in measured_shifts if g in self.index]

    def __call__(self, gene):
        if gene in self.measured:
            return self.measured[gene], 1.0
        if gene not in self.index or len(self.measured_idx) == 0:
            return None, 0.0
        sims = self.E[self.measured_idx] @ self.E[self.index[gene]]
        order = np.argsort(-sims)[: self.k]
        sims_k, names_k = sims[order], [self.measured_names[i] for i in order]
        keep = sims_k >= self.min_sim
        if not keep.any():
            return None, float(sims_k.max())
        w = sims_k[keep] / sims_k[keep].sum()
        shift = sum(
            wi * self.measured[n] for wi, n in zip(w, np.array(names_k)[keep])
        )
        return shift, float(sims_k[keep].mean())


class CrossModalDirectionPrior:
    """Reduced-rank ridge map from an RNA shift to shifts in other modalities.

    Fitted on paired (tumour, matched normal) shift vectors, one observation
    per cancer type.  ``rank`` and ``alpha`` are selected by
    leave-one-cancer-type-out CV in ``fit``.
    """

    def __init__(self, rank: int = 4, alpha: float = 10.0):
        self.rank, self.alpha = rank, alpha
        self.W = None
        self.x_mean = self.y_mean = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (C, d_rna) RNA shifts. Y: (C, d_target) target-modality shifts."""
        self.x_mean, self.y_mean = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.x_mean, Y - self.y_mean
        G = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        B = np.linalg.solve(G, Xc.T @ Yc)              # ridge solution
        U, S, Vt = np.linalg.svd(Xc @ B, full_matrices=False)
        r = min(self.rank, len(S))
        P = Vt[:r].T @ Vt[:r]                          # rank-r projector
        self.W = B @ P
        return self

    def predict(self, x):
        return (x - self.x_mean) @ self.W + self.y_mean

    @staticmethod
    def select_hyperparams(X, Y, groups, ranks=(1, 2, 4, 8), alphas=(1, 10, 100, 1000)):
        """Leave-one-group-out CV; groups is e.g. the cancer-type label."""
        groups = np.asarray(groups)
        best, best_score = None, -np.inf
        for r in ranks:
            for a in alphas:
                scores = []
                for g in np.unique(groups):
                    tr, te = groups != g, groups == g
                    if tr.sum() < 2:
                        continue
                    m = CrossModalDirectionPrior(r, a).fit(X[tr], Y[tr])
                    p, y = m.predict(X[te]), Y[te]
                    cos = (p * y).sum(1) / (
                        np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1) + 1e-8
                    )
                    scores.append(cos.mean())
                s = float(np.mean(scores)) if scores else -np.inf
                if s > best_score:
                    best, best_score = (r, a), s
        return best, best_score
