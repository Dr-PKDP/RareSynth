"""
Remaining generative baselines.

FlatUNet1D
    The denoiser from the original draft: a 1-D U-Net over the concatenated
    3328-dimensional vector.  Retained deliberately as a baseline so the
    architecture change is quantified rather than asserted.  If MoDiT does not
    beat this, the paper should say so.

IndependentDiffusion
    One diffusion model per modality, sampled independently and concatenated.
    This is the "no cross-modal coherence" ablation and simultaneously the
    strongest naive baseline.  It should score well on per-modality FID and
    badly on CMCS; if it does not, the paper's central premise is wrong and
    that is worth knowing before submission rather than after review.

GaussianCopula
    Rank-based marginals with a Gaussian dependence structure.  Cheap, fast,
    and surprisingly competitive on latent vectors.  Papers that omit it tend
    to be the ones where it would have won.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


# --------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    def __init__(self, cin, cout, d_cond):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.film = nn.Linear(d_cond, cout * 2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = torch.nn.functional.silu(self.norm1(self.conv1(x)))
        g, b = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + g[..., None]) + b[..., None]
        h = torch.nn.functional.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class FlatUNet1D(nn.Module):
    """1-D U-Net over the concatenated latent (the original draft's denoiser)."""

    def __init__(self, spec, base_ch=64, d_cond=256, d_hpo=768, d_gene=256,
                 n_tissues=64, depth=4):
        super().__init__()
        from ..model.dit import ConditionEncoder

        self.spec = spec
        self.D = spec.total_dim
        self.cond = ConditionEncoder(d_cond, d_hpo, d_gene, n_tissues, p_uncond=0.1)
        chs = [base_ch * (2**i) for i in range(depth)]
        self.inp = nn.Conv1d(1, base_ch, 3, padding=1)
        self.downs = nn.ModuleList()
        cin = base_ch
        for c in chs:
            self.downs.append(nn.ModuleList([_ResBlock1D(cin, c, d_cond),
                                             nn.Conv1d(c, c, 4, stride=2, padding=1)]))
            cin = c
        self.mid = _ResBlock1D(cin, cin, d_cond)
        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.ModuleList([nn.ConvTranspose1d(cin, c, 4, stride=2,
                                                              padding=1),
                                           _ResBlock1D(c * 2, c, d_cond)]))
            cin = c
        self.out = nn.Conv1d(cin, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, avail=None, c_hpo=None, c_gene=None, tissue=None,
                force_uncond=False):
        cond = self.cond(t, c_hpo, c_gene, tissue, force_uncond=force_uncond)
        h = self.inp(x.unsqueeze(1))
        skips = []
        for blk, ds in self.downs:
            h = blk(h, cond)
            skips.append(h)
            h = ds(h)
        h = self.mid(h, cond)
        for (us, blk), sk in zip(self.ups, reversed(skips)):
            h = us(h)
            if h.shape[-1] != sk.shape[-1]:
                h = torch.nn.functional.pad(h, (0, sk.shape[-1] - h.shape[-1]))
            h = blk(torch.cat([h, sk], dim=1), cond)
        return self.out(h).squeeze(1)[:, : self.D]


# --------------------------------------------------------------------------

class GaussianCopula:
    """Rank-based marginals + Gaussian dependence."""

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray):
        self.X_sorted = np.sort(X, axis=0)
        n, d = X.shape
        ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
        u = ranks / (n + 1)
        Z = stats.norm.ppf(u)
        C = np.corrcoef(Z, rowvar=False)
        self.Sigma = (1 - self.shrinkage) * C + self.shrinkage * np.eye(d)
        self.L = np.linalg.cholesky(self.Sigma + 1e-6 * np.eye(d))
        self.n, self.d = n, d
        return self

    def sample(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, self.d)) @ self.L.T
        U = stats.norm.cdf(Z)
        idx = np.clip((U * self.n).astype(int), 0, self.n - 1)
        return np.take_along_axis(self.X_sorted, idx, axis=0)


class IndependentDiffusion:
    """Container that trains and samples one diffusion model per modality."""

    def __init__(self, spec, model_factory, diffusion):
        self.spec = spec
        self.models = {m: model_factory(d) for m, d in spec.dims.items()}
        self.diffusion = diffusion

    def sample(self, n, device="cpu", **kw):
        parts = []
        for m, d in self.spec.dims.items():
            parts.append(
                self.diffusion.ddim_sample(
                    self.models[m], (n, d), device=device, cfg_scale=None, **kw
                )
            )
        return torch.cat(parts, dim=-1)
