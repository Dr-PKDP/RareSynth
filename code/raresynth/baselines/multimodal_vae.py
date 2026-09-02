"""
Multimodal VAE baselines: MVAE (product of experts) and MoPoE (mixture of
products of experts).

These are the real competition for joint multimodal generation.  Omitting them
in favour of comparing only against unimodal GANs and tabular synthesisers is
the most likely reason a methods reviewer rejects the paper, because they are
the established way to model a joint distribution over heterogeneous
modalities with missing-modality support -- exactly the problem RareSynth
claims.

Both share the same per-modality encoder/decoder stack so that any difference
in the results comes from the fusion rule, not from unequal capacity.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, d_in, d_latent, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.mu = nn.Linear(hidden, d_latent)
        self.logvar = nn.Linear(hidden, d_latent)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)


class ModalityDecoder(nn.Module):
    def __init__(self, d_latent, d_out, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, z):
        return self.net(z)


def product_of_experts(mus, logvars, mask=None):
    """PoE fusion including the standard N(0, I) prior expert."""
    prec = torch.exp(-logvars)
    if mask is not None:
        m = mask.unsqueeze(-1)
        prec, mus = prec * m, mus * m
    prior_mu = torch.zeros_like(mus[:, :1])
    prior_prec = torch.ones_like(prec[:, :1])
    mus = torch.cat([prior_mu, mus], dim=1)
    prec = torch.cat([prior_prec, prec], dim=1)
    prec_sum = prec.sum(1)
    mu = (mus * prec).sum(1) / prec_sum
    return mu, torch.log(1.0 / prec_sum)


class MultimodalVAE(nn.Module):
    """fusion='poe' -> MVAE (Wu & Goodman);  fusion='mopoe' -> MoPoE (Sutter)."""

    def __init__(self, spec, d_latent=256, hidden=1024, fusion="poe", beta=1.0):
        super().__init__()
        self.spec, self.d_latent, self.fusion, self.beta = spec, d_latent, fusion, beta
        dims = list(spec.dims.values())
        self.encoders = nn.ModuleList([ModalityEncoder(d, d_latent, hidden) for d in dims])
        self.decoders = nn.ModuleList([ModalityDecoder(d_latent, d, hidden) for d in dims])
        self.M = len(dims)

    def _encode_all(self, parts):
        mus, lvs = zip(*[enc(p) for enc, p in zip(self.encoders, parts)])
        return torch.stack(mus, 1), torch.stack(lvs, 1)

    def _fuse(self, mus, lvs, mask):
        if self.fusion == "poe":
            return [product_of_experts(mus, lvs, mask)]
        subsets = []
        idx = list(range(self.M))
        for r in range(1, self.M + 1):
            for sub in itertools.combinations(idx, r):
                sm = torch.zeros_like(mask)
                sm[:, list(sub)] = 1.0
                sm = sm * mask
                if (sm.sum(1) == 0).all():
                    continue
                subsets.append(product_of_experts(mus, lvs, sm))
        return subsets

    def forward(self, x, mask=None):
        parts = self.spec.split(x)
        B = x.shape[0]
        mask = torch.ones(B, self.M, device=x.device) if mask is None else mask
        mus, lvs = self._encode_all(parts)
        fused = self._fuse(mus, lvs, mask)

        recon_loss = x.new_zeros(())
        kl_loss = x.new_zeros(())
        for mu, lv in fused:
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            rec = torch.cat([dec(z) for dec in self.decoders], dim=-1)
            recon_loss = recon_loss + F.mse_loss(rec, x)
            kl_loss = kl_loss + (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(-1)).mean()
        n = len(fused)
        return recon_loss / n + self.beta * kl_loss / n, {
            "recon": (recon_loss / n).item(),
            "kl": (kl_loss / n).item(),
        }

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        z = torch.randn(n, self.d_latent, device=device)
        return torch.cat([dec(z) for dec in self.decoders], dim=-1)

    @torch.no_grad()
    def impute(self, x, mask):
        """Generate missing modalities conditioned on the observed ones."""
        parts = self.spec.split(x)
        mus, lvs = self._encode_all(parts)
        mu, lv = product_of_experts(mus, lvs, mask)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        rec = torch.cat([dec(z) for dec in self.decoders], dim=-1)
        keep = torch.cat(
            [
                mask[:, i : i + 1].expand(-1, d)
                for i, d in enumerate(self.spec.dims.values())
            ],
            dim=-1,
        )
        return keep * x + (1 - keep) * rec
