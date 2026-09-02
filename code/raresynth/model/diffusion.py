"""
Gaussian diffusion over multimodal latents, with two guidance channels.

Two things are separated here that the original draft conflated:

1. *Classifier-free guidance* on the semantic condition (HPO terms, causal
   gene, tissue).  This sharpens generation toward the requested phenotype and
   is standard.

2. *Mechanism guidance* -- a gradient term  -s * grad_x E(x; g, t_tissue)
   injected into the reverse process, where E is a differentiable
   mechanism-consistency energy (see guidance.py).  This is what carries
   rare-disease specificity.

The point of (2) is that the diffusion model itself is trained only on **real**
embeddings (TCGA, GTEx, CPTAC).  In the original design the model was trained
on PPN/CPM output, so its distribution was upper-bounded by the PPN/CPM
distribution and the diffusion stage added nothing that could not be obtained
by sampling the MLPs directly and adding noise.  Moving the mechanism to
sampling time removes that circularity: the prior is real data, the
perturbation is a steering force, and the guidance scale s gives a continuous
ablation axis from s = 0 (pure real-data prior) upward.

Also implemented: inpainting-style conditional sampling for missing-modality
imputation, following the replacement scheme of Lugmayr et al. (RePaint) --
observed modality slices are re-noised to the current level at every step so
that the generated modalities are conditioned on them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cosine_alpha_bar(T: int, s: float = 0.008, device="cpu"):
    steps = torch.arange(T + 1, dtype=torch.float64, device=device) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    # The floor matters: DDIM recovers x0 as (x - sqrt(1-ab)*eps)/sqrt(ab), so an
    # alpha_bar of 1e-8 multiplies the epsilon-prediction error by 1e4 at the
    # first sampling step and the trajectory never recovers. 1e-4 keeps the
    # amplification bounded at ~100x, which the x0 clamp then absorbs.
    return ab.clamp(1e-4, 1.0)


class GaussianDiffusion:
    def __init__(self, T: int = 1000, schedule: str = "cosine", device="cpu"):
        self.T = T
        if schedule != "cosine":
            raise ValueError("only the cosine schedule is implemented")
        ab = cosine_alpha_bar(T, device=device)
        self.alpha_bar = ab[1:].float()                 # \bar{alpha}_t, t = 1..T
        self.alpha_bar_prev = ab[:-1].float()
        self.betas = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999).float()
        self.alphas = 1.0 - self.betas
        self.device = device

    def to(self, device):
        for k in ("alpha_bar", "alpha_bar_prev", "betas", "alphas"):
            setattr(self, k, getattr(self, k).to(device))
        self.device = device
        return self

    # -- forward ---------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        ab = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def training_loss(
        self,
        model,
        x0,
        avail=None,
        c_hpo=None,
        c_gene=None,
        tissue=None,
        loss_weights=None,
    ):
        """Simple epsilon-MSE loss.

        ``loss_weights`` optionally reweights the coordinate-wise loss per
        modality.  Modalities differ in dimensionality by a factor of two, so
        an unweighted MSE gives pathology (1024 dims) twice the gradient of
        radiology (512 dims) purely from dimension count.  Weighting by
        1/d_m equalises per-modality contribution.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred = model(x_t, t, avail=avail, c_hpo=c_hpo, c_gene=c_gene, tissue=tissue)
        se = (pred - noise) ** 2
        if loss_weights is not None:
            se = se * loss_weights.view(1, -1)
        return se.mean(), {"t_mean": t.float().mean().item()}

    # -- reverse ---------------------------------------------------------
    @torch.no_grad()
    def _eps_with_cfg(self, model, x, t, cfg_scale, **kw):
        if cfg_scale is None or cfg_scale == 1.0:
            return model(x, t, **kw)
        eps_c = model(x, t, force_uncond=False, **kw)
        eps_u = model(x, t, force_uncond=True, **kw)
        return eps_u + cfg_scale * (eps_c - eps_u)

    def ddim_sample(
        self,
        model,
        shape,
        n_steps: int = 200,
        eta: float = 0.0,
        cfg_scale: float | None = 2.0,
        guidance=None,
        guidance_scale: float = 0.0,
        guidance_kwargs: dict | None = None,
        x_obs=None,
        obs_mask=None,
        avail=None,
        c_hpo=None,
        c_gene=None,
        tissue=None,
        device=None,
        generator=None,
        return_trajectory=False,
        x0_clip: float | None = 4.0,
    ):
        """Deterministic (eta=0) DDIM sampling with optional mechanism guidance.

        x_obs / obs_mask
            For conditional imputation.  ``obs_mask`` is a (B, D) 0/1 tensor
            marking observed coordinates; at each step the observed slice of
            x_t is replaced by a correctly-noised version of x_obs.
        guidance
            Callable ``E(x0_hat, **guidance_kwargs) -> scalar tensor``.  Its
            gradient w.r.t. x0_hat is subtracted from the predicted x0.
        """
        device = device or self.device
        gkw = guidance_kwargs or {}
        kw = dict(avail=avail, c_hpo=c_hpo, c_gene=c_gene, tissue=tissue)

        x = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, n_steps, device=device).long()
        traj = []

        for i, t_cur in enumerate(ts):
            t_batch = t_cur.repeat(shape[0])
            ab_t = self.alpha_bar[t_cur]
            t_next = ts[i + 1] if i + 1 < len(ts) else None
            ab_next = self.alpha_bar[t_next] if t_next is not None else torch.tensor(
                1.0, device=device
            )

            if x_obs is not None and obs_mask is not None:
                noisy_obs, _ = self.q_sample(x_obs, t_batch)
                x = obs_mask * noisy_obs + (1 - obs_mask) * x

            eps = self._eps_with_cfg(model, x, t_batch, cfg_scale, **kw)
            x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if x0_clip is not None:
                x0_hat = x0_hat.clamp(-x0_clip, x0_clip)
                eps = (x - ab_t.sqrt() * x0_hat) / (1 - ab_t).sqrt()

            if guidance is not None and guidance_scale != 0.0:
                with torch.enable_grad():
                    xg = x0_hat.detach().requires_grad_(True)
                    energy = guidance(xg, **gkw)
                    (grad,) = torch.autograd.grad(energy.sum(), xg)
                # scale the step by sqrt(1 - ab_t): guidance is strongest early
                # (high noise) and tapers as the sample resolves
                x0_hat = x0_hat - guidance_scale * (1 - ab_t).sqrt() * grad
                if x0_clip is not None:
                    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)
                eps = (x - ab_t.sqrt() * x0_hat) / (1 - ab_t).sqrt()

            if t_next is None:
                x = x0_hat
            else:
                sigma = (
                    eta
                    * ((1 - ab_next) / (1 - ab_t)).sqrt()
                    * (1 - ab_t / ab_next).sqrt()
                )
                dir_xt = (1 - ab_next - sigma**2).clamp(min=0).sqrt() * eps
                x = ab_next.sqrt() * x0_hat + dir_xt
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)

            if return_trajectory:
                traj.append(x0_hat.detach().cpu())

        if x_obs is not None and obs_mask is not None:
            x = obs_mask * x_obs + (1 - obs_mask) * x
        return (x, traj) if return_trajectory else x


class EMA:
    """Exponential moving average of model weights, with warmup.

    The warmup is not cosmetic.  A fixed decay of 0.9999 has an effective
    averaging window of 1/(1-d) = 10,000 steps, so after 8,550 steps the
    shadow weights are still 0.9999^8550 = 0.43, i.e. 43% initialisation.
    Sampling from that checkpoint produces a distribution that saturates the
    x0 clamp and looks exactly like a diverging sampler, which is a
    time-consuming thing to misdiagnose.  Ramping the decay as
    (1+step)/(10+step) capped at ``decay`` makes the average behave like a
    running mean early and like the requested EMA once enough steps have
    accumulated.
    """

    def __init__(self, model, decay=0.9999, warmup=True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def current_decay(self):
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = self.current_decay()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
