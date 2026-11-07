"""Guidance toward principal-component-matched controls: response to each control and its cross-response along the target direction."""

import argparse
import json
import sys
import time

sys.path.insert(0, "/data/pduttapramanik/raresynth/code")

import numpy as np
import torch

from raresynth.model.dit import MoDiT, ModalitySpec
from raresynth.model.diffusion import GaussianDiffusion
from raresynth.model.guidance import _cos_energy

PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
GTEX_BASELINE = "/data/pduttapramanik/raresynth/data/embeddings/gtex_fibroblast/fibroblast_baseline.npz"
BEST = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/best.pt"
DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
OUT_PATH = "/data/pduttapramanik/raresynth/runs/rna_dit_attempt1/guidance_pc_matched_spill.json"
SCALES = [150, 200, 300]
SEED = 5000

def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cos_rows(u, dhat):
    return (u @ dhat) / (np.linalg.norm(u, axis=1) + 1e-12)


def pc_matched_controls(d, Vt, n, seed=0, cos_max=0.2):
    rng = np.random.default_rng(seed)
    c = Vt @ unit(d)
    out = ((rng.choice([-1.0, 1.0], size=(n, len(c))) * c) @ Vt)
    keep = np.abs(out @ unit(d)) <= cos_max
    return out[keep]


def summarize(own_true, own_ctrl, spill, cos_ctrl_true):
    own_ctrl, spill, cos_ctrl_true = map(np.asarray, (own_ctrl, spill, cos_ctrl_true))
    p95 = float(np.percentile(np.abs(spill), 95))
    r = float(np.corrcoef(spill, cos_ctrl_true)[0, 1]) if len(spill) > 2 and spill.std() > 0 and cos_ctrl_true.std() > 0 else float("nan")
    ratio = p95 / own_true if own_true > 0 else float("nan")
    reading = ("no spill beyond geometric overlap" if ratio <= 0.25 else
               "partly shared" if ratio <= 0.5 else "largely shared") if np.isfinite(ratio) else "not computable"
    return {"own_true": float(own_true), "own_ctrl_mean": float(own_ctrl.mean()), "own_ctrl_min": float(own_ctrl.min()),
            "own_ctrl_max": float(own_ctrl.max()), "spill_mean": float(spill.mean()), "spill_abs_p95": p95,
            "spill_abs_max": float(np.abs(spill).max()), "ratio": float(ratio), "spill_vs_cos_corr": r,
            "followability": float(own_ctrl.mean() / own_true) if own_true > 0 else float("nan"), "reading": reading}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--best", default=BEST)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--k", type=int, default=16, help="number of PC-matched controls")
    args = ap.parse_args()
    n_gen = 16 if args.smoke else 200
    n_steps = 20 if args.smoke else 200
    scales = [150] if args.smoke else SCALES
    k_ctrl = 2 if args.smoke else args.k

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.best, map_location=dev, weights_only=False)
    a, mu, sd, bound = ck["args"], ck["mu"], ck["sd"], ck["bound"]
    x_tr = np.load(DATA)["rna_train"].astype(np.float64)
    _, _, Vt = np.linalg.svd(x_tr - x_tr.mean(axis=0), full_matrices=False)
    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    b = np.load(GTEX_BASELINE)["baseline"].astype(np.float64)
    dh_true = unit(d_true)
    ctrls = pc_matched_controls(d_true, Vt, 4000)[:k_ctrl]
    assert len(ctrls) == k_ctrl, f"only {len(ctrls)} usable controls"
    cos_ctrl_true = ctrls @ dh_true
    print(f"{k_ctrl} PC-matched controls; their cosine with the true direction: min {cos_ctrl_true.min():+.3f}, max {cos_ctrl_true.max():+.3f}", flush=True)

    spec = ModalitySpec(dims={"rna": 768})
    model = MoDiT(spec=spec, d_model=a["d_model"], depth=a["depth"], n_heads=a["heads"],
                  tokens_per_modality=a["tokens_per_modality"], d_hpo=768, d_gene=256,
                  n_tissues=8, p_uncond=a["p_uncond"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    diffusion = GaussianDiffusion(a["timesteps"]).to(dev)
    B = torch.from_numpy(bound).float().to(dev)
    mu_t, sd_t = torch.from_numpy(mu).float().to(dev), torch.from_numpy(sd).float().to(dev)
    b_t = torch.from_numpy(b).float().to(dev)

    class RnaEnergy:
        def __init__(self, direction):
            self.d = torch.from_numpy(direction).float().to(dev)

        def __call__(self, z):
            delta = (z * sd_t + mu_t) - b_t
            return _cos_energy(delta, self.d.unsqueeze(0).expand_as(delta))

    def generate(scale, energy):
        g = torch.Generator(device=dev).manual_seed(SEED)
        x = torch.randn((n_gen, 768), device=dev, generator=g)
        ts = torch.linspace(diffusion.T - 1, 0, n_steps, device=dev).long()
        kw = dict(avail=torch.ones(n_gen, 1, device=dev), c_hpo=None, c_gene=None,
                  tissue=torch.zeros(n_gen, dtype=torch.long, device=dev))
        for i, t_cur in enumerate(ts):
            tb = t_cur.repeat(n_gen)
            ab_t = diffusion.alpha_bar[t_cur]
            t_next = ts[i + 1] if i + 1 < len(ts) else None
            ab_next = diffusion.alpha_bar[t_next] if t_next is not None else torch.tensor(1.0, device=dev)
            with torch.no_grad():
                eps = diffusion._eps_with_cfg(model, x, tb, 0.0, **kw)
                x0 = torch.maximum(torch.minimum((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt(), B), -B)
            if energy is not None and scale != 0.0:
                xg = x0.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    (grad,) = torch.autograd.grad(energy(xg).sum(), xg)
                with torch.no_grad():
                    x0 = torch.maximum(torch.minimum(x0 - scale * (1 - ab_t).sqrt() * grad, B), -B)
            with torch.no_grad():
                eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
                x = x0 if t_next is None else ab_next.sqrt() * x0 + (1 - ab_next).clamp(min=0).sqrt() * eps
        return x.double().cpu().numpy() * sd + mu

    t0 = time.time()
    x_u = generate(0.0, None)
    u_true = cos_rows(x_u - b, dh_true).mean()
    u_ctrl = np.array([cos_rows(x_u - b, c).mean() for c in ctrls])
    print(f"unguided generated ({time.time()-t0:.0f}s): mean cos(d_true) = {u_true:+.4f}", flush=True)

    E_true = RnaEnergy(d_true)
    E_ctrl = [RnaEnergy(c) for c in ctrls]
    results = {"k_controls": int(k_ctrl), "cos_ctrl_true": cos_ctrl_true.tolist(), "scales": {}}
    print(f"\n{'scale':>6} | {'own true':>8} | {'ctrl own mean [min, max]':>26} | {'followability':>13} | {'spill mean':>10} | {'P95|spill|':>10} | "
          f"{'ratio':>6} | {'corr(spill, cos)':>16} | reading")
    for scale in scales:
        own_true = cos_rows(generate(float(scale), E_true) - b, dh_true).mean() - u_true
        own_c, spill = [], []
        for j, (c, E) in enumerate(zip(ctrls, E_ctrl)):
            x_c = generate(float(scale), E)
            own_c.append(cos_rows(x_c - b, c).mean() - u_ctrl[j])
            spill.append(cos_rows(x_c - b, dh_true).mean() - u_true)
        r = summarize(own_true, own_c, spill, cos_ctrl_true)
        results["scales"][str(scale)] = {**r, "own_ctrl": [float(v) for v in own_c], "spill": [float(v) for v in spill]}
        print(f"{scale:>6} | {r['own_true']:>+8.4f} | {r['own_ctrl_mean']:>+8.4f} [{r['own_ctrl_min']:+.3f}, {r['own_ctrl_max']:+.3f}] | "
              f"{r['followability']:>13.2f} | {r['spill_mean']:>+10.4f} | {r['spill_abs_p95']:>10.4f} | {r['ratio']:>6.2f} | "
              f"{r['spill_vs_cos_corr']:>16.2f} | {r['reading']}", flush=True)

    print("\nREADING (rules fixed before the run): ratio <= 0.25 no spill beyond geometric overlap (controls are capped at |cos| <= 0.2, so pure overlap alone gives a ratio of at most about 0.2), 0.25-0.5 partly shared, > 0.5 largely shared; "
          "followability >= 0.8 means the matched controls are followed about as well as the true direction. A spill that tracks "
          "each control's cosine with the true direction (correlation near 1) is geometric overlap.")
    if not args.smoke:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"saved -> {args.out}")
    else:
        print("smoke run finished (nothing saved)")


if __name__ == "__main__":
    main()
