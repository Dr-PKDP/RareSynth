"""Principal-component profile of the target direction and construction of principal-component-matched control directions."""

import sys

import numpy as np

DATA = "/data/pduttapramanik/raresynth/data/derived/rna_clean_arrays.npz"
PPN_DIR = "/data/pduttapramanik/raresynth/data/raw/mechanism/ppn_extracted"
KS = (5, 10, 20, 50)

def unit(v):
    return v / np.linalg.norm(v)


def stratified_shuffle(d, sd, block=32, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(sd)
    out = d.copy()
    for start in range(0, len(d), block):
        idx = order[start:start + block]
        out[idx] = d[rng.permutation(idx)]
    return out


def pc_share(d, Vt, ks=KS):
    d = unit(d)
    return {k: float(((Vt[:k] @ d) ** 2).sum()) for k in ks}


def pc_matched_controls(d, Vt, n, seed=0, cos_max=None):
    rng = np.random.default_rng(seed)
    c = Vt @ unit(d)
    signs = rng.choice([-1.0, 1.0], size=(n, len(c)))
    out = (signs * c) @ Vt
    if cos_max is None:
        return out, 1.0
    keep = np.abs(out @ unit(d)) <= cos_max
    return out[keep], float(keep.mean())


def main():
    d = np.load(DATA)
    x = d["rna_train"].astype(np.float64)
    sd = np.maximum(x.std(axis=0), 1e-4)
    _, _, Vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    d_true = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance.npz")["rna_embedding_guidance"].astype(np.float64)
    d_shuf = np.load(f"{PPN_DIR}/ufm1_rna_embedding_guidance_SHUFFLED_CONTROL.npz")["rna_embedding_guidance"].astype(np.float64)
    d_strat = stratified_shuffle(d_true, sd)
    ctrl_all, _ = pc_matched_controls(d_true, Vt, 2000)
    ctrl, acc = pc_matched_controls(d_true, Vt, 2000, cos_max=0.2)

    print(f"principal components of {len(x)} training embeddings; share of each direction's squared norm in the top-k components")
    print(f"{'direction':>28} | " + " | ".join(f"k={k:<3}" for k in KS))
    rows = {"true": pc_share(d_true, Vt), "coordinate shuffle": pc_share(d_shuf, Vt),
            "variance-stratified shuffle": pc_share(d_strat, Vt),
            "PC-matched (first kept control)": pc_share(ctrl[0], Vt)}
    for name, r in rows.items():
        print(f"{name:>28} | " + " | ".join(f"{r[k]:.3f}" for k in KS))
    print(f"{'no preference (k/768)':>28} | " + " | ".join(f"{k/768:.3f}" for k in KS))

    cs = ctrl_all @ unit(d_true)
    print(f"\nPC-matched controls, unfiltered (n=2000): cosine with the true direction: mean {cs.mean():+.3f}, sd {cs.std():.3f}, "
          f"95th percentile of |cos| {np.percentile(np.abs(cs), 95):.3f}, max |cos| {np.abs(cs).max():.3f}")
    print(f"after rejecting draws with |cos| > 0.2 with the true direction: {len(ctrl)} of 2000 kept (acceptance {acc:.3f}); "
          f"if the acceptance is very low the direction's energy is concentrated in too few components for sign randomization alone to give usable controls")
    c = Vt @ unit(d_true)
    top = np.argsort(-(c ** 2))[:5]
    print("largest single-component shares of the true direction (component index: share): "
          + ", ".join(f"{int(i)}: {c[i]**2:.3f}" for i in top))
    print(f"cos(true, coordinate shuffle) = {unit(d_true) @ unit(d_shuf):+.3f}; cos(true, stratified shuffle) = {unit(d_true) @ unit(d_strat):+.3f}")
    print("\nReading: a control is matched in the relevant respect only if its row equals the true direction's. "
          "If the stratified shuffle's profile is close to 'no preference', it is not matched, and the PC-matched "
          "control is the fairer comparator for specificity claims.")


if __name__ == "__main__":
    main()
