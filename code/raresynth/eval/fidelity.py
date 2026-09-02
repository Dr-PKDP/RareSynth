"""
Distributional fidelity: Frechet distance, PRDC, and a classifier two-sample
test with an honest null.

Notes on what these do and do not show
--------------------------------------
FID here is computed in the penultimate feature space of an independently
trained tissue-of-origin classifier.  That makes the number comparable across
methods *within this paper* and meaningless across papers, which should be
stated in the caption.  Reporting it per modality as well as jointly is the
part that carries information, because a generator can match every marginal
and still get the joint wrong -- which is exactly the failure mode the paper
is about, and which FID alone will not detect.  See coherence.py for the
metric that does.

C2ST is run with a *learned* classifier (gradient-boosted trees), not 1-NN.
1-NN two-sample tests are known to be weak in high dimension and will report
chance-level accuracy for generators that a stronger classifier separates
trivially, which would flatter the method.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------

def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    """Precision, recall, density, coverage (Naeem et al., 2020)."""

    def knn_radius(X):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        d, _ = nn.kneighbors(X)
        return d[:, -1]

    r_rad, f_rad = knn_radius(real), knn_radius(fake)
    d_rf = np.linalg.norm(real[:, None, :] - fake[None, :, :], axis=2)

    precision = (d_rf < r_rad[:, None]).any(axis=0).mean()
    recall = (d_rf < f_rad[None, :]).any(axis=1).mean()
    density = (1.0 / k) * (d_rf < r_rad[:, None]).sum(axis=0).mean()
    coverage = (d_rf.min(axis=1) < r_rad).mean()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }


def c2st(
    real: np.ndarray,
    fake: np.ndarray,
    n_splits: int = 5,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """Classifier two-sample test with a permutation null.

    Returns accuracy, the permutation p-value, and the null distribution mean,
    so the reader can see that 0.5 is the empirical and not merely the
    theoretical chance level for this sample size.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([real, fake])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])

    def cv_acc(labels):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(X, labels):
            clf = HistGradientBoostingClassifier(
                max_depth=6, max_iter=200, random_state=seed
            )
            clf.fit(X[tr], labels[tr])
            accs.append((clf.predict(X[te]) == labels[te]).mean())
        return float(np.mean(accs))

    obs = cv_acc(y)
    null = []
    n_perm_cheap = min(n_permutations, 200)  # full CV per permutation is costly
    for _ in range(n_perm_cheap):
        null.append(cv_acc(rng.permutation(y)))
    null = np.array(null)
    p = float(((null >= obs).sum() + 1) / (len(null) + 1))
    return {
        "c2st_accuracy": obs,
        "p_value": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Patient-level bootstrap CI for any metric function of arrays."""
    rng = np.random.default_rng(seed)
    vals = []
    n = [len(a) for a in arrays]
    for _ in range(n_boot):
        samples = [a[rng.integers(0, ni, ni)] for a, ni in zip(arrays, n)]
        vals.append(fn(*samples))
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals)), float(np.quantile(vals, alpha / 2)), float(
        np.quantile(vals, 1 - alpha / 2)
    )


def per_modality_report(real, fake, spec, feature_fn=None, k=5, seed=0):
    """Joint + per-modality fidelity table (populates Table 4)."""
    rows = {}
    slices = {"joint": slice(0, real.shape[1])}
    slices.update(spec.slices())
    for name, sl in slices.items():
        r, f = real[:, sl], fake[:, sl]
        if feature_fn is not None:
            r, f = feature_fn(r, name), feature_fn(f, name)
        row = {"fid": frechet_distance(r, f)}
        row.update(prdc(r, f, k=k))
        row.update(c2st(r, f, seed=seed))
        rows[name] = row
    return rows
