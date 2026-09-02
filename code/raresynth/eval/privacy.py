"""
Privacy evaluation for synthetic clinical data.

Any synthetic-patient paper without this gets a reviewer request for it, and
the cost is one afternoon.  Two standard attacks:

Distance to Closest Record (DCR)
    For each synthetic sample, distance to its nearest real training record,
    compared against the distance from a held-out real record to its nearest
    training record.  If the synthetic DCR distribution sits systematically
    below the holdout DCR distribution, the generator is copying.  Reported as
    the 5th percentile and as the DCR ratio, plus the fraction of synthetic
    samples that are closer to a training record than the median holdout
    record is.

Membership Inference Attack (MIA)
    A distance-threshold attack: for a record, decide "was it in training?" by
    thresholding the distance to the nearest synthetic sample.  Reported as
    AUC over a balanced set of training members and held-out non-members.
    0.5 means the synthetic data leaks no membership signal; anything above
    ~0.6 needs to be addressed in the paper rather than omitted.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def _nn_dist(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(reference)
    d, _ = nn.kneighbors(query)
    return d[:, 0]


def dcr_report(synth, real_train, real_holdout, seed: int = 0) -> dict:
    d_synth = _nn_dist(synth, real_train)
    d_hold = _nn_dist(real_holdout, real_train)
    med_hold = float(np.median(d_hold))
    return {
        "dcr_synth_p5": float(np.percentile(d_synth, 5)),
        "dcr_synth_median": float(np.median(d_synth)),
        "dcr_holdout_p5": float(np.percentile(d_hold, 5)),
        "dcr_holdout_median": med_hold,
        "dcr_ratio_median": float(np.median(d_synth) / (med_hold + 1e-12)),
        "frac_synth_closer_than_median_holdout": float((d_synth < med_hold).mean()),
        "n_exact_duplicates": int((d_synth < 1e-8).sum()),
    }


def membership_inference_auc(synth, members, non_members) -> dict:
    """Distance-threshold MIA. AUC near 0.5 = no membership leakage."""
    d_mem = _nn_dist(members, synth)
    d_non = _nn_dist(non_members, synth)
    y = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
    # smaller distance => more likely a member, so negate as the score
    score = -np.concatenate([d_mem, d_non])
    auc = float(roc_auc_score(y, score))
    return {
        "mia_auc": auc,
        "mean_dist_members": float(d_mem.mean()),
        "mean_dist_nonmembers": float(d_non.mean()),
    }


def privacy_report(synth, real_train, real_holdout, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = min(len(real_train), len(real_holdout))
    idx_tr = rng.choice(len(real_train), n, replace=False)
    idx_ho = rng.choice(len(real_holdout), n, replace=False)
    out = dcr_report(synth, real_train, real_holdout, seed=seed)
    out.update(
        membership_inference_auc(synth, real_train[idx_tr], real_holdout[idx_ho])
    )
    return out
