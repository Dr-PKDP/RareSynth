"""
Cross-modal coherence and mechanism retrieval.

These two metrics are the evaluation contribution of the paper.  FID, PRDC and
C2ST all operate on marginals or on the joint treated as an undifferentiated
vector; none of them detects the specific failure the paper claims to fix,
which is a synthetic patient whose pathology latent belongs to a different
virtual person than their transcriptomic latent.  A generator that samples
each modality from the correct marginal but pairs them at random can score a
low joint FID while being biologically meaningless.

Cross-Modal Coherence Score (CMCS)
----------------------------------
Fit modality-pair predictors f_{m->m'} on *real* paired data only.  On real
held-out pairs these achieve some accuracy R_real (they are imperfect: the
relationship between transcriptome and tissue morphology is real but noisy).
Apply the same frozen predictors to synthetic samples to get R_synth.  CMCS is
the ratio R_synth / R_real, clipped at 1.

The ratio, not the raw R_synth, is the right quantity: it asks whether the
synthetic pairs satisfy the conditional relationships *to the same degree that
real pairs do*.  A synthetic set that is more predictable than real data is
not better, it is over-smoothed, so scores above 1 are reported rather than
rewarded and are flagged as mode simplification.

Shuffled-pair control
---------------------
Every CMCS is reported alongside the score obtained by randomly permuting the
modality assignment within the synthetic set.  This is the null the metric
exists to reject, and any paper making a cross-modal coherence claim should
show it.

Mechanism retrieval
-------------------
Given a synthetic sample generated under guidance toward causal gene g, can a
retrieval model trained only on real data recover g from among all candidate
disease genes?  Reported as top-1 / top-10 accuracy and mean reciprocal rank.
This tests whether the mechanism signal survived generation, which is a
stronger and more interpretable claim than "the pathway was enriched".
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


class ModalityPairPredictor:
    """Ridge map from modality m to modality m', fitted on real pairs."""

    def __init__(self, alpha: float = 10.0):
        self.model = Ridge(alpha=alpha)
        self.r2_real = None

    def fit(self, src_train, tgt_train, src_val, tgt_val):
        self.model.fit(src_train, tgt_train)
        pred = self.model.predict(src_val)
        self.r2_real = float(
            r2_score(tgt_val, pred, multioutput="variance_weighted")
        )
        return self

    def score(self, src, tgt):
        pred = self.model.predict(src)
        return float(r2_score(tgt, pred, multioutput="variance_weighted"))


def cross_modal_coherence(
    real_train, real_val, synth, spec, alpha: float = 10.0, seed: int = 0, pairs=None
):
    """Compute CMCS for every ordered modality pair.

    Returns a dict with, per pair: r2 on real validation data, r2 on synthetic
    data, the coherence ratio, and the shuffled-pair control.
    """
    rng = np.random.default_rng(seed)
    slices = spec.slices()
    names = spec.names
    pairs = pairs or [(a, b) for a in names for b in names if a != b]

    out = {}
    for src, tgt in pairs:
        s, t = slices[src], slices[tgt]
        pred = ModalityPairPredictor(alpha).fit(
            real_train[:, s], real_train[:, t], real_val[:, s], real_val[:, t]
        )
        r2_synth = pred.score(synth[:, s], synth[:, t])

        perm = rng.permutation(len(synth))
        r2_shuffled = pred.score(synth[:, s], synth[perm][:, t])

        denom = max(pred.r2_real, 1e-6)
        out[f"{src}->{tgt}"] = {
            "r2_real": pred.r2_real,
            "r2_synth": r2_synth,
            "cmcs": float(r2_synth / denom),
            "r2_shuffled_control": r2_shuffled,
        }

    valid = [v["cmcs"] for v in out.values() if v["r2_real"] > 0.05]
    out["_mean_cmcs"] = float(np.mean(valid)) if valid else float("nan")
    out["_mean_shuffled"] = float(
        np.mean([v["r2_shuffled_control"] for v in out.values() if isinstance(v, dict)])
    )
    return out


class MechanismRetriever:
    """Recover the causal gene of a synthetic sample from its latent.

    Trained on real perturbation-labelled data (LINCS/DepMap latent shifts
    mapped into the RNA slice) using a bilinear scoring function between the
    sample's deviation-from-baseline and the gene embedding.
    """

    def __init__(self, d_latent: int, d_gene: int, alpha: float = 1.0):
        self.alpha = alpha
        self.W = None
        self.d_latent, self.d_gene = d_latent, d_gene

    def fit(self, deltas: np.ndarray, gene_embs: np.ndarray):
        """Closed-form ridge fit of W in score(x, g) = (x^T W g)."""
        X, G = deltas, gene_embs
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ G)
        return self

    def rank_genes(self, delta: np.ndarray, gene_bank: np.ndarray):
        proj = delta @ self.W                                    # (N, d_gene)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        bank = gene_bank / (np.linalg.norm(gene_bank, axis=1, keepdims=True) + 1e-8)
        return proj @ bank.T                                     # (N, n_genes)

    def evaluate(self, deltas, true_idx, gene_bank, ks=(1, 5, 10)):
        scores = self.rank_genes(deltas, gene_bank)
        order = np.argsort(-scores, axis=1)
        ranks = np.array(
            [np.where(order[i] == true_idx[i])[0][0] + 1 for i in range(len(true_idx))]
        )
        res = {f"top{k}": float((ranks <= k).mean()) for k in ks}
        res["mrr"] = float((1.0 / ranks).mean())
        res["median_rank"] = float(np.median(ranks))
        res["n_candidates"] = int(gene_bank.shape[0])
        return res
