"""
Downstream tasks: missing-modality imputation and few-shot diagnosis.

Imputation is the task that turns this from "we generated data" into "we built
a joint model that does something".  It is evaluated on *real* held-out pairs,
needs no rare-disease cohort, and has established baselines (MVAE, MoPoE,
TotalVI-style shared-latent models).  Radiology is available for only a
minority of TCGA cases, so imputation is not a bonus experiment here -- it is
the setting the data actually presents.

Few-shot diagnosis is retained from the draft but with the sample sizes stated
honestly.  With k classes and n test patients the confidence interval on
accuracy is wide; the script reports the interval rather than the point
estimate alone, and refuses to run if n / k falls below a threshold that would
make the number uninterpretable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score, r2_score


# --------------------------------------------------------------------------
# Imputation
# --------------------------------------------------------------------------

def evaluate_imputation(
    imputed: np.ndarray, truth: np.ndarray, spec, target_modalities
):
    """Per-modality imputation quality on held-out real pairs."""
    slices = spec.slices()
    out = {}
    for m in target_modalities:
        sl = slices[m]
        p, t = imputed[:, sl], truth[:, sl]
        cos = (p * t).sum(1) / (
            np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-8
        )
        out[m] = {
            "r2": float(r2_score(t, p, multioutput="variance_weighted")),
            "cosine_mean": float(cos.mean()),
            "cosine_std": float(cos.std()),
            "rmse": float(np.sqrt(((p - t) ** 2).mean())),
        }
    return out


def retrieval_at_k(imputed, truth, ks=(1, 5, 10)):
    """Does the imputed modality retrieve its own true partner?

    Stronger than R^2: a model that predicts the dataset mean for every sample
    scores a respectable R^2 on high-dimensional latents but retrieves nothing.
    """
    P = imputed / (np.linalg.norm(imputed, axis=1, keepdims=True) + 1e-8)
    T = truth / (np.linalg.norm(truth, axis=1, keepdims=True) + 1e-8)
    S = P @ T.T
    order = np.argsort(-S, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(len(P))])
    res = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    res["median_rank"] = float(np.median(ranks))
    return res


# --------------------------------------------------------------------------
# Downstream classifier
# --------------------------------------------------------------------------

class DiagnosticMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=1024, dropout=0.3, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                       nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(X, y, n_classes, epochs=100, lr=1e-3, bs=256, device="cpu",
                     init_state=None, label_smoothing=0.1, seed=0):
    torch.manual_seed(seed)
    model = DiagnosticMLP(X.shape[1], n_classes).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def evaluate_classifier(model, X, y, n_classes, ks=(1, 3, 5), n_boot=1000,
                        device="cpu", seed=0):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, -1).cpu().numpy()
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0]

    res = {}
    for k in ks:
        res[f"top{k}"] = float(np.mean([y[i] in order[i, :k] for i in range(len(y))]))
    res["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))
    res["kappa"] = float(cohen_kappa_score(y, pred))
    res["chance_top1"] = 1.0 / n_classes
    res["n_test"] = len(y)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(np.mean(pred[idx] == y[idx]))
    res["top1_ci"] = (
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )
    if len(y) < 5 * n_classes:
        res["_warning"] = (
            f"n_test={len(y)} for {n_classes} classes: the accuracy interval is "
            "too wide to support a claim of superiority over a baseline. Report "
            "the interval, not the point estimate, and consider collapsing to "
            "disease groups."
        )
    return res


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on paired per-patient correctness."""
    from scipy import stats

    a, b = pred_a == y_true, pred_b == y_true
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    p = float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p_value": p}
