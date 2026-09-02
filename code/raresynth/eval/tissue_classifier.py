"""
Tissue-of-origin classifier, trained on REAL data, whose penultimate
layer serves as the feature space fidelity.py's FID computation should
run in -- matching what per_modality_report's own docstring always said
it did ("FID here is computed in the penultimate feature space of an
independently trained tissue-of-origin classifier"), which the actual
code never implemented (it called frechet_distance directly on raw,
full-dimensional embeddings).

This matters, confirmed by direct demonstration, not theory: computing
FID in the RAW 3584-dim concatenated modality space with only ~500-4000
real samples is a severe small-n-large-p covariance estimation problem.
Live check: two samples drawn from the IDENTICAL distribution produced
FID=4939 (should be ~0), and the real covariance matrix's rank was
capped at n-1=499 instead of the full 3584 dimensions -- the computation
was numerically meaningless at that dimensionality/sample-size ratio.

The classifier's penultimate layer is kept deliberately narrow (default
64-dim) specifically so n (hundreds to thousands of real cases) stays
much larger than d (64), keeping the FID covariance estimate
well-conditioned -- this is the same design principle as image-domain
FID using Inception's ~2048-dim penultimate layer with tens of thousands
of images, scaled down appropriately for this project's real sample
sizes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TissueClassifier(nn.Module):
    def __init__(self, input_dim, n_tissues, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.penultimate_dim = prev
        self.head = nn.Linear(prev, n_tissues)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits

    @torch.no_grad()
    def embed(self, x, batch_size=256, device="cpu"):
        """x: (N, input_dim) numpy array or tensor -> (N, penultimate_dim)
        numpy array of penultimate-layer features, the space FID should
        be computed in.
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        out = []
        for i in range(0, len(x), batch_size):
            b = x[i:i + batch_size].to(device)
            _, feats = self.forward(b, return_features=True)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0)


def train_tissue_classifier(x_train, y_train, x_val, y_val, n_tissues,
                            device="cpu", epochs=100, lr=1e-3, batch_size=64,
                            seed=0):
    """x_*: (N, input_dim) real data (already at the real, unscaled
    embedding magnitude -- this classifier is a diagnostic tool, not part
    of the generative model, so it does not need data_scale applied).
    y_*: (N,) integer tissue labels.
    """
    torch.manual_seed(seed)
    input_dim = x_train.shape[1]
    model = TissueClassifier(input_dim, n_tissues).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    n = len(x_train_t)
    best_val_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train_t[idx].to(device), y_train_t[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"tissue classifier: best val accuracy {best_val_acc:.3f} "
         f"(chance level: {1.0/n_tissues:.3f})")
    return model, best_val_acc
