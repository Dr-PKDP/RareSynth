import torch.nn.functional as F


def _cos_energy(delta, direction, eps=1e-8):
    return 1.0 - F.cosine_similarity(delta, direction, dim=-1, eps=eps)
