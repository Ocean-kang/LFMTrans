import random
from typing import Tuple

import torch
import torch.nn.functional as F

def L2_compute(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix."""
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq.sqrt()

def ip_compute(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine-similarity matrix."""
    X = F.normalize(X, p=2, dim=1)
    return X @ X.T

def anchor_embeddings_compute_unsupervised(
    cfg, feat_a: torch.Tensor, feat_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        cfg        : config object, needs .seed and .anchor.metric/n_anchors
        feat_a     : [N_cls, D] features from model A
        feat_b     : [N_cls, D] features from model B
    Returns:
        anchor_embeddings_a : [N_cls, n_anchor] distances to anchors (model A)
        anchor_embeddings_b : [N_cls, n_anchor] distances to anchors (model B)
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    n_anchor = cfg.anchor.n_anchors
    n_cls, d = feat_a.shape
    assert feat_b.shape == (n_cls, d), "feat_a/feat_b shape mismatch"
    assert 0 < n_anchor <= n_cls, f"n_anchor={n_anchor} out of range 1..{n_cls}"

    anchor_idx = torch.randperm(n_cls, device=feat_a.device)[:n_anchor]

    if cfg.anchor.metric == 'L2':
        distance_a = L2_compute(feat_a)
        distance_b = L2_compute(feat_b)
    elif cfg.anchor.metric == 'ip':
        distance_a = ip_compute(feat_a)
        distance_b = ip_compute(feat_b)
    else:
        raise ValueError(f"Unsupported metric: {cfg.anchor.metric}")

    anchor_embeddings_a = distance_a[:, anchor_idx]
    anchor_embeddings_b = distance_b[:, anchor_idx]

    return anchor_embeddings_a, anchor_embeddings_b