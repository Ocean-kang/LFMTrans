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

def L2_compute_each(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm_sq = (Y ** 2).sum(dim=1, keepdim=True).T 
    # ||x-y||² = ||x||² + ||y||² - 2<x,y>
    dist_sq = X_norm_sq + Y_norm_sq - 2 * (X @ Y.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)             
    return dist_sq.sqrt()                               

def ip_compute(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine-similarity matrix."""
    X = F.normalize(X, p=2, dim=1)
    return X @ X.T

def ip_compute_each(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        X: [n, d]
        Y: [m, d]
    Returns:
        sim: [n, m],  sim[i, j] = cos(X[i], Y[j])
    """
    X = F.normalize(X, p=2, dim=1)  # [n, d]
    Y = F.normalize(Y, p=2, dim=1)  # [m, d]
    return X @ Y.T       

def select_anchor(cfg, feat: torch.Tensor) -> torch.Tensor:
    """
    select anchor for embedding_computing each cls
    Args:
        cfg  : config object
        feat : [N_cls, M, D]  features from model
    Returns:
        anchor_embedding [N_cls, cls_anchor, D]
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    cls_anchor = cfg.anchor.cls_anchors
    N, M, D = feat.shape
    assert cls_anchor <= M

    idx = torch.randperm(M, device=feat.device)[:cls_anchor].expand(N, -1)

    return feat.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

# TODO supervised: pair data in using anchors
# TODO unsupervised: unpair data cluster center for anchors
def anchor_embeddings_compute_supervised(
        cfg, 
        feat_a: torch.Tensor, 
        feat_a_raw: torch.Tensor, 
        feat_b: torch.Tensor,
        feat_b_raw: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        cfg              : config object, needs .seed and .anchor.metric/n_anchors
        feat_a           : [N_cls, D] features from model A
        feat_a_raw       : [N_cls, M, D] features from model A
        feat_b           : [N_cls, D] features from model B
        feat_b_raw       : [N_cls, M, D] features from model B
    Returns:
        anchor_embeddings_a : [N_cls, n_anchor] distances to anchors (model A)
        anchor_embeddings_b : [N_cls, n_anchor] distances to anchors (model B)
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    n_anchors = cfg.anchor.n_anchors

    # extract anchor
    anchor_a = select_anchor(cfg, feat_a_raw)
    anchor_b = select_anchor(cfg, feat_b_raw)

    # reshape
    anchor_a = anchor_a.view(-1, anchor_a.shape[2])[:n_anchors]
    anchor_b = anchor_b.view(-1, anchor_b.shape[2])[:n_anchors]

    if cfg.anchor.metric == 'L2':
        distance_a = L2_compute_each(feat_a, anchor_a)
        distance_b = L2_compute_each(feat_b, anchor_b)
    elif cfg.anchor.metric == 'ip':
        distance_a = ip_compute_each(feat_a, anchor_a)
        distance_b = ip_compute_each(feat_b, anchor_b)
    else:
        raise ValueError(f"Unsupported metric: {cfg.anchor.metric}")

    anchor_embeddings_a = distance_a
    anchor_embeddings_b = distance_b

    return anchor_embeddings_a, anchor_embeddings_b