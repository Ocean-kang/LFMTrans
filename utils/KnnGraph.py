import torch
import torch.nn.functional as F


def _symmetrize_knn(weight_matrix: torch.Tensor, k: int) -> torch.Tensor:
    values, indices = torch.topk(weight_matrix, k=min(k, weight_matrix.shape[1]), dim=1)
    W = torch.zeros_like(weight_matrix)
    W.scatter_(1, indices, values)
    return 0.5 * (W + W.T)


def knngraph(cfg, X, device):
    assert cfg.knngraph.metric_knn in ["ip", "L2"], f"Unsupported metric: {cfg.knngraph.metric_knn}"
    if cfg.knngraph.metric_knn == "ip":
        return knngraph_ip(cfg, X, device)
    return knngraph_L2(cfg, X, device)


def knngraph_ip(cfg, X, device):
    """Build a non-negative cosine-affinity kNN graph."""
    X = F.normalize(X.to(device), p=2, dim=1)
    sim = torch.matmul(X, X.T)
    weight_matrix = 0.5 * (sim + 1.0)
    return _symmetrize_knn(weight_matrix, cfg.knngraph.k)


def knngraph_L2(cfg, X, device):
    X = F.normalize(X.to(device), p=2, dim=1)
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    positive_dist = dist_sq[dist_sq > 0]
    sigma = torch.median(positive_dist) if positive_dist.numel() > 0 else torch.tensor(1.0, device=device)
    sigma = torch.clamp(sigma, min=1e-12)

    weight_matrix = torch.exp(-dist_sq / (2.0 * sigma))
    return _symmetrize_knn(weight_matrix, cfg.knngraph.k)
