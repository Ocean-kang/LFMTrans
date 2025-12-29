import torch
import torch.nn.functional as F

def knngraph(cfg, X, device):

    assert cfg.knngraph.metric_knn in ['ip', 'L2'], f"Unsupported metric: {cfg.knngraph.metric_knn}"

    if cfg.knngraph.metric_knn == "ip":
        return knngraph_ip(cfg, X, device)
    elif cfg.knngraph.metric_knn == "L2":
        return knngraph_L2(cfg, X, device)

def knngraph_ip(cfg, X, device):

    X = X.to(device)
    # X = F.normalize(X, p=2, dim=1)
    weight_matrix = X @ X.T

    k = cfg.knngraph.k
    values, indices = torch.topk(weight_matrix, k=k, dim=1)
    W = torch.zeros_like(weight_matrix)
    W.scatter_(1, indices, values)
    cost_matrix = (W + W.T) / 2

    return cost_matrix

def knngraph_L2(cfg, X, device):

    X = X.to(device)
    X = F.normalize(X, dim=1)
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    sigma = torch.median(dist_sq)
    weight_matrix = torch.exp(- dist_sq / (2 * sigma))

    k = cfg.knngraph.k
    values, indices = torch.topk(weight_matrix, k=k, dim=1)
    W = torch.zeros_like(weight_matrix)
    W.scatter_(1, indices, values)
    cost_matrix = (W + W.T) / 2

    return cost_matrix
