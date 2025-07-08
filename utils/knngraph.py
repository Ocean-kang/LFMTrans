from typing import Tuple, Literal

import faiss
import numpy as np
import torch
import torch.nn.functional as F

def knn_graph_making(cfg, X: torch.Tensor, metric: str = 'ip') -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a k-nearest neighbor (kNN) graph using FAISS.

    Args:
        cfg: Configuration object
        X (torch.Tensor): Feature tensor of shape [n_samples, dim], generated from a pretrained model.
        metric (str): Distance metric for neighbor search. One of ['L2', 'ip'].
                      - 'L2': Euclidean distance
                      - 'ip': Inner product(default)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - distances or similarity scores of shape [n_samples, k]
            - indices of nearest neighbors of shape [n_samples, k]
    """
    metric = cfg.knngraph.knn_fn

    assert metric in ['L2', 'ip'], f"Unsupported metric: {metric}, choose 'L2' or 'ip'."
    assert hasattr(cfg, 'knngraph') and hasattr(cfg.knngraph, 'k'), "cfg.knngraph.k not found"

    # Convert torch tensor to numpy and ensure float32 for FAISS
    X = X.detach().contiguous().cpu().numpy().astype('float32')

    if metric == 'L2':
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
    elif metric == 'ip':
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
   
    sim, idx = index.search(X, cfg.knngraph.k + 1)
    #escape myself
    idx = idx[:, 1:]
    sim_matrix = sim[:, 1:]
    return (sim_matrix, idx)

def weight_matrix_construct(cfg, X: torch.Tensor, device, metric: Literal['L2', 'ip'] = 'ip') -> torch.tensor:
    """
    Construct a full pairwise weight matrix from feature vectors.

    Depending on the metric:
        - 'L2': Returns pairwise L2 distance matrix (n x n)
        - 'ip': Returns pairwise cosine similarity matrix (n x n)

    Args:
        cfg: Unused config (placeholder for future use)
        X (torch.Tensor): Input feature matrix of shape [n, d]
        metric (str): 'L2' for distance, 'ip' for cosine similarity

    Returns:
        torch.Tensor: [n, n] matrix of distances or similarities
    """
    metric = cfg.knngraph.W_full_fn
    assert metric in ['L2', 'ip'], f"Unsupported metric: {metric}, choose 'L2' or 'ip'."

    # Detach and ensure contiguous for safety
    X = X.to(device).detach()

    if metric == 'L2':
        X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
        dist_sq = X_norm_sq + X_norm_sq.T - 2 * (X @ X.T)
        dist_sq = torch.clamp(dist_sq, min=0.0)
        weight_matrix = dist_sq.sqrt()
    elif metric == 'ip':
        X = F.normalize(X, p=2, dim=1)
        weight_matrix = X @ X.T
    return weight_matrix

def knn_weight_matrix_construct(
        cfg,
        W_full: torch.tensor,
        knn_idx: np.array,
        device,
        fn: str='heat_kernal',
        symmetrize: bool = True
    ) -> torch.Tensor:
    """
    knn weight matrix construction

    """
    sigma = cfg.knngraph.sigma
    fn = cfg.knngraph.W_fn
    W_full = W_full = W_full.to(device)
    n, k = knn_idx.shape

    row, col, data = [], [], []
    for i in range(n):
        for j in range(k):
            j_idx = knn_idx[i, j]
            dist = W_full[i, j_idx]

            # Apply weighting function
            if fn == 'heat_kernel':
                w = torch.exp(- dist**2 / sigma**2)
            elif fn == 'inv':
                w = 1.0 / (dist + 1e-6)
            elif fn == 'raw':
                w = dist
            else:
                raise ValueError(f"Unsupported weighting function: {fn}")
            
            row.append(i)
            col.append(j_idx)
            data.append(w)

    # Convert lists to PyTorch tensors
    row = torch.tensor(row, dtype=torch.long, device=device)
    col = torch.tensor(col, dtype=torch.long, device=device)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # Create a COO format sparse tensor
    A = torch.sparse_coo_tensor(torch.stack([row, col]), data, size=(n, n), device=device) 
    if symmetrize:
        A = 0.5 * (A + A.T)
    return A

def Latent_knn_graph_construct(cfg, feat, device, symmetrize):
    '''
    General Function!!!
    '''
    # knn graph construction
    (_, knn_idx) = knn_graph_making(cfg, X=feat, metric=cfg.knngraph.knn_fn)
    # W_full weight construction
    W_full = weight_matrix_construct(cfg, X=feat, device=device, metric=cfg.knngraph.W_full_fn)
    # W_knn weight construction
    W = knn_weight_matrix_construct(cfg, W_full, knn_idx, device, fn=cfg.knngraph.W_fn, symmetrize=symmetrize)
    W = W.to_dense()
    return W

