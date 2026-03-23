import torch


def build_laplacian_matrix(W: torch.Tensor, normalize: str = 'none', device: str = 'cpu') -> torch.Tensor:
    if not isinstance(W, torch.Tensor):
        raise ValueError('W must be a torch.Tensor')
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError('W must be a square matrix')

    W = W.to(device)
    W = 0.5 * (W + W.T)
    W = torch.clamp(W, min=0.0)

    d = torch.sum(W, dim=1)
    n = W.shape[0]
    d_safe = torch.clamp(d, min=1e-10)
    I = torch.eye(n, device=device, dtype=W.dtype)

    if normalize == 'none':
        return torch.diag(d) - W
    if normalize == 'sym':
        d_inv_sqrt = torch.diag(torch.rsqrt(d_safe))
        return I - d_inv_sqrt @ W @ d_inv_sqrt
    if normalize == 'rw':
        d_inv = torch.diag(1.0 / d_safe)
        return I - d_inv @ W
    raise ValueError("normalize parameter must be 'none', 'sym' or 'rw'")


def laplacian_construction_decomposition(cfg, cost_matrix, device, ret_L=False):
    assert cfg.knngraph.metric_laplacian in ['none', 'sym', 'rw'], f"Unsupported metric: {cfg.knngraph.metric_laplacian}"

    L_matrix = build_laplacian_matrix(cost_matrix, cfg.knngraph.metric_laplacian, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_matrix)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    if ret_L:
        return eigenvectors, eigenvalues, L_matrix
    return eigenvectors, eigenvalues
