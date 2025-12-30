import torch
import torch.nn.functional as F

def build_laplacian_matrix(W: torch.Tensor, normalize: str = 'none', device: str = 'cpu') -> torch.Tensor:

    # Input validation
    if not isinstance(W, torch.Tensor):
        raise ValueError("W must be a torch.Tensor")
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    if torch.any(W < 0):
        import warnings
        warnings.warn("Weight matrix W contains negative values, which may lead to unexpected results", UserWarning)
    
    W = W.to(device)
    
    d = torch.sum(W, dim=1)  # [n]
    n = W.shape[0]
    
    d_safe = d.clone()
    d_safe[d_safe == 0] = 1e-10
    
    I = torch.eye(n, device=device)
    
    if normalize == 'none':
        # L = D - W
        D = torch.diag(d)
        L = D - W
        return L
    
    elif normalize == 'sym':
        # L_sym = I - D^{-1/2} W D^{-1/2}
        d_inv_sqrt = 1.0 / torch.sqrt(d_safe)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        D_inv_sqrt_W_D_inv_sqrt = torch.mm(D_inv_sqrt, torch.mm(W, D_inv_sqrt))
        L_sym = I - D_inv_sqrt_W_D_inv_sqrt
        return L_sym
    
    elif normalize == 'rw':
        # L_rw = I - D^{-1} W
        d_inv = 1.0 / d_safe
        D_inv = torch.diag(d_inv)
        D_inv_W = torch.mm(D_inv, W)
        L_rw = I - D_inv_W
        return L_rw
    
    else:
        raise ValueError("normalize parameter must be 'none', 'sym' or 'rw'")
    

def laplacian_construction_decomposition(cfg, cost_matrix, device, ret_L=False):

    assert cfg.knngraph.metric_laplacian in ['none', 'sym', 'rw'], f"Unsupported metric: {cfg.knngraph.metric_laplacian}"

    # L Matrix construction
    L_matrix = build_laplacian_matrix(cost_matrix, cfg.knngraph.metric_laplacian, device)

    # L Matrix decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    if ret_L:
        return eigenvectors, eigenvalues, L_matrix
    else:
        return eigenvectors, eigenvalues

