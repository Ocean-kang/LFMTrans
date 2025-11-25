import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# GPU(Pytorch) Version
def build_normalized_laplacian_matrix(W: torch.tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Building Laplacion Matrix by using W_knn Matrix.

    Arg:
        W(torch.tesnor): Weight_KNN sparse Matrix.
        device(str): 'cuda' or 'cpu'.

    Returns:
        L_G(torch.tesnor): Laplcion Matrix [n, n].
    """

    # Ensure W is a square matrix
    assert W.shape[0] == W.shape[1]

    W = W.to(device)

    # Compute the degree matrix D, Sum along rows to get the degree of each node
    D = torch.sum(W, dim=1) # [n]
    # Create the degree matrix as a diagonal matrix
    D = torch.diag(D)

    # Compute D^(-1/2) and Add a small value to avoid division by zero
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-6))
    # Compute D^(-1/2) W D^(-1/2)
    D_inv_sqrt_W_D_inv_sqrt = torch.mm(D_inv_sqrt, torch.mm(W, D_inv_sqrt))

    I = torch.eye(W.shape[0], device=device)
    L_G = I - D_inv_sqrt_W_D_inv_sqrt

    return L_G

def laplacian_eigendecomposition(L: torch.tensor, k: int, device) -> torch.Tensor:
    """
    Obtain the first k eigenvectors of the laplacian matrix.

    Args:
        L (torch.Tensor): Normalized laplacian Matrix
        k (int): Number of feature dimension

    Returns:
        eigvecs (torch.Tensor): The eigenvector matrix [k,n]
    """

    L = L.to(device)
    n = L.shape[0]
    assert k <= n, f"Cannot compute {k} eigenvectors for a matrix of size {n}"

    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the first k eigenvectors and eigenvalues
    eigvecs = eigenvectors[:, :k]
    eigvecs = eigvecs.transpose(1, 0)
    # eigvals_diag = torch.diag(eigenvalues[:k])
    eigvals = eigenvalues[:k]

    return eigvecs, eigvals

def laplacian_main(W: torch.tensor, k: int, device: str='cpu'):
    L = build_normalized_laplacian_matrix(W, device)
    return laplacian_eigendecomposition(L, k)


# CPU(numpy) Vserion
def build_normalized_laplacian_matrix_sparse_cpu(W: sp.spmatrix) -> sp.csr_matrix:
    """
    Build normalized Laplacian matrix from sparse weight matrix W.

    Args:
        W (scipy.sparse.spmatrix): sparse weight matrix [n, n]

    Returns:
        L_G (scipy.sparse.csr_matrix): normalized Laplacian [n, n]
    """
    assert W.shape[0] == W.shape[1]

    D = np.array(W.sum(axis=1)).flatten()  # degree vector [n]
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-6)

    # D^(-1/2)
    D_inv_sqrt_mat = sp.diags(D_inv_sqrt)

    # L = I - D^(-1/2) W D^(-1/2)
    L_G = sp.eye(W.shape[0]) - D_inv_sqrt_mat @ W @ D_inv_sqrt_mat

    return L_G

def laplacian_eigendecomposition_sparse_cpu(L: sp.spmatrix, k: int):
    """
    Compute first k eigenvectors and eigenvalues of sparse Laplacian matrix L.

    Args:
        L (scipy.sparse.spmatrix): normalized Laplacian sparse matrix [n, n]
        k (int): number of eigenvectors to compute

    Returns:
        eigvecs (np.ndarray): shape [k, n]
        eigvals (np.ndarray): shape [k]
    """
    n = L.shape[0]
    assert k <= n, f"Cannot compute {k} eigenvectors for matrix size {n}"

    # argmin k eigvals and eigvecs（默认返回 k 个最大，参数 which='SM'求最小）
    eigvals, eigvecs = eigsh(L, k=k, which='SM')

    # eigsh returns eigvecs are [n, k] -> [k, n]
    eigvecs = eigvecs.T

    # 按 eigenvalues 排序
    sorted_idx = np.argsort(eigvals)
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[sorted_idx]

    return eigvecs, eigvals

def laplacian_main_sparse(W: sp.spmatrix, k: int):
    L = build_normalized_laplacian_matrix_sparse_cpu(W)
    return laplacian_eigendecomposition_sparse_cpu(L, k)

