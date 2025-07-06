
import torch

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
    
    # Select the first k eigenvectors
    eigvecs = eigenvectors[:, :k]
    eigvecs = eigvecs.transpose(1,0)

    return eigvecs

