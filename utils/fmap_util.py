# code adapted from original EOT-Correspondence implementation https://github.com/Tungthanhlee/EOT-Correspondence
import numpy as np
import torch


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)

def deepfmap2pointmap(C12, evecs_x, evecs_y, feat_x, feat_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
        feat_x: deep feature of source shape X [Vx, Dx]
        feat_y: deep feature of source shape Y [Vy, Dy]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    # Transport into eigenvector space
    feat_x_trans = torch.matmul(evecs_x, feat_x) # [n, Dx]

    # [n, V] --> [V, n]
    feat_x_trans = feat_x_trans.permute(1, 0)

    # Transport X space to Y space
    feat_x_trans_C = torch.matmul(feat_x_trans, C12.t()) #[V, n]
    
    return nn_query(torch.matmul(evecs_y.t(), feat_x_trans_C.t()), feat_y)

def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21

