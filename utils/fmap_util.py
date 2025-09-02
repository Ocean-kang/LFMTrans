# code adapted from original EOT-Correspondence implementation https://github.com/Tungthanhlee/EOT-Correspondence
import numpy as np
import torch
import torch.nn.functional as F


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
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y, dim=1)

def fmap2pointmap_norm(C12, evecs_x, evecs_y):
    # C12 = C12 / (C12.norm() + 1e-8)
    evecs_x = F.normalize(evecs_x, dim=0)
    evecs_y = F.normalize(evecs_y, dim=0)
    proj = torch.matmul(evecs_x, C12.t())
    proj = torch.nan_to_num(proj)
    return nn_query(proj, evecs_y, dim=1)

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

def fmap_deep_feature_nn(C12, evecs_x, evecs_y, feat_x, feat_y):
    """
    Map source deep features to target domain using functional map,
    then compute NN with target deep features.

    Args:
        C12: [K, K] functional map (X->Y)
        evecs_x: [K, Vx] eigenvectors of source shape X
        evecs_y: [K, Vy] eigenvectors of target shape Y
        feat_x: [Vx, Dx] deep feature of source shape
        feat_y: [Vy, Dy] deep feature of target shape (Dy should match Dx or projected)

    Returns:
        p2p: LongTensor [Vy], index in X for each Y
    """
    # Step 1: project deep feature into spectral space of X
    feat_x_spec = evecs_x @ feat_x  # [K, Dx]

    # Step 2: apply functional map
    feat_y_spec = C12 @ feat_x_spec   # [K, Dx]

    # Step 3: reconstruct in target domain
    mapped_feat_y = evecs_y.T @ feat_y_spec  # [Vy, Dx]

    # Step 4: compute nearest neighbor
    dists = torch.cdist(mapped_feat_y, feat_y)  # [Vy, Vy]
    p2p = torch.argmin(dists, dim=1)

    return p2p


# General Version of different [evecs_x, evecs_y] inputs "[Vx, K] or [K, Vx]"
def fmap_deep_feature_nn_general(C12, evecs_x, evecs_y, feat_x, feat_y, return_mapped_feat=False):
    """
    Map source deep features to target domain using functional map,
    then compute NN with target deep features.

    Args:
        C12: [K, K] functional map (X->Y)
        evecs_x: [Vx, K] or [K, Vx] eigenvectors of source shape X
        evecs_y: [Vy, K] or [K, Vy] eigenvectors of target shape Y
        feat_x: [Vx, Dx] deep feature of source shape
        feat_y: [Vy, Dx] deep feature of target shape
        return_mapped_feat: if True, return mapped features as well

    Returns:
        p2p: LongTensor [Vy], index in X for each Y
        mapped_feat_y (optional): [Vy, Dx]
    """
    # Ensure evecs_x is [Vx, K], evecs_y is [Vy, K]
    if evecs_x.shape[0] != feat_x.shape[0]:  # means it's [K, Vx]
        evecs_x = evecs_x.T
    if evecs_y.shape[0] != feat_y.shape[0]:  # means it's [K, Vy]
        evecs_y = evecs_y.T

    # --- Step 1: project deep feature into spectral space of X ---
    feat_x_spec = evecs_x.T @ feat_x  # [K, Dx]

    # --- Step 2: apply functional map ---
    feat_y_spec = C12 @ feat_x_spec   # [K, Dx]

    # --- Step 3: reconstruct in target domain ---
    mapped_feat_y = evecs_y @ feat_y_spec  # [Vy, Dx]

    # --- Step 4: compute nearest neighbor ---
    dists = torch.cdist(mapped_feat_y, feat_y)  # [Vy, Vy]
    p2p = torch.argmin(dists, dim=1)

    if return_mapped_feat:
        return p2p, mapped_feat_y
    return p2p