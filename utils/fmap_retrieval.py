import torch
from torch.functional import F

from .fmap_util import fmap2pointmap

def fmap_retrieval(cfg, Cxy, vec_x, vec_y, metric: str='L2'):
    """
    Use Fuction map to retrieve poitwise correspondence based on metric

    Args:
        cfg: configuration object.
        Cxy: Latent Function Map Matrix. [n, n]
        vec_x: eigenvectors of source shape X [n, Vx]
        vec_y: eigenvectors of target shape Y [n, Vy]
        metric: 'L2' or 'ip'

    Returns:
        Correspendece index: feat_x Corresponding to feat_y.
    """
    metric = cfg.fm_retrieval.metric
    assert metric in ['L2', 'ip'], f"Unsupported metric: {metric}, choose 'L2' or 'ip'."

    # [n, V] --> [V, n]
    vec_x = vec_x.permute(1, 0)
    vec_y = vec_y.permute(1, 0)

    if metric == 'L2':
        return fmap2pointmap(Cxy, vec_x, vec_y)
    
    elif metric == 'ip':
        Tran_x = torch.matmul(vec_x, Cxy.transpose(1, 0)) # [Vx, n]

        # Normalize both to unit vectors
        Tran_x_n = F.normalize(Tran_x, p=2, dim=1) # [Vx, n]
        vec_y_n = F.normalize(vec_y, p=2, dim=1)   # [Vy, n]

        # Cosine similarity = dot product of normalized vectors
        sim = torch.matmul(Tran_x_n, vec_y_n.T) # [Vx, Vy]

        # Get most similar Y point for each transformed X point
        p2p = torch.argmax(sim, dim=1) # [Vx]

        return p2p

def accrucy_fn(shuffle_idx, fm_index):
    """
    Judge accrucy of function map method in retrieval

    Arg:
        shuffle_idx: Ground-Truth index.
        fm_index: predicted index.

    Returns:
        Numbers: Correct Numbers.
    """
    equal_count = (shuffle_idx == fm_index).sum().item()

    return equal_count