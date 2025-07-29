import torch
from torch.functional import F

from .fmap_util import fmap2pointmap, deepfmap2pointmap

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
    
def deepfmap_retrieval(cfg, Cxy, vec_x, vec_y, feat_x, feat_y, metric: str='L2'):
    """
    Use Fuction map to retrieve poitwise correspondence based on metric
    Vx, Vy = Number of point
    Dx = Dimension of shape X and Y
    n = number of eigenvals and eigenvectors

    Args:
        cfg: configuration object.
        Cxy: Latent Function Map Matrix. [n, n]
        vec_x: eigenvectors of source shape X [n, Vx]
        vec_y: eigenvectors of target shape Y [n, Vy]
        feat_x: deep feature of source shape X [Vx, Dx]
        feat_y: deep feature of source shape Y [Vy, Dy]
        metric: 'L2' or 'ip'

    Returns:
        Correspendece index: feat_x Corresponding to feat_y.
    """
    metric = cfg.fm_retrieval.metric
    assert metric in ['L2', 'ip'], f"Unsupported metric: {metric}, choose 'L2' or 'ip'."

    if metric == 'L2':
        return deepfmap2pointmap(Cxy, vec_x, vec_y, feat_x, feat_y)
    
    elif metric == 'ip':

        # Transport into eigenvector space
        feat_x_trans = torch.matmul(vec_x, feat_x) # [n, Dx]

        # [n, V] --> [V, n]
        feat_x_trans = feat_x_trans.permute(1, 0)
        
        feat_x_trans_C = torch.matmul(feat_x_trans, Cxy.transpose(1, 0)) # [Vx, n]
        feat_x_trans_C_Y = torch.matmul(vec_y.t(), feat_x_trans_C.t()) # [Vx, Dx]

        # Normalize both to unit vectors
        feat_x_trans_C_Y_n = F.normalize(feat_x_trans_C_Y, p=2, dim=1) # [Vx, Dx]
        feat_y_n = F.normalize(feat_y, p=2, dim=1)  # [Vx, Dx]

        # Cosine similarity = dot product of normalized vectors
        sim = torch.matmul(feat_x_trans_C_Y_n, feat_y_n.T) # [Vx, Vy]

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
    n_cls = shuffle_idx.shape[0]
    equal_count = (shuffle_idx == fm_index).sum().item()
    res = equal_count / n_cls
    return res

def cos_sim_retrieval(feat_1, feat_2):
    '''

    Args:
        feat_1: N_cls x D2
        feat_2: N_cls x D1

    Returns:
        retrieval ratio
    '''
    feat_1_ = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    feat_2_ = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    sim = (feat_1_ @ feat_2_.transpose(1, 0)).cpu()
    idx = sim.argmax(0)
    retrieval_sim = (idx == torch.Tensor(range(len(sim)))).sum() / len(sim)

    return retrieval_sim