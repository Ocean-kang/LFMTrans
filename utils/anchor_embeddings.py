import random
from typing import Tuple

import torch
import torch.nn.functional as F

from utils.shuffle_utils import shuffle_features_and_labels
from model.fmap_network import RegularizedFMNet
from utils.fmap_retrieval import accrucy_fn, fmap_retrieval_norm, fmap_retrieval
from utils.knngraph import Latent_knn_sysmmetric_graph_construct_numpy
from utils.laplacian_utils import laplacian_main_sparse

def L2_compute(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix."""
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq.sqrt()

def L2_compute_each(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm_sq = (Y ** 2).sum(dim=1, keepdim=True).T 
    # ||x-y||² = ||x||² + ||y||² - 2<x,y>
    dist_sq = X_norm_sq + Y_norm_sq - 2 * (X @ Y.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)             
    return dist_sq.sqrt()                               

def ip_compute(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine-similarity matrix."""
    X = F.normalize(X, p=2, dim=1)
    return X @ X.T

def ip_compute_each(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        X: [n, d]
        Y: [m, d]
    Returns:
        sim: [n, m],  sim[i, j] = cos(X[i], Y[j])
    """
    X = F.normalize(X, p=2, dim=1)  # [n, d]
    Y = F.normalize(Y, p=2, dim=1)  # [m, d]
    return X @ Y.T                  # [n, m]

def select_anchor(cfg, feat: torch.Tensor) -> torch.Tensor:
    """
    select anchor for embedding_computing each cls
    Args:
        cfg  : config object
        feat : [N_cls, M, D]  features from model
    Returns:
        anchor_embedding [N_cls, cls_anchor, D]
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    cls_anchor = cfg.anchor.cls_anchors
    N, M, D = feat.shape
    assert cls_anchor <= M

    idx = torch.randperm(M, device=feat.device)[:cls_anchor].expand(N, -1)

    return feat.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

def anchor_embeddings_compute_unsupervised(
    cfg, feat_a: torch.Tensor, feat_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        cfg        : config object, needs .seed and .anchor.metric/n_anchors
        feat_a     : [N_cls, D] features from model A
        feat_b     : [N_cls, D] features from model B
    Returns:
        anchor_embeddings_a : [N_cls, n_anchor] distances to anchors (model A)
        anchor_embeddings_b : [N_cls, n_anchor] distances to anchors (model B)
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    n_anchor = cfg.anchor.n_anchors
    n_cls, d = feat_a.shape
    assert feat_b.shape[0] == n_cls, "feat_a/feat_b shape mismatch"
    assert 0 < n_anchor <= n_cls, f"n_anchor={n_anchor} out of range 1..{n_cls}"

    anchor_idx = torch.randperm(n_cls, device=feat_a.device)[:n_anchor]

    if cfg.anchor.metric == 'L2':
        distance_a = L2_compute(feat_a)
        distance_b = L2_compute(feat_b)
    elif cfg.anchor.metric == 'ip':
        distance_a = ip_compute(feat_a)
        distance_b = ip_compute(feat_b)
    else:
        raise ValueError(f"Unsupported metric: {cfg.anchor.metric}")

    anchor_embeddings_a = distance_a[:, anchor_idx]
    anchor_embeddings_b = distance_b[:, anchor_idx]

    return anchor_embeddings_a, anchor_embeddings_b

def anchor_embeddings_compute_supervised(
        cfg, 
        feat_a: torch.Tensor, 
        feat_a_raw: torch.Tensor, 
        feat_b: torch.Tensor,
        feat_b_raw: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        cfg              : config object, needs .seed and .anchor.metric/n_anchors
        feat_a           : [N_cls, D] features from model A
        feat_a_raw       : [N_cls, M, D] features from model A
        feat_b           : [N_cls, D] features from model B
        feat_b_raw       : [N_cls, M, D] features from model B
    Returns:
        anchor_embeddings_a : [N_cls, n_anchor] distances to anchors (model A)
        anchor_embeddings_b : [N_cls, n_anchor] distances to anchors (model B)
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # extract anchor
    anchor_a = select_anchor(cfg, feat_a_raw)
    anchor_b = select_anchor(cfg, feat_b_raw)
    # reshape
    anchor_a = anchor_a.view(-1, anchor_a.shape[2])
    anchor_b = anchor_b.view(-1, anchor_b.shape[2])

    if cfg.anchor.metric == 'L2':
        distance_a = L2_compute_each(feat_a, anchor_a)
        distance_b = L2_compute_each(feat_b, anchor_b)
    elif cfg.anchor.metric == 'ip':
        distance_a = ip_compute_each(feat_a, anchor_a)
        distance_b = ip_compute_each(feat_b, anchor_b)
    else:
        raise ValueError(f"Unsupported metric: {cfg.anchor.metric}")

    anchor_embeddings_a = distance_a
    anchor_embeddings_b = distance_b

    return anchor_embeddings_a, anchor_embeddings_b


def anchor_matching(cfg, feature_dict, feat_vision_raw, feat_language_raw, device, _log):

    # ADE20k-150
    feat_v_a150 = feature_dict["dinov2_features"]["dinov2_ade150"].to(device).float()
    feat_t_a150 = feature_dict["llama3_features"]["llama3_ade150"].to(device).float()
    # ADE20k-847
    feat_v_a847 = feature_dict["dinov2_features"]["dinov2_ade847"].to(device).float()
    feat_t_a847 = feature_dict["llama3_features"]["llama3_ade847"].to(device).float()
    # coco
    feat_v_coco = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()
    feat_t_coco = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
    # pc59
    feat_v_pc59 = feature_dict["dinov2_features"]["dinov2_pc59"].to(device).float()
    feat_t_pc59 = feature_dict["llama3_features"]["llama3_pc59"].to(device).float()
    # pc459
    feat_v_pc459 = feature_dict["dinov2_features"]["dinov2_pc459"].to(device).float()
    feat_t_pc459 = feature_dict["llama3_features"]["llama3_pc459"].to(device).float()
    # voc20
    feat_v_voc20 = feature_dict["dinov2_features"]["dinov2_voc20"].to(device).float()
    feat_t_voc20 = feature_dict["llama3_features"]["llama3_voc20"].to(device).float()
    # voc20b
    feat_v_voc20b = feature_dict["dinov2_features"]["dinov2_voc20b"].to(device).float()
    feat_t_voc20b = feature_dict["llama3_features"]["llama3_voc20b"].to(device).float()

    # match_dataset = ['a150', 'a847', 'coco', 'pc59', 'pc459', 'voc20', 'voc20b']
    match_dataset = ['a150', 'a847', 'coco', 'pc59', 'pc459', 'voc20b']

    import tqdm
    for i in tqdm.tqdm(match_dataset):
        if i == 'a150':
            feat_vision = feat_v_a150
            feat_language = feat_t_a150
        elif i == 'a847':
            feat_vision = feat_v_a847
            feat_language = feat_t_a847
        elif i == 'coco':
            feat_vision = feat_v_coco
            feat_language = feat_t_coco
        elif i == 'pc59':
            feat_vision = feat_v_pc59
            feat_language = feat_t_pc59
        elif i == 'pc459':
            feat_vision = feat_v_pc459
            feat_language = feat_t_pc459
        elif i == 'voc20':
            feat_vision = feat_v_voc20
            feat_language = feat_t_voc20
        elif i == 'voc20b':
            feat_vision = feat_v_voc20b
            feat_language = feat_t_voc20b

        # labels
        n_cls, dimension = feat_language.shape
        feat_labels_v = torch.arange(n_cls).to(device).float()

        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

        #language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_language, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_vision = feat_vision.to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)

        # language
        feat_language = feat_language.float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # anchor descriptor
        feat_v_anchor, feat_t_anchor = anchor_embeddings_compute_supervised(cfg, feat_vision, feat_vision_raw, feat_language, feat_language_raw)

        # shuffle features
        if cfg.anchor.shuffle:
            feat_v_anchor, feat_labels_v = shuffle_features_and_labels(feat_v_anchor, feat_labels_v, cfg.seed)

        # add btach size
        feat_v_anchor = feat_v_anchor.unsqueeze(0)
        feat_t_anchor = feat_t_anchor.unsqueeze(0)

        # build regularized_funciton_map model
        fm_net = RegularizedFMNet(bidirectional=True)
        Cxy, Cyx = fm_net(feat_v_anchor, feat_t_anchor, v_vals, t_vals, v_vecs, t_vecs)

        Cxy = Cxy.squeeze(0)
        Cyx = Cyx.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)

        # Cxy
        csr_index_Cxy = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)
        # Cyx
        csr_index_Cyx = fmap_retrieval(cfg, Cyx, t_vecs, v_vecs)
        # Cxy
        csr_index_Cxy_norm = fmap_retrieval_norm(cfg, Cxy, v_vecs, t_vecs)
        # Cyx
        csr_index_Cyx_norm = fmap_retrieval_norm(cfg, Cyx, t_vecs, v_vecs)

        accurcy_Cxy = accrucy_fn(feat_labels_v, csr_index_Cxy)
        accurcy_Cyx = accrucy_fn(feat_labels_v, csr_index_Cyx)
        accurcy_Cxy_norm = accrucy_fn(feat_labels_v, csr_index_Cxy_norm)
        accurcy_Cyx_norm = accrucy_fn(feat_labels_v, csr_index_Cyx_norm)

        accrucy_norm = (accurcy_Cxy_norm + accurcy_Cyx_norm) / 2
        accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
        print(f'{i}')
        _log.info(f"validation - {i} - {cfg.train.type} - Cxy/Cyx/Avg accrucy: {accurcy_Cxy:.4f}/{accurcy_Cyx:.4f}/{accrucy:.4f}")
        _log.info(f"validation - {i} - {cfg.train.type} - Cxy_norm/Cyx_norm/Avg accrucy_norm: {accurcy_Cxy_norm:.4f}/{accurcy_Cyx_norm:.4f}/{accrucy_norm:.4f}")

    return None