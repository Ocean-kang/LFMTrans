import pickle as pkl
import yaml

import torch
from easydict import EasyDict as edict

from utils.knngraph import Latent_knn_graph_construct
from utils.LatentFuncitonMap import build_normalized_laplacian_matrix, laplacian_eigendecomposition



if __name__ == '__main__':

    # Loading configs
    with open('./configs/LFMTrans_cfg.yaml','r') as config:
        cfg_dict = yaml.safe_load(config)
    cfg = edict(cfg_dict)

    with open('./feature/feat_clipv_150_B.pkl', 'rb') as file:
        tmp = pkl.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W = Latent_knn_graph_construct(cfg, tmp, device, symmetrize=True)
    L = build_normalized_laplacian_matrix(W, device)
    base = laplacian_eigendecomposition(L, cfg.laplacian_mat.k, device)

    breakpoint()