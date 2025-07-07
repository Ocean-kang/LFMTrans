import pickle as pkl
import yaml

import torch
from easydict import EasyDict as edict

from utils.knngraph import Latent_knn_graph_construct
from utils.LatentFuncitonMap import build_normalized_laplacian_matrix, laplacian_eigendecomposition
from model.fmap_network import RegularizedFMNet



if __name__ == '__main__':

    # Loading configs
    with open('./configs/LFMTrans_cfg.yaml','r') as config:
        cfg_dict = yaml.safe_load(config)
    cfg = edict(cfg_dict)

    with open('./feature/feat_clipv_150_B.pkl', 'rb') as file:
        feat_v = pkl.load(file)

    with open('./feature/feat_clipv_150_B.pkl', 'rb') as file:
        feat_t = pkl.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # vision
    W_v = Latent_knn_graph_construct(cfg, feat_v, device, symmetrize=True)
    L_v = build_normalized_laplacian_matrix(W_v, device)
    v_vecs, v_vals = laplacian_eigendecomposition(L_v, cfg.laplacian_mat.k, device)

    #language
    W_t = Latent_knn_graph_construct(cfg, feat_t, device, symmetrize=True)
    L_t = build_normalized_laplacian_matrix(W_t, device)
    t_vecs, t_vals = laplacian_eigendecomposition(L_t, cfg.laplacian_mat.k, device)

    # adding batch dimension
    # vision
    feat_v = feat_v.to(device).unsqueeze(0)
    v_vecs = v_vecs.unsqueeze(0)
    v_vals = v_vals.unsqueeze(0)

    # language
    feat_t = feat_t.float().to(device).unsqueeze(0)
    t_vecs = t_vecs.unsqueeze(0)
    t_vals = t_vals.unsqueeze(0)
    # build regularized_funciton_map model
    fm_net = RegularizedFMNet()
    Cxy, Cyx = fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)
