import pickle as pkl
import yaml

import torch
from easydict import EasyDict as edict

from utils.knngraph import Latent_knn_graph_construct, Latent_knn_graph_construct_numpy
from utils.LatentFuncitonMap import build_normalized_laplacian_matrix, laplacian_eigendecomposition, laplacian_main_sparse
from model.fmap_network import RegularizedFMNet
from loss.fmap_loss import SURFMNetLoss
from utils.shuffle_utils import shuffle_tensor
from utils.fmap_retrieval import fmap_retrieval, accrucy_fn, cos_sim_retrieval
from model.Encoder import LinearProj



if __name__ == '__main__':

    # Loading configs
    with open('./configs/LFMTrans_cfg.yaml','r') as config:
        cfg_dict = yaml.safe_load(config)
    cfg = edict(cfg_dict)

    with open('./feature/feat_dinov2_patch_cocostuff_L.pkl', 'rb') as file:
        feat_v = pkl.load(file)

    with open('./feature/feat_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb') as file:
        feat_t = pkl.load(file)
    # with open('./feature/feat_dinov2_patch_cocostuff_L.pkl', 'rb') as file:
    #     feat_t = pkl.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = LinearProj(cfg=cfg)
    model.load_state_dict(torch.load("./weight/fmap/proj6.pth", map_location=device))
    model.eval()

    model = model.to(device)
    feat_t = feat_t.to(device).float()
    feat_v = feat_v.to(device).float()

    feat_t_trans = model(feat_t)

    # vision
    W_v = Latent_knn_graph_construct_numpy(cfg, feat_v, device, symmetrize=True)
    v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

    #language
    W_t = Latent_knn_graph_construct_numpy(cfg, feat_t_trans, device, symmetrize=True)
    t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

    # cpu -> gpu and np.arrary -> torch.tensor
    # vision
    v_vecs = torch.from_numpy(v_vecs).to(device).float()
    v_vals = torch.from_numpy(v_vals).to(device).float()
    # language
    t_vecs = torch.from_numpy(t_vecs).to(device).float()
    t_vals = torch.from_numpy(t_vals).to(device).float()

    # # vision
    # W_v = Latent_knn_graph_construct(cfg, feat_v, device, symmetrize=True)
    # L_v = build_normalized_laplacian_matrix(W_v, device)
    # v_vecs, v_vals = laplacian_eigendecomposition(L_v, cfg.laplacian_mat.k, device)

    # #language
    # W_t = Latent_knn_graph_construct(cfg, feat_t_trans, device, symmetrize=True)
    # L_t = build_normalized_laplacian_matrix(W_t, device)
    # t_vecs, t_vals = laplacian_eigendecomposition(L_t, cfg.laplacian_mat.k, device)

    # adding batch dimension
    # vision
    feat_v = feat_v.to(device).unsqueeze(0)
    v_vecs = v_vecs.unsqueeze(0)
    v_vals = v_vals.unsqueeze(0)

    # language
    feat_t_trans = feat_t_trans.float().to(device).unsqueeze(0)
    t_vecs = t_vecs.unsqueeze(0)
    t_vals = t_vals.unsqueeze(0)

    # shuffle vision side
    feat_v_shuffled, shuffle_idx = shuffle_tensor(cfg, device, feat_v)
    shuffle_idx = shuffle_idx.squeeze(0)


    # build regularized_funciton_map model
    fm_net = RegularizedFMNet(bidirectional=True)
    Cxy, Cyx = fm_net(feat_v_shuffled, feat_t_trans, v_vals, t_vals, v_vecs, t_vecs)

    # # fm_loss
    # # fucmap_loss = SURFMNetLoss()
    # # fm_loss_dict = fucmap_loss(Cxy, Cyx, v_vals, t_vals)

    Cxy = Cxy.squeeze(0)
    v_vecs = v_vecs.squeeze(0)
    t_vecs = t_vecs.squeeze(0)
    csr_index = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)

    accurcy = accrucy_fn(shuffle_idx, csr_index)
    print(accurcy)
    breakpoint()