import pickle as pkl
import yaml

import torch
from easydict import EasyDict as edict

from model.fmap_network import RegularizedFMNet
from model.Encoder import LinearProj

from utils.knngraph import Latent_knn_graph_construct, Latent_knn_graph_construct_numpy
from utils.LatentFuncitonMap import build_normalized_laplacian_matrix, laplacian_eigendecomposition, laplacian_main_sparse
from utils.shuffle_utils import shuffle_tensor
from utils.fmap_retrieval import fmap_retrieval, accrucy_fn, cos_sim_retrieval
from utils.permutation_compute import compute_permutation_matrices

from loss.fmap_loss import SURFMNetLoss
from loss.proj_loss import proj_loss_sparse_oncefmap



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
    model.load_state_dict(torch.load("./weight/fmap/proj10.pth", map_location=device))
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        feat_t = feat_t.to(device).float()
        feat_v = feat_v.to(device).float()

        feat_t_trans = model(feat_t)

    # vision
    W_v = Latent_knn_graph_construct_numpy(cfg, feat_v, device, symmetrize=False)
    v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

    #language
    W_t = Latent_knn_graph_construct_numpy(cfg, feat_t_trans, device, symmetrize=False)
    t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

    # cpu -> gpu and np.arrary -> torch.tensor
    # vision
    v_vecs = torch.from_numpy(v_vecs).to(device).float()
    v_vals = torch.from_numpy(v_vals).to(device).float()
    # language
    t_vecs = torch.from_numpy(t_vecs).to(device).float()
    t_vals = torch.from_numpy(t_vals).to(device).float()

    # adding batch dimension
    # vision
    feat_v = feat_v.to(device).unsqueeze(0)
    v_vecs = v_vecs.unsqueeze(0)
    v_vals = v_vals.unsqueeze(0)

    # language
    feat_t_trans = feat_t_trans.float().to(device).unsqueeze(0)
    t_vecs = t_vecs.unsqueeze(0)
    t_vals = t_vals.unsqueeze(0)

    Pxy, Pyx = compute_permutation_matrices(cfg, feat_x=feat_t_trans, feat_y=feat_v, with_refine='ip')

    csr_t2v = cos_sim_retrieval(feat_t_trans.squeeze(0), feat_v.squeeze(0))
    csr_v2t = cos_sim_retrieval(feat_v.squeeze(0), feat_t_trans.squeeze(0))

    # shuffle vision side
    feat_v_shuffled, shuffle_idx = shuffle_tensor(cfg, device, feat_v)
    shuffle_idx = shuffle_idx.squeeze(0)

    # build regularized_funciton_map model
    fm_net = RegularizedFMNet(bidirectional=True)
    Cxy, Cyx = fm_net(feat_v_shuffled, feat_t_trans, v_vals, t_vals, v_vecs, t_vecs)

    # general loss
    loss_fn = proj_loss_sparse_oncefmap(cfg=cfg)
    loss_dict = loss_fn(feat_t_trans, feat_v, t_vals, v_vals, t_vecs, v_vecs, Pxy, Pyx)

    # Cxy = Cxy.squeeze(0)
    # v_vecs = v_vecs.squeeze(0)
    # t_vecs = t_vecs.squeeze(0)
    # csr_index = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)

    # accurcy = accrucy_fn(shuffle_idx, csr_index)

    # print(f't2v: {csr_t2v} -- v2t: {csr_v2t} -- accuracy: {accurcy}')
    (W_lap, W_orth, W_bij, W_align, W_ot) = (cfg.loss.w_lap, cfg.loss.w_orth, cfg.loss.w_bij, cfg.loss.w_align, cfg.loss.w_ot)
    breakpoint()