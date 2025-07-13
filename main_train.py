import os
import random
import logging
import warnings
import pickle as pkl

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict

from model.Encoder import LinearProj
from loss.proj_loss import proj_loss, proj_loss_sparse, proj_loss_sparse_oncefmap
from utils.knngraph import Latent_knn_graph_construct_numpy
from utils.LatentFuncitonMap import laplacian_main_sparse
from utils.shuffle_utils import shuffle_tensor
from utils.fmap_retrieval import fmap_retrieval, accrucy_fn, cos_sim_retrieval
from utils.permutation_compute import compute_permutation_matrices
from model.fmap_network import RegularizedFMNet

warnings.filterwarnings("ignore", category=UserWarning)
ex = Experiment('uvlt')

def create_basic_stream_logger(format):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

ex.logger = create_basic_stream_logger('%(levelname)s - %(name)s - %(message)s')
ex.add_config('./configs/LFMTrans_cfg.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

@ex.capture
def train_proj(cfg, model, feature_dict, criterion, device, _log):
    """
    Train the Projector.

    Args:
        model: the LinearProj model
        feature_dict: feature_dict
        device: 'cuda' or 'cpu'
    """
    num_epochs = cfg.model.nums_epoch
    lr = cfg.model.lr_proj

    # Load features
    feat_language = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
    feat_vision = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        x = feat_language.to(device)  # [B, text_dimension]
        y = feat_vision.to(device)  # [B, vision_dimension]

        optimizer.zero_grad()
        output = model(x)  # [B, vision_dimension]
        loss_dict = criterion(output, y)

        # Weight_init
        (W_lap, W_orth, W_bij) = (1.0, 1.0, 0.0)


        loss = loss_dict['l_lap'] * W_lap + loss_dict['l_orth'] * W_orth + loss_dict['l_bij'] * W_bij
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _log.info(f"Epoch {epoch+1}/{num_epochs} - Per Loss: {loss:.4f}")
    _log.info(f"Train Finished - Avg Loss: {total_loss / num_epochs:.4f}")

    return None

@ex.capture
def train_proj_onceafmap(cfg, model, feature_dict, criterion, device, _log):
    """
    Train the Projector.

    Args:
        model: the LinearProj model
        feature_dict: feature_dict
        device: 'cuda' or 'cpu'
    """
    num_epochs = cfg.model.nums_epoch
    lr = cfg.model.lr_proj

    # Load features
    feat_language = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
    feat_vision = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()

    # adding batch dimension
    if cfg.train.batchsize == 1:
        feat_language = feat_language.unsqueeze(0)
        feat_vision = feat_vision.unsqueeze(0)
    
    # Load model
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # compute once eigenvecs and eigenvals in each multimodels
    # vision
    W_v = Latent_knn_graph_construct_numpy(cfg, feat_vision, device, symmetrize=True)
    v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

    #language
    W_t = Latent_knn_graph_construct_numpy(cfg, feat_language, device, symmetrize=True)
    t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

    # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
    # vision
    v_vecs = torch.from_numpy(v_vecs).to(device).float().unsqueeze(0)
    v_vals = torch.from_numpy(v_vals).to(device).float().unsqueeze(0)
    # language
    t_vecs = torch.from_numpy(t_vecs).to(device).float().unsqueeze(0)
    t_vals = torch.from_numpy(t_vals).to(device).float().unsqueeze(0)

    total_loss = 0.0
    for epoch in range(num_epochs):

        x = feat_language.to(device)  # [B, text_dimension]
        y = feat_vision.to(device)  # [B, vision_dimension]

        optimizer.zero_grad()
        output = model(x)  # [B, vision_dimension]

        # compute permutation
        Pxy, Pyx = compute_permutation_matrices(cfg, output, y)

        # Loss
        loss_dict = criterion(output, y, t_vals, v_vals, t_vecs, v_vecs, Pxy, Pyx)

        # Weight_init
        (W_lap, W_orth, W_bij, W_align, W_ot) = (1.0, 1.0, 0.0, 1.0, 1.0)
        loss_fm = loss_dict['l_lap'] * W_lap + loss_dict['l_orth'] * W_orth + loss_dict['l_bij'] * W_bij
        loss_align = loss_dict['l_align'] * W_align
        loss_ot = loss_dict['l_ot'] * W_ot
        loss = loss_fm + loss_align + loss_ot
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _log.info(f"Epoch {epoch+1}/{num_epochs} - Per Loss: {loss:.4f}")
    _log.info(f"Train Finished - Avg Loss: {total_loss / num_epochs:.4f}")

    return None

@ex.capture
def eval_proj(cfg, model, feature_dict, device, _log):

    # Load features
    feat_language = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
    feat_vision = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()

    feat_v = feat_vision
    feat_t_trans = model(feat_language)
    csr_accuracy_t2v = cos_sim_retrieval(feat_t_trans, feat_v)
    csr_accuracy_v2t = cos_sim_retrieval(feat_v, feat_t_trans)

    # compute once eigenvecs and eigenvals in each multimodels
    # vision
    W_v = Latent_knn_graph_construct_numpy(cfg, feat_vision, device, symmetrize=True)
    v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

    #language
    W_t = Latent_knn_graph_construct_numpy(cfg, feat_language, device, symmetrize=True)
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

    Cxy = Cxy.squeeze(0)
    Cyx = Cyx.squeeze(0)
    v_vecs = v_vecs.squeeze(0)
    t_vecs = t_vecs.squeeze(0)
    # Cxy
    csr_index_Cxy = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)
    # Cyx
    csr_index_Cyx = fmap_retrieval(cfg, Cyx, t_vecs, v_vecs)

    accurcy_Cxy = accrucy_fn(shuffle_idx, csr_index_Cxy)
    accurcy_Cyx = accrucy_fn(shuffle_idx, csr_index_Cyx)

    accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
    _log.info(f"Train Finished - Avg accrucy: {accrucy:.4f} - CSR_t2v: {csr_accuracy_t2v:.4f} - CSR_v2t: {csr_accuracy_v2t:.4f}")

    return None

@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = cfg.model.DINOv2_BACKBONE_SIZE[0].upper()

    TRAINSETPTH = './feature/train'
    TESTSETPATH = './feature/val'

    feature_dict = {
        "dinov2_features": {
            "dinov2_coco": pkl.load(open(f'./feature/feat_dinov2_patch_cocostuff_{size}.pkl', 'rb')),
            "dinov2_ade150": pkl.load(open(f'./feature/feat_dinov2_patch_150_{size}.pkl', 'rb')),
            "dinov2_ade847": pkl.load(open(f'./feature/feat_dinov2_patch_847_{size}.pkl', 'rb')),
            "dinov2_voc20": pkl.load(open(f'./feature/feat_dinov2_patch_voc20_{size}.pkl', 'rb')),
            "dinov2_voc20b": pkl.load(open(f'./feature/feat_dinov2_patch_voc20b_{size}.pkl', 'rb')),
            "dinov2_pc59": pkl.load(open(f'./feature/feat_dinov2_patch_pc59_{size}.pkl', 'rb')),
            "dinov2_pc459": pkl.load(open(f'./feature/feat_dinov2_patch_pc459_{size}.pkl', 'rb')),
        },
        "llama3_features": {
            "llama3_coco_unmean": pkl.load(open(f'./feature/feat_llama3_cocostuff_S2P0_27_28_29_8_12_26_30_unmean.pkl', 'rb')),
            "llama3_coco": pkl.load(open(f'./feature/feat_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_ade150": pkl.load(open(f'./feature/feat_llama3_150_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_ade847": pkl.load(open(f'./feature/feat_llama3_847_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_voc20": pkl.load(open(f'./feature/feat_llama3_voc20_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_voc20b": pkl.load(open(f'./feature/feat_llama3_voc20b_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_pc59": pkl.load(open(f'./feature/feat_llama3_pc59_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_pc459": pkl.load(open(f'./feature/feat_llama3_pc459_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
        },
        "llama3_synonym_features": {
            "llama3_coco": pkl.load(open(f'./feature/feat_synonyms_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
        },
        "llama3_subpart_features": {
            "llama3_coco": pkl.load(open(f'./feature/feat_subparts_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
        },
        "dinov2_subpart_features": {
            "dinov2_coco": pkl.load(open(f'./feature/feat_subpart_dinov2_cocostuff_20k.pkl', 'rb')),
        },
        "dinov2_synonym_features": {
            "dinov2_coco": pkl.load(open(f'./feature/feat_synonym_dinov2_cocostuff.pkl', 'rb')),
        },
        "dinov2_patchtoken_clustering": {
            "dinov2_coco": pkl.load(open(f'./feature/dinov2_cocostuff_patchtoken_clustering_16_{size}.pkl', 'rb'))
        },
        "cinic-10": {
            "train":{
                "dinov2": torch.load(TRAINSETPTH + '/CINIC-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TRAINSETPTH + '/CINIC-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TRAINSETPTH + '/CINIC-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TRAINSETPTH + '/CINIC-10/labels.pt'),
                "prototype": torch.load(TRAINSETPTH + '/CINIC-10/vision/Prototype_dinov2_vit-g14.pt')

            },
            "test":{
                "dinov2": torch.load(TESTSETPATH + '/CINIC-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TESTSETPATH + '/CINIC-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TESTSETPATH + '/CINIC-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TESTSETPATH + '/CINIC-10/labels.pt')
            }
        },
        "cifar-10":{
            "test":{
                "dinov2": torch.load(TESTSETPATH + '/CIFAR-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TESTSETPATH + '/CIFAR-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TESTSETPATH + '/CIFAR-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TESTSETPATH + '/CIFAR-10/labels.pt')
            },
            "train":{
                "dinov2": torch.load(TRAINSETPTH + '/CIFAR-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TRAINSETPTH + '/CIFAR-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TRAINSETPTH + '/CIFAR-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TRAINSETPTH + '/CIFAR-10/labels.pt'),
                "prototype": torch.load(TRAINSETPTH + '/CIFAR-10/vision/Prototype_dinov2_vit-g14.pt')
            }
        }
    }
    
    
    model = LinearProj(cfg=cfg)

    # --dense matrix--
    # criterion = proj_loss(cfg=cfg)
    # --sparse matrix--
    # criterion = proj_loss_sparse(cfg=cfg)
    # --sparse matrix once enginvecs and enginvals--
    criterion_once = proj_loss_sparse_oncefmap(cfg=cfg)

    # train_proj(cfg, model, feature_dict, criterion, device, _log)
    train_proj_onceafmap(cfg, model, feature_dict, criterion_once, device, _log)


    if os.path.isdir('./weight'):
        torch.save(model.state_dict(), f"./weight/proj10.pth")
    else:
        os.makedirs('./weight', exist_ok=True)
        torch.save(model.state_dict(), f"./weight/proj10.pth")

    with torch.no_grad():
        model.eval()
        eval_proj(cfg, model, feature_dict, device, _log)




