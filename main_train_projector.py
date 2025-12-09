import random
import logging
import warnings

import torch
import torch.backends.cudnn as cudnn

import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict

from model.Encoder import LinearProjText

from utils.LatentFunctionMap import DeepFunctionMap
from utils.permutation_compute import compute_permutation_matrices
from utils.load_feature import mpnet_features, llama_features, llama_unmean_features, mpnet_unmean_features
from utils.misc import cos_sim_retrieval, structure_retrieval
from utils.fmap_retrieval import fmap_retrieval, accrucy_fn
from utils.pairs_utils import get_retrieval_matrix

from loss.proj_loss import projector_loss

warnings.filterwarnings("ignore", category=UserWarning)
ex = Experiment('LFMtrans')

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
ex.add_config('./configs/LFMTrans_cfg_projector.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

@ex.capture
def train(cfg, feature_dict, functionmap, model, criterions, epoch, device, _log):
    
    train_dataset = cfg.train.dataset
    train_text_model = cfg.train.text_model
    train_type = cfg.train.type
    num_epochs_iter = cfg.model.nums_epoch_iters
    lr = cfg.model.lr_projector

    # Raw data
    v_unmean = feature_dict[f'{train_dataset}'][f'{train_type}'].float()
    t_unmean = feature_dict[f'{train_dataset}'][f'{train_text_model}_unmean'].float()

    _, NUMS_v, D_v = v_unmean.shape
    _, NUMS_t, D_t = t_unmean.shape
    indices_v = torch.randperm(NUMS_v)[:cfg.train.sample]
    indices_t = torch.randperm(NUMS_t)[:cfg.train.sample]

    v = v_unmean[:, indices_v, :]
    t = t_unmean[:, indices_t, :]
    v = v.reshape(-1, D_v)
    t = t.reshape(-1, D_t)

    # Load model
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    total_loss = 0.0
    for epoch_iter in range(num_epochs_iter):
        v = v.to(device)  # [B, text_dimension]
        t = t.to(device)  # [B, vision_dimension]

        optimizer.zero_grad()
        feat_v = v # [N_cls, vision_dimension]
        feat_t = model(t)  # [N_cls, vision_dimension]

        # FunctionMap & Permutation
        Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals = functionmap.FunctionMap(feat_v, feat_t, v, t, device)
        # Add batchsize
        feat_v = feat_v.unsqueeze(0)
        feat_t = feat_t.unsqueeze(0)
        Pxy, Pyx = compute_permutation_matrices(cfg, feat_v, feat_t)

        # Loss
        loss_dict = criterions(feat_v, feat_t, Cxy, Cyx, t_vals, v_vals, t_vecs, v_vecs, Pxy, Pyx)
        # Weight_init
        (W_lap, W_orth, W_bij, W_align, W_ot) = (cfg.loss.w_lap, cfg.loss.w_orth, cfg.loss.w_bij, cfg.loss.w_align, 1.0)

        loss_fm = loss_dict['l_lap'] * W_lap + loss_dict['l_orth'] * W_orth + loss_dict['l_bij'] * W_bij
        loss_align = loss_dict['l_align'] * W_align
        loss_ot = loss_dict['l_ot'] * W_ot
        loss = loss_fm + loss_align + loss_ot
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _log.info(f"Epoch - {epoch} - Epoch_iter {epoch_iter+1}/{num_epochs_iter} -Fmap/Align/Ot: {loss_fm:.3f}/{loss_align:.3f}/{loss_ot:.3f} TotalLoss: {loss:.4f}")
    _log.info(f"Train Finished - Epoch - {epoch} - Avg Loss: {total_loss / num_epochs_iter:.4f}")

    return None

@ex.capture
def eval(cfg, feature_dict, model, epoch, device, _log):

    train_text_model = cfg.train.text_model
    train_type = 'patch'
    # Load Patch dataset
    v_coco = feature_dict[f'cocostuff'][f'{train_type}'].to(device)
    t_coco = feature_dict[f'cocostuff'][f'{train_text_model}'].to(device)

    v_a150 = feature_dict[f'150'][f'{train_type}'].to(device)
    t_a150 = feature_dict[f'150'][f'{train_text_model}'].to(device)

    # v_a847 = feature_dict[f'847'][f'{train_type}'].to(device)
    # t_a847 = feature_dict[f'847'][f'{train_text_model}'].to(device)

    # v_pc59 = feature_dict[f'pc59'][f'{train_type}'].to(device)
    # t_pc59 = feature_dict[f'pc59'][f'{train_text_model}'].to(device)

    # v_voc20 = feature_dict[f'voc20'][f'{train_type}'].to(device)
    # t_voc20 = feature_dict[f'voc20'][f'{train_text_model}'].to(device)

    # v_voc20b = feature_dict[f'voc20b'][f'{train_type}'].to(device)
    # t_voc20b = feature_dict[f'voc20b'][f'{train_text_model}'].to(device)

    # v_ImageNet100 = feature_dict[f'ImageNet-100'][f'{train_type}'].to(device)
    # t_ImageNet100 = feature_dict[f'ImageNet-100'][f'{train_text_model}'].to(device)

    # v_CIFAR100 = feature_dict[f'CIFAR-100'][f'{train_type}'].to(device)
    # t_CIFAR100 = feature_dict[f'CIFAR-100'][f'{train_text_model}'].to(device)

    # Project to Dinov2 space
    t_coco_proj = model(t_coco)
    t_a150_proj = model(t_a150)
    # t_a847_proj = model(t_a847)
    # t_pc59_proj = model(t_pc59)
    # t_voc20_proj = model(t_voc20)
    # t_voc20b_proj = model(t_voc20b)
    # t_ImageNet100_proj = model(t_ImageNet100)
    # t_CIFAR100_proj = model(t_CIFAR100)

    # CSR_Retrieval
    csr_coco = cos_sim_retrieval(t_coco_proj.cpu(), v_coco.cpu())
    csr_a150 = cos_sim_retrieval(t_a150_proj.cpu(), v_a150.cpu())
    # csr_a847 = cos_sim_retrieval(t_a847_proj.cpu(), v_a847.cpu())
    # csr_pc59 = cos_sim_retrieval(t_pc59_proj.cpu(), v_pc59.cpu())
    # csr_voc20 = cos_sim_retrieval(t_voc20_proj.cpu(), v_voc20.cpu())
    # csr_voc20b = cos_sim_retrieval(t_voc20b_proj.cpu(), v_voc20b.cpu())
    # csr_ImageNet100 = cos_sim_retrieval(t_ImageNet100_proj.cpu(), v_ImageNet100.cpu())
    # csr_CIFAR100 = cos_sim_retrieval(t_CIFAR100_proj.cpu(), v_CIFAR100.cpu())

    # STR_Retrieval
    str_coco = structure_retrieval(t_coco_proj.cpu(), v_coco.cpu())
    str_a150 = structure_retrieval(t_a150_proj.cpu(), v_a150.cpu())
    # str_a847 = structure_retrieval(t_a847_proj.cpu(), v_a847.cpu())
    # str_pc59 = structure_retrieval(t_pc59_proj.cpu(), v_pc59.cpu())
    # str_voc20 = structure_retrieval(t_voc20_proj.cpu(), v_voc20.cpu())
    # str_voc20b = structure_retrieval(t_voc20b_proj.cpu(), v_voc20b.cpu())
    # str_ImageNet100 = structure_retrieval(t_ImageNet100_proj.cpu(), v_ImageNet100.cpu())
    # str_CIFAR100 = structure_retrieval(t_CIFAR100_proj.cpu(), v_CIFAR100.cpu())

    # TODO: FunctionMap L2 retrieval
    deepfuctionmap = DeepFunctionMap(cfg=cfg)
    Cxy_coco, Cyx_coco, v_vecs_coco, v_vals_coco, t_vecs_coco, t_vals_coco = deepfuctionmap.FunctionMap(v_coco, t_coco_proj, v_coco, t_coco, device)
    Cxy_a150, Cyx_a150, v_vecs_a150, v_vals_a150, t_vecs_a150, t_vals_a150 = deepfuctionmap.FunctionMap(v_a150, t_a150_proj, v_a150, t_a150, device)
    # Cxy_a847, Cyx_a847, v_vecs_a847, v_vals_a847, t_vecs_a847, t_vals_a847 = deepfuctionmap.FunctionMap(v_a847, t_a847_proj, v_a847, t_a847, device)
    # COCO
    P_coco = fmap_retrieval(cfg, Cyx_coco.squeeze(0), t_vecs_coco.squeeze(0), v_vecs_coco.squeeze(0), cfg.fm_retrieval.metric)
    fmr_coco = accrucy_fn(torch.arange(P_coco.shape[0]), P_coco.cpu())
    # a150
    P_a150 = fmap_retrieval(cfg, Cyx_a150.squeeze(0), t_vecs_a150.squeeze(0), v_vecs_a150.squeeze(0), cfg.fm_retrieval.metric)
    fmr_a150 = accrucy_fn(torch.arange(P_a150.shape[0]), P_a150.cpu())

    # _log.info(f"[coco/150/847/pc59/voc20/voc20b/ImageNet-100/CIFAR-100]    str:/{str_coco:.4f}/{str_a150:.4f}/{str_a847:.4f}/{str_pc59:.4f}/{str_voc20:.4f}/{str_voc20b:.4f}/{str_ImageNet100:.4f}/{str_CIFAR100:.4f}")
    # _log.info(f"[coco/150/847/pc59/voc20/voc20b/ImageNet-100/CIFAR-100]    csr:/{csr_coco:.4f}/{csr_a150:.4f}/{csr_a847:.4f}/{csr_pc59:.4f}/{csr_voc20:.4f}/{csr_voc20b:.4f}/{csr_ImageNet100:.4f}/{csr_CIFAR100:.4f}")
    # _log.info(f"[coco/150/847/pc59/voc20/voc20b/ImageNet-100/CIFAR-100]    fmr:/{fmr_coco:.4f}/{fmr_a150:.4f}/{fmr_a847:.4f}/{fmr_pc59:.4f}/{fmr_voc20:.4f}/{fmr_voc20b:.4f}/{fmr_ImageNet100:.4f}/{fmr_CIFAR100:.4f}")

    _log.info(f"Epoch - {epoch} - [coco/150]    str:/{str_coco:.4f}/{str_a150:.4f}")
    _log.info(f"Epoch - {epoch} - [coco/150]    csr:/{csr_coco:.4f}/{csr_a150:.4f}")
    _log.info(f"Epoch - {epoch} - [coco/150]    fmr:/{fmr_coco:.4f}/{fmr_a150:.4f}")

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

    # Load Models
    deepfuctionmap = DeepFunctionMap(cfg=cfg)
    model_proj = LinearProjText(cfg=cfg)
    model_proj.to(device)

    # Load criterions
    criterion_proj = projector_loss(cfg=cfg)

    # Load Features_dict
    datasets_train = ['cocostuff']
    datasets_eval = ['cocostuff', '150'] # ['cocostuff', '150', '847', 'voc20', 'voc20b', 'pc59', 'ImageNet-100', 'CIFAR-100']
    if cfg.train.text_model == "llama":
        feature_dict_train = llama_unmean_features(datasets_train)
        feature_dict_eval = llama_features(datasets_eval)
    else:
        feature_dict_train = mpnet_unmean_features(datasets_train)
        feature_dict_eval = mpnet_features(datasets_eval)
    
    # Training
    for epoch in range(cfg.model.nums_epoch_proj):

        with torch.no_grad():
            model_proj.eval()
            eval(cfg, feature_dict_eval, model_proj, epoch, device, _log)

        train(cfg, feature_dict_train, deepfuctionmap, model_proj, criterion_proj, epoch, device, _log)
    
    torch.save(model_proj.state_dict(), f'./weight/proj_weight_{cfg.save}.pt')



