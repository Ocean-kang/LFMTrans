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
from utils.load_feature import mpnet_features, llama_features

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
def train(cfg, feature_dict, functionmap, model, criterions, device, _log):
    
    train_dataset = cfg.train.dataset
    train_text_model = cfg.train.text_model
    train_type = cfg.train.type
    num_epochs = cfg.model.nums_epoch_proj
    lr = cfg.model.lr_projector

    # Raw data
    v = feature_dict[f'{train_dataset}'][f'{train_type}']
    t = feature_dict[f'{train_dataset}'][f'{train_text_model}']

    # Load model
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    total_loss = 0.0
    for epoch in range(num_epochs):
        v = v.to(device)  # [B, text_dimension]
        t = t.to(device)  # [B, vision_dimension]

        optimizer.zero_grad()
        feat_v = v # [B, vision_dimension]
        feat_t = model(t)  # [B, vision_dimension]

        # FunctionMap & Permutation
        Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals = functionmap.FunctionMap(feat_v, feat_t, v, t, device)
        Pxy, Pyx = compute_permutation_matrices(cfg, feat_v, feat_t)

        # Add batchsize
        feat_v = feat_v.unsqueeze(0)
        feat_t = feat_t.unsqueeze(0)

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
        _log.info(f"Epoch {epoch+1}/{num_epochs} -Fmap/Align/Ot: {loss_fm:.3f}/{loss_align:.3f}/{loss_ot:.3f} TotalLoss: {loss:.4f}")
    _log.info(f"Train Finished - Avg Loss: {total_loss / num_epochs:.4f}")

    return None

@ex.capture
def eval(cfg, feature_dict, model, device, _log):
    
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

    # Load criterions
    criterion_proj = projector_loss(cfg=cfg)

    # Load Features_dict
    datasets = ['cocostuff'] # ['cocostuff', '150', '847', 'voc20', 'voc20b', 'pc59', 'ImageNet-100', 'CIFAR-100']
    if cfg.train.text_model == "llama":
        feature_dict = llama_features(datasets)
    else:
        feature_dict = mpnet_features(datasets)
    train(cfg, feature_dict, deepfuctionmap, model_proj, criterion_proj, device, _log)

    with torch.no_grad:
        eval()




