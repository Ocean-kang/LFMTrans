import random
import logging
import warnings

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict

from utils.fmap_retrieval import deepfmap_retrieval, accrucy_fn, fmap_retrieval_norm, fmap_retrieval
from utils.LatentFunctionMap import LatentFunctionMap
from utils.load_feature import llama_features, mpnet_features, llama_trans_features
from model.fmap_network import RegularizedFMNet

warnings.filterwarnings("ignore", category=UserWarning)
ex = Experiment('LFMTrans')

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
ex.add_config('./configs/LFMTrans_cfg_ZoomOut.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True


@ex.capture
def eval_proj(cfg, feature_dict, device, _log):

    TRAINTYPE = cfg.train.type
    LF_map = LatentFunctionMap(cfg)
    # Load features
    if TRAINTYPE == 'prototype':
        feat_language = feature_dict[f"{cfg.train.dataset}"]["llama_trans"].to(device)
        feat_vision = feature_dict[f"{cfg.train.dataset}"]["patch"].to(device)

        n_cls, dimension = feat_language.shape
        feat_labels_v = torch.arange(n_cls).to(device).float()
        feat_labels_t = torch.arange(n_cls).to(device).float()

        Pxy_zoomout = LF_map.LFM_zoomout(feat_vision, feat_language, 2, device)

        accurcy_Cxy_zoomout = accrucy_fn(feat_labels_v, Pxy_zoomout)

        _log.info(f"Train Finished - prototype - Cxy: {accurcy_Cxy_zoomout:.4f}")
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

    datasets = ["cocostuff"] # ['cocostuff', '150', '847', 'voc20', 'voc20b', 'pc59', 'ImageNet-100', 'CIFAR-100']
    feature_dict = llama_trans_features(datasets)

    eval_proj(cfg, feature_dict, device, _log)