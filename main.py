import random
import logging
import warnings

import torch
import torch.backends.cudnn as cudnn

import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

from utils.load_feature import mpnet_features, llama_features, llama_unmean_features, mpnet_unmean_features
from utils.fmap_util import fmap2pointmap
from utils.fmap_retrieval import accrucy_fn

from src.anchor_supervised import LFMAnchor
from src.l2ipcombinemapping import LFMapIpL2Combination

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
ex.add_config('./configs/LFMTrans_cfg.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Features_dict
    datasets_train = [f'{cfg.train.dataset}']
    datasets_eval = ['cocostuff', '150'] # ['cocostuff', '150', '847', 'voc20', 'voc20b', 'pc59', 'ImageNet-100', 'CIFAR-100']
    if cfg.train.text_model == "llama_unmean":
        feature_dict_train = llama_unmean_features(datasets_train)
        feature_dict_eval = llama_features(datasets_eval)
    else:
        feature_dict_train = mpnet_unmean_features(datasets_train)
        feature_dict_eval = mpnet_features(datasets_eval)

    
    if cfg.fmap.type == "anchor":
        LFM_Anchor = LFMAnchor(cfg=cfg)
        Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals, labels_t, labels_v = LFM_Anchor(feature_dict_train, feature_dict_eval, device)

    elif cfg.fmap.type == "Ip_and_L2":
        LFM_Combine = LFMapIpL2Combination(cfg=cfg)
        Cxy, Cyx, v_vecs, t_vecs = LFM_Combine(feature_dict_eval, device)
        labels_t = None
    
    # C -> P
    Pxy = fmap2pointmap(Cxy, v_vecs.permute(1, 0), t_vecs.permute(1, 0))
    Pyx = fmap2pointmap(Cyx, t_vecs.permute(1, 0), v_vecs.permute(1, 0))

    # Evaluation
    if labels_t is not None:
        gt_labels = labels_t # torch.arange(171)
        pred_xy = Pxy % 171
        pred_yx = Pyx % 171
    else:
        gt_labels = torch.arange(171)
        pred_xy = Pxy % 171
        pred_yx = Pyx % 171
    breakpoint()
    tmp1 = accrucy_fn(gt_labels, pred_xy)
    print(tmp1)

    tmp2 = accrucy_fn(gt_labels, pred_yx)
    print(tmp2)

    # Heat Map
    plt.imshow(Cxy.abs().cpu())
    plt.colorbar()
    plt.savefig('./train_fig/HeatMap.png')

    
