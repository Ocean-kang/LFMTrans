import os
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

warnings.filterwarnings('ignore', category=UserWarning)
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
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets_train = [f'{cfg.train.dataset}']
    datasets_eval = ['cocostuff', '150']
    if cfg.train.text_model == 'llama_unmean':
        feature_dict_train = llama_unmean_features(datasets_train)
        feature_dict_eval = llama_features(datasets_eval)
    else:
        feature_dict_train = mpnet_unmean_features(datasets_train)
        feature_dict_eval = mpnet_features(datasets_eval)

    if cfg.fmap.type == 'anchor':
        lfm_anchor = LFMAnchor(cfg=cfg)
        Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals, labels_t, labels_v = lfm_anchor(feature_dict_train, feature_dict_eval, device)
    elif cfg.fmap.type == 'Ip_and_L2':
        lfm_combine = LFMapIpL2Combination(cfg=cfg)
        Cxy, Cyx, v_vecs, t_vecs = lfm_combine(feature_dict_eval, device)
        labels_t = None
    else:
        raise ValueError(f'Unsupported fmap.type: {cfg.fmap.type}')

    Pxy = fmap2pointmap(Cxy, v_vecs.permute(1, 0), t_vecs.permute(1, 0))
    Pyx = fmap2pointmap(Cyx, t_vecs.permute(1, 0), v_vecs.permute(1, 0))

    if labels_t is not None:
        num_classes = int(labels_t.max().item()) + 1
        gt_labels = labels_t
        pred_xy = Pxy % num_classes
        pred_yx = Pyx % num_classes
    else:
        gt_labels = torch.arange(Pxy.shape[0], device=Pxy.device)
        pred_xy = Pxy
        pred_yx = Pyx

    print(accrucy_fn(gt_labels, pred_xy))
    print(accrucy_fn(gt_labels, pred_yx))

    os.makedirs('./train_fig', exist_ok=True)
    plt.imshow(Cxy.abs().detach().cpu())
    plt.colorbar()
    plt.savefig('./train_fig/HeatMap.png')
