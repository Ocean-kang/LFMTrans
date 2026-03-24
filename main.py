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

from model.Encoder import build_text_projector
from utils.load_feature import mpnet_features, llama_features, llama_unmean_features, mpnet_unmean_features
from utils.fmap_util import fmap2pointmap
from utils.fmap_retrieval import accrucy_fn
from src.anchor_supervised import LFMAnchor
from src.l2ipcombinemapping import LFMapIpL2Combination
from src.proj_train import ProjectorFMTrainer

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


def _load_train_features(cfg):
    datasets = [cfg.train.dataset]
    if cfg.train.text_model == 'llama_unmean':
        return llama_unmean_features(datasets)
    return mpnet_unmean_features(datasets)



def _load_eval_features(cfg):
    datasets = [cfg.validation.dataset]
    if cfg.validation.text_model == 'llama':
        return llama_features(datasets)
    return mpnet_features(datasets)



def _load_projector_if_needed(cfg, feature_dict_eval, device):
    ckpt_path = getattr(getattr(cfg, 'projector', None), 'checkpoint_path', '')
    if not ckpt_path:
        return None

    dataset = cfg.validation.dataset
    feat_t = feature_dict_eval[dataset][cfg.validation.text_model]
    feat_v = feature_dict_eval[dataset][cfg.validation.type]
    projector = build_text_projector(cfg, feat_t.shape[-1], feat_v.shape[-1]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    projector.load_state_dict(state_dict)
    projector.eval()
    return projector


@ex.automain
def main(_run, _log):
    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device(f'cuda:{cfg.device_gpu}' if torch.cuda.is_available() else 'cpu')

    if cfg.fmap.type == 'train':
        feature_dict_train = _load_train_features(cfg)
        feature_dict_eval = _load_eval_features(cfg)
        dataset = cfg.train.dataset
        text_dim = feature_dict_train[dataset][cfg.train.text_model].shape[-1]
        vision_dim = feature_dict_train[dataset][cfg.train.type].shape[-1]
        trainer = ProjectorFMTrainer(cfg, device, text_dim=text_dim, vision_dim=vision_dim)
        result = trainer.fit(feature_dict_train, feature_dict_eval)
        print(f"vision->text acc: {result['acc_v_to_t']:.4f}")
        print(f"text->vision acc: {result['acc_t_to_v']:.4f}")
        print(f"checkpoint: {result['checkpoint_path']}")
        Cxy = result['Cxy']
        basis_x = result['x_basis']
        basis_y = result['y_basis']
    elif cfg.fmap.type == 'anchor':
        feature_dict_train = _load_train_features(cfg)
        feature_dict_eval = _load_eval_features(cfg)
        lfm_anchor = LFMAnchor(cfg=cfg)
        Cxy, Cyx, x_basis, _, y_basis, _, labels_t, labels_v = lfm_anchor(feature_dict_train, feature_dict_eval, device)
        pred_y_to_x = fmap2pointmap(Cxy, x_basis.permute(1, 0), y_basis.permute(1, 0))
        pred_x_to_y = fmap2pointmap(Cyx, y_basis.permute(1, 0), x_basis.permute(1, 0))
        num_classes = int(labels_t.max().item()) + 1
        print(accrucy_fn(labels_t, pred_y_to_x % num_classes))
        print(accrucy_fn(labels_t, pred_x_to_y % num_classes))
        basis_x = x_basis
        basis_y = y_basis
    elif cfg.fmap.type == 'Ip_and_L2':
        feature_dict_eval = _load_eval_features(cfg)
        projector = _load_projector_if_needed(cfg, feature_dict_eval, device)
        lfm_combine = LFMapIpL2Combination(cfg=cfg)
        Cxy, Cyx, basis_x, basis_y = lfm_combine(feature_dict_eval, device, projector=projector)
        pred_y_to_x = fmap2pointmap(Cxy, basis_x.permute(1, 0), basis_y.permute(1, 0))
        pred_x_to_y = fmap2pointmap(Cyx, basis_y.permute(1, 0), basis_x.permute(1, 0))
        gt = torch.arange(min(pred_y_to_x.shape[0], pred_x_to_y.shape[0]), device=pred_y_to_x.device)
        print(accrucy_fn(gt, pred_y_to_x[:gt.shape[0]]))
        print(accrucy_fn(gt, pred_x_to_y[:gt.shape[0]]))
    else:
        raise ValueError(f'Unsupported fmap.type: {cfg.fmap.type}')

    os.makedirs('./train_fig', exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(Cxy.abs().detach().cpu())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./train_fig/HeatMap.png')
