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
from utils.load_feature import load_features_by_model, ensure_dataset_list
from src.anchor_supervised import LFMAnchor
from src.l2ipcombinemapping import LFMapIpL2Combination
from src.proj_train import ProjectorFMTrainer, ProjectorFMEvaluator

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


def _validation_datasets(cfg):
    datasets = ensure_dataset_list(getattr(cfg.validation, 'datasets', []))
    if datasets:
        return datasets
    if getattr(cfg.validation, 'dataset', None):
        return [cfg.validation.dataset]
    raise ValueError('validation.datasets is empty and validation.dataset is not set')


def _load_train_features(cfg):
    return load_features_by_model([cfg.train.dataset], cfg.train.text_model)



def _load_eval_features(cfg):
    return load_features_by_model(_validation_datasets(cfg), cfg.validation.text_model)



def _build_projector_from_feature_dict(cfg, feature_dict_eval, device, dataset):
    feat_t = feature_dict_eval[dataset][cfg.validation.text_model]
    feat_v = feature_dict_eval[dataset][cfg.validation.type]
    feat_t_dim = feat_t.shape[-1]
    feat_v_dim = feat_v.shape[-1]
    return build_text_projector(cfg, feat_t_dim, feat_v_dim).to(device)



def _load_projector_if_needed(cfg, feature_dict_eval, device):
    ckpt_path = getattr(getattr(cfg, 'projector', None), 'checkpoint_path', '')
    if not ckpt_path:
        return None
    dataset = _validation_datasets(cfg)[0]
    projector = _build_projector_from_feature_dict(cfg, feature_dict_eval, device, dataset)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    projector.load_state_dict(state_dict)
    projector.eval()
    return projector



def _print_eval_results(title, results):
    print(title)
    for dataset, metrics in results.items():
        print(
            f"  [{dataset}] "
            f"vision->text={metrics['acc_v_to_t']:.4f} "
            f"text->vision={metrics['acc_t_to_v']:.4f}"
        )


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
        result = trainer.fit(feature_dict_train, feature_dict_eval, final_eval_datasets=_validation_datasets(cfg))
        print(f"checkpoint: {result['checkpoint_path']}")
        if result.get('train_eval_history'):
            last_train_eval = result['train_eval_history'][-1]
            print(
                f"last train-set eval ({dataset}): "
                f"vision->text={last_train_eval['acc_v_to_t']:.4f} "
                f"text->vision={last_train_eval['acc_t_to_v']:.4f}"
            )
        _print_eval_results('final multi-dataset eval:', result['final_eval_results'])
        Cxy = result['Cxy']
    elif cfg.fmap.type == 'anchor':
        feature_dict_train = _load_train_features(cfg)
        feature_dict_eval = _load_eval_features(cfg)
        lfm_anchor = LFMAnchor(cfg=cfg)
        Cxy, Cyx, x_basis, _, y_basis, _, labels_t, labels_v = lfm_anchor(feature_dict_train, feature_dict_eval, device)
        print('anchor mode finished')
    elif cfg.fmap.type == 'Ip_and_L2':
        feature_dict_eval = _load_eval_features(cfg)
        projector = _load_projector_if_needed(cfg, feature_dict_eval, device)
        fm_helper = LFMapIpL2Combination(cfg=cfg)
        evaluator = ProjectorFMEvaluator(cfg, device, fm_helper)
        results = evaluator.evaluate_feature_dict(
            feature_dict_eval=feature_dict_eval,
            projector=projector,
            text_key=cfg.validation.text_model,
            vision_key=cfg.validation.type,
            datasets=_validation_datasets(cfg),
        )
        _print_eval_results('multi-dataset eval:', results)
        first_dataset = _validation_datasets(cfg)[0]
        Cxy = results[first_dataset]['Cxy']
    else:
        raise ValueError(f'Unsupported fmap.type: {cfg.fmap.type}')

    os.makedirs('./train_fig', exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(Cxy.abs().detach().cpu())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./train_fig/HeatMap.png')
