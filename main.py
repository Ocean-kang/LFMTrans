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

from model.DINO_V2 import DINO_V2 as VisualEncoder
from model.Encoder import build_text_projector
from utils.load_feature import load_features_by_model, ensure_dataset_list
from utils.load_feature_one_photo import load_train_one_photo_features
from src.anchor_supervised import LFMAnchor
from src.l2ipcombinemapping import LFMapIpL2Combination
from src.proj_train import ProjectorFMTrainer, ProjectorFMEvaluator
from utils.eval import eval_on_datasets

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



def _load_train_one_photo_features(cfg, device, _log):
    return load_train_one_photo_features(cfg, device=device, _log=_log)



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
    cfg.train.dataset=str(cfg.train.dataset)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device(f'cuda:{cfg.device_gpu}' if torch.cuda.is_available() else 'cpu')
    exp_ckpt_dir = os.path.join(
        _run.meta_info['options']['--file_storage'], str(_run._id)
    ) if _run._id else os.path.join('train_proj_init', 'public')
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    
    if not cfg.eval.enabled:
        if cfg.fmap.type in ['train', 'train_one_photo']:
            if cfg.fmap.type == 'train':
                feature_dict_train = _load_train_features(cfg)
            else:
                feature_dict_train = _load_train_one_photo_features(cfg, device=device, _log=_log)
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

            if result.get('final_cluster_metrics') is not None:
                cm = result['final_cluster_metrics']
                print(
                    f"last cluster eval ({dataset}): "
                    f"cluster->gt={cm['cluster_gt_accuracy']:.4f} "
                    f"cluster->pred={cm['cluster_pred_accuracy']:.4f} "
                    f"pred->gt={cm['pred_gt_accuracy']:.4f}"
                )

            _print_eval_results('final multi-dataset eval:', result['final_eval_results'])

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
        else:
            raise ValueError(f'Unsupported fmap.type: {cfg.fmap.type}')
    else:
        # Eval
        # 1. 加载Projector
        projector = build_text_projector(cfg, input_dim=cfg.eval.input_dim, output_dim=cfg.eval.output_dim)
        state_dict = torch.load(f"{cfg.eval.checkpoint_path}", map_location='cpu')
        projector.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
        projector.eval()
        projector.to(device)
        # 2.加载DINOv2
        visual_encoder = VisualEncoder(cfg=cfg.eval)
        visual_encoder.eval()
        visual_encoder.to(device)
        # 3. evaluation
        feature_dict_eval = _load_eval_features(cfg)
        eval_on_datasets(cfg, visual_encoder, projector, feature_dict_eval, device, _log)

