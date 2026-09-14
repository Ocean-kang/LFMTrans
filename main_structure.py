import logging
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from sacred import Experiment

from src.proj_train_structure import ProjectorFMStructureTrainer
from utils.experiment import apply_runtime_paths, resolve_experiment_dir
from utils.load_feature import ensure_dataset_list, load_features_by_model
from utils.load_feature_one_photo import load_train_one_photo_features

warnings.filterwarnings('ignore', category=UserWarning)

ex = Experiment('LFMtrans_structure')


def create_basic_stream_logger(fmt):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


ex.logger = create_basic_stream_logger('%(levelname)s - %(name)s - %(message)s')
ex.add_config('./configs/LFMTrans_cfg.yaml')
ex.add_config('./configs/LFMTrans_structure.yaml')

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


def _feature_root(cfg):
    return getattr(getattr(cfg, 'paths', None), 'feature_root', './feature')


def _load_train_features(cfg):
    return load_features_by_model(
        [cfg.train.dataset],
        cfg.train.text_model,
        feature_root=_feature_root(cfg),
    )


def _load_eval_features(cfg):
    return load_features_by_model(
        _validation_datasets(cfg),
        cfg.validation.text_model,
        feature_root=_feature_root(cfg),
    )


def _print_eval_results(results):
    print('final multi-dataset eval:')
    for dataset, metrics in results.items():
        print(
            f"  [{dataset}] "
            f"vision->text={metrics['acc_v_to_t']:.4f} "
            f"text->vision={metrics['acc_t_to_v']:.4f}"
        )


@ex.automain
def main(_run, _log):
    cfg = edict(_run.config)
    cfg.train.dataset = str(cfg.train.dataset)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = resolve_experiment_dir(
        _run,
        default_root=getattr(getattr(cfg, 'paths', {}), 'log_root', 'train_log'),
    )
    cfg = apply_runtime_paths(cfg, exp_dir)
    _log.info('experiment_dir=%s', exp_dir)

    if cfg.eval.enabled:
        raise ValueError('main_structure.py is a training-only experiment entry.')
    if cfg.fmap.type not in ['train', 'train_one_photo']:
        raise ValueError('main_structure.py supports fmap.type=train or train_one_photo only.')

    if cfg.fmap.type == 'train':
        feature_dict_train = _load_train_features(cfg)
    else:
        feature_dict_train = load_train_one_photo_features(cfg, device=device, _log=_log)

    feature_dict_eval = _load_eval_features(cfg)
    dataset = cfg.train.dataset

    text_dim = feature_dict_train[dataset][cfg.train.text_model].shape[-1]
    vision_dim = feature_dict_train[dataset][cfg.train.type].shape[-1]

    trainer = ProjectorFMStructureTrainer(
        cfg,
        device,
        text_dim=text_dim,
        vision_dim=vision_dim,
    )
    result = trainer.fit(
        feature_dict_train,
        feature_dict_eval,
        final_eval_datasets=_validation_datasets(cfg),
    )

    print(f"checkpoint: {result['checkpoint_path']}")

    if result.get('train_eval_history'):
        last = result['train_eval_history'][-1]
        print(
            f"last train-set eval ({dataset}): "
            f"vision->text={last['acc_v_to_t']:.4f} "
            f"text->vision={last['acc_t_to_v']:.4f}"
        )

    if result.get('final_cluster_metrics') is not None:
        cm = result['final_cluster_metrics']
        print(
            f"last cluster eval ({dataset}): "
            f"cluster->gt={cm['cluster_gt_accuracy']:.4f} "
            f"cluster->pred={cm['cluster_pred_accuracy']:.4f} "
            f"pred->gt={cm['pred_gt_accuracy']:.4f}"
        )

    _print_eval_results(result['final_eval_results'])
