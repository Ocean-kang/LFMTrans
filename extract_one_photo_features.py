import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sacred import Experiment, SETTINGS
from easydict import EasyDict as edict

from utils.experiment import apply_runtime_paths, resolve_experiment_dir
from utils.load_feature_one_photo import build_train_one_photo_features

SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment('extract_one_photo')
ex.add_config('./configs/extract_one_photo_cfg.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

@ex.automain
def main(_run, _log):
    cfg = edict(_run.config)
    cfg.train.dataset = str(cfg.train.dataset)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = resolve_experiment_dir(
        _run,
        default_root=getattr(getattr(cfg, 'paths', {}), 'log_root', 'train_log'),
    )
    cfg = apply_runtime_paths(cfg, exp_dir)
    _log.info('experiment_dir=%s', exp_dir)

    save_path = build_train_one_photo_features(cfg, device=device, _log=_log)
