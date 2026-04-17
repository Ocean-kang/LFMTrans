import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sacred import Experiment
from easydict import EasyDict as edict

from utils.load_feature_one_photo import build_train_one_photo_features

ex = Experiment('extract_one_photo')
ex.add_config('./configs/LFMTrans_cfg.yaml')

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
    device = torch.device(f'cuda:{cfg.device_gpu}' if torch.cuda.is_available() else 'cpu')

    save_path = build_train_one_photo_features(cfg, device=device, _log=_log)
    print(f'one_photo feature file: {save_path}')
