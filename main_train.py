import os
import random
import logging
import warnings
import pickle as pkl

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict

from model.Encoder import LinearProj
from loss.proj_loss import proj_loss

warnings.filterwarnings("ignore", category=UserWarning)
ex = Experiment('uvlt')

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

@ex.capture
def train_proj(cfg, model, feature_dict, criterion, device, _log):
    """
    Train the Projector.

    Args:
        model: the LinearProj model
        feature_dict: feature_dict
        device: 'cuda' or 'cpu'
    """
    num_epochs = cfg.model.nums_epoch
    lr = cfg.model.lr_proj

    # Load features
    feat_language = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
    feat_vision = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        x = feat_language.to(device)  # [B, text_dimension]
        y = feat_vision.to(device)  # [B, vision_dimension]

        optimizer.zero_grad()
        output = model(x)  # [B, vision_dimension]
        loss_dict = criterion(output, y)

        # Weight_init
        (W_lap, W_orth, W_bij) = (1.0, 1.0, 0.0)


        loss = loss_dict['l_lap'] * W_lap + loss_dict['l_orth'] * W_orth + loss_dict['l_bij'] * W_bij
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _log.info(f"Epoch {epoch+1}/{num_epochs} - Per Loss: {loss:.4f}")
    _log.info(f"Train Finished - Avg Loss: {total_loss / num_epochs:.4f}")

    return None

# @ex.capture
# def eval_proj(cfg, model, feature_dict, device, _log):
#     with torch.no_grad:


#     return None

@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = cfg.model.DINOv2_BACKBONE_SIZE[0].upper()

    TRAINSETPTH = './feature/train'
    TESTSETPATH = './feature/val'

    feature_dict = {
        "dinov2_features": {
            "dinov2_coco": pkl.load(open(f'./feature/feat_dinov2_patch_cocostuff_{size}.pkl', 'rb')),
            "dinov2_ade150": pkl.load(open(f'./feature/feat_dinov2_patch_150_{size}.pkl', 'rb')),
            "dinov2_ade847": pkl.load(open(f'./feature/feat_dinov2_patch_847_{size}.pkl', 'rb')),
            "dinov2_voc20": pkl.load(open(f'./feature/feat_dinov2_patch_voc20_{size}.pkl', 'rb')),
            "dinov2_voc20b": pkl.load(open(f'./feature/feat_dinov2_patch_voc20b_{size}.pkl', 'rb')),
            "dinov2_pc59": pkl.load(open(f'./feature/feat_dinov2_patch_pc59_{size}.pkl', 'rb')),
            "dinov2_pc459": pkl.load(open(f'./feature/feat_dinov2_patch_pc459_{size}.pkl', 'rb')),
        },
        "llama3_features": {
            "llama3_coco_unmean": pkl.load(open(f'./feature/feat_llama3_cocostuff_S2P0_27_28_29_8_12_26_30_unmean.pkl', 'rb')),
            "llama3_coco": pkl.load(open(f'./feature/feat_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_ade150": pkl.load(open(f'./feature/feat_llama3_150_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_ade847": pkl.load(open(f'./feature/feat_llama3_847_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_voc20": pkl.load(open(f'./feature/feat_llama3_voc20_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_voc20b": pkl.load(open(f'./feature/feat_llama3_voc20b_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_pc59": pkl.load(open(f'./feature/feat_llama3_pc59_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
            "llama3_pc459": pkl.load(open(f'./feature/feat_llama3_pc459_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
        },
        "llama3_synonym_features": {
            "llama3_coco": pkl.load(open(f'./feature/feat_synonyms_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
        },
        "llama3_subpart_features": {
            "llama3_coco": pkl.load(open(f'./feature/feat_subparts_llama3_cocostuff_S2P0_27_28_29_8_12_26_30.pkl', 'rb')),
        },
        "dinov2_subpart_features": {
            "dinov2_coco": pkl.load(open(f'./feature/feat_subpart_dinov2_cocostuff_20k.pkl', 'rb')),
        },
        "dinov2_synonym_features": {
            "dinov2_coco": pkl.load(open(f'./feature/feat_synonym_dinov2_cocostuff.pkl', 'rb')),
        },
        "dinov2_patchtoken_clustering": {
            "dinov2_coco": pkl.load(open(f'./feature/dinov2_cocostuff_patchtoken_clustering_16_{size}.pkl', 'rb'))
        },
        "cinic-10": {
            "train":{
                "dinov2": torch.load(TRAINSETPTH + '/CINIC-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TRAINSETPTH + '/CINIC-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TRAINSETPTH + '/CINIC-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TRAINSETPTH + '/CINIC-10/labels.pt'),
                "prototype": torch.load(TRAINSETPTH + '/CINIC-10/vision/Prototype_dinov2_vit-g14.pt')

            },
            "test":{
                "dinov2": torch.load(TESTSETPATH + '/CINIC-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TESTSETPATH + '/CINIC-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TESTSETPATH + '/CINIC-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TESTSETPATH + '/CINIC-10/labels.pt')
            }
        },
        "cifar-10":{
            "test":{
                "dinov2": torch.load(TESTSETPATH + '/CIFAR-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TESTSETPATH + '/CIFAR-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TESTSETPATH + '/CIFAR-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TESTSETPATH + '/CIFAR-10/labels.pt')
            },
            "train":{
                "dinov2": torch.load(TRAINSETPTH + '/CIFAR-10/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TRAINSETPTH + '/CIFAR-10/language/sentencet_all-mpnet-base-v2.pt'),
                "all-Roberta-large-v1": torch.load(TRAINSETPTH + '/CIFAR-10/language/sentencet_all-roberta-large-v1.pt'),
                "labels": torch.load(TRAINSETPTH + '/CIFAR-10/labels.pt'),
                "prototype": torch.load(TRAINSETPTH + '/CIFAR-10/vision/Prototype_dinov2_vit-g14.pt')
            }
        }
    }
    
    
    model = LinearProj(cfg=cfg)
    criterion = proj_loss(cfg=cfg)

    train_proj(cfg, model, feature_dict, criterion, device, _log)

    if os.path.isdir('./weight'):
        torch.save(model.state_dict(), f"./weight/proj4.pth")
    else:
        os.makedirs('./weight', exist_ok=True)
        torch.save(model.state_dict(), f"./weight/proj.pth")




