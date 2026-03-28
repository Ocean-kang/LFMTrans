import pickle as pkl
from pathlib import Path
from typing import Iterable, List

import torch


def pkl_feat_load(PATH: Path):
    with open(PATH, 'rb') as f:
        ret = pkl.load(f)
    return ret

def pt_feat_load(PATH: Path):
    return torch.load(PATH)

def load_feature_llama(dataset):

    ret_dict ={}
    llama_path = Path(f'./feature/feat_llama3_{dataset}_S2P0_27_28_29_8_12_26_30.pkl')
    dinov2_patchtoken_path = Path(f'./feature/feat_dinov2_patch_{dataset}_L.pkl')
    if dataset in ['voc20b']:
        dinov2_patchtoken_path = Path(f'./feature/feat_dinov2_patch_{dataset}_1_L.pkl')
    ret_dict['llama'] = pkl_feat_load(llama_path).cpu().float()
    ret_dict['patch'] = pkl_feat_load(dinov2_patchtoken_path).cpu().float()
    return ret_dict

def load_feature_llama_unmean(dataset):

    ret_dict ={}
    llama_unmean_path = Path(f'./feature/feat_llama3_{dataset}_S2P0_27_28_29_8_12_26_30_unmean.pkl')
    dinov2_patchtoken_unmean_path = Path(f'./feature/feat_synonym_dinov2_{dataset}_L.pkl')
    if dataset in ['cocostuff', '150', '847', 'voc20', 'pc59']:
        ret_dict['llama_unmean'] = pkl_feat_load(llama_unmean_path).cpu().float()
        ret_dict['patch_unmean'] = pkl_feat_load(dinov2_patchtoken_unmean_path).cpu().float()
    return ret_dict

def load_feature_llama_trans(dataset):

    ret_dict ={}
    llama_path = Path(f'./itsamatch/feat_llama3_{dataset}_S2P0_27_28_29_8_12_26_30_trans.pkl')
    dinov2_patchtoken_path = Path(f'./feature/feat_dinov2_patch_{dataset}_L.pkl')
    if dataset in ['voc20b']:
        dinov2_patchtoken_path = Path(f'./feature/feat_dinov2_patch_{dataset}_1_L.pkl')

    ret_dict['llama_trans'] = pkl_feat_load(llama_path).squeeze(0).cpu().float()
    ret_dict['patch'] = pkl_feat_load(dinov2_patchtoken_path).cpu().float()
    return ret_dict


def load_feature_mpnet(dataset):

    ret_dict ={}
    mpnet_path = Path(f'./itsamatch_more/feat_mpnet_{dataset}_S2P0_27_28_29_8_12_26_30.pkl')
    dinov2_patchtoken_path = Path(f'./itsamatch_more/feat_dinov2_patch_{dataset}_L.pkl')        
    ret_dict['mpnet'] = pkl_feat_load(mpnet_path).cpu().float()
    ret_dict['patch'] = pkl_feat_load(dinov2_patchtoken_path).cpu().float()
    return ret_dict

def load_feature_mpnet_unmean(dataset):
    ret_dict ={}
    if dataset in ['ImageNet-100', 'CIFAR-100']:
        mpnet_path_unmean = Path(f'./itsamatch_more/feat_mpnet_{dataset}_S2P0_27_28_29_8_12_26_30_unmean.pkl')
        dinov2_patchtoken_unmean_path = Path(f'./itsamatch_more/feat_synonym_dinov2_{dataset}_L.pkl')
        ret_dict['mpnet_unmean'] = pkl_feat_load(mpnet_path_unmean).cpu().float()
        ret_dict['patch_unmean'] = pkl_feat_load(dinov2_patchtoken_unmean_path).cpu().float()
    return ret_dict

def llama_features(datasets):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_llama(dataset)
    return feat_dict

def llama_unmean_features(datasets):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_llama_unmean(dataset)
    return feat_dict

def llama_trans_features(datasets):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_llama_trans(dataset)
    return feat_dict

def mpnet_features(datasets):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_mpnet(dataset)
    return feat_dict

def mpnet_unmean_features(datasets):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_mpnet_unmean(dataset)
    return feat_dict

if __name__ == '__main__':
    datasets = ['cocostuff', '150', '847', 'voc20', 'voc20b', 'pc59', 'ImageNet-100', 'CIFAR-100']
    feature_dict = mpnet_features(datasets) # llama_trans_features(datasets)


def ensure_dataset_list(datasets):
    if datasets is None:
        return []
    if isinstance(datasets, (list, tuple)):
        return list(datasets)
    return [datasets]


def load_features_by_model(datasets, text_model: str):
    datasets = ensure_dataset_list(datasets)
    loader_map = {
        'llama': llama_features,
        'llama_unmean': llama_unmean_features,
        'llama_trans': llama_trans_features,
        'mpnet': mpnet_features,
        'mpnet_unmean': mpnet_unmean_features,
    }
    if text_model not in loader_map:
        raise ValueError(f'Unsupported text_model: {text_model}')
    return loader_map[text_model](datasets)


def maybe_mean_pool_features(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 2:
        return feat
    if feat.ndim == 3:
        return feat.mean(dim=1)
    raise ValueError(f'Expected feature tensor with ndim 2 or 3, got shape {tuple(feat.shape)}')
