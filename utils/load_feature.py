import pickle as pkl
from pathlib import Path
from typing import Iterable, List, Optional

import torch


FEATURE_SPECS = {
    'llama': {
        'text_key': 'llama',
        'vision_key': 'patch',
        'text_template': 'feat_llama3_{dataset}_S2P0_27_28_29_8_12_26_30.pkl',
        'vision_template': 'feat_dinov2_patch_{dataset}_L.pkl',
        'supported_datasets': None,
    },
    'llama_unmean': {
        'text_key': 'llama_unmean',
        'vision_key': 'patch_unmean',
        'text_template': 'feat_llama3_{dataset}_S2P0_27_28_29_8_12_26_30_unmean.pkl',
        'vision_template': 'feat_synonym_dinov2_{dataset}_L.pkl',
        'supported_datasets': {'cocostuff', '150', '847', 'voc20', 'pc59', 'ImageNet-100', 'CIFAR-100'},
    },
    'mpnet': {
        'text_key': 'mpnet',
        'vision_key': 'patch',
        'text_template': 'feat_mpnet_{dataset}_S2P0_27_28_29_8_12_26_30.pkl',
        'vision_template': 'feat_dinov2_patch_{dataset}_L.pkl',
        'supported_datasets': None,
    },
    'mpnet_unmean': {
        'text_key': 'mpnet_unmean',
        'vision_key': 'patch_unmean',
        'text_template': 'feat_mpnet_{dataset}_S2P0_27_28_29_8_12_26_30_unmean.pkl',
        'vision_template': 'feat_synonym_dinov2_{dataset}_L.pkl',
        'supported_datasets': {'ImageNet-100', 'CIFAR-100'},
    },
    'clipb': {
        'text_key': 'clipb',
        'vision_key': 'patch',
        'text_template': 'CLIP/feat_clipb_{dataset}_L.pkl',
        'vision_template': 'feat_dinov2_patch_{dataset}_L.pkl',
        'supported_datasets': None,
    },
    'clipb_unmean': {
        'text_key': 'clipb_unmean',
        'vision_key': 'patch_unmean',
        'text_template': 'CLIP/feat_clipb_{dataset}_L_unmean.pkl',
        'vision_template': 'feat_synonym_dinov2_{dataset}_L.pkl',
        'supported_datasets': None,
    },
}


def pkl_feat_load(path: Path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def pt_feat_load(path: Path):
    return torch.load(path)


def ensure_dataset_list(datasets):
    if datasets is None:
        return []
    if isinstance(datasets, (list, tuple)):
        return list(datasets)
    return [datasets]


def _resolve_feature_root(feature_root: Optional[str] = None) -> Path:
    return Path(feature_root or './feature').expanduser()


def _feature_path(feature_root: Path, template: str, dataset: str) -> Path:
    path = feature_root / template.format(dataset=dataset)
    if not path.is_file():
        raise FileNotFoundError(f'Feature file not found: {path}')
    return path


def _validate_dataset(dataset: str, spec_name: str):
    supported = FEATURE_SPECS[spec_name]['supported_datasets']
    if supported is not None and dataset not in supported:
        raise ValueError(f'{spec_name} features are not available for dataset {dataset}')


def _load_feature_pair(dataset: str, spec_name: str, feature_root: Optional[str] = None):
    _validate_dataset(dataset, spec_name)
    spec = FEATURE_SPECS[spec_name]
    root = _resolve_feature_root(feature_root)

    text_path = _feature_path(root, spec['text_template'], dataset)
    vision_path = _feature_path(root, spec['vision_template'], dataset)

    return {
        spec['text_key']: pkl_feat_load(text_path).cpu().float(),
        spec['vision_key']: pkl_feat_load(vision_path).cpu().float(),
    }


def load_feature_llama(dataset, feature_root: Optional[str] = None):
    return _load_feature_pair(dataset, 'llama', feature_root=feature_root)


def load_feature_llama_unmean(dataset, feature_root: Optional[str] = None):
    return _load_feature_pair(dataset, 'llama_unmean', feature_root=feature_root)


def load_feature_mpnet(dataset, feature_root: Optional[str] = None):
    return _load_feature_pair(dataset, 'mpnet', feature_root=feature_root)


def load_feature_mpnet_unmean(dataset, feature_root: Optional[str] = None):
    return _load_feature_pair(dataset, 'mpnet_unmean', feature_root=feature_root)


def load_feature_clipb(dataset, feature_root: Optional[str] = None):
    return _load_feature_pair(dataset, 'clipb', feature_root=feature_root)


def load_feature_clipb_unmean(dataset, feature_root: Optional[str] = None):
    return _load_feature_pair(dataset, 'clipb_unmean', feature_root=feature_root)


def llama_features(datasets, feature_root: Optional[str] = None):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_llama(dataset, feature_root=feature_root)
    return feat_dict


def llama_unmean_features(datasets, feature_root: Optional[str] = None):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_llama_unmean(dataset, feature_root=feature_root)
    return feat_dict


def mpnet_features(datasets, feature_root: Optional[str] = None):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_mpnet(dataset, feature_root=feature_root)
    return feat_dict


def mpnet_unmean_features(datasets, feature_root: Optional[str] = None):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_mpnet_unmean(dataset, feature_root=feature_root)
    return feat_dict


def clipb_features(datasets, feature_root: Optional[str] = None):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_clipb(dataset, feature_root=feature_root)
    return feat_dict


def clipb_unmean_features(datasets, feature_root: Optional[str] = None):
    feat_dict = {}
    for dataset in datasets:
        feat_dict[f'{dataset}'] = load_feature_clipb_unmean(dataset, feature_root=feature_root)
    return feat_dict


def load_features_by_model(datasets, text_model: str, feature_root: Optional[str] = None):
    datasets = ensure_dataset_list(datasets)
    loader_map = {
        'llama': llama_features,
        'llama_unmean': llama_unmean_features,
        'mpnet': mpnet_features,
        'mpnet_unmean': mpnet_unmean_features,
        'clipb': clipb_features,
        'clipb_unmean': clipb_unmean_features,
    }
    if text_model not in loader_map:
        raise ValueError(f'Unsupported text_model: {text_model}')
    return loader_map[text_model](datasets, feature_root=feature_root)


def maybe_mean_pool_features(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 2:
        return feat
    if feat.ndim == 3:
        return feat.mean(dim=1)
    raise ValueError(f'Expected feature tensor with ndim 2 or 3, got shape {tuple(feat.shape)}')


if __name__ == '__main__':
    datasets = ['cocostuff', '150', '847', 'voc20', 'voc20b', 'pc59', 'ImageNet-100', 'CIFAR-100']
    feature_dict = mpnet_features(datasets)
