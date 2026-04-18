from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_plain(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def resolve_experiment_dir(_run, default_root: str = 'train_log') -> str:
    """Resolve a Sacred run directory.

    If Sacred file storage is enabled (e.g. ``python main.py -F train_log/exp``),
    outputs are written under ``<file_storage>/<run_id>``. Otherwise a stable
    fallback directory is created under ``train_log/public``.
    """
    storage_root = None
    meta_info = getattr(_run, 'meta_info', {}) or {}
    options = meta_info.get('options', {}) or {}
    storage_root = options.get('--file_storage')

    if storage_root and getattr(_run, '_id', None):
        exp_dir = Path(storage_root).expanduser() / str(_run._id)
    else:
        exp_dir = Path(default_root).expanduser() / 'public'

    exp_dir.mkdir(parents=True, exist_ok=True)
    return str(exp_dir)


def resolve_dir_under_exp(exp_dir: str, value: str | None, default_relative: str) -> str:
    raw_value = (value or '').strip()
    if not raw_value:
        raw_value = default_relative

    path = Path(raw_value).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        resolved = Path(exp_dir) / raw_value.lstrip('./')

    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def resolve_file_path(value: str | None, base_dir: str | None = None) -> str:
    raw_value = (value or '').strip()
    if not raw_value:
        return ''

    path = Path(raw_value).expanduser()
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((Path(base_dir) / raw_value).resolve())


def apply_runtime_paths(cfg, exp_dir: str):
    cfg.runtime = getattr(cfg, 'runtime', {})
    cfg.runtime['exp_dir'] = exp_dir

    if getattr(cfg, 'paths', None) is not None:
        feature_root = getattr(cfg.paths, 'feature_root', './feature')
        cfg.paths.feature_root = str(Path(feature_root).expanduser())

    if getattr(cfg, 'projector_train', None) is not None:
        cfg.projector_train.ckpt_dir = resolve_dir_under_exp(
            exp_dir,
            getattr(cfg.projector_train, 'ckpt_dir', ''),
            'checkpoints',
        )

    if (getattr(cfg, 'train_one_photo', None) is not None) and (getattr(cfg, 'projector_train', None) is None):
        cfg.train_one_photo.save_dir = resolve_dir_under_exp(
            exp_dir,
            getattr(cfg.train_one_photo, 'save_dir', ''),
            'features/one_photo',
        )
        feature_path = getattr(cfg.train_one_photo, 'feature_path', '')
        if feature_path:
            cfg.train_one_photo.feature_path = resolve_file_path(
                feature_path,
                base_dir=cfg.train_one_photo.save_dir,
            )
    return cfg
