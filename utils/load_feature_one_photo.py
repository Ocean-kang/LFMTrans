import os
import math
import pickle as pkl
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import tqdm

from dataloaders.datasets import (
    load_dataset_ade20k,
    load_dataset_coco,
    load_train_dataset_pc,
    load_train_dataset_voc,
)
from dataloaders.cls_feat_prompts import *  # noqa: F401,F403
from model.DINO_V2 import DINO_V2 as VisualEncoder


IGNORE_LABELS = {255, 459, 65535}


@dataclass
class OnePhotoSelection:
    dataset_name: str
    sampled_indices: List[int]
    class_ids: List[int]
    class_names: List[str]
    per_image_class_ids: Dict[int, List[int]]


def _dataset_dispatch(cfg, dataset_name: str, split: str):
    if dataset_name == 'cocostuff':
        return load_dataset_coco(cfg, split=split)
    if dataset_name == '150':
        return load_dataset_ade20k(cfg, set='A150', split=split)
    if dataset_name == '847':
        return load_dataset_ade20k(cfg, set='A847', split=split)
    if dataset_name in ['pc59', 'pc459']:
        return load_train_dataset_pc(cfg, set=dataset_name, split=split)
    if dataset_name in ['voc20', 'voc20b']:
        return load_train_dataset_voc(cfg, set=dataset_name, split=split)
    raise ValueError(
        f"train_one_photo currently supports semantic segmentation datasets only, got {dataset_name}"
    )


def _valid_class_ids(label_cat: torch.Tensor, name_list: List[str]) -> List[int]:
    unique_ids = torch.unique(label_cat.cpu()).tolist()
    valid = []
    for cid in unique_ids:
        cid = int(cid)
        if cid in IGNORE_LABELS:
            continue
        if 0 <= cid < len(name_list):
            valid.append(cid)
    return sorted(valid)


def _select_random_images_and_classes(cfg, dataset, dataset_name: str) -> OnePhotoSelection:
    one_cfg = cfg.train_one_photo
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(getattr(one_cfg, 'sample_seed', getattr(cfg, 'seed', 0))))
    perm = torch.randperm(len(dataset), generator=generator).tolist()

    target_num_classes = int(getattr(one_cfg, 'target_num_classes', 10))
    num_images = int(getattr(one_cfg, 'num_images', 1))
    expand_until_target = bool(getattr(one_cfg, 'expand_until_target', True))

    sampled_indices: List[int] = []
    class_ids: List[int] = []
    class_id_set = set()
    per_image_class_ids: Dict[int, List[int]] = {}

    for idx in perm:
        sample = dataset[idx]
        valid_ids = _valid_class_ids(sample['label_cat'], dataset.name_list)
        if not valid_ids:
            continue

        sampled_indices.append(idx)
        per_image_class_ids[idx] = valid_ids
        for cid in valid_ids:
            if cid not in class_id_set:
                class_id_set.add(cid)
                class_ids.append(cid)
            if len(class_ids) >= target_num_classes:
                break

        enough_images = len(sampled_indices) >= num_images
        enough_classes = len(class_ids) >= target_num_classes
        if enough_images and (enough_classes or not expand_until_target):
            break
        if enough_classes and expand_until_target:
            break

    if len(class_ids) == 0:
        raise RuntimeError('Failed to find any valid semantic classes from the sampled images.')

    class_ids = class_ids[:target_num_classes]
    class_names = [dataset.name_list[cid] for cid in class_ids]

    if len(class_ids) < target_num_classes:
        raise RuntimeError(
            f"Requested target_num_classes={target_num_classes}, but only found {len(class_ids)} classes "
            f"from {len(sampled_indices)} sampled images. Increase train_one_photo.num_images or enable "
            f"expand_until_target."
        )

    return OnePhotoSelection(
        dataset_name=dataset_name,
        sampled_indices=sampled_indices,
        class_ids=class_ids,
        class_names=class_names,
        per_image_class_ids=per_image_class_ids,
    )


@torch.no_grad()
def _build_visual_bank(cfg, device, dataset, selection: OnePhotoSelection) -> torch.Tensor:
    one_cfg = cfg.train_one_photo
    bank_size = int(getattr(one_cfg, 'bank_size', 100))
    use_prototype = bool(getattr(one_cfg, 'visual_use_prototype', True))

    visual_encoder = VisualEncoder(cfg=cfg.dataset).to(device)
    visual_encoder.eval()

    dim = int(visual_encoder.backnbone_dim)
    class_token_lists: Dict[int, List[torch.Tensor]] = {cid: [] for cid in selection.class_ids}
    patch_size = int(visual_encoder.patch_size)

    for data_idx in tqdm.tqdm(selection.sampled_indices, desc='extract visual bank'):
        sample = dataset[data_idx]
        image = sample['images'].unsqueeze(0).to(device)
        label_cat = sample['label_cat'].unsqueeze(0).to(device)

        _, patchtokens = visual_encoder(image, ret_dense_feat=True)
        _, _, H, W = image.shape
        h, w = H // patch_size, W // patch_size
        patchtokens = patchtokens[0].view(h, w, -1)
        label_small = F.interpolate(label_cat[:, None].float(), size=(h, w), mode='nearest')[0, 0].long()

        for cid in selection.class_ids:
            mask = (label_small == cid)
            if not mask.any():
                continue
            feats = patchtokens[mask].detach().cpu().float()
            if use_prototype:
                feats = feats.mean(dim=0, keepdim=True)
            class_token_lists[cid].append(feats)

    visual_bank = torch.zeros((len(selection.class_ids), bank_size, dim), dtype=torch.float32)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(int(getattr(one_cfg, 'sample_seed', getattr(cfg, 'seed', 0))) + 17)

    for row, cid in enumerate(selection.class_ids):
        if len(class_token_lists[cid]) == 0:
            raise RuntimeError(
                f'Class id {cid} ({dataset.name_list[cid]}) has no visual tokens in the sampled image set.'
            )
        feat = torch.cat(class_token_lists[cid], dim=0).float()
        valid_mask = feat.abs().sum(dim=-1) > 0
        feat = feat[valid_mask]
        if feat.shape[0] == 0:
            raise RuntimeError(
                f'Class id {cid} ({dataset.name_list[cid]}) only produced all-zero visual features.'
            )
        if feat.shape[0] >= bank_size:
            idx = torch.randperm(feat.shape[0], generator=rng)[:bank_size]
        else:
            idx = torch.randint(low=0, high=feat.shape[0], size=(bank_size,), generator=rng)
        visual_bank[row] = feat[idx]

    return visual_bank


def _build_prompt_pool():
    top_k_prompts_0 = [6, 7, 11, 12, 13, 18, 20, 46, 47, 51, 56, 58, 59]
    top_k_prompts_27 = [0, 1, 7, 11, 17, 21, 24, 28, 30, 41, 42, 46, 49, 50, 51, 53, 55, 58, 60, 62]
    top_k_prompts_28 = [1, 10, 12, 20, 26, 38]
    top_k_prompts_29 = [26, 49]
    prompts = [
        [_ for i, _ in enumerate(globals()['prompts_0'][0]) if i in top_k_prompts_0]
        + [_ for i, _ in enumerate(globals()['prompts_27'][0]) if i in top_k_prompts_27]
        + [_ for i, _ in enumerate(globals()['prompts_28'][0]) if i in top_k_prompts_28]
        + [_ for i, _ in enumerate(globals()['prompts_29'][0]) if i in top_k_prompts_29]
    ]
    return prompts[0]


def _run_llama_dialogs(generator, dialogs, temperature: float, top_p: float, max_gen_len: int):
    outputs = generator.chat_completion_feat(
        dialogs,
        max_gen_len=None if max_gen_len == 0 else max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    ret = []
    for out in outputs:
        generation = out['generation']
        ret.append({
            'role': generation['role'],
            'content': generation['content'],
            'corr_feats': generation['corr_feats'],
        })
    return ret


def _build_text_bank(cfg, selection: OnePhotoSelection) -> torch.Tensor:
    from llama import Llama

    one_cfg = cfg.train_one_photo
    bank_size = int(getattr(one_cfg, 'bank_size', 100))
    prompt_ids = list(getattr(one_cfg, 'llama_prompt_ids', [8, 12, 26, 30]))
    system_msg_id = int(getattr(one_cfg, 'llama_system_msg_id', 2))
    temperature = float(getattr(one_cfg, 'llama_temperature', 0.6))
    top_p = float(getattr(one_cfg, 'llama_top_p', 0.9))
    max_gen_len = int(getattr(one_cfg, 'llama_max_gen_len', 0))

    all_prompts = _build_prompt_pool()
    prompts = [all_prompts[i] for i in prompt_ids]
    system_msg = globals()[f'system_msg_{system_msg_id}']

    if len(prompts) == 0:
        raise ValueError('No prompts were selected for llama text bank.')
    if bank_size <= 0:
        raise ValueError(f'bank_size must be positive, got {bank_size}.')

    generator = Llama.build(
        ckpt_dir=one_cfg.llama_ckpt_dir,
        tokenizer_path=one_cfg.llama_tokenizer_path,
        max_seq_len=int(getattr(one_cfg, 'llama_max_seq_len', 2048)),
        max_batch_size=int(getattr(one_cfg, 'llama_max_batch_size', 8)),
        seed=int(getattr(cfg, 'seed', 0)),
        local_rank=0,
        MASTER_PORT=str(getattr(cfg, 'MASTER_PORT', 29500)),
    )

    # 和老代码一致：每一轮只跑一批 prompts
    # 不同的是这里为了凑满 bank_size，会多跑几轮
    rounds = int(math.ceil(bank_size / len(prompts)))

    text_bank = None

    for cls_idx, cls_name in enumerate(tqdm.tqdm(selection.class_names)):
        collected = []

        for _ in range(rounds):
            dialogs = [[] for _ in prompts]

            # 每轮固定按 prompt 顺序构造输入
            for i, prompt in enumerate(prompts):
                dialogs[i].append({
                    'role': 'system',
                    'content': system_msg.replace('[prompt]', prompt),
                })
                dialogs[i].append({
                    'role': 'user',
                    'content': prompt.replace('[cls]', f'[{cls_name}]'),
                })

            # 一次性跑这一轮的全部 prompt
            generations = _run_llama_dialogs(
                generator, dialogs, temperature, top_p, max_gen_len
            )

            # 按返回顺序收集特征，和老代码一样：每个 prompt -> 一个均值特征
            for generation in generations:
                feat = torch.stack(generation['corr_feats'], dim=0).mean(dim=0).cpu().float()
                collected.append(feat)

                if len(collected) >= bank_size:
                    break

            if len(collected) >= bank_size:
                break

        if len(collected) == 0:
            raise RuntimeError(f'No llama features were generated for class {cls_name}.')

        if text_bank is None:
            text_dim = int(collected[0].shape[-1])
            text_bank = torch.zeros(
                (len(selection.class_names), bank_size, text_dim),
                dtype=torch.float32
            )

        # 如果 bank_size 不是 len(prompts) 的整数倍，就循环复制已有结果补齐
        if len(collected) < bank_size:
            base = list(collected)
            ptr = 0
            while len(collected) < bank_size:
                collected.append(base[ptr % len(base)].clone())
                ptr += 1

        text_bank[cls_idx] = torch.stack(collected[:bank_size], dim=0)

    if text_bank is None:
        raise RuntimeError('Failed to build any llama text features.')

    return text_bank


def _default_feature_prefix(cfg, selection: OnePhotoSelection) -> str:
    one_cfg = cfg.train_one_photo
    return (
        f"{selection.dataset_name}_split-{getattr(one_cfg, 'split', 'train')}"
        f"_img-{len(selection.sampled_indices)}"
        f"_cls-{len(selection.class_ids)}"
        f"_bank-{int(getattr(one_cfg, 'bank_size', 100))}"
        f"_seed-{int(getattr(one_cfg, 'sample_seed', getattr(cfg, 'seed', 0)))}"
    )


def _save_feature_payload(cfg, selection: OnePhotoSelection, text_bank: torch.Tensor, visual_bank: torch.Tensor) -> str:
    one_cfg = cfg.train_one_photo
    save_dir = getattr(one_cfg, 'save_dir', './feature_one_photo')
    os.makedirs(save_dir, exist_ok=True)
    prefix = getattr(one_cfg, 'save_name', '').strip() or _default_feature_prefix(cfg, selection)
    save_path = os.path.join(save_dir, f'{prefix}.pkl')
    payload = {
        'dataset_name': selection.dataset_name,
        'sampled_indices': selection.sampled_indices,
        'class_ids': selection.class_ids,
        'class_names': selection.class_names,
        'per_image_class_ids': selection.per_image_class_ids,
        'text_bank': text_bank.cpu().float(),
        'visual_bank': visual_bank.cpu().float(),
    }
    with open(save_path, 'wb') as f:
        pkl.dump(payload, f)
    return save_path


def build_train_one_photo_features(cfg, device, _log=None) -> str:
    dataset_name = str(cfg.train.dataset)
    split = getattr(cfg.train_one_photo, 'split', 'train')
    dataset = _dataset_dispatch(cfg, dataset_name, split)
    selection = _select_random_images_and_classes(cfg, dataset, dataset_name)

    if _log is not None:
        _log.info(
            'build train_one_photo: dataset=%s split=%s sampled_indices=%s class_ids=%s class_names=%s',
            dataset_name,
            split,
            selection.sampled_indices,
            selection.class_ids,
            selection.class_names,
        )

    visual_bank = _build_visual_bank(cfg, device, dataset, selection)
    text_bank = _build_text_bank(cfg, selection)
    save_path = _save_feature_payload(cfg, selection, text_bank, visual_bank)

    if _log is not None:
        _log.info('saved one_photo features to %s', save_path)

    return save_path


def load_train_one_photo_features(cfg, device, _log=None):
    one_cfg = cfg.train_one_photo
    feature_path = getattr(one_cfg, 'feature_path', '')
    if not feature_path:
        raise ValueError('train_one_photo.feature_path is empty. Please run one_photo extraction first.')
    if not os.path.isfile(feature_path):
        raise FileNotFoundError(f'one_photo feature file not found: {feature_path}')

    with open(feature_path, 'rb') as f:
        payload = pkl.load(f)

    dataset_name = str(cfg.train.dataset)
    payload_dataset = str(payload.get('dataset_name', dataset_name))
    if payload_dataset != dataset_name:
        raise ValueError(
            f'train.dataset={dataset_name} but one_photo feature file was built for dataset={payload_dataset}'
        )

    text_bank = payload['text_bank'].float().to(device)
    visual_bank = payload['visual_bank'].float().to(device)

    if _log is not None:
        _log.info(
            'loaded one_photo features from %s: classes=%d text_shape=%s vision_shape=%s',
            feature_path,
            len(payload.get('class_ids', [])),
            tuple(text_bank.shape),
            tuple(visual_bank.shape),
        )

    return {
        dataset_name: {
            cfg.train.text_model: text_bank,
            cfg.train.type: visual_bank,
        }
    }
