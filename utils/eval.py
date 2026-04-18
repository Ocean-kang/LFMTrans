import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataloaders.datasets import load_dataset_ade20k, load_dataset_coco, load_train_dataset_pc, load_train_dataset_voc
from utils.load_feature import ensure_dataset_list


SUPPORTED_SEG_DATASETS = {'cocostuff', '150', '847', 'pc59', 'pc459', 'voc20', 'voc20b'}


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask] + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def get_result_metrics(histogram):
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn)
    opc = np.sum(tp) / np.sum(histogram)

    result = {
        'iou': iou,
        'mean_iou': np.nanmean(iou),
        'precision_per_class (per class accuracy)': prc,
        'mean_precision (class-avg accuracy)': np.nanmean(prc),
        'overall_precision (pixel accuracy)': opc,
    }
    return {k: 100 * v for k, v in result.items()}


def _num_categories(dataset):
    if dataset == 'cocostuff':
        return 171
    if dataset == '150':
        return 150
    if dataset == '847':
        return 847
    if dataset == 'pc59':
        return 59
    if dataset == 'pc459':
        return 459
    if dataset == 'voc20':
        return 20
    if dataset == 'voc20b':
        return 21
    raise NotImplementedError(f'Unsupported segmentation dataset: {dataset}')


def _build_eval_loader(cfg, dataset_name):
    if dataset_name == 'cocostuff':
        dataset = load_dataset_coco(cfg, split='val')
    elif dataset_name == '150':
        dataset = load_dataset_ade20k(cfg, set='A150', split='val')
    elif dataset_name == '847':
        dataset = load_dataset_ade20k(cfg, set='A847', split='val')
    elif dataset_name in {'pc59', 'pc459'}:
        dataset = load_train_dataset_pc(cfg, set=dataset_name, split='val')
    elif dataset_name in {'voc20', 'voc20b'}:
        dataset = load_train_dataset_voc(cfg, set=dataset_name, split='val')
    else:
        raise NotImplementedError(f'Unsupported segmentation dataset: {dataset_name}')

    return DataLoader(
        dataset,
        batch_size=cfg.dataset.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def eval_on_dataset(visual_encoder, projector, data_loader, feat_text, dataset, device_img, method, _log):
    num_cat = _num_categories(dataset)
    patch_size = visual_encoder.module.patch_size if hasattr(visual_encoder, 'module') else visual_encoder.patch_size

    visual_encoder.eval()
    projector.eval()

    feat_text = feat_text.float().to(device_img)

    if 'uvlt' in method:
        feat_text_trans = projector(feat_t=feat_text.float().unsqueeze(0).to(device_img))[0]
    else:
        if method == 'talk2dino':
            feat_text_trans = feat_text @ projector.transpose(1, 0)
        elif method == 'LFMTrans':
            feat_text_trans = projector(feat_text)
        else:
            raise FileNotFoundError(f"Method '{method}' not recognized. No corresponding file or mapping found.")

    feat_text_trans_norm = feat_text_trans / feat_text_trans.norm(dim=-1, keepdim=True)
    feat_text_trans_norm = feat_text_trans_norm.to(device_img)
    histogram = np.zeros((num_cat, num_cat))

    for _, sample in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        label_cat = sample['label_cat']
        images = sample['images'].to(device_img)

        N, _, H, W = images.shape
        assert H % patch_size == 0 and W % patch_size == 0
        h, w = H // patch_size, W // patch_size

        if N == 1 and hasattr(visual_encoder, 'module'):
            _, patchtokens = visual_encoder.module(images, augment=False, ret_dense_feat=True)
        else:
            _, patchtokens = visual_encoder(images, augment=False, ret_dense_feat=True)
        patchtokens_norm = patchtokens / patchtokens.norm(dim=-1, keepdim=True)

        pred_masks = patchtokens_norm @ feat_text_trans_norm.transpose(1, 0)
        pred_masks = pred_masks.view(N, h, w, num_cat)
        pred_masks = torch.nn.functional.interpolate(pred_masks.permute(0, 3, 1, 2), (H, W), mode='bilinear')
        pred_masks = pred_masks.permute(0, 2, 3, 1).max(-1)[1]

        if dataset == 'voc20b':
            pred_masks[pred_masks >= 20] = 20

        pred_masks = pred_masks.view(N, -1).cpu().numpy()
        label_cat = label_cat.view(N, -1).cpu().numpy()
        histogram += scores(label_cat, pred_masks, num_cat)

    results = get_result_metrics(histogram)
    _log.info(
        'Epoch: %02d\t- mIoU: translated_text: %.4f\tdataset: %s',
        0,
        results['mean_iou'],
        dataset,
    )
    return results


def eval_on_datasets(cfg, visual_encoder, projector, feature_dict, device_img, _log):
    requested = ensure_dataset_list(getattr(cfg.validation, 'datasets', []))
    if not requested and getattr(cfg.validation, 'dataset', None):
        requested = [cfg.validation.dataset]

    text_key = cfg.validation.text_model
    method = cfg.eval.method
    results = {}

    for dataset in requested:
        if dataset not in SUPPORTED_SEG_DATASETS:
            _log.info('Skip segmentation eval for unsupported dataset: %s', dataset)
            continue
        loader = _build_eval_loader(cfg, dataset)
        results[dataset] = eval_on_dataset(
            visual_encoder,
            projector,
            loader,
            feature_dict[dataset][text_key],
            dataset,
            device_img,
            method,
            _log,
        )

    return results
