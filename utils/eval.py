import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataloaders.datasets import load_dataset_ade20k, load_dataset_coco, load_train_dataset_pc, load_train_dataset_voc

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)  # Exclude unlabelled data.
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

    result = {"iou": iou,
             "mean_iou": np.nanmean(iou),
             "precision_per_class (per class accuracy)": prc,
             "mean_precision (class-avg accuracy)": np.nanmean(prc),
             "overall_precision (pixel accuracy)": opc}

    result = {k: 100*v for k, v in result.items()}

    return result

def eval_on_dataset(visual_encoder, projector, data_loader, feat_text, dataset, device_img, method, _log):

    if dataset == 'cocostuff':
        num_cat = 171
    elif '150' in dataset:
        num_cat = 150
    elif '847' in dataset:
        num_cat = 847
    elif 'pc' in dataset:
        num_cat = 459 if '459' in dataset else 59
    elif 'voc' in dataset:
        num_cat = 59 if '20b' in dataset else 20
    else:
        raise NotImplementedError
    patch_size = visual_encoder.module.patch_size if hasattr(visual_encoder, 'module') else visual_encoder.patch_size

    visual_encoder.eval()
    projector.eval()

    feat_text = feat_text.float().to(device_img)

    if 'uvlt' in method:
        feat_text_trans = projector(feat_t=feat_text.float().unsqueeze(0).to(device_img))[0]
    else:
        if method == "talk2dino":
            feat_text_trans = feat_text @ projector.transpose(1, 0)
        elif method == 'LFMTrans':
            feat_text_trans = projector(feat_text)
        else:
            raise FileNotFoundError(f"Method '{method}' not recognized. No corresponding file or mapping found.")

    feat_text_trans_norm = feat_text_trans / feat_text_trans.norm(dim=-1, keepdim=True)
    feat_text_trans_norm = feat_text_trans_norm.to(device_img)

    if dataset == 'voc20b':
        num_cat_voc20b = 21
        histogram_llama = np.zeros((num_cat_voc20b, num_cat_voc20b))
    else:
        histogram_llama = np.zeros((num_cat, num_cat))
    
    for index, sample in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
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

        # patchtokens_norm: (B, h, w, D_v) -> B * (hw, D_v)
        # feat_text_trans_norm: (B, K, D_v) -> B * (K, D_v)
        # (hw, D_v) * (D_v, K) -> (hw, K)
        pred_masks_llama = patchtokens_norm @ feat_text_trans_norm.transpose(1, 0)
        pred_masks_llama = pred_masks_llama.view(N, h, w, num_cat)

        # image: patchtokens_norm:      (1, K'(I,M), D_v) -> 1 * (K, D_t)
        # text:  feat_text_trans_norm:  (1, K, D_t) -> 1 * (K, D_t)
        # (K', D_t) * (K, D_t) -> (K', K)

        # upsample
        pred_masks_llama = torch.nn.functional.interpolate(pred_masks_llama.permute(0, 3, 1, 2), (H, W),
                                                           mode='bilinear')
        pred_masks_llama = pred_masks_llama.permute(0, 2, 3, 1)

        pred_masks_llama = pred_masks_llama.max(-1)[1]

        # pc59 --> voc20b
        if dataset == "voc20b":
            pred_masks_llama[pred_masks_llama >= 20] = 20

        pred_masks_llama = pred_masks_llama.view(N, -1).cpu().numpy()
        label_cat = label_cat.view(N, -1).cpu().numpy()

        # TODO: use mmseg eval instead
        if dataset=='voc20b':
            histogram_llama += scores(label_cat, pred_masks_llama, num_cat_voc20b)
        else:
            histogram_llama += scores(label_cat, pred_masks_llama, num_cat)

    res_llama_trans = get_result_metrics(histogram_llama)
    _log.info(f"Epoch: {0:2d}\t- "
              f"mIoU: "
              f"llama_trans: {res_llama_trans['mean_iou']:.4f} \t"
              f"dataset: {dataset}")

def eval_on_datasets(cfg, visual_encoder, projector, feature_dict, device_img, _log):

    # load validation datasets
    val_dataset_150 = load_dataset_ade20k(cfg, set=f'A150', split='val')
    val_dataset_847 = load_dataset_ade20k(cfg, set=f'A847', split='val')
    val_dataset_coco = load_dataset_coco(cfg, split='val')
    val_dataset_pc59 = load_train_dataset_pc(cfg, set='pc59', split='val')
    val_dataset_pc459 = load_train_dataset_pc(cfg, set='pc459', split='val')
    val_dataset_voc20 = load_train_dataset_voc(cfg, set='voc20', split='val')
    val_dataset_voc20b = load_train_dataset_voc(cfg, set='voc20b', split='val')

    val_dataset_150 = DataLoader(val_dataset_150, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
    val_dataset_847 = DataLoader(val_dataset_847, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
    val_dataset_coco = DataLoader(val_dataset_coco, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
    val_dataset_voc20 = DataLoader(val_dataset_voc20, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False, drop_last=False)
    val_dataset_voc20b = DataLoader(val_dataset_voc20b, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
    val_dataset_pc59 = DataLoader(val_dataset_pc59, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
    val_dataset_pc459 = DataLoader(val_dataset_pc459, batch_size=cfg.dataset.val_batch_size,
                                  shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
    
    # evaluation
    method = cfg.eval.method
    eval_on_dataset(visual_encoder, projector, val_dataset_coco,
                    feature_dict["cocostuff"]["llama"], 'cocostuff', device_img, method, _log)
    eval_on_dataset(visual_encoder, projector, val_dataset_150,
                    feature_dict["150"]["llama"], '150', device_img, method, _log)
    eval_on_dataset(visual_encoder, projector, val_dataset_voc20,
                    feature_dict["voc20"]["llama"], 'voc20', device_img, method, _log)
    # eval_on_dataset(visual_encoder, projector, val_dataset_voc20b,
    #                 feature_dict["voc20b"]["llama"], 'voc20b', device_img, method)
    eval_on_dataset(visual_encoder, projector, val_dataset_pc59,
                    feature_dict["pc59"]["llama"], 'pc59', device_img, method, _log)
    return None