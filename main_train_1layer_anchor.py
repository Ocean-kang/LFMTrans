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

from model.Encoder import LinearProjText
from loss.gromov_loss import SGW
from utils.knngraph import Latent_knn_sysmmetric_graph_construct_numpy
from utils.laplacian_utils import laplacian_main_sparse
from utils.shuffle_utils import select_samples_per_class, map_indices_to_class_labels, sample_features_per_class_coco, shuffle_features_and_labels, select_samples_per_class_mean
from utils.fmap_retrieval import deepfmap_retrieval, accrucy_fn, fmap_retrieval_norm, fmap_retrieval
from utils.anchor_embeddings import anchor_embeddings_compute_unsupervised
from model.fmap_network import RegularizedFMNet

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
ex.add_config('./configs/LFMTrans_cfg_1layer_anchor.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

@ex.capture
def train_projector_oncefmap(cfg, model, feature_dict, criterion, device, _log):
    """
    Train the Projector.

    Args:
        model: the LinearProj model
        feature_dict: feature_dict
        device: 'cuda' or 'cpu'
    """
    num_epochs = cfg.model.nums_epoch_proj
    lr = cfg.model.lr_projector
    TRAINTYPE = cfg.train.type

    # Load features
    if TRAINTYPE == 'prototype':
        if cfg.train.dataset == 'CIFAR-10':
            feat_language = feature_dict["cifar-10"]["train"]["all-Roberta-large-v1"].to(device).float()
            feat_vision = feature_dict["cifar-10"]["train"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-10"]["train"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            feat_language = feat_language.mean(dim=1) # [n_cls, n_prompt, text_dimension] -> [n_cls, text_dimension]
            feat_vision, feat_labels = select_samples_per_class(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed) # [180, 1024], [180]
        elif cfg.train.dataset == 'CIFAR-100':
            feat_language = feature_dict["cifar-100"]["test"]["all-mpnet-base-v2"].to(device).float()
            feat_vision = feature_dict["cifar-100"]["test"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-100"]["test"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            feat_language = feat_language.view(-1, cfg.model.text_dimension) # [n_cls, n_prompt, text_dimension] -> [180, text_dimension]
            feat_vision, feat_labels = select_samples_per_class(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed) # [180, 1024], [180]            
        elif cfg.train.dataset == 'cocostuff':   
            feat_language = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
            feat_vision = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()

    elif TRAINTYPE == 'photo':
        if cfg.train.dataset == 'CIFAR-10':
            feat_language = feature_dict["cifar-10"]["train"]["all-Roberta-large-v1"].to(device).float()
            feat_vision = feature_dict["cifar-10"]["train"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-10"]["train"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            feat_language = feat_language.view(-1, cfg.model.text_dimension) # [n_cls, n_prompt, text_dimension] -> [180, text_dimension]
            feat_vision, feat_labels = select_samples_per_class(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed) # [180, 1024], [180]

        elif cfg.train.dataset == 'CIFAR-100':
            feat_language = feature_dict["cifar-100"]["test"]["all-mpnet-base-v2"].to(device).float()
            feat_vision = feature_dict["cifar-100"]["test"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-100"]["test"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            feat_language = feat_language.view(-1, cfg.model.text_dimension) # [n_cls, n_prompt, text_dimension] -> [180, text_dimension]
            feat_vision, feat_labels = select_samples_per_class(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed) # [180, 1024], [180]

        elif cfg.train.dataset == 'cocostuff':
            feat_llama3_train = feature_dict["llama3_synonym_features"]["llama3_coco"].to('cpu').float()
            feat_dinov2_train = feature_dict["dinov2_synonym_features"]["dinov2_coco"].to('cpu').float()

            # Find zero embeddings
            if (feat_dinov2_train.sum(-1) == 0).sum() > 0:
                for i in range(len(feat_dinov2_train)):
                    if (feat_dinov2_train[i].sum(-1) == 0).sum() > 0:
                        zero_idx = torch.where((feat_dinov2_train[i].sum(-1) == 0))[0]
                        nonzero_idx = torch.where((feat_dinov2_train[i].sum(-1) != 0))[0]
                        pad_idx = np.array(random.choices(list(nonzero_idx.numpy()), k=len(zero_idx)))
                        feat_dinov2_train[i][zero_idx] = feat_dinov2_train[i][pad_idx]

            if (feat_llama3_train.sum(-1) == 0).sum() > 0:
                for i in range(len(feat_llama3_train)):
                    if (feat_llama3_train[i].sum(-1) == 0).sum() > 0:
                        zero_idx = torch.where((feat_llama3_train[i].sum(-1) == 0))[0]
                        nonzero_idx = torch.where((feat_llama3_train[i].sum(-1) != 0))[0]
                        pad_idx = np.array(random.choices(list(nonzero_idx.numpy()), k=len(zero_idx)))
                        feat_llama3_train[i][zero_idx] = feat_llama3_train[i][pad_idx]
            feat_language = feat_llama3_train.to(device).float()
            feat_vision = feat_dinov2_train.to(device).float()
            
            feat_vision, feat_labels = sample_features_per_class_coco(feat_vision, 10, cfg.seed)
            feat_language, feat_labels_language = sample_features_per_class_coco(feat_language, 10, cfg.seed)

    # Load model
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    total_loss = 0.0
    for epoch in range(num_epochs):

        x = feat_language.to(device)  # [B, text_dimension]
        y = feat_vision.to(device)  # [B, vision_dimension]

        optimizer.zero_grad()
        output = model(x)  # [B, vision_dimension]

        # Loss
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _log.info(f"Epoch {epoch+1}/{num_epochs} - ProjectorLoss - Loss: {loss:.4f}")
    _log.info(f"Train Projector Finished - Avg Loss: {total_loss / num_epochs:.4f}")

    return None

@ex.capture
def eval_proj(cfg, model_1, feature_dict, device, _log):

    TRAINTYPE = cfg.train.type
    # Load features
    if TRAINTYPE == 'prototype':
        if cfg.train.dataset == 'CIFAR-10':
            # v 1536, t 1024
            feat_language = feature_dict["cifar-10"]["train"]["all-Roberta-large-v1"].to(device).float()
            feat_vision = feature_dict["cifar-10"]["train"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-10"]["train"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            _, _, feat_vision_mean = select_samples_per_class_mean(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed, return_class_mean=True) # [180, 1024]
            feat_labels_v = torch.arange(n_cls).to(device).float()
            # prototype
            feat_language = feat_language.mean(dim=1)
            feat_vision = feat_vision_mean

            # projector and translator
            feat_v = feat_vision
            feat_t_trans = model_1(feat_language)

            # compute once eigenvecs and eigenvals in each multimodels
            # vision
            W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
            v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

            #language
            W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_t_trans, device, symmetrize=False)
            t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

            # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
            # vision
            v_vecs = torch.from_numpy(v_vecs).to(device).float()
            v_vals = torch.from_numpy(v_vals).to(device).float()
            # language
            t_vecs = torch.from_numpy(t_vecs).to(device).float()
            t_vals = torch.from_numpy(t_vals).to(device).float()

            # adding batch dimension
            # vision
            feat_v = feat_v.to(device)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_t_trans.float().to(device)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # anchor descriptor
            feat_v_anchor, feat_t_trans_anchor = anchor_embeddings_compute_unsupervised(cfg, feat_v, feat_t_trans)

            # shuffle features
            feat_v_anchor, feat_labels_v = shuffle_features_and_labels(feat_v_anchor, feat_labels_v, cfg.seed)

            # add btach size
            feat_v_anchor = feat_v_anchor.unsqueeze(0)
            feat_t_trans_anchor = feat_t_trans_anchor.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_trans_anchor, v_vals, t_vals, v_vecs, t_vecs)

            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            feat_v = feat_v.squeeze(0)
            feat_t_trans = feat_t_trans.squeeze(0)

            # Cxy
            csr_index_Cxy = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx = fmap_retrieval(cfg, Cyx, t_vecs, v_vecs)
            # Cxy
            csr_index_Cxy_norm = fmap_retrieval_norm(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx_norm = fmap_retrieval_norm(cfg, Cyx, t_vecs, v_vecs)

            accurcy_Cxy = accrucy_fn(feat_labels_v, csr_index_Cxy)
            accurcy_Cyx = accrucy_fn(feat_labels_v, csr_index_Cyx)
            accurcy_Cxy_norm = accrucy_fn(feat_labels_v, csr_index_Cxy_norm)
            accurcy_Cyx_norm = accrucy_fn(feat_labels_v, csr_index_Cyx_norm)

            accrucy_norm = (accurcy_Cxy_norm + accurcy_Cyx_norm) / 2
            accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
            _log.info(f"Train Finished - prototype - Cxy/Cyx/Avg accrucy: {accurcy_Cxy:.4f}/{accurcy_Cyx:.4f}/{accrucy:.4f}")
            _log.info(f"Train Finished - prototype - Cxy_norm/Cyx_norm/Avg accrucy_norm: {accurcy_Cxy_norm:.4f}/{accurcy_Cyx_norm:.4f}/{accrucy_norm:.4f}")

        elif cfg.train.dataset == 'CIFAR-100':
            # v 1536, t 768
            feat_language = feature_dict["cifar-100"]["test"]["all-mpnet-base-v2"].to(device).float()
            feat_vision = feature_dict["cifar-100"]["test"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-100"]["test"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            _, _, feat_vision_mean = select_samples_per_class_mean(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed, return_class_mean=True)
            feat_labels_v = torch.arange(n_cls).to(device).float()

            # prototype
            feat_language = feat_language.mean(dim=1)
            feat_vision = feat_vision_mean

            # projector and translator
            feat_v = feat_vision
            feat_t_trans = model_1(feat_language)

            # compute once eigenvecs and eigenvals in each multimodels
            # vision
            W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
            v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

            #language
            W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_t_trans, device, symmetrize=False)
            t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

            # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
            # vision
            v_vecs = torch.from_numpy(v_vecs).to(device).float()
            v_vals = torch.from_numpy(v_vals).to(device).float()
            # language
            t_vecs = torch.from_numpy(t_vecs).to(device).float()
            t_vals = torch.from_numpy(t_vals).to(device).float()

            # adding batch dimension
            # vision
            feat_v = feat_v.to(device)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_t_trans.float().to(device)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # anchor descriptor
            feat_v_anchor, feat_t_trans_anchor = anchor_embeddings_compute_unsupervised(cfg, feat_v, feat_t_trans)

            # shuffle features
            # feat_v_anchor, feat_labels_v = shuffle_features_and_labels(feat_v_anchor, feat_labels_v, cfg.seed)

            # add btach size
            feat_v_anchor = feat_v_anchor.unsqueeze(0)
            feat_t_trans_anchor = feat_t_trans_anchor.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_trans_anchor, v_vals, t_vals, v_vecs, t_vecs)

            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            feat_v = feat_v.squeeze(0)
            feat_t_trans = feat_t_trans.squeeze(0)

            # Cxy
            csr_index_Cxy = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx = fmap_retrieval(cfg, Cyx, t_vecs, v_vecs)
            # Cxy
            csr_index_Cxy_norm = fmap_retrieval_norm(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx_norm = fmap_retrieval_norm(cfg, Cyx, t_vecs, v_vecs)

            accurcy_Cxy = accrucy_fn(feat_labels_v, csr_index_Cxy)
            accurcy_Cyx = accrucy_fn(feat_labels_v, csr_index_Cyx)
            accurcy_Cxy_norm = accrucy_fn(feat_labels_v, csr_index_Cxy_norm)
            accurcy_Cyx_norm = accrucy_fn(feat_labels_v, csr_index_Cyx_norm)

            accrucy_norm = (accurcy_Cxy_norm + accurcy_Cyx_norm) / 2
            accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
            _log.info(f"Train Finished - prototype - Cxy/Cyx/Avg accrucy: {accurcy_Cxy:.4f}/{accurcy_Cyx:.4f}/{accrucy:.4f}")
            _log.info(f"Train Finished - prototype - Cxy_norm/Cyx_norm/Avg accrucy_norm: {accurcy_Cxy_norm:.4f}/{accurcy_Cyx_norm:.4f}/{accrucy_norm:.4f}")
            
        elif cfg.train.dataset == 'cocostuff':
            # v 1024, t 4096
            feat_language = feature_dict["llama3_features"]["llama3_coco"].to(device).float()
            feat_vision = feature_dict["dinov2_features"]["dinov2_coco"].to(device).float()
            n_cls, dimension = feat_language.shape
            feat_labels_v = torch.arange(n_cls).to(device).float()
            feat_labels_t = torch.arange(n_cls).to(device).float()

            # projector and translator
            feat_v = feat_vision
            feat_t_trans = model_1(feat_language)

            # compute once eigenvecs and eigenvals in each multimodels
            # vision
            W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
            v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

            #language
            W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_t_trans, device, symmetrize=False)
            t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

            # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
            # vision
            v_vecs = torch.from_numpy(v_vecs).to(device).float()
            v_vals = torch.from_numpy(v_vals).to(device).float()
            # language
            t_vecs = torch.from_numpy(t_vecs).to(device).float()
            t_vals = torch.from_numpy(t_vals).to(device).float()

            # adding batch dimension
            # vision
            feat_v = feat_v.to(device)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_t_trans.float().to(device)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # anchor descriptor
            feat_v_anchor, feat_t_trans_anchor = anchor_embeddings_compute_unsupervised(cfg, feat_v, feat_t_trans)

            # shuffle features
            feat_v_anchor, feat_labels_v = shuffle_features_and_labels(feat_v_anchor, feat_labels_v, cfg.seed)

            # add btach size
            feat_v_anchor = feat_v_anchor.unsqueeze(0)
            feat_t_trans_anchor = feat_t_trans_anchor.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_trans_anchor, v_vals, t_vals, v_vecs, t_vecs)

            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            feat_v = feat_v.squeeze(0)
            feat_t_trans = feat_t_trans.squeeze(0)

            # Cxy
            csr_index_Cxy = fmap_retrieval(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx = fmap_retrieval(cfg, Cyx, t_vecs, v_vecs)
            # Cxy
            csr_index_Cxy_norm = fmap_retrieval_norm(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx_norm = fmap_retrieval_norm(cfg, Cyx, t_vecs, v_vecs)

            accurcy_Cxy = accrucy_fn(feat_labels_v, csr_index_Cxy)
            accurcy_Cyx = accrucy_fn(feat_labels_v, csr_index_Cyx)
            accurcy_Cxy_norm = accrucy_fn(feat_labels_v, csr_index_Cxy_norm)
            accurcy_Cyx_norm = accrucy_fn(feat_labels_v, csr_index_Cyx_norm)

            accrucy_norm = (accurcy_Cxy_norm + accurcy_Cyx_norm) / 2
            accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
            _log.info(f"Train Finished - prototype - Cxy/Cyx/Avg accrucy: {accurcy_Cxy:.4f}/{accurcy_Cyx:.4f}/{accrucy:.4f}")
            _log.info(f"Train Finished - prototype - Cxy_norm/Cyx_norm/Avg accrucy_norm: {accurcy_Cxy_norm:.4f}/{accurcy_Cyx_norm:.4f}/{accrucy_norm:.4f}")

    elif TRAINTYPE == 'photo':
        if cfg.train.dataset == 'CIFAR-10':
            feat_language = feature_dict["cifar-10"]["train"]["all-Roberta-large-v1"].to(device).float()
            feat_vision = feature_dict["cifar-10"]["train"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-10"]["train"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            feat_language = feat_language.view(-1, cfg.model.text_dimension) # [n_cls, n_prompt, text_dimension] -> [180, text_dimension]
            feat_vision, feat_labels = select_samples_per_class(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed) # [180, 1024], [180]

            # projector and translator
            feat_v = feat_vision
            feat_t_trans = model_1(feat_language)

            # shuffle feats and labels
            # feat_v, feat_labels = shuffle_features_and_labels(feat_vision, feat_labels, cfg.seed)

            # compute once eigenvecs and eigenvals in each multimodels
            # vision
            W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
            v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

            #language
            W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_language, device, symmetrize=False)
            t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

            # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
            # vision
            v_vecs = torch.from_numpy(v_vecs).to(device).float()
            v_vals = torch.from_numpy(v_vals).to(device).float()
            # language
            t_vecs = torch.from_numpy(t_vecs).to(device).float()
            t_vals = torch.from_numpy(t_vals).to(device).float()

            # adding batch dimension
            # vision
            feat_v = feat_v.to(device).unsqueeze(0)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_t_trans.float().to(device).unsqueeze(0)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v, feat_t_trans, v_vals, t_vals, v_vecs, t_vecs)
            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            feat_t_trans = feat_t_trans.squeeze(0)
            feat_v = feat_v.squeeze(0)

            # Cxy_deep
            csr_index_Cxy = deepfmap_retrieval(cfg, Cxy, v_vecs, t_vecs, feat_v, feat_t_trans)
            # Cyx_deep
            csr_index_Cyx = deepfmap_retrieval(cfg, Cyx, t_vecs, v_vecs, feat_t_trans, feat_v)
            # Cxy
            csr_index_Cxy_test = fmap_retrieval_norm(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx_test = fmap_retrieval_norm(cfg, Cyx, t_vecs, v_vecs)

            # map indexes to classes
            csr_index_Cxy = map_indices_to_class_labels(csr_index_Cxy, n_prompt)
            csr_index_Cyx = map_indices_to_class_labels(csr_index_Cyx, n_prompt)
            csr_index_Cxy_test = map_indices_to_class_labels(csr_index_Cxy_test, n_prompt)
            csr_index_Cyx_test = map_indices_to_class_labels(csr_index_Cyx_test, n_prompt)
            
            accurcy_Cxy_deep = accrucy_fn(feat_labels, csr_index_Cxy)
            accurcy_Cyx_deep = accrucy_fn(feat_labels, csr_index_Cyx)
            accurcy_Cxy = accrucy_fn(feat_labels, csr_index_Cxy_test)
            accurcy_Cyx = accrucy_fn(feat_labels, csr_index_Cyx_test)

            accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
            accrucy_deep = (accurcy_Cxy_deep + accurcy_Cyx_deep) / 2
            _log.info(f"Train Finished - photo - Cxy_deep/Cyx_deep/Avg accrucy_deep: {accurcy_Cxy_deep:.4f}/{accurcy_Cyx_deep:.4f}/{accrucy_deep:.4f}")
            _log.info(f"Train Finished - photo - Cxy/Cyx/Avg accrucy: {accurcy_Cxy:.4f}/{accurcy_Cyx:.4f}/{accrucy:.4f}")

        elif cfg.train.dataset == 'CIFAR-100':
            
            feat_language = feature_dict["cifar-100"]["test"]["all-mpnet-base-v2"].to(device).float()
            feat_vision = feature_dict["cifar-100"]["test"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-100"]["test"]["labels"].to(device).float()
            n_cls, n_prompt, dimension = feat_language.shape
            feat_language = feat_language.view(-1, cfg.model.text_dimension) # [n_cls, n_prompt, text_dimension] -> [180, text_dimension]
            feat_vision, feat_labels = select_samples_per_class(feat_vision, feat_labels, n_cls, n_prompt, cfg.seed) # [180, 1024], [180]

            # projector and translator
            feat_v = feat_vision
            feat_t_trans = model_1(feat_language)

            # compute once eigenvecs and eigenvals in each multimodels
            # vision
            W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
            v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

            #language
            W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_language, device, symmetrize=False)
            t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

            # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
            # vision
            v_vecs = torch.from_numpy(v_vecs).to(device).float()
            v_vals = torch.from_numpy(v_vals).to(device).float()
            # language
            t_vecs = torch.from_numpy(t_vecs).to(device).float()
            t_vals = torch.from_numpy(t_vals).to(device).float()

            # adding batch dimension
            # vision
            feat_v = feat_v.to(device).unsqueeze(0)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_t_trans.float().to(device).unsqueeze(0)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v, feat_t_trans, v_vals, t_vals, v_vecs, t_vecs)
            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            feat_t_trans = feat_t_trans.squeeze(0)
            feat_v = feat_v.squeeze(0)
            # Cxy
            csr_index_Cxy = deepfmap_retrieval(cfg, Cxy, v_vecs, t_vecs, feat_v, feat_t_trans)
            # Cyx
            csr_index_Cyx = deepfmap_retrieval(cfg, Cyx, t_vecs, v_vecs, feat_t_trans, feat_v)
            # map indexes to classes
            csr_index_Cxy = map_indices_to_class_labels(csr_index_Cxy, n_prompt)
            csr_index_Cyx = map_indices_to_class_labels(csr_index_Cyx, n_prompt)
            
            accurcy_Cxy = accrucy_fn(feat_labels, csr_index_Cxy)
            accurcy_Cyx = accrucy_fn(feat_labels, csr_index_Cyx)

            accrucy = (accurcy_Cxy + accurcy_Cyx) / 2
            _log.info(f"Train Finished - photo - Cxy/Cyx/Avg accrucy: {accurcy_Cxy:.4f}/{accurcy_Cyx:.4f}/{accrucy:.4f}")

        elif cfg.train.dataset == 'cocostuff':
            feat_llama3_train = feature_dict["llama3_synonym_features"]["llama3_coco"].to('cpu').float()
            feat_dinov2_train = feature_dict["dinov2_synonym_features"]["dinov2_coco"].to('cpu').float()
            feat_dinov2_train = torch.cat([feat_dinov2_train], 1)
            feat_llama3_train = torch.cat([feat_llama3_train], 1)
            # Find zero embeddings
            if (feat_dinov2_train.sum(-1) == 0).sum() > 0:
                for i in range(len(feat_dinov2_train)):
                    if (feat_dinov2_train[i].sum(-1) == 0).sum() > 0:
                        zero_idx = torch.where((feat_dinov2_train[i].sum(-1) == 0))[0]
                        nonzero_idx = torch.where((feat_dinov2_train[i].sum(-1) != 0))[0]
                        pad_idx = np.array(random.choices(list(nonzero_idx.numpy()), k=len(zero_idx)))
                        feat_dinov2_train[i][zero_idx] = feat_dinov2_train[i][pad_idx]
            if (feat_llama3_train.sum(-1) == 0).sum() > 0:
                for i in range(len(feat_llama3_train)):
                    if (feat_llama3_train[i].sum(-1) == 0).sum() > 0:
                        zero_idx = torch.where((feat_llama3_train[i].sum(-1) == 0))[0]
                        nonzero_idx = torch.where((feat_llama3_train[i].sum(-1) != 0))[0]
                        pad_idx = np.array(random.choices(list(nonzero_idx.numpy()), k=len(zero_idx)))
                        feat_llama3_train[i][zero_idx] = feat_llama3_train[i][pad_idx]

            feat_language = feat_llama3_train.to(device).float()
            feat_vision = feat_dinov2_train.to(device).float()
            n_prompt = cfg.train.samples
            feat_vision, feat_labels = sample_features_per_class_coco(feat_vision, n_prompt, cfg.seed)
            feat_language, feat_labels_language = sample_features_per_class_coco(feat_language, n_prompt, cfg.seed)
            feat_labels = feat_labels.to(device)

            # projector and translator
            feat_v = feat_vision
            feat_t_trans = model_1(feat_language)

            # compute once eigenvecs and eigenvals in each multimodels
            # vision
            W_v = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_vision, device, symmetrize=False)
            v_vecs, v_vals = laplacian_main_sparse(W_v, cfg.laplacian_mat.k)

            #language
            W_t = Latent_knn_sysmmetric_graph_construct_numpy(cfg, feat_language, device, symmetrize=False)
            t_vecs, t_vals = laplacian_main_sparse(W_t, cfg.laplacian_mat.k)

            # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
            # vision
            v_vecs = torch.from_numpy(v_vecs).to(device).float()
            v_vals = torch.from_numpy(v_vals).to(device).float()
            # language
            t_vecs = torch.from_numpy(t_vecs).to(device).float()
            t_vals = torch.from_numpy(t_vals).to(device).float()

            # adding batch dimension
            # vision
            feat_v = feat_v.to(device)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_t_trans.float().to(device)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # anchor descriptor
            feat_v_anchor, feat_t_trans_anchor = anchor_embeddings_compute_unsupervised(cfg, feat_v, feat_t_trans)

            # shuffle features
            # feat_v_anchor, feat_labels = shuffle_features_and_labels(feat_v_anchor, feat_labels, cfg.seed)

            feat_v_anchor = feat_v_anchor.unsqueeze(0)
            feat_t_trans_anchor = feat_t_trans_anchor.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)

            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_trans_anchor, v_vals, t_vals, v_vecs, t_vecs)
            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            feat_t_trans = feat_t_trans.squeeze(0)
            feat_v = feat_v.squeeze(0)

            # Cxy_deep
            csr_index_Cxy = deepfmap_retrieval(cfg, Cxy, v_vecs, t_vecs, feat_v, feat_t_trans)
            # Cyx_deep
            csr_index_Cyx = deepfmap_retrieval(cfg, Cyx, t_vecs, v_vecs, feat_t_trans, feat_v)
            # Cxy
            csr_index_Cxy_norm = fmap_retrieval_norm(cfg, Cxy, v_vecs, t_vecs)
            # Cyx
            csr_index_Cyx_norm = fmap_retrieval_norm(cfg, Cyx, t_vecs, v_vecs)

            # map indexes to classes
            csr_index_Cxy = map_indices_to_class_labels(csr_index_Cxy, n_prompt)
            csr_index_Cyx = map_indices_to_class_labels(csr_index_Cyx, n_prompt)
            csr_index_Cxy_norm = map_indices_to_class_labels(csr_index_Cxy_norm, n_prompt)
            csr_index_Cyx_norm = map_indices_to_class_labels(csr_index_Cyx_norm, n_prompt)
            
            accurcy_Cxy_deep = accrucy_fn(feat_labels, csr_index_Cxy)
            accurcy_Cyx_deep = accrucy_fn(feat_labels, csr_index_Cyx)
            accurcy_Cxy_norm = accrucy_fn(feat_labels, csr_index_Cxy_norm)
            accurcy_Cyx_norm = accrucy_fn(feat_labels, csr_index_Cyx_norm)

            accrucy_norm = (accurcy_Cxy_norm + accurcy_Cyx_norm) / 2
            accrucy_deep = (accurcy_Cxy_deep + accurcy_Cyx_deep) / 2
            _log.info(f"Train Finished - photo - Cxy_deep/Cyx_deep/Avg accrucy_deep: {accurcy_Cxy_deep:.4f}/{accurcy_Cyx_deep:.4f}/{accrucy_deep:.4f}")
            _log.info(f"Train Finished - photo - Cxy_Norm/Cyx_Norm/Avg accrucy_Norm: {accurcy_Cxy_norm:.4f}/{accurcy_Cyx_norm:.4f}/{accrucy_norm:.4f}")

    return None

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
        },
        "cifar-100":{
            "test":{
                "dinov2": torch.load(TESTSETPATH + '/CIFAR-100/vision/dinov2_vit-g14.pt'),
                "all-mpnet-base-v2": torch.load(TESTSETPATH + '/CIFAR-100/language/sentencet_all-mpnet-base-v2.pt'),
                "labels": torch.load(TESTSETPATH + '/CIFAR-100/labels.pt')
            }
        }
    }
    
    model_proj = LinearProjText(cfg=cfg)

    # --dense matrix--
    # criterion = proj_loss(cfg=cfg)
    # --sparse matrix--
    # criterion = proj_loss_sparse(cfg=cfg)
    # --sparse matrix once enginvecs and enginvals--
    criterion_proj = SGW(cfg=cfg)

    # train_proj(cfg, model, feature_dict, criterion, device, _log)
    train_projector_oncefmap(cfg, model_proj, feature_dict, criterion_proj, device, _log)


    if os.path.isdir('./weight'):
        torch.save(model_proj.state_dict(), f"./weight/projector_1layer_anchor{cfg.save}.pth")
    else:
        os.makedirs('./weight', exist_ok=True)
        torch.save(model_proj.state_dict(), f"./weight/projector_1layer_anchor{cfg.save}.pth")

    with torch.no_grad():
        model_proj.eval()
        eval_proj(cfg, model_proj, feature_dict, device, _log)