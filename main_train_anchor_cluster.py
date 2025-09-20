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
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from utils.kmeans import eval_kmeans_itsamatch
from utils.knngraph import Latent_knn_sysmmetric_graph_construct_numpy
from utils.LatentFuncitonMap import laplacian_main_sparse
from utils.shuffle_utils import select_samples_per_class, map_indices_to_class_labels, sample_features_per_class_coco, shuffle_features_and_labels, select_samples_per_class_mean
from utils.fmap_retrieval import deepfmap_retrieval, accrucy_fn, fmap_retrieval_norm, fmap_retrieval, fmap_retrieval_unsupervised
from utils.anchor_embeddings import anchor_embeddings_compute_unsupervised, anchor_embeddings_compute_supervised, anchor_matching
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
ex.add_config('./configs/LFMTrans_cfg_anchor_cluster.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

@ex.capture
def eval_proj_unsupervised(cfg, feature_dict, clusterer, device, _log):

    TRAINTYPE = cfg.train.type
    # Load features
    if TRAINTYPE == 'prototype':
        if cfg.train.dataset == 'CIFAR-10':
            # v 1536, t 1024 loading embeddings
            feat_language = feature_dict["cifar-10"]["train"]["all-Roberta-large-v1"].to(device).float()
            feat_vision = feature_dict["cifar-10"]["train"]["dinov2"].to(device).float()
            feat_labels = feature_dict["cifar-10"]["train"]["labels"].to(device).float()

            # clusterer
            print('start clustering')
            # cluster_dict = eval_kmeans_itsamatch(feat_language, feat_vision, feat_labels, clusterer, cfg.seed)
            # feat_v_clustered = cluster_dict['feat_v_clustered']
            # feat_v_subsampled = cluster_dict['feat_v_subsampled']
            # cluster_assignments = cluster_dict['cluster_assignments']
            # labels_subsampled = cluster_dict['labels_subsampled']
            with open('./cluster/feat_cluster.pkl', 'rb') as f1:
                feat_v_clustered = pkl.load(f1)
            with open('./cluster/cluster_assignments.pkl', 'rb') as f2:
                cluster_assignments = pkl.load(f2)
            with open('./cluster/labels_subsampled.pkl', 'rb') as f3:
                labels_subsampled = pkl.load(f3)
            print('clustering finished')

            # prototype
            feat_language = feat_language.mean(dim=1)
            feat_vision = feat_v_clustered

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
            feat_v = feat_vision.to(device)
            v_vecs = v_vecs.unsqueeze(0)
            v_vals = v_vals.unsqueeze(0)

            # language
            feat_t_trans = feat_language.float().to(device)
            t_vecs = t_vecs.unsqueeze(0)
            t_vals = t_vals.unsqueeze(0)

            # anchor descriptor
            feat_v_anchor, feat_t_trans_anchor = anchor_embeddings_compute_unsupervised(cfg, feat_v, feat_t_trans)

            # add btach size
            feat_v_anchor = feat_v_anchor.unsqueeze(0)
            feat_t_trans_anchor = feat_t_trans_anchor.unsqueeze(0)

            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_trans_anchor, v_vals, t_vals, v_vecs, t_vecs)

            Cxy = Cxy.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)

            # Cxy
            sim_matrix = fmap_retrieval_unsupervised(cfg, Cxy, v_vecs, t_vecs)
            # best matching 
            _, col_ind = linear_sum_assignment(sim_matrix.cpu())
            permutation = torch.as_tensor(col_ind, device=cluster_assignments.device)

            prediction = permutation[cluster_assignments]

            accuracy = (prediction == labels_subsampled).float().mean().item()

            _log.info(f"Finished - prototype - accrucy: {accuracy:.4f}")

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
            feat_t_trans = feat_language

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
            
        elif cfg.train.dataset == 'cocostuff':
            feat_language_raw = feature_dict["llama3_features"]["llama3_coco_unmean"].to(device).float()
            feat_vision_raw = feature_dict["dinov2_synonym_features"]["dinov2_coco"].to(device).float()
            anchor_matching(cfg, feature_dict, feat_vision_raw, feat_language_raw, device, _log)

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

    clusterer_kmean = KMeans(
        n_clusters=None,
        init="k-means++",
        n_init=100,
        random_state=cfg.seed,
    )

    eval_proj_unsupervised(cfg, feature_dict, clusterer_kmean, device, _log)