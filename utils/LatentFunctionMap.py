
import torch
import numpy as np
import tqdm

from utils.knngraph import Latent_knn_sysmmetric_graph_construct_numpy
from utils.laplacian_utils import laplacian_main_sparse
from utils.shuffle_utils import shuffle_features_and_labels
from utils.anchor_embeddings import anchor_embeddings_compute_unsupervised, anchor_embeddings_compute_supervised
from utils.fmap_util import pointmap2fmap, fmap2pointmap

from model.fmap_network import RegularizedFMNet


class LatentFunctionMap():
    def __init__(self, cfg):
        self.cfg = cfg
        self.fm_net = RegularizedFMNet(bidirectional=True)
    
    def LFM(self, feat_v, feat_t, device):
        '''
        feat_v: feature of vision,
        feat_t: feature of language,
        '''
        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_v = feat_v.unsqueeze(0).to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)

        # language
        feat_t = feat_t.unsqueeze(0).float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # build regularized_funciton_map model
        Cxy, Cyx = self.fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)
        
        # reduce batchsize
        Cxy = Cxy.squeeze(0)
        Cyx = Cyx.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)        

        return Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals

    def LFM_anchor(self, feat_v, feat_t, feat_labels_v, device):
        '''
        feat_v: feature of vision,
        feat_t: feature of language,
        '''
        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

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
        feat_t = feat_t.float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # anchor descriptor
        feat_v_anchor, feat_t_anchor = anchor_embeddings_compute_unsupervised(self.cfg, feat_v, feat_t)

        # shuffle features
        feat_v_anchor, feat_labels_v = shuffle_features_and_labels(feat_v_anchor, feat_labels_v, self.cfg.seed)

        # add btach size
        feat_v_anchor = feat_v_anchor.unsqueeze(0)
        feat_t_anchor = feat_t_anchor.unsqueeze(0)

        # build regularized_funciton_map model
        Cxy, Cyx = self.fm_net(feat_v_anchor, feat_t_anchor, v_vals, t_vals, v_vecs, t_vecs)
        
        # reduce batchsize
        Cxy = Cxy.squeeze(0)
        Cyx = Cyx.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)

        return Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals, feat_labels_v
    
    def LFM_zoomout(self, feat_v, feat_t, step, device):
        '''
        feat_v: feature of vision,
        feat_t: feature of language,
        '''

        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_v = feat_v.unsqueeze(0).to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)

        # language
        feat_t = feat_t.unsqueeze(0).float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        Cxy_k, _ = self.fm_net(feat_v, feat_t, v_vals[:, :step], t_vals[:, :step], v_vecs[:, :step, :], t_vecs[:, :step, :])
        Cxy_k = Cxy_k.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)

        for k in tqdm.tqdm(range(step, self.cfg.laplacian_mat.k-step, step), colour='red'):
            next_k = min(k+step, self.cfg.laplacian_mat.k)
            Pxy_k = fmap2pointmap(Cxy_k, v_vecs[:k].t(), t_vecs[:k].t())
            Cxy_k = pointmap2fmap(Pxy_k, v_vecs[:next_k].t(), t_vecs[:next_k].t())
        return Pxy_k