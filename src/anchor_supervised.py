import torch
import torch.nn as nn

from utils.KnnGraph import knngraph
from utils.shuffle_utils import sample_features_per_class_coco
from utils.laplacian_utils import laplacian_construction_decomposition
from utils.anchors_utils import anchor_embeddings_compute_supervised

from model.fmap_network import RegularizedFMNet, LatentFunctionMap


class LFMAnchor(nn.Module):
    def __init__(self, cfg):
        super(LFMAnchor, self).__init__()
        self.cfg = cfg

    def forward(self, feature_dict_train, feature_dict_val, device):
        
        # TODO Unmean and path all in training
        # Load Feature
        feat_t_unmean = feature_dict_train[f'{self.cfg.train.dataset}'][f'{self.cfg.train.text_model}']
        feat_v_unmean = feature_dict_train[f'{self.cfg.train.dataset}'][f'{self.cfg.train.type}']
        feat_t, labels_t = sample_features_per_class_coco(feat_t_unmean, self.cfg.train.sample, self.cfg.seed)
        feat_v, labels_v = sample_features_per_class_coco(feat_v_unmean, self.cfg.train.sample, self.cfg.seed)

        # KNN Graph Construction
        Knn_v = knngraph(self.cfg, feat_v, device)
        Knn_t = knngraph(self.cfg, feat_t, device)

        # Laplacian Matrix and eigendecomposition
        v_vecs, v_vals = laplacian_construction_decomposition(self.cfg, Knn_v, device)
        t_vecs, t_vals = laplacian_construction_decomposition(self.cfg, Knn_t, device)
        
        # TODO randon selection in anchors
        # Anchors Selected and computation
        feat_v_anchor, feat_t_anchor = anchor_embeddings_compute_supervised(self.cfg, feat_v, feat_v_unmean, feat_t, feat_t_unmean)
        
        # Function Map Comuptation
        
        # n_eigens choose
        n_eigens = self.cfg.laplacian_mat.k

        # add batchsize vision
        v_vecs = v_vecs[:n_eigens, :].unsqueeze(0).cpu()
        v_vals = v_vals[:n_eigens].unsqueeze(0).cpu()
        feat_v_anchor = feat_v_anchor.unsqueeze(0).cpu()

        # add batchsize text
        t_vecs = t_vecs[:n_eigens, :].unsqueeze(0).cpu()
        t_vals = t_vals[:n_eigens].unsqueeze(0).cpu()
        feat_t_anchor = feat_t_anchor.unsqueeze(0).cpu()

        if self.cfg.fmap.method == "EOT":
            # build regularized_funciton_map model
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_anchor, v_vals, t_vals, v_vecs, t_vecs)

            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            v_vals = v_vals.squeeze(0)
            t_vals = t_vals.squeeze(0)

        elif self.cfg.fmap.method == "LFMtrans":
            fm_net = LatentFunctionMap(cfg=self.cfg)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_anchor, v_vecs, t_vecs)

            Cxy = Cxy.squeeze(0)
            Cyx = Cyx.squeeze(0)
            v_vecs = v_vecs.squeeze(0)
            t_vecs = t_vecs.squeeze(0)
            v_vals = v_vals.squeeze(0)
            t_vals = t_vals.squeeze(0)

        return Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals, labels_t, labels_v