'''
 L2 and Ip distance combination in KNN Graph for Laplacian Matrix
'''


import torch
import torch.nn as nn

from utils.KnnGraph import knngraph
from utils.laplacian_utils import laplacian_construction_decomposition

from model.fmap_network import MultiConstraintFM


class LFMapIpL2Combination(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fm_net = MultiConstraintFM()
    
    def forward(self, feature_dict_val, device):

        # Load Feature
        feat_t = feature_dict_val[f'{self.cfg.validation.dataset}'][f'{self.cfg.validation.text_model}']
        feat_v = feature_dict_val[f'{self.cfg.validation.dataset}'][f'{self.cfg.validation.type}']

        # KNN Graph Construction
        # IP KNNGraph
        cfg_ip = self.cfg
        cfg_ip['knngraph']['metric_knn'] = 'ip'
        Knn_v_ip = knngraph(cfg_ip, feat_v, device)
        Knn_t_ip = knngraph(cfg_ip, feat_t, device)

        # L2 KNNGraph
        cfg_l2 = self.cfg
        cfg_l2['knngraph']['metric_knn'] = 'L2'
        Knn_v_l2 = knngraph(cfg_l2, feat_v, device)
        Knn_t_l2 = knngraph(cfg_l2, feat_t, device)

        # Laplacian Matrix and eigendecomposition
        # Ip
        v_vecs_ip, v_vals_ip = laplacian_construction_decomposition(self.cfg, Knn_v_ip, device)
        t_vecs_ip, t_vals_ip = laplacian_construction_decomposition(self.cfg, Knn_t_ip, device)
        # L2
        v_vecs_l2, v_vals_l2 = laplacian_construction_decomposition(self.cfg, Knn_v_l2, device)
        t_vecs_l2, t_vals_l2 = laplacian_construction_decomposition(self.cfg, Knn_t_l2, device)

        # Function Map Comuptation
        # n_eigens choose
        n_eigens = self.cfg.laplacian_mat.k

        # add batchsize vision (ip)
        v_vecs_ip = v_vecs_ip[:n_eigens, :].unsqueeze(0).cpu()
        v_vals_ip = v_vals_ip[:n_eigens].unsqueeze(0).cpu()
        # add batchsize text (ip)
        t_vecs_ip = t_vecs_ip[:n_eigens, :].unsqueeze(0).cpu()
        t_vals_ip = t_vals_ip[:n_eigens].unsqueeze(0).cpu()

        # add batchsize vision (L2)
        v_vecs_l2 = v_vecs_l2[:n_eigens, :].unsqueeze(0).cpu()
        v_vals_l2 = v_vals_l2[:n_eigens].unsqueeze(0).cpu()
        # add batchsize text (L2)
        t_vecs_l2 = t_vecs_l2[:n_eigens, :].unsqueeze(0).cpu()
        t_vals_l2 = t_vals_l2[:n_eigens].unsqueeze(0).cpu()

        Cxy, Cyx = self.fm_net(v_vecs_ip, t_vecs_ip, v_vecs_l2, t_vecs_l2)
        Cxy = Cxy.squeeze(0)
        Cyx = Cyx.squeeze(0)
        v_vecs_ip = v_vecs_ip.squeeze(0)
        t_vecs_ip = t_vecs_ip.squeeze(0)


        return Cxy, Cyx, v_vecs_ip, t_vecs_ip