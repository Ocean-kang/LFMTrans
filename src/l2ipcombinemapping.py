"""L2 and IP graph combination with a single functional basis and auxiliary regularization."""

import copy

import torch
import torch.nn as nn

from utils.KnnGraph import knngraph
from utils.laplacian_utils import laplacian_construction_decomposition
from model.fmap_network import MultiConstraintFM


class LFMapIpL2Combination(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fm_net = MultiConstraintFM(
            lmbda=float(getattr(cfg.fmap, 'lmbda', 100.0)),
            aux_lmbda=float(getattr(cfg.fmap, 'aux_lmbda', 25.0)),
            reg=float(getattr(cfg.fmap, 'reg', 1e-6)),
            resolvant_gamma=float(getattr(cfg.fmap, 'resolvant_gamma', 0.5)),
            bidirectional=True,
        )

    @staticmethod
    def _topk_basis(evecs: torch.Tensor, evals: torch.Tensor, k: int):
        basis = evecs[:, :k].T.contiguous().unsqueeze(0)
        spectrum = evals[:k].contiguous().unsqueeze(0)
        return basis, spectrum

    @staticmethod
    def _rayleigh_from_aux_laplacian(main_evecs: torch.Tensor, aux_laplacian: torch.Tensor, k: int) -> torch.Tensor:
        phi = main_evecs[:, :k]
        aux_phi = aux_laplacian @ phi
        rayleigh = torch.sum(phi * aux_phi, dim=0)
        return torch.clamp(rayleigh, min=0.0).unsqueeze(0)

    def forward(self, feature_dict_val, device):
        feat_t = feature_dict_val[f'{self.cfg.validation.dataset}'][f'{self.cfg.validation.text_model}'].to(device)
        feat_v = feature_dict_val[f'{self.cfg.validation.dataset}'][f'{self.cfg.validation.type}'].to(device)

        cfg_ip = copy.deepcopy(self.cfg)
        cfg_ip.knngraph.metric_knn = 'ip'
        cfg_l2 = copy.deepcopy(self.cfg)
        cfg_l2.knngraph.metric_knn = 'L2'

        Knn_v_ip = knngraph(cfg_ip, feat_v, device)
        Knn_t_ip = knngraph(cfg_ip, feat_t, device)
        Knn_v_l2 = knngraph(cfg_l2, feat_v, device)
        Knn_t_l2 = knngraph(cfg_l2, feat_t, device)

        v_vecs_ip, v_vals_ip = laplacian_construction_decomposition(cfg_ip, Knn_v_ip, device)
        t_vecs_ip, t_vals_ip = laplacian_construction_decomposition(cfg_ip, Knn_t_ip, device)
        _, _, L_v_l2 = laplacian_construction_decomposition(cfg_l2, Knn_v_l2, device, ret_L=True)
        _, _, L_t_l2 = laplacian_construction_decomposition(cfg_l2, Knn_t_l2, device, ret_L=True)

        n_eigens = min(self.cfg.laplacian_mat.k, v_vecs_ip.shape[1], t_vecs_ip.shape[1])

        v_basis, v_vals = self._topk_basis(v_vecs_ip, v_vals_ip, n_eigens)
        t_basis, t_vals = self._topk_basis(t_vecs_ip, t_vals_ip, n_eigens)
        v_aux_vals = self._rayleigh_from_aux_laplacian(v_vecs_ip, L_v_l2, n_eigens)
        t_aux_vals = self._rayleigh_from_aux_laplacian(t_vecs_ip, L_t_l2, n_eigens)

        feat_v_batch = feat_v.unsqueeze(0)
        feat_t_batch = feat_t.unsqueeze(0)

        Cxy, Cyx = self.fm_net(
            feat_v_batch,
            feat_t_batch,
            v_vals,
            t_vals,
            v_basis,
            t_basis,
            v_aux_vals,
            t_aux_vals,
        )

        return Cxy.squeeze(0), Cyx.squeeze(0), v_basis.squeeze(0), t_basis.squeeze(0)
