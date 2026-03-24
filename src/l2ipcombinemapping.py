"""L2 and IP graph combination with a single functional basis and auxiliary regularization."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _build_cfg_pair(self):
        cfg_ip = copy.deepcopy(self.cfg)
        cfg_ip.knngraph.metric_knn = 'ip'
        cfg_l2 = copy.deepcopy(self.cfg)
        cfg_l2.knngraph.metric_knn = 'L2'
        return cfg_ip, cfg_l2

    def build_spectral_system(self, feat_x: torch.Tensor, feat_y: torch.Tensor, device, detach_basis: bool = True):
        if feat_x.ndim != 2 or feat_y.ndim != 2:
            raise ValueError(f'Expected 2D features [N, D]. Got {feat_x.shape} and {feat_y.shape}.')
        if feat_x.shape[-1] != feat_y.shape[-1]:
            raise ValueError(
                f'Feature dimensions must match before FM. Got x={feat_x.shape[-1]}, y={feat_y.shape[-1]}. '
                'Use a text->vision projector first.'
            )

        feat_x = F.normalize(feat_x.to(device), p=2, dim=-1)
        feat_y = F.normalize(feat_y.to(device), p=2, dim=-1)
        cfg_ip, cfg_l2 = self._build_cfg_pair()

        context = torch.no_grad() if detach_basis else torch.enable_grad()
        with context:
            Knn_x_ip = knngraph(cfg_ip, feat_x.detach() if detach_basis else feat_x, device)
            Knn_y_ip = knngraph(cfg_ip, feat_y.detach() if detach_basis else feat_y, device)
            Knn_x_l2 = knngraph(cfg_l2, feat_x.detach() if detach_basis else feat_x, device)
            Knn_y_l2 = knngraph(cfg_l2, feat_y.detach() if detach_basis else feat_y, device)

            x_vecs_ip, x_vals_ip = laplacian_construction_decomposition(cfg_ip, Knn_x_ip, device)
            y_vecs_ip, y_vals_ip = laplacian_construction_decomposition(cfg_ip, Knn_y_ip, device)
            _, _, L_x_l2 = laplacian_construction_decomposition(cfg_l2, Knn_x_l2, device, ret_L=True)
            _, _, L_y_l2 = laplacian_construction_decomposition(cfg_l2, Knn_y_l2, device, ret_L=True)

            n_eigens = min(self.cfg.laplacian_mat.k, x_vecs_ip.shape[1], y_vecs_ip.shape[1], feat_x.shape[0], feat_y.shape[0])
            x_basis, x_vals = self._topk_basis(x_vecs_ip, x_vals_ip, n_eigens)
            y_basis, y_vals = self._topk_basis(y_vecs_ip, y_vals_ip, n_eigens)
            x_aux_vals = self._rayleigh_from_aux_laplacian(x_vecs_ip, L_x_l2, n_eigens)
            y_aux_vals = self._rayleigh_from_aux_laplacian(y_vecs_ip, L_y_l2, n_eigens)

        return {
            'feat_x': feat_x.unsqueeze(0),
            'feat_y': feat_y.unsqueeze(0),
            'x_basis': x_basis.detach() if detach_basis else x_basis,
            'y_basis': y_basis.detach() if detach_basis else y_basis,
            'x_vals': x_vals.detach() if detach_basis else x_vals,
            'y_vals': y_vals.detach() if detach_basis else y_vals,
            'x_aux_vals': x_aux_vals.detach() if detach_basis else x_aux_vals,
            'y_aux_vals': y_aux_vals.detach() if detach_basis else y_aux_vals,
        }

    def solve_from_features(self, feat_x: torch.Tensor, feat_y: torch.Tensor, device, detach_basis: bool = True):
        system = self.build_spectral_system(feat_x, feat_y, device, detach_basis=detach_basis)
        Cxy, Cyx = self.fm_net(
            system['feat_x'],
            system['feat_y'],
            system['x_vals'],
            system['y_vals'],
            system['x_basis'],
            system['y_basis'],
            system['x_aux_vals'],
            system['y_aux_vals'],
        )
        system['Cxy'] = Cxy
        system['Cyx'] = Cyx
        return system

    def forward(self, feature_dict_val, device, projector=None):
        feat_x = feature_dict_val[f'{self.cfg.validation.dataset}'][f'{self.cfg.validation.text_model}'].to(device)
        feat_y = feature_dict_val[f'{self.cfg.validation.dataset}'][f'{self.cfg.validation.type}'].to(device)

        if projector is not None:
            was_training = projector.training
            projector.eval()
            with torch.no_grad():
                feat_x = projector(feat_x)
            if was_training:
                projector.train()

        system = self.solve_from_features(feat_x, feat_y, device, detach_basis=True)
        return (
            system['Cxy'].squeeze(0),
            system['Cyx'].squeeze(0),
            system['x_basis'].squeeze(0),
            system['y_basis'].squeeze(0),
        )
