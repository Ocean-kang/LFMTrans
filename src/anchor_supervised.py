import torch
import torch.nn as nn

from utils.KnnGraph import knngraph
from utils.shuffle_utils import sample_features_per_class_coco
from utils.laplacian_utils import laplacian_construction_decomposition
from utils.anchors_utils import anchor_embeddings_compute_supervised
from model.fmap_network import RegularizedFMNet, LatentFunctionMap


class LFMAnchor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _topk_basis(evecs: torch.Tensor, evals: torch.Tensor, k: int):
        basis = evecs[:, :k].T.contiguous().unsqueeze(0)
        spectrum = evals[:k].contiguous().unsqueeze(0)
        return basis, spectrum

    def forward(self, feature_dict_train, feature_dict_val, device):
        feat_t_unmean = feature_dict_train[f'{self.cfg.train.dataset}'][f'{self.cfg.train.text_model}']
        feat_v_unmean = feature_dict_train[f'{self.cfg.train.dataset}'][f'{self.cfg.train.type}']
        feat_t, labels_t = sample_features_per_class_coco(feat_t_unmean, self.cfg.train.sample, self.cfg.seed)
        feat_v, labels_v = sample_features_per_class_coco(feat_v_unmean, self.cfg.train.sample, self.cfg.seed)

        feat_t = feat_t.to(device)
        feat_v = feat_v.to(device)
        feat_t_unmean = feat_t_unmean.to(device)
        feat_v_unmean = feat_v_unmean.to(device)

        Knn_v = knngraph(self.cfg, feat_v, device)
        Knn_t = knngraph(self.cfg, feat_t, device)
        v_vecs, v_vals = laplacian_construction_decomposition(self.cfg, Knn_v, device)
        t_vecs, t_vals = laplacian_construction_decomposition(self.cfg, Knn_t, device)

        feat_v_anchor, feat_t_anchor = anchor_embeddings_compute_supervised(
            self.cfg, feat_v, feat_v_unmean, feat_t, feat_t_unmean
        )

        n_eigens = min(self.cfg.laplacian_mat.k, v_vecs.shape[1], t_vecs.shape[1])
        v_basis, v_vals = self._topk_basis(v_vecs, v_vals, n_eigens)
        t_basis, t_vals = self._topk_basis(t_vecs, t_vals, n_eigens)
        feat_v_anchor = feat_v_anchor.unsqueeze(0).to(device)
        feat_t_anchor = feat_t_anchor.unsqueeze(0).to(device)

        if self.cfg.fmap.method == 'EOT':
            fm_net = RegularizedFMNet(bidirectional=True)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_anchor, v_vals, t_vals, v_basis, t_basis)
        elif self.cfg.fmap.method == 'LFMtrans':
            fm_net = LatentFunctionMap(cfg=self.cfg)
            Cxy, Cyx = fm_net(feat_v_anchor, feat_t_anchor, v_basis, t_basis)
        else:
            raise ValueError(f'Unsupported fmap.method: {self.cfg.fmap.method}')

        return (
            Cxy.squeeze(0),
            Cyx.squeeze(0),
            v_basis.squeeze(0),
            v_vals.squeeze(0),
            t_basis.squeeze(0),
            t_vals.squeeze(0),
            labels_t,
            labels_v,
        )
