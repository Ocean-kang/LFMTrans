# code adapted from original EOT-Correspondence implementation https://github.com/Tungthanhlee/EOT-Correspondence

import torch
import torch.nn as nn


def _safe_evals(evals: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(evals, nan=eps, posinf=1e6, neginf=eps), min=eps)


def _get_mask(evals1, evals2, resolvant_gamma):
    evals1 = _safe_evals(evals1)
    evals2 = _safe_evals(evals2)

    scaling_factor = torch.maximum(evals1.max(), evals2.max())
    scaling_factor = torch.clamp(scaling_factor, min=1e-10)
    evals1 = evals1 / scaling_factor
    evals2 = evals2 / scaling_factor

    evals_gamma1 = evals1.pow(resolvant_gamma)[None, :]
    evals_gamma2 = evals2.pow(resolvant_gamma)[:, None]

    denom1 = evals_gamma1.square() + 1.0
    denom2 = evals_gamma2.square() + 1.0
    M_re = evals_gamma2 / denom2 - evals_gamma1 / denom1
    M_im = 1.0 / denom2 - 1.0 / denom1
    return torch.nan_to_num(M_re.square() + M_im.square(), nan=0.0, posinf=1e6, neginf=0.0)


def get_mask(evals1, evals2, resolvant_gamma):
    evals1 = _safe_evals(evals1)
    evals2 = _safe_evals(evals2)
    masks = [_get_mask(evals1[bs], evals2[bs], resolvant_gamma) for bs in range(evals1.shape[0])]
    return torch.stack(masks, dim=0)


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM."""
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super().__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        A = torch.bmm(evecs_trans_x, feat_x)
        B = torch.bmm(evecs_trans_y, feat_y)

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)
        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype).unsqueeze(0)
        rows = []
        for i in range(evals_x.shape[1]):
            D_i = torch.diag_embed(D[:, i, :])
            lhs = A_A_t + self.lmbda * D_i + 1e-8 * eye
            rhs = B_A_t[:, i, :].unsqueeze(-1)
            row = torch.linalg.solve(lhs, rhs).transpose(1, 2)
            rows.append(row)
        return torch.cat(rows, dim=1)

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x) if self.bidirectional else None
        return Cxy, Cyx


class LatentFunctionMap(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reg = float(getattr(getattr(cfg, 'fmap', {}), 'reg', 1e-6))

    def compute_functional_map_batch(self, A: torch.Tensor, B: torch.Tensor, reg: float = None) -> torch.Tensor:
        """Solve min_C ||CA - B||_F^2 + reg ||C||_F^2 for batched spectral coefficients."""
        if reg is None:
            reg = self.reg
        assert A.dim() == 3 and B.dim() == 3, 'A/B must be 3D tensors [B, K, C]'
        assert A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1], 'batch and spectral dims must match'

        k = A.shape[1]
        eye = torch.eye(k, device=A.device, dtype=A.dtype).unsqueeze(0)
        AtA = torch.bmm(A, A.transpose(1, 2)) + reg * eye
        BAt = torch.bmm(B, A.transpose(1, 2))
        return torch.linalg.solve(AtA.transpose(1, 2), BAt.transpose(1, 2)).transpose(1, 2)

    def forward(self, feat_x, feat_y, evecs_x, evecs_y):
        A = torch.bmm(evecs_x, feat_x)
        B = torch.bmm(evecs_y, feat_y)
        Cxy = self.compute_functional_map_batch(A, B)
        Cyx = self.compute_functional_map_batch(B, A)
        return Cxy, Cyx


class MultiConstraintFM(nn.Module):
    """Single-basis functional map with an auxiliary graph used only as a spectral regularizer."""
    def __init__(self, lmbda: float = 100.0, aux_lmbda: float = 25.0, reg: float = 1e-6,
                 resolvant_gamma: float = 0.5, bidirectional: bool = True):
        super().__init__()
        self.lmbda = lmbda
        self.aux_lmbda = aux_lmbda
        self.reg = reg
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def _solve(self, A, B, evals_x, evals_y, aux_evals_x=None, aux_evals_y=None):
        D_main = get_mask(evals_x, evals_y, self.resolvant_gamma)
        D_aux = get_mask(aux_evals_x, aux_evals_y, self.resolvant_gamma) if aux_evals_x is not None and aux_evals_y is not None else None

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)
        eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype).unsqueeze(0)

        rows = []
        for i in range(A.shape[1]):
            reg_i = self.lmbda * torch.diag_embed(D_main[:, i, :])
            if D_aux is not None:
                reg_i = reg_i + self.aux_lmbda * torch.diag_embed(D_aux[:, i, :])
            lhs = A_A_t + reg_i + self.reg * eye
            rhs = B_A_t[:, i, :].unsqueeze(-1)
            row = torch.linalg.solve(lhs, rhs).transpose(1, 2)
            rows.append(row)
        return torch.cat(rows, dim=1)

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_x, evecs_y, aux_evals_x=None, aux_evals_y=None):
        A = torch.bmm(evecs_x, feat_x)
        B = torch.bmm(evecs_y, feat_y)
        Cxy = self._solve(A, B, evals_x, evals_y, aux_evals_x, aux_evals_y)
        Cyx = self._solve(B, A, evals_y, evals_x, aux_evals_y, aux_evals_x) if self.bidirectional else None
        return Cxy, Cyx


class MultiConstraintDFM(nn.Module):
    """Deep FM variant kept for backward compatibility."""
    def __init__(self, reg: float = 1e-6, alpha: float = 1.0, bidirectional: bool = True):
        super().__init__()
        self.reg = reg
        self.alpha = alpha
        self.bidirectional = bidirectional

    def compute_fm_batch(self, A_ip, B_ip, aux_X=None, aux_Y=None, alpha=1.0, reg=1e-6):
        B, K_ip, _ = A_ip.shape
        if aux_X is not None and aux_Y is not None:
            A_aug = torch.cat([A_ip, alpha * aux_X], dim=2)
            B_aug = torch.cat([B_ip, alpha * aux_Y], dim=2)
        else:
            A_aug, B_aug = A_ip, B_ip

        reg_eye = reg * torch.eye(K_ip, device=A_ip.device, dtype=A_ip.dtype).unsqueeze(0)
        AtA = torch.bmm(A_aug, A_aug.transpose(1, 2)) + reg_eye
        BAt = torch.bmm(B_aug, A_aug.transpose(1, 2))
        return torch.linalg.solve(AtA.transpose(1, 2), BAt.transpose(1, 2)).transpose(1, 2)

    def forward(self, feat_x, feat_y, evecs_ip_x, evecs_ip_y, evecs_l2_x=None, evecs_l2_y=None):
        A_ip = torch.bmm(evecs_ip_x, feat_x)
        B_ip = torch.bmm(evecs_ip_y, feat_y)
        if evecs_l2_x is not None and evecs_l2_y is not None:
            A_l2 = torch.bmm(evecs_l2_x, feat_x)
            B_l2 = torch.bmm(evecs_l2_y, feat_y)
        else:
            A_l2 = B_l2 = None
        Cxy = self.compute_fm_batch(A_ip, B_ip, A_l2, B_l2, alpha=self.alpha, reg=self.reg)
        Cyx = self.compute_fm_batch(B_ip, A_ip, B_l2, A_l2, alpha=self.alpha, reg=self.reg) if self.bidirectional else None
        return Cxy, Cyx
