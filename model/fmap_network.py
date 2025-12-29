# code adapted from original EOT-Correspondence implementation https://github.com/Tungthanhlee/EOT-Correspondence

import torch
import torch.nn as nn

# def _get_mask(evals1, evals2, resolvant_gamma):
#     scaling_factor = max(torch.max(evals1), torch.max(evals2))
#     evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
#     evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
#     evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

#     M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
#     M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
#     return M_re.square() + M_im.square()


# def get_mask(evals1, evals2, resolvant_gamma):
#     masks = []
#     for bs in range(evals1.shape[0]):
#         masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
#     return torch.stack(masks, dim=0)

def _get_mask(evals1, evals2, resolvant_gamma):
    """专项修复：处理极小负数导致的开平方NaN"""
    # ========== 核心修复：极小负数→非负极小值（针对第一个元素） ==========
    # 阈值：绝对值<1e-6的数视为"数值误差导致的0"，强制置为1e-10
    eps = 1e-6
    evals1 = torch.where(torch.abs(evals1) < eps, torch.tensor(1e-10, device=evals1.device), evals1)
    evals2 = torch.where(torch.abs(evals2) < eps, torch.tensor(1e-10, device=evals2.device), evals2)
    # 二次兜底：确保无负数（拉普拉斯特征值理论上≥0）
    evals1 = torch.clamp(evals1, min=1e-10)
    evals2 = torch.clamp(evals2, min=1e-10)

    # ========== 原有逻辑（保留） ==========
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    
    # ========== 幂运算前再次防护（避免遗漏） ==========
    evals1_pow = evals1 ** resolvant_gamma
    evals2_pow = evals2 ** resolvant_gamma
    # 清理幂运算后可能的NaN/Inf（兜底）
    evals1_pow = torch.nan_to_num(evals1_pow, nan=1e-10, posinf=1e6, neginf=1e-10)
    evals2_pow = torch.nan_to_num(evals2_pow, nan=1e-10, posinf=1e6, neginf=1e-10)

    evals_gamma1 = evals1_pow[None, :]  # (1, K)
    evals_gamma2 = evals2_pow[:, None]  # (K, 1)

    # ========== 分母运算防护（避免Inf/Inf→NaN） ==========
    denom1 = evals_gamma1.square() + 1
    denom2 = evals_gamma2.square() + 1
    # 强制分母≥1e-10，避免除以0
    denom1 = torch.clamp(denom1, min=1e-10)
    denom2 = torch.clamp(denom2, min=1e-10)

    M_re = evals_gamma2 / denom2 - evals_gamma1 / denom1
    M_im = 1 / denom2 - 1 / denom1
    
    # ========== 最终输出清理 ==========
    mask = M_re.square() + M_im.square()
    mask = torch.nan_to_num(mask, nan=0.0, posinf=1e6, neginf=0.0)
    return mask

def get_mask(evals1, evals2, resolvant_gamma):
    masks = []
    # 先整体处理输入的极小负数
    eps = 1e-6
    evals1 = torch.where(torch.abs(evals1) < eps, torch.tensor(1e-10, device=evals1.device), evals1)
    evals2 = torch.where(torch.abs(evals2) < eps, torch.tensor(1e-10, device=evals2.device), evals2)
    evals1 = torch.clamp(evals1, min=1e-10)
    evals2 = torch.clamp(evals2, min=1e-10)

    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    masks = torch.stack(masks, dim=0)
    masks = torch.nan_to_num(masks, nan=0.0, posinf=1e6, neginf=0.0)
    return masks

class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
        return Cxy

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        if self.bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        else:
            Cyx = None

        return Cxy, Cyx
    

# FunctionMap by Myself

class LatentFunctionMap(nn.Module):
    def __init__(self, cfg):
        super(LatentFunctionMap, self).__init__()
        self.cfg = cfg
    
    def compute_functional_map(
        self,
        Phi_X: torch.Tensor,  # (n, k)
        Phi_Y: torch.Tensor,  # (m, k)
        reg: float = 1e-6
    ):
        """
        Solve: min_C || C Phi_X^T - Phi_Y^T ||_F^2 + reg ||C||_F^2
        """
        # (k, n)
        A = Phi_X.T
        # (k, m)
        B = Phi_Y.T

        # Solve C A A^T = B A^T
        AtA = A @ A.T
        AtA = AtA + reg * torch.eye(AtA.shape[0], device=AtA.device)

        C = (B @ A.T) @ torch.linalg.inv(AtA)
        return C
    
    def compute_functional_map_batch(
        self,
        Phi_X: torch.Tensor,  # (B, n, k) 批量特征矩阵X: B=batch size, n=X节点数, k=谱特征维度
        Phi_Y: torch.Tensor,  # (B, m, k) 批量特征矩阵Y: B=batch size, m=Y节点数, k=谱特征维度
        reg: float = 1e-6
    ):
        """
        批量版本Functional Map计算:
        Solve for each batch: min_C || C Phi_X^T - Phi_Y^T ||_F^2 + reg ||C||_F^2
        
        Args:
            Phi_X: (B, n, k) 批量X的谱特征矩阵(每行是节点的谱特征)
            Phi_Y: (B, m, k) 批量Y的谱特征矩阵
            reg: 正则化系数,防止AtA奇异
        
        Returns:
            C: (B, m, n) 批量Functional Map矩阵,每个batch对应一个m*n的C
        """
        # 1. 校验输入维度（保证批量维度一致）
        assert Phi_X.dim() == 3 and Phi_Y.dim() == 3, "Phi_X/Phi_Y must be 3D (B, num_nodes, k)"
        assert Phi_X.shape[0] == Phi_Y.shape[0], "Batch size of Phi_X and Phi_Y must match"
        assert Phi_X.shape[2] == Phi_Y.shape[2], "Spectral dimension k of Phi_X and Phi_Y must match"
        
        B = Phi_X.shape[0]  # batch size
        k = Phi_X.shape[2]  # 谱特征维度
        device = Phi_X.device
         
        # 重新实现正确的批量逻辑：
        PhiX_T = Phi_X.transpose(1, 2)  # (B, k, n)
        PhiX_PhiX_T = torch.bmm(Phi_X, PhiX_T)  # (B, n, k) @ (B, k, n) → (B, n, n)
        # 批量正则化单位矩阵 (B, n, n)
        eye_n = torch.eye(Phi_X.shape[1], device=device).unsqueeze(0).repeat(B, 1, 1)
        PhiX_PhiX_T_reg = PhiX_PhiX_T + reg * eye_n
        # 计算 Phi_Y @ Phi_X^T (B, m, k) @ (B, k, n) → (B, m, n)
        PhiY_PhiX_T = torch.bmm(Phi_Y, PhiX_T)
        # 批量求逆 + 最终计算C
        PhiX_PhiX_T_inv = torch.linalg.inv(PhiX_PhiX_T_reg)
        C = torch.bmm(PhiY_PhiX_T, PhiX_PhiX_T_inv)  # (B, m, n)
        
        return C
    
    def forward(self, feat_x, feat_y, evecs_x, evecs_y):
        A = torch.bmm(evecs_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_y, feat_y)  # [B, K, C]
        Cxy = self.compute_functional_map_batch(A, B)
        Cyx = self.compute_functional_map_batch(B, A)

        return Cxy, Cyx