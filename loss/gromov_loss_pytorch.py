# entropic_gw_torch_vs_pot.py
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import ot
import matplotlib.pyplot as plt
import math

# -----------------------------
# Utilities
# -----------------------------
def pairwise_sq_dists_torch(X: torch.Tensor) -> torch.Tensor:
    x2 = (X * X).sum(dim=1, keepdim=True)           # [n, 1]
    C = x2 + x2.t() - 2.0 * (X @ X.t())             # [n, n]
    return C.clamp_min(0.0)


def init_uniform_weights(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        return torch.full((n,), 1.0 / n, dtype=torch.float64)
    return torch.full((n,), 1.0 / n, device=device, dtype=torch.float64)


# -----------------------------
# GW surrogate (square loss)
# -----------------------------
def gw_tensor_square_loss(C1: torch.Tensor, C2: torch.Tensor, a: torch.Tensor, b: torch.Tensor, T: torch.Tensor):
    # All inputs expected as double tensors
    const1 = (C1 * C1) @ a  # [ns]
    const2 = (C2 * C2) @ b  # [nt]
    M = const1[:, None] + const2[None, :] - 2.0 * (C1 @ T @ C2.t())
    gw_loss = (M * T).sum()
    return M, gw_loss


# -----------------------------
# Log-domain Sinkhorn (with warm start)
# -----------------------------
def sinkhorn_log_domain(
    logK: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-9,
    f_init: Optional[torch.Tensor] = None,
    g_init: Optional[torch.Tensor] = None,
    verbose: bool = False,
):
    """
    Inputs:
      logK: [ns, nt] = -M / epsilon  (torch.double recommended)
      a: [ns], b: [nt]  (torch.double)
    Returns:
      T: [ns, nt] transport plan (torch.double, sums to 1)
      f, g: log potentials for warm start
    """
    device = logK.device
    dtype = logK.dtype
    ns, nt = logK.shape

    if f_init is None:
        f = torch.zeros(ns, dtype=dtype, device=device)
    else:
        f = f_init.to(device).clone().to(dtype)

    if g_init is None:
        g = torch.zeros(nt, dtype=dtype, device=device)
    else:
        g = g_init.to(device).clone().to(dtype)

    log_a = torch.log(a.clamp_min(1e-300))
    log_b = torch.log(b.clamp_min(1e-300))

    for it in range(max_iter):
        # update f
        # row-wise: row = logsumexp(logK + g[None, :], dim=1)
        row = torch.logsumexp(logK + g[None, :], dim=1)  # [ns]
        f_new = log_a - row

        # update g
        col = torch.logsumexp(logK + f_new[:, None], dim=0)  # [nt]
        g_new = log_b - col

        # check convergence of potentials
        max_change = max(torch.max(torch.abs(f_new - f)).item(), torch.max(torch.abs(g_new - g)).item())
        f, g = f_new, g_new

        if max_change < tol:
            if verbose:
                print(f"[sinkhorn] converged at iter {it+1}, max_change={max_change:.3e}")
            break

    # reconstruct T in a stable way
    logT = f[:, None] + logK + g[None, :]
    # shift to avoid overflow / underflow
    logT = logT - torch.max(logT)
    T = torch.exp(logT)
    T = T / T.sum()
    return T, f, g


# -----------------------------
# Entropic GW (torch, log-domain Sinkhorn)
# -----------------------------

def entropic_gw_torch(
    C1: torch.Tensor,
    C2: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 100,
    sinkhorn_maxiter: int = 1000,
    tol: float = 1e-9,
    normalize_cost: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Returns:
      T_np: numpy array coupling (ns x nt)
      obj: float objective (gw loss + entropic term)
      log: dict with 'gw_dist' (no entropy), 'obj', 'iters', 'history'
    """
    if device is None:
        device = torch.device("cpu")
    dtype = torch.float64

    C1 = C1.to(device=device, dtype=dtype)
    C2 = C2.to(device=device, dtype=dtype)
    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)

    if normalize_cost:
        C1 = C1 / (C1.max() + 1e-12)
        C2 = C2 / (C2.max() + 1e-12)

    ns = C1.size(0)
    nt = C2.size(0)

    # init T as outer product
    T = torch.outer(a, b).to(device=device, dtype=dtype)

    # warm start potentials
    f = None
    g = None

    history = []

    for it in range(max_iter):
        M, _ = gw_tensor_square_loss(C1, C2, a, b, T)  # [ns, nt]
        logK = (-M / float(epsilon)).to(dtype=dtype, device=device)
        T_new, f, g = sinkhorn_log_domain(logK, a, b, max_iter=sinkhorn_maxiter, tol=tol, f_init=f, g_init=g, verbose=False)

        delta = torch.norm(T_new - T, p=1).item()
        T = T_new

        # compute gw_loss (no entropy)
        gw_loss = (M * T).sum().item()
        ent = (T * (torch.log(T.clamp_min(1e-300)) - 1.0)).sum().item()
        obj = gw_loss + epsilon * ent

        history.append({'iter': it + 1, 'gw_loss': gw_loss, 'entropy_term': (epsilon * ent), 'obj': obj, 'delta': delta})


        if delta < tol:
            break

    # final
    M_final, gw_loss_final = gw_tensor_square_loss(C1, C2, a, b, T)
    ent_final = (T * (torch.log(T.clamp_min(1e-300)) - 1.0)).sum().item()
    obj_final = gw_loss_final.item() + epsilon * ent_final

    log = {'gw_dist': float(gw_loss_final.item()), 'obj': float(obj_final), 'iters': it + 1, 'history': history}
    T_torch = T.detach()
    return T_torch, float(obj_final), log

def draw(X_s, X_t, gw_coupling, log, savepth):
    plt.figure(figsize=(15, 5))

    # 4.1 绘制第一个点集
    plt.subplot(131)
    plt.scatter(X_s[:, 0], X_s[:, 1], c='blue', s=50, alpha=0.7, label='Source distribution')
    plt.title('Source distribution')
    plt.legend()

    # 4.2 绘制第二个点集
    plt.subplot(132)
    plt.scatter(X_t[:, 0], X_t[:, 1], c='red', s=50, alpha=0.7, label='Target distribution')
    plt.title('Target distribution')
    plt.legend()

    # 4.3 绘制耦合（匹配）关系
    plt.subplot(133)
    # 绘制点集
    plt.scatter(X_s[:, 0], X_s[:, 1], c='blue', s=50, alpha=0.7, label='Source')
    plt.scatter(X_t[:, 0], X_t[:, 1], c='red', s=50, alpha=0.7, label='Target')

    # 绘制连接线：只显示耦合强度大于阈值的匹配
    threshold = 0.01  # 耦合强度的阈值
    for i in range(n_samples):
        for j in range(n_samples):
            if gw_coupling[i, j] > threshold:
                plt.plot([X_s[i, 0], X_t[j, 0]], 
                        [X_s[i, 1], X_t[j, 1]], 
                        'k-', alpha=gw_coupling[i, j] * 5, linewidth=1)

    plt.title('Coupling (Matching) with Entropic Regularization GW\n' +
            f'(epsilon={epsilon}, GW-distance={log["gw_dist"]:.3f})')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f'/20230031/code/LFMTrans/loss/tmp/{savepth}.png')



# -----------------------------
# Example / compare with POT
# -----------------------------
if __name__ == "__main__":
    # generate the same data as you used
    n_samples = 30
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_s = np.random.randn(n_samples, 2) * 0.5
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    X_t = np.dot(np.random.randn(n_samples, 2) * 0.6, rotation_matrix.T) + np.array([3, 3])

    # POT versions (numpy)
    a_np = np.ones(n_samples) / n_samples
    b_np = np.ones(n_samples) / n_samples
    C1_np = ot.dist(X_s, X_s, metric='sqeuclidean')
    C2_np = ot.dist(X_t, X_t, metric='sqeuclidean')

    # run POT
    epsilon = 0.05
    gw_coupling_pot, log_pot = ot.gromov.entropic_gromov_wasserstein(
        C1_np, C2_np, a_np, b_np, 'square_loss', epsilon=epsilon, max_iter=100, log=True
    )

    # run torch implementation (cpu double)
    Xs_t = torch.from_numpy(X_s).double().to(device)
    Xt_t = torch.from_numpy(X_t).double().to(device)
    C1_t = pairwise_sq_dists_torch(Xs_t).to(device)
    C2_t = pairwise_sq_dists_torch(Xt_t).to(device)
    a_t = torch.from_numpy(a_np).double().to(device)
    b_t = torch.from_numpy(b_np).double().to(device)

    # try with normalize_cost False first; set True if POT did normalization
    T_torch, obj_torch, log_torch = entropic_gw_torch(
        C1_t, C2_t, a_t, b_t, epsilon=epsilon, max_iter=100, sinkhorn_maxiter=1000,
        tol=1e-9, normalize_cost=False, device=device, verbose=True
    )

    # SGW implement
    from gromov_loss_SGW import SGW
    SGW_implement = SGW()
    loss = SGW_implement(Xs_t, Xt_t)

    # Compare couplings
    T_torch_np = T_torch
    T_pot_np = gw_coupling_pot

    draw(X_s, X_t, T_torch.to('cpu').numpy(), log_torch, 'torch')
    draw(X_s, X_t, T_pot_np, log_pot, 'ot')

    breakpoint()