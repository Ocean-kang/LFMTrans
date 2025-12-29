import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment as linear_assignment

# GPU(Pytorch) Version
def build_laplacian_matrix(W: torch.Tensor, normalize: str = 'none', device: str = 'cpu') -> torch.Tensor:

    # Input validation
    if not isinstance(W, torch.Tensor):
        raise ValueError("W must be a torch.Tensor")
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    if torch.any(W < 0):
        import warnings
        warnings.warn("Weight matrix W contains negative values, which may lead to unexpected results", UserWarning)
    
    W = W.to(device)
    
    d = torch.sum(W, dim=1)  # [n]
    n = W.shape[0]
    
    d_safe = d.clone()
    d_safe[d_safe == 0] = 1e-10
    
    I = torch.eye(n, device=device)
    
    if normalize == 'none':
        # L = D - W
        D = torch.diag(d)
        L = D - W
        return L
    
    elif normalize == 'sym':
        # L_sym = I - D^{-1/2} W D^{-1/2}
        d_inv_sqrt = 1.0 / torch.sqrt(d_safe)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        D_inv_sqrt_W_D_inv_sqrt = torch.mm(D_inv_sqrt, torch.mm(W, D_inv_sqrt))
        L_sym = I - D_inv_sqrt_W_D_inv_sqrt
        return L_sym
    
    elif normalize == 'rw':
        # L_rw = I - D^{-1} W
        d_inv = 1.0 / d_safe
        D_inv = torch.diag(d_inv)
        D_inv_W = torch.mm(D_inv, W)
        L_rw = I - D_inv_W
        return L_rw
    
    else:
        raise ValueError("normalize parameter must be 'none', 'sym' or 'rw'")

# Lcompute
def L_decomposition_ip(X, metric_c, metric_l, device):

    assert metric_c in ['raw', 'Heat', 'knn'], f"Unsupported metric: {metric_c}"
    assert metric_l in ['none', 'sym', 'rw'], f"Unsupported metric: {metric_l}"

    X = X.to(device)
    X = F.normalize(X, p=2, dim=1)
    weight_matrix = X @ X.T
    # Cost Matrix construction
    if metric_c == "knn":
        k = 10
        values, indices = torch.topk(weight_matrix, k=k, dim=1)
        W = torch.zeros_like(weight_matrix)
        W.scatter_(1, indices, values)
        cost_matrix = (W + W.T) / 2
    elif metric_c == "Heat":
        sigma = weight_matrix.mean()
        cost_matrix = torch.exp(weight_matrix / sigma)
    else:
        cost_matrix = weight_matrix

    # Laplacian Matrix Construction
    L_matrix = build_laplacian_matrix(cost_matrix, metric_l, device)
    # L Matrix decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L_matrix)
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvectors, eigenvalues

def L_decomposition_L2(X, metric_c, metric_l, device):
    assert metric_c in ['knn'], f"Unsupported metric: {metric_c}"
    assert metric_l in ['none', 'sym', 'rw'], f"Unsupported metric: {metric_l}"

    X = X.to(device)
    # weight matrix construction
    # Normalize
    X = F.normalize(X, dim=1)
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    sigma = torch.median(dist_sq)
    weight_matrix = torch.exp(- dist_sq / (2 * sigma))

    # Cost Matrix construction
    k = 10
    values, indices = torch.topk(weight_matrix, k=k, dim=1)
    W = torch.zeros_like(weight_matrix)
    W.scatter_(1, indices, values)
    cost_matrix = (W + W.T) / 2

    # Laplacian Matrix Construction
    L_matrix = build_laplacian_matrix(cost_matrix, metric_l, device)
    # L Matrix decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L_matrix)
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvectors, eigenvalues

# Fiedler Vector Cluster Method
def threshold_split(fiedler_vector: torch.Tensor, method='zero'):
    """
    方法:
        'zero': 阈值为0
        'median': 中位数分割
        'mean': 均值分割
    """
    if method == 'zero':
        threshold = 0.0
    elif method == 'median':
        threshold = torch.median(fiedler_vector)
    elif method == 'mean':
        threshold = torch.mean(fiedler_vector)
    
    labels = (fiedler_vector > threshold).long()
    return labels

def spectral_clustering_ip(X, metric_w: str='ip', metric_c: str='raw', metric_l: str='none', device: str='cpu'):

    assert metric_w in ['ip'], f"Unsupported metric: {metric_w}"
    assert metric_c in ['raw', 'Heat', 'knn'], f"Unsupported metric: {metric_c}"
    assert metric_l in ['none', 'sym', 'rw'], f"Unsupported metric: {metric_l}"

    eigenvectors, eigenvalues = L_decomposition_ip(X, metric_c, metric_l, device)

    # Clustering
    fiedler_vector = eigenvectors[:, 1]

    # Change Direction
    idx = torch.argmax(torch.abs(fiedler_vector))
    if fiedler_vector[idx] < 0:
        fiedler_vector = -fiedler_vector

    # Generate Labels
    labels = threshold_split(fiedler_vector, method='median')
    return labels

def spectral_clustering_L2(X, metric_w: str='L2', metric_c: str='raw', metric_l: str='none', device: str='cpu'):

    assert metric_w in ['L2'], f"Unsupported metric: {metric_w}"
    assert metric_c in ['knn'], f"Unsupported metric: {metric_c}"
    assert metric_l in ['none', 'sym', 'rw'], f"Unsupported metric: {metric_l}"

    eigenvectors, eigenvalues = L_decomposition_L2(X, metric_c, metric_l, device)

    # Clustering
    fiedler_vector = eigenvectors[:, 1]

    # Change Direction
    idx = torch.argmax(torch.abs(fiedler_vector))
    if fiedler_vector[idx] < 0:
        fiedler_vector = -fiedler_vector

    # Generate Labels
    labels = threshold_split(fiedler_vector, method='median')
    return labels

def spectral_clustering(X, metric_w: str='L2', metric_c: str='raw', metric_l: str='none', device: str='cpu'):
     
    assert metric_w in ['L2', 'ip'], f"Unsupported metric: {metric_w}"

    if metric_w == 'L2':
        return spectral_clustering_L2(X, metric_w, metric_c, metric_l, device)
    elif metric_w == 'ip':
        return spectral_clustering_ip(X, metric_w, metric_c, metric_l, device)
    
# FunctionMap
def compute_functional_map(
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

def functional_map_to_labels(
    Phi_X: torch.Tensor,  # (n, k)
    Phi_Y: torch.Tensor,  # (m, k)
    C: torch.Tensor
):
    """
    Returns:
        labels: (n,) each x maps to an index in Y
    """
    # Map X embedding to Y space
    mapped_X = Phi_X @ C.T  # (n, k)

    # Normalize for cosine stability
    mapped_X = torch.nn.functional.normalize(mapped_X, dim=1)
    Phi_Y = torch.nn.functional.normalize(Phi_Y, dim=1)

    # Pairwise distances
    dist = torch.cdist(mapped_X, Phi_Y, p=2)  # (n, m)

    labels = dist.argmin(dim=1)
    return labels

def functional_map_pipline(
    eigenvecs_X,
    eigenvecs_Y,
    k=70,
):
    Phi_X = eigenvecs_X[:, :k]
    Phi_Y = eigenvecs_Y[:, :k]

    C = compute_functional_map(Phi_X, Phi_Y)
    labels = functional_map_to_labels(Phi_X, Phi_Y, C)
    return labels, C


def zoomout_functional_map_strict(
    Phi_X,          # (n, Kx)
    Phi_Y,          # (m, Ky)
    k_start=10,
    k_max=None,
    step=5,
    device='cpu'
):
    """
    Strict (canonical) ZoomOut refinement.
    Returns:
        labels : (n,) hard point-to-point map X -> Y
        C      : (k_max, k_max) refined functional map
    """

    Phi_X = Phi_X.to(device)
    Phi_Y = Phi_Y.to(device)

    n, Kx = Phi_X.shape
    m, Ky = Phi_Y.shape

    if k_max is None:
        k_max = min(Kx, Ky)

    # --------------------------------------------------
    # 1. 初始化低频 Functional Map（正交）
    # --------------------------------------------------
    k = k_start
    Phi_X_k = Phi_X[:, :k]
    Phi_Y_k = Phi_Y[:, :k]

    # canonical initialization: identity (or plug in your fm_net output here)
    C = torch.eye(k, device=device)

    # --------------------------------------------------
    # 2. ZoomOut 主循环
    # --------------------------------------------------
    while k < k_max:
        k_next = min(k + step, k_max)

        # ---------- (a) fmap -> pointmap (当前频率) ----------
        Phi_X_k = Phi_X[:, :k]
        Phi_Y_k = Phi_Y[:, :k]

        mapped_X = Phi_X_k @ C.T              # (n, k)
        dist = torch.cdist(mapped_X, Phi_Y_k) # (n, m)
        nn_idx = dist.argmin(dim=1)            # (n,)

        # 构造点映射矩阵 P (Y <- X)
        P = torch.zeros(m, n, device=device)
        P[nn_idx, torch.arange(n)] = 1.0

        # ---------- (b) 扩频后 pointmap -> fmap ----------
        Phi_X_k_next = Phi_X[:, :k_next]
        Phi_Y_k_next = Phi_Y[:, :k_next]

        C = Phi_Y_k_next.T @ P @ Phi_X_k_next  # (k_next, k_next)

        # ---------- (c) 正交化（关键） ----------
        U, _, Vt = torch.linalg.svd(C, full_matrices=False)
        C = U @ Vt

        k = k_next

    # --------------------------------------------------
    # 3. 最终点映射（X -> Y）
    # --------------------------------------------------
    mapped_X = Phi_X[:, :k_max] @ C.T
    dist = torch.cdist(mapped_X, Phi_Y[:, :k_max])
    labels = dist.argmin(dim=1)

    return labels, C

def main_pipline(X, Y, metric_w: str='L2', metric_c: str='raw', metric_l: str='none', device: str='cpu', zoom: bool=False):
    assert metric_w in ['L2', 'ip'], f"Unsupported metric: {metric_w}"
    assert metric_c in ['raw', 'Heat', 'knn'], f"Unsupported metric: {metric_c}"
    assert metric_l in ['none', 'sym', 'rw'], f"Unsupported metric: {metric_l}"

    vec_X, val_X = L_decomposition_ip(X, metric_c, metric_l, device)
    vec_Y, val_Y = L_decomposition_ip(Y, metric_c, metric_l, device)

    if zoom:
        labels, C = zoomout_functional_map_strict(vec_X, vec_Y, device=device)
    else:
        labels, C = functional_map_pipline(vec_X, vec_Y)
    return labels, C

def compute_pointwise_map_hungarian(C, Vec_X, Vec_Y, X, Y, normalize=True, device='cpu'):
    """
    Compute strict one-to-one pointwise correspondence P using Hungarian algorithm.

    Args:
        C (torch.Tensor): Functional map (k x k)
        Vec_X (torch.Tensor): Source spectral basis (n x k)
        Vec_Y (torch.Tensor): Target spectral basis (m x k)
        X (torch.Tensor): Source features (n x d)
        Y (torch.Tensor): Target features (m x d)
        normalize (bool): Whether to L2 normalize features
        device (str): 'cpu' or 'cuda'

    Returns:
        P (torch.Tensor): Binary correspondence matrix (n x m), one-to-one
        row_ind, col_ind: Hungarian matching indices
    """
    # Move to device
    X, Y, Vec_X, Vec_Y, C = X.to(device), Y.to(device), Vec_X.to(device), Vec_Y.to(device), C.to(device)

    # Step 1: Project source features to spectral space
    F_X_S = Vec_X.T @ X  # (k, d)

    # Step 2: Map to target spectral space
    F_Y_S_hat = C @ F_X_S  # (k, d)

    # Step 3: Map back to target point space
    X_mapped = Vec_Y @ F_Y_S_hat  # (m, d)

    # Step 4: Optional L2 normalization
    if normalize:
        X_mapped = X_mapped / (X_mapped.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    else:
        Y_norm = Y

    # Step 5: Compute similarity matrix
    sim_matrix = X_mapped @ Y_norm.T  # (m, m)

    # Step 6: Hungarian algorithm requires a cost matrix
    cost_matrix = -sim_matrix.cpu().numpy()  # maximize similarity -> minimize -sim

    row_ind, col_ind = linear_assignment(cost_matrix)

    # Step 7: Build binary P
    n, m = X.shape[0], Y.shape[0]
    P = torch.zeros((n, m), device=device, dtype=torch.int)
    P[row_ind, col_ind] = 1

    return P

def LatentFunctionMap(F_X: torch.Tensor, F_Y: torch.Tensor, metric_c: str='raw', metric_l: str='none', k: int=30, device: str='cpu'):

    F_X = (F_X / F_X.norm(dim=1, keepdim=True)).to(device).float()
    F_Y = (F_Y / F_Y.norm(dim=1, keepdim=True)).to(device).float()

    vec_X, val_X = L_decomposition_ip(F_X, metric_c, metric_l, device)
    vec_Y, val_Y = L_decomposition_ip(F_Y, metric_c, metric_l, device)

    # Proj to Spectral Space
    F_X_S = vec_X[:,:k].T @ F_X
    F_Y_S = vec_Y[:,:k].T @ F_Y

    # FunctionMap Compute
    C = compute_functional_map(F_X_S.T, F_Y_S.T)

    P= compute_pointwise_map_hungarian(C, vec_X[:,:k], vec_Y[:,:k], F_X, F_Y, normalize=True, device=device)
    labels = P.argmax(dim=1)

    return labels, C

# Ortho
def apply_ortho_L2(P, X, Y, k=5):
    """
    简单 Ortho 修正硬匹配 P,使匹配更平滑几何一致
    Args:
        P: (n, m) 硬匹配矩阵,P[i,j]=1表示X[i]->Y[j]
        X: (n, d) 源点特征或坐标
        Y: (m, d) 目标点特征或坐标
        k: 邻居数量，用于局部平滑

    Returns:
        P_ortho: 修正后的硬匹配矩阵
    """
    device = P.device
    n, m = P.shape
    P_ortho = P.clone()

    # Step 1: 得到每个 X 对应的 Y 索引
    labels = P.argmax(dim=1)  # labels[i] = j 表示 X[i] -> Y[j]

    # Step 2: 构建源点的局部邻居索引 (基于欧氏距离)
    with torch.no_grad():
        # 计算 X 的 pairwise 距离
        dist_X = torch.cdist(X, X)  # (n, n)
        _, neighbors = dist_X.topk(k+1, largest=False)  # 每个点的最近 k+1 个邻居
        neighbors = neighbors[:, 1:]  # 排除自己

        # Step 3: 局部平滑
        for i in range(n):
            # X[i] 的邻居在 Y 中对应的索引
            neigh_labels = labels[neighbors[i]]  # shape: (k,)
            # X[i] 当前匹配的 Y 点
            current = labels[i]
            # 找邻居最常见的 Y 点索引
            vals, counts = torch.unique(neigh_labels, return_counts=True)
            majority = vals[counts.argmax()]
            # 如果当前匹配不在邻居主流中，可以替换
            if current != majority:
                labels[i] = majority

    # Step 4: 构建新的 P_ortho
    P_ortho.zero_()
    P_ortho[torch.arange(n, device=device), labels] = 1

    return P_ortho, labels

def apply_ortho_ip(P, X, Y, k=5):
    """
    简单 Ortho 修正硬匹配 P,使匹配更平滑几何一致
    Args:
        P: (n, m) 硬匹配矩阵,P[i,j]=1表示X[i]->Y[j]
        X: (n, d) 源点特征或坐标
        Y: (m, d) 目标点特征或坐标
        k: 邻居数量，用于局部平滑

    Returns:
        P_ortho: 修正后的硬匹配矩阵
    """
    device = P.device
    n, m = P.shape
    P_ortho = P.clone()

    # Step 1: 得到每个 X 对应的 Y 索引
    labels = P.argmax(dim=1)  # labels[i] = j 表示 X[i] -> Y[j]

    # Step 2: 构建源点的局部邻居索引 (基于欧氏距离)
    with torch.no_grad():
        # 计算 X 的 pairwise 距离
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)  # (n, n)
        sim_matrix = X_norm @ X_norm.T
        sim_matrix.fill_diagonal_(-float('inf'))
        _, neighbors = sim_matrix.topk(k, dim=1)

        # Step 3: 局部平滑
        for i in range(n):
            # X[i] 的邻居在 Y 中对应的索引
            neigh_labels = labels[neighbors[i]]  # shape: (k,)
            # X[i] 当前匹配的 Y 点
            current = labels[i]
            # 找邻居最常见的 Y 点索引
            vals, counts = torch.unique(neigh_labels, return_counts=True)
            majority = vals[counts.argmax()]
            # 如果当前匹配不在邻居主流中，可以替换
            if current != majority:
                labels[i] = majority

    # Step 4: 构建新的 P_ortho
    P_ortho.zero_()
    P_ortho[torch.arange(n, device=device), labels] = 1

    return P_ortho, labels

