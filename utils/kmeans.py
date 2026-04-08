import torch

import faiss
import faiss.contrib.torch_utils

import numpy as np

def train_kmeans_faiss_gpu():
    import numpy as np
    import faiss

    res = faiss.StandardGpuResources()

    d = 128
    n_data = 10000
    np.random.seed(1234)
    data = np.random.random((n_data, d)).astype('float32')

    k = 5
    max_iter = 100
    gpu_id = 0

    kmeans = faiss.Clustering(d, k)
    kmeans.verbose = True
    kmeans.max_points_per_centroid = 1000000
    kmeans.seed = 1234

    cfg = faiss.GpuClonerOptions()
    cfg.useFloat16 = False
    cfg.usePrecomputed = False

    index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index, cfg)

    kmeans.train(data, gpu_index)

    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(k, d)
    print(f"Final centroids shape: {centroids.shape}")

    test_data = np.random.random((5, d)).astype('float32')
    _, labels = gpu_index.search(test_data, 1)
    print("Predicted cluster labels:", labels.flatten())


def train_kmeans_faiss(x, k, niter=100, metric='l2', return_idx=True, min_points_per_centroid=None, seed=1,
                       device='cpu', gpu_index=None, verbose=False):
    '''
    Runs kmeans on one or several GPUs
    :param x:           Tensor, N x d, float
    :param k:           number of cluster centroid
    :param niter:
    :param metric:      l2 or ip (for inner product)
    :param gpu_id:
    :param seed:        integer, greater than 0
    :param verbose:
    :return:            cluster centroid with k x d, indice with N x 1
    '''
    metric_list = ['l2', 'ip', 'cos']
    assert device in ['cpu', 'cuda']
    assert metric in metric_list
    d = x.shape[1]
    # device = x.device
    clus = faiss.Clustering(d, k)
    clus.seed = int(np.array(seed)) if seed is not None else np.random.randint(2021)
    clus.verbose = verbose
    clus.niter = niter

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 20000
    if min_points_per_centroid is not None:
        clus.min_points_per_centroid = min_points_per_centroid

    if device == 'cpu':
        if metric == 'l2':
            index = faiss.IndexFlatL2(d)
        elif metric == 'ip' or metric == 'cos':
            index = faiss.IndexFlatIP(d)
        else:
            raise NotImplementedError(f"metric must be in the range of {metric_list}")
        # perform the training
        input = np.ascontiguousarray(x.detach().cpu().numpy())
        clus.train(x=input, index=index)
        centroids = faiss.vector_float_to_array(clus.centroids)
        D, I = index.search(input, 1)
        centroids = torch.Tensor(centroids).view(k, -1).to(x.device)
        if return_idx:
            return centroids, torch.Tensor(I).squeeze(1).to(x.device)
        else:
            return centroids
    else:
        assert type(gpu_index) == list and len(gpu_index) > 0
        res = faiss.StandardGpuResources()

        cfg = faiss.GpuClonerOptions()
        cfg.useFloat16 = True  # 是否禁用半精度计算（保持精度）
        cfg.usePrecomputed = False

        if metric == 'l2':
            index = faiss.IndexFlatL2(d)
        elif metric == 'ip' or metric == 'cos':
            index = faiss.IndexFlatIP(d)
        else:
            raise NotImplementedError(f"metric must be in the range of {metric_list}")
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_index[0], index, cfg)
        clus.train(x=x.cpu().numpy(), index=gpu_index)
        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = torch.Tensor(centroids).view(k, -1).to(x.device)

        if return_idx:
            search_index = faiss.IndexFlatIP(d)
            search_index.add(centroids)
            _, I = search_index.search(x, 1)
            return centroids, torch.Tensor(I).squeeze(1).to(x.device)
        else:
            return centroids

