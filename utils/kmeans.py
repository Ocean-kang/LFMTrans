import torch
import numpy as np
import faiss
import faiss.contrib.torch_utils


def get_seeds(
    seed, num_seeds
) -> list[int]:
    """Generate a list of random seeds from an initial seed.

    Args:
        seed: Initial seed to initialize the random generator
        num_seeds: Number of seeds to generate

    Returns:
        A list of randomly generated seeds
    """
    if seed is None:
        seed = torch.initial_seed()

    if num_seeds is None:
        num_seeds = 1

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Generate random integers up to the maximum uint32 value
    return torch.randint(
        np.iinfo(np.uint32).max,
        (num_seeds,),
        dtype=torch.long,
        generator=generator,
    ).tolist()

def train_kmeans_itsamatch(feat_t_input:torch.tensor, feat_v_input:torch.tensor, feat_v_labels:torch.tensor, clusterer, seed, epoch):

    # prprocessing
    num_samples = feat_v_input.shape[0]
    n_cls, t1, t2 = feat_t_input.shape # capture language embeddings imformation of dimenssion
    # init clusterer center
    clusterer.n_clusters = n_cls
    # random samples
    subseed = get_seeds(seed, epoch+1)[epoch]
    sampled_indices = torch.randperm(
        num_samples,
        generator=torch.Generator().manual_seed(subseed),
    )[: 10000] # 1w, half, whole

    feat_v_subsampled = feat_v_input[sampled_indices]
    labels_subsampled = feat_v_labels[sampled_indices]

    # cluster vision embeddings
    clusterer.fit(feat_v_subsampled.cpu().numpy())
    feat_v_clustered = torch.tensor(
        clusterer.cluster_centers_
    ).to(feat_v_input)
    cluster_assignments = torch.tensor(clusterer.labels_).to(
        feat_v_subsampled.device
    )

    return_dict = dict()
    return_dict['feat_v_subsampled'] = feat_v_subsampled
    return_dict['feat_v_clustered'] = feat_v_clustered
    return_dict['cluster_assignments'] = cluster_assignments
    return_dict['labels_subsampled'] = labels_subsampled

    return return_dict

def eval_kmeans_itsamatch(feat_t_input:torch.tensor, feat_v_input:torch.tensor, feat_v_labels:torch.tensor, clusterer, seed):

    # prprocessing
    num_samples = feat_v_input.shape[0]
    n_cls, t1, t2 = feat_t_input.shape # capture language embeddings imformation of dimenssion
    # init clusterer center
    clusterer.n_clusters = n_cls
    # random samples
    sampled_indices = torch.randperm(
        num_samples,
        generator=torch.Generator().manual_seed(seed),
    )[: num_samples // 2]

    feat_v_subsampled = feat_v_input[sampled_indices]
    labels_subsampled = feat_v_labels[sampled_indices]

    # cluster vision embeddings
    clusterer.fit(feat_v_subsampled.cpu().numpy())
    feat_v_clustered = torch.tensor(
        clusterer.cluster_centers_
    ).to(feat_v_input)
    cluster_assignments = torch.tensor(clusterer.labels_).to(
        feat_v_subsampled.device
    )

    return_dict = dict()
    return_dict['feat_v_subsampled'] = feat_v_subsampled
    return_dict['feat_v_clustered'] = feat_v_clustered
    return_dict['cluster_assignments'] = cluster_assignments
    return_dict['labels_subsampled'] = labels_subsampled

    return return_dict

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