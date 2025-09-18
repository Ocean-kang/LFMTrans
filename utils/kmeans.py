import torch
import numpy as np


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