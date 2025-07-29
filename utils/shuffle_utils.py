import random
import torch


def shuffle_tensor(cfg, device, feats: torch.Tensor):
    """
    A function for shuffle input_feats

    Arg:
        cfg: configure file's pointer.
        feats: input feature of pretrain feature[b, n, d].

    Returns:
        feats_shuffled: shuffled features [b, n, d].
        shuffle_idx: the index used to shuffle each sample (optional, for reverse)
    """
    seed = cfg.seed
    torch.manual_seed(seed)
    B, N, D = feats.shape

    # random idx
    shuffle_idx = torch.stack([torch.randperm(N) for _ in range(B)]).to(device)

    # shuffle feats
    feats_shuffled = torch.stack([
        feats[i, shuffle_idx[i]] for i in range(B)
    ], dim=0)  # shape: [B, N, D]

    return feats_shuffled, shuffle_idx

def unshuffle_tensor(feats_shuffled, shuffle_idx):
    """
    Reverse the shuffle using shuffle_idx.
    """
    B, N, D = feats_shuffled.shape
    device = feats_shuffled.device

    # 构建反索引
    unshuffle_idx = torch.argsort(shuffle_idx, dim=1)  # [B, N]

    feats_unshuffled = torch.stack([
        feats_shuffled[i, unshuffle_idx[i]] for i in range(B)
    ], dim=0)

    return feats_unshuffled

def select_samples_per_class(features: torch.Tensor, 
                            labels: torch.Tensor, 
                            num_classes: int = 10, 
                            n_samples_per_class: int = 18, 
                            seed: int = None):
    """
    Randomly select a fixed number of samples from each class.

    Args:
        features (torch.Tensor): Feature tensor of shape [N, D], where N is the number of samples, D is feature dimension.
        labels (torch.Tensor): Label tensor of shape [N], with class indices ranging from 0 to num_classes - 1.
        num_classes (int): Total number of distinct classes.
        n_samples_per_class (int): Number of samples to select per class.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        selected_features (torch.Tensor): Tensor of selected features with shape [num_classes * n_samples_per_class, D].
        selected_labels (torch.Tensor): Tensor of selected labels with shape [num_classes * n_samples_per_class].
    """
    if seed is not None:
        torch.manual_seed(seed)

    selected_features = []
    selected_labels = []

    for cls in range(num_classes):
        # Find indices of all samples belonging to class `cls`
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        assert len(idx) >= n_samples_per_class, f"Not enough samples in class {cls}"

        # Randomly sample n_samples_per_class indices
        chosen_idx = idx[torch.randperm(len(idx))[:n_samples_per_class]]

        # Collect the selected features and labels
        selected_features.append(features[chosen_idx])
        selected_labels.append(labels[chosen_idx])

    # Concatenate the selected samples from all classes
    selected_features = torch.cat(selected_features, dim=0)  # Shape: [num_classes * n_samples_per_class, D]
    selected_labels = torch.cat(selected_labels, dim=0)      # Shape: [num_classes * n_samples_per_class]

    return selected_features, selected_labels

def map_indices_to_class_labels(indices: torch.Tensor, block_size: int = 18) -> torch.Tensor:
    """
    Map flat indices (e.g., 0-179) to class IDs based on block boundaries.

    Args:
        indices (torch.Tensor): Tensor of indices (e.g., predicted indices), shape [N]
        block_size (int): Number of samples per class block (default = 18)

    Returns:
        torch.Tensor: Corresponding class labels, shape [N]
    """
    return indices // block_size

