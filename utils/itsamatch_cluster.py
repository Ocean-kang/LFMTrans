from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class ClusterPackResult:
    centers: torch.Tensor
    assignments: torch.Tensor
    feat_vision_cluster: torch.Tensor
    sampled_indices: torch.Tensor
    sampled_features: torch.Tensor
    sampled_labels: torch.Tensor
    per_class_counts: torch.Tensor
    valid_counts_per_class: torch.Tensor
    num_zero_padded_removed: int


def _nonzero_row_mask(features: torch.Tensor) -> torch.Tensor:
    return features.abs().sum(dim=-1) > 0


def flatten_classwise_features(
    vision_features: torch.Tensor,
    labels: torch.Tensor | None = None,
    remove_zero_padding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Normalize different feature layouts to flat sample-wise tensors.

    Args:
        vision_features: [C, S, D] or [N, D]
        labels: optional [N] labels when vision_features is [N, D]
        remove_zero_padding: whether to drop all-zero rows before clustering

    Returns:
        flat_features: [N_total, D]
        flat_labels: [N_total]
        valid_counts_per_class: [C]
        num_zero_padded_removed: int
    """
    if vision_features.ndim == 2:
        flat_features = vision_features
        if labels is None:
            flat_labels = torch.zeros(flat_features.shape[0], dtype=torch.long, device=flat_features.device)
            valid_counts_per_class = torch.tensor([flat_features.shape[0]], dtype=torch.long, device=flat_features.device)
        else:
            flat_labels = labels.to(device=flat_features.device, dtype=torch.long).view(-1)
            if flat_labels.shape[0] != flat_features.shape[0]:
                raise ValueError(
                    f"labels length {flat_labels.shape[0]} does not match features {flat_features.shape[0]}"
                )
            valid_counts_per_class = torch.bincount(flat_labels, minlength=int(flat_labels.max().item()) + 1)

        if remove_zero_padding:
            keep_mask = _nonzero_row_mask(flat_features)
            num_removed = int((~keep_mask).sum().item())
            flat_features = flat_features[keep_mask]
            flat_labels = flat_labels[keep_mask]
            if flat_labels.numel() > 0:
                valid_counts_per_class = torch.bincount(flat_labels, minlength=valid_counts_per_class.shape[0])
            else:
                valid_counts_per_class = torch.zeros_like(valid_counts_per_class)
        else:
            num_removed = 0
        return flat_features, flat_labels, valid_counts_per_class.cpu(), num_removed

    if vision_features.ndim != 3:
        raise ValueError(
            f"vision_features must be [C, S, D] or [N, D], got shape={tuple(vision_features.shape)}"
        )

    num_classes, _, feat_dim = vision_features.shape
    per_class_features = []
    per_class_labels = []
    valid_counts = []
    num_removed = 0

    for class_idx in range(num_classes):
        class_features = vision_features[class_idx]
        if remove_zero_padding:
            keep_mask = _nonzero_row_mask(class_features)
            num_removed += int((~keep_mask).sum().item())
            class_features = class_features[keep_mask]
        if class_features.shape[0] == 0:
            raise ValueError(
                f"All features for class {class_idx} are zero-padded after filtering; cannot cluster this class."
            )
        per_class_features.append(class_features)
        per_class_labels.append(
            torch.full((class_features.shape[0],), class_idx, dtype=torch.long, device=vision_features.device)
        )
        valid_counts.append(class_features.shape[0])

    flat_features = torch.cat(per_class_features, dim=0).reshape(-1, feat_dim)
    flat_labels = torch.cat(per_class_labels, dim=0)
    valid_counts_per_class = torch.as_tensor(valid_counts, dtype=torch.long)
    return flat_features, flat_labels, valid_counts_per_class, num_removed


def build_subsample_indices_from_labels(
    labels: torch.Tensor,
    ratio: float,
    seed: int,
) -> torch.Tensor:
    """Sample each class independently to keep class balance."""
    generator = torch.Generator(device=labels.device if labels.is_cuda else "cpu")
    generator.manual_seed(seed)

    unique_labels = torch.unique(labels)
    sampled_indices = []

    for cls_id in unique_labels.tolist():
        cls_indices = torch.where(labels == cls_id)[0]
        num_cls = cls_indices.numel()
        if num_cls == 0:
            continue
        num_take = max(1, int(num_cls * ratio))
        perm = torch.randperm(num_cls, generator=generator, device=cls_indices.device)
        sampled_indices.append(cls_indices[perm[:num_take]])

    if not sampled_indices:
        raise ValueError("No valid samples remained after filtering zero-padded rows.")

    sampled_indices = torch.cat(sampled_indices, dim=0)
    shuffle_perm = torch.randperm(sampled_indices.numel(), generator=generator, device=sampled_indices.device)
    return sampled_indices[shuffle_perm]


def run_itsamatch_kmeans(features: torch.Tensor, k: int, seed: int, n_init: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.float().cpu()
    kmeans = KMeans(n_clusters=int(k), init="k-means++", n_init=int(n_init), random_state=int(seed))
    assignments_np = kmeans.fit_predict(features.numpy())
    centers_np = kmeans.cluster_centers_
    centers = torch.from_numpy(np.asarray(centers_np)).float()
    assignments = torch.from_numpy(np.asarray(assignments_np)).long()
    return centers, assignments


def pack_cluster_features(features: torch.Tensor, assignments: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.float().cpu()
    assignments = assignments.long().cpu()

    counts = torch.bincount(assignments, minlength=int(k))
    if counts.numel() == 0:
        raise ValueError("KMeans returned no assignments.")

    # 这里必须提前拦截空簇，否则后面会生成整类全0 support
    if (counts == 0).any():
        raise ValueError(
            f"Empty cluster detected after KMeans: counts={counts.tolist()}. "
            "Please retry with another seed / ratio / n_init."
        )

    max_count = int(counts.max().item())
    feat_vision_cluster = torch.zeros(int(k), max_count, features.shape[-1], dtype=features.dtype)

    for cluster_idx in range(int(k)):
        members = features[assignments == cluster_idx]
        feat_vision_cluster[cluster_idx, : members.shape[0]] = members

    return feat_vision_cluster, counts


def build_cluster_training_tensor(
    vision_features: torch.Tensor,
    n_cls: int,
    sub_seed: int,
    subsample_ratio: float = 0.5,
    n_init: int = 100,
    vision_labels: torch.Tensor | None = None,
    remove_zero_padding: bool = True,
) -> ClusterPackResult:
    flat_features, flat_labels, valid_counts_per_class, num_zero_padded_removed = flatten_classwise_features(
        vision_features,
        labels=vision_labels,
        remove_zero_padding=remove_zero_padding,
    )

    if flat_features.shape[0] < n_cls:
        raise ValueError(
            f"Not enough flattened samples for clustering: {flat_features.shape[0]} < {n_cls}"
        )

    if flat_labels.numel() > 0 and torch.unique(flat_labels).numel() < n_cls:
        raise ValueError(
            f"Number of classes after zero-pad filtering is {torch.unique(flat_labels).numel()}, expected at least {n_cls}."
        )

    sampled_indices = build_subsample_indices_from_labels(
        flat_labels,
        ratio=subsample_ratio,
        seed=sub_seed,
    )

    sampled_features = flat_features[sampled_indices]
    sampled_labels = flat_labels[sampled_indices]

    if sampled_features.shape[0] < n_cls:
        raise ValueError(
            f"Not enough samples after subsampling: {sampled_features.shape[0]} < {n_cls}"
        )

    centers, assignments = run_itsamatch_kmeans(
        sampled_features,
        k=n_cls,
        seed=sub_seed,
        n_init=n_init,
    )

    feat_vision_cluster, counts = pack_cluster_features(
        sampled_features,
        assignments,
        k=n_cls,
    )

    return ClusterPackResult(
        centers=centers,
        assignments=assignments,
        feat_vision_cluster=feat_vision_cluster,   # [K, max_count, D]
        sampled_indices=sampled_indices,
        sampled_labels=sampled_labels,
        sampled_features=sampled_features,
        per_class_counts=counts,
        valid_counts_per_class=valid_counts_per_class,
        num_zero_padded_removed=num_zero_padded_removed,
    )

def assign_to_centers(features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    features = features.float().cpu()
    centers = centers.float().cpu()
    dists = torch.cdist(features, centers)
    return dists.argmin(dim=1)
