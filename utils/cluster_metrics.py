from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import one_hot


@dataclass
class ClusterEvaluation:
    cluster_gt_accuracy: float
    cluster_pred_accuracy: float
    pred_gt_accuracy: float
    oracle_permutation: torch.Tensor
    predicted_permutation: torch.Tensor
    prediction: torch.Tensor
    assignments: torch.Tensor


def hungarian_match_from_similarity(similarity: torch.Tensor) -> torch.Tensor:
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2D, got shape={tuple(similarity.shape)}")
    _, col_ind = linear_sum_assignment(-similarity.detach().cpu().numpy())
    return torch.as_tensor(col_ind, device=similarity.device, dtype=torch.long)


def evaluate_cluster_predictions(
    assignments: torch.Tensor,
    labels: torch.Tensor,
    predicted_permutation: torch.Tensor,
    num_classes: int,
) -> ClusterEvaluation:
    assignments = assignments.long()
    labels = labels.long().to(assignments.device)
    predicted_permutation = predicted_permutation.long().to(assignments.device)

    frequencies = one_hot(assignments, num_classes=num_classes).T @ one_hot(labels, num_classes=num_classes)
    _, col_ind = linear_sum_assignment(-frequencies.detach().cpu().numpy())
    oracle_permutation = torch.as_tensor(col_ind, device=assignments.device, dtype=torch.long)

    prediction = predicted_permutation[assignments]
    cluster_gt_accuracy = (oracle_permutation[assignments] == labels).float().mean().item()
    cluster_pred_accuracy = (predicted_permutation == oracle_permutation).float().mean().item()
    pred_gt_accuracy = (prediction == labels).float().mean().item()

    return ClusterEvaluation(
        cluster_gt_accuracy=cluster_gt_accuracy,
        cluster_pred_accuracy=cluster_pred_accuracy,
        pred_gt_accuracy=pred_gt_accuracy,
        oracle_permutation=oracle_permutation,
        predicted_permutation=predicted_permutation,
        prediction=prediction,
        assignments=assignments,
    )


def summarize_cluster_metrics(metrics: Dict[str, ClusterEvaluation]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name, metric in metrics.items():
        summary[f"{name}/cluster_gt_accuracy"] = metric.cluster_gt_accuracy
        summary[f"{name}/cluster_pred_accuracy"] = metric.cluster_pred_accuracy
        summary[f"{name}/pred_gt_accuracy"] = metric.pred_gt_accuracy
    return summary
