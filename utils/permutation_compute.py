# code adapted from original EOT-Correspondence implementation https://github.com/Tungthanhlee/EOT-Correspondence

import torch
import torch.nn.functional as F
from typing import Tuple

from utils.sinkhorn_util import sinkhorn_OT, dist_mat


def compute_permutation_matrices(cfg, feat_x: torch.Tensor, feat_y: torch.Tensor, with_refine: str='ip') -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute permutation matrices between feature sets.
        
        Args:
            feat_x: Features from first shape
            feat_y: Features from second shape
            
        Returns:
            Tuple of permutation matrices (Pxy, Pyx)
        """
        
        with_refine = cfg.permute.metric
        assert with_refine in ['L2', 'ip'], f"Unsupported metric: {with_refine}, choose 'L2' or 'ip'."

        if with_refine == 'L2':
            distance_matrix = dist_mat(feat_x.squeeze(0), feat_y.squeeze(0), False)
            Pxy, Pyx = sinkhorn_OT(distance_matrix, sigma=0.1, num_sink=10)
            return Pxy.unsqueeze(0), Pyx.unsqueeze(0)
        elif with_refine == 'ip':
            return compute_permutation_matrix(feat_x, feat_y, bidirectional=True)
    

def compute_permutation_matrix(feat_x: torch.Tensor, feat_y: torch.Tensor, 
                                bidirectional: bool = False, normalize: bool = True) -> torch.Tensor:
    """Compute permutation matrix between feature sets.
    
    Args:
        feat_x: Features from first shape
        feat_y: Features from second shape
        bidirectional: Whether to compute both Pxy and Pyx
        normalize: Whether to normalize features
        
    Returns:
        Permutation matrix or tuple of permutation matrices
    """
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
    # Compute similarity matrix
    similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

    # Apply Sinkhorn normalization
    similarity = similarity.squeeze(0)
    Pxy, Pyx = sinkhorn_OT(similarity, sigma=1, num_sink=10)
    similarity = similarity.unsqueeze(0)

    if bidirectional:
        return Pxy.unsqueeze(0), Pyx.unsqueeze(0)
    else:
        return Pxy
    
