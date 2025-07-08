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