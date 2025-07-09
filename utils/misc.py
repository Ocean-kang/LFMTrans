'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import shutil

import math
import numpy as np
import os
import hostlist
import torch.distributed as dist
import torch
from pathlib import Path
import cv2
from collections import OrderedDict


def init_process(backend="nccl"):
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    # world_size = int(os.environ.get("SLURM_NTASKS", 1))
    world_size = torch.cuda.device_count()
    print(f"Starting process with rank {dist_rank}...", flush=True)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["MASTER_PORT"] = str(12345 + int(gpu_ids[0]))
    elif "SLURM_STEPS_GPUS" in os.environ:
        gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
        os.environ["MASTER_PORT"] = str(12345 + int(min(gpu_ids)))
    else:
        os.environ["MASTER_PORT"] = str(12345 + np.random.randint(0, 100))

    if "SLURM_JOB_NODELIST" in os.environ:
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group(
        backend,
        rank=dist_rank,
        world_size=world_size,
    )
    print(f"Process {dist_rank} is connected, world_size is {world_size}", flush=True)
    dist.barrier()

    # silence_print(dist_rank == 0)
    # if dist_rank == 0:
    #     print(f"All processes are connected.", flush=True)


def sync_model(sync_dir, model):
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    sync_path = Path(sync_dir).resolve() / "sync_model.pkl"
    if dist_rank == 0 and world_size > 1:
        torch.save(model.state_dict(), sync_path)
    dist.barrier()
    if dist_rank > 0:
        model.load_state_dict(torch.load(sync_path))
    dist.barrier()
    if dist_rank == 0 and world_size > 1:
        sync_path.unlink()
    return model


class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -np.inf
        self.min = np.inf

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

def get_va_params_groups(model, cfg=None):
    text_projector = []
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif 'text_projector' in name:
            text_projector.append(param)
        else:
            params.append(param)
    return [
        {'params': params, 'lr': cfg.lr},
        {'params': text_projector, 'weight_decay': 0., 'lr': 1e-7},
    ]

def get_params_groups(model, cfg=None):
    text_projector = []
    mask_decoder = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'text_projector' in name:
            text_projector.append(param)
        elif 'mask_decoder' in name:
            mask_decoder.append(param)
        else:
            pass
    return [
        {'params': mask_decoder, 'lr': cfg.lr},
        {'params': text_projector, 'weight_decay': 0., 'lr': 1e-7},
    ]


def get_params_groups_uvlt(model, cfg=None, include_D=True):
    param_list = []
    param_list_sem_D = []
    param_list_sem_wgan_D = []
    param_list_G = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif 'no_grad' in name:
            print(name)
            continue
        elif 'text_projector' in name:
            param_list_G.append(param)
        elif 'discriminator_sem' in name:
            if 'wgan' in name:
                param_list_sem_wgan_D.append(param)
            else:
                param_list_sem_D.append(param)
        else:
            param_list.append(param)
    return [
        {'params': param_list, 'lr': cfg.lr, 'initial_lr': cfg.lr},
        {'params': param_list_G, 'lr': cfg.lr * 1e0, 'initial_lr': cfg.lr* 1e0},
        {'params': param_list_sem_wgan_D, 'lr': cfg.lr * 1e1, 'initial_lr': cfg.lr * 1e1},
        {'params': param_list_sem_D, 'lr': cfg.lr * 1e2, 'initial_lr': cfg.lr * 1e2},
    ]


def adjustable_exponential_decay_lr(epoch, optimizer,
                                    num_epochs=1000,
                                    exponent=1.0, final_decay_rate=0.1):

    for param_group in optimizer.param_groups:
        initial_lr = param_group['initial_lr']
        final_lr = initial_lr * final_decay_rate
        base_decay = math.pow(final_lr / initial_lr, 1 / exponent)

        t = min(epoch / num_epochs, 1.0)

        param_group['lr'] = initial_lr * math.pow(base_decay, math.pow(t, exponent))
        print(f"epoch:{epoch:4d}, initial_lr:{param_group['initial_lr']}, running_lr:{param_group['lr']}")