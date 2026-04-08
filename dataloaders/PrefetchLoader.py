'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
'''
import torch

class PrefetchLoader():

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_data in self.loader:
            with torch.cuda.stream(stream):
                next_data['feat_t'] = next_data['feat_t'].cuda(device=self.device, non_blocking=True)
                next_data['feat_v'] = next_data['feat_v'].cuda(device=self.device, non_blocking=True)

            if not first:
                yield data
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data

        yield data

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

