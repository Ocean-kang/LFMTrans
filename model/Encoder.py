import torch
import torch.nn as nn

class LinearProj(nn.Module):
    def __init__(self, cfg):
        super(LinearProj, self).__init__()
        self.text_dimension = cfg.model.text_dimension
        self.vision_dimension = cfg.model.vision_dimension
        self.mapper = nn.Linear(self.text_dimension, self.vision_dimension)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, cfg.model.text_dimension]
        Returns:
            out: Tensor of shape [batch_size, cfg.model.vision_dimension]
        """
        return self.mapper(x)
