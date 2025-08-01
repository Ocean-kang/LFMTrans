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


'''
Project Text Feature into vision Feature dimension
'''

class LinearProjText(nn.Module):
    def __init__(self, cfg):
        super(LinearProjText, self).__init__()
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

class Translator(nn.Module):
    def __init__(self, cfg):
        super(Translator, self).__init__()
        self.dimension1 = cfg.model.vision_dimension
        self.dimension2 = cfg.model.vision_dimension
        self.mapper = nn.Linear(self.dimension1, self.dimension2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, cfg.model.dimension1]
        Returns:
            out: Tensor of shape [batch_size, cfg.model.dimension2]
        """
        return self.mapper(x)