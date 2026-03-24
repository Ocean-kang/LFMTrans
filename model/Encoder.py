import torch
import torch.nn as nn
import torch.nn.functional as F


class TextVisionProjector(nn.Module):
    """Project text embeddings into the vision latent space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048,
                 dropout: float = 0.0, normalize_output: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else max(input_dim, output_dim)
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        return x


class LinearProj(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mapper = nn.Linear(cfg.model.text_dimension, cfg.model.vision_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapper(x)


class LinearProjText(LinearProj):
    pass


class Translator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mapper = nn.Linear(cfg.model.vision_dimension, cfg.model.vision_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapper(x)


def build_text_projector(cfg, input_dim: int, output_dim: int) -> TextVisionProjector:
    projector_cfg = getattr(cfg, 'projector', None)
    hidden_dim = getattr(projector_cfg, 'hidden_dim', max(input_dim, output_dim))
    dropout = float(getattr(projector_cfg, 'dropout', 0.0))
    normalize_output = bool(getattr(projector_cfg, 'normalize_output', True))
    return TextVisionProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        normalize_output=normalize_output,
    )
