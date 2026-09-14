import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """Preserve source neighborhood geometry after the text projector."""

    def __init__(self, temperature=0.1, levels=3, eps=1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.levels = int(levels)
        self.eps = float(eps)

        if self.temperature <= 0:
            raise ValueError('temperature must be > 0')
        if self.levels < 1:
            raise ValueError('levels must be >= 1')

    def _transition(self, x):
        x = F.normalize(x, dim=-1)
        sim = x @ x.transpose(0, 1)
        sim = sim / self.temperature

        # Exclude the trivial self-neighbor.
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
        return torch.softmax(sim, dim=-1)

    def _js(self, p, q):
        p = p.clamp_min(self.eps)
        q = q.clamp_min(self.eps)
        m = 0.5 * (p + q)

        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm).mean()

    def forward(self, source, projected):
        if source.shape[0] != projected.shape[0]:
            raise ValueError('source and projected must have the same batch size')
        if source.shape[0] < 2:
            return projected.sum() * 0.0

        # Frozen text geometry is the teacher; gradients only update projector.
        p = self._transition(source.detach())
        q = self._transition(projected)

        p_level, q_level = p, q
        loss = projected.new_zeros(())

        for level in range(self.levels):
            if level > 0:
                p_level = p_level @ p
                q_level = q_level @ q
            loss = loss + self._js(p_level, q_level)

        return loss / self.levels
