import torch
import torch.nn.functional as F

from loss.structure_loss import StructureLoss
from src.proj_train import ProjectorFMTrainer


class ProjectorFMStructureTrainer(ProjectorFMTrainer):
    """Original LFMTrans trainer + source-geometry STRUCTURE regularization."""

    def __init__(self, cfg, device, text_dim: int, vision_dim: int):
        super().__init__(cfg, device, text_dim=text_dim, vision_dim=vision_dim)

        loss_cfg = cfg.loss
        self.w_structure = float(getattr(loss_cfg, 'w_structure', 0.1))
        self.structure_loss = StructureLoss(
            temperature=float(getattr(loss_cfg, 'structure_temperature', 0.1)),
            levels=int(getattr(loss_cfg, 'structure_levels', 3)),
        )
        self._structure_step = 0

    def _train_step(self, feat_t: torch.Tensor, feat_v: torch.Tensor):
        self.projector.train()
        self.optimizer.zero_grad(set_to_none=True)

        feat_t_proj = self.projector(feat_t)
        structure = self.structure_loss(feat_t, feat_t_proj)
        feat_v = F.normalize(feat_v, dim=-1)

        system = self.fm_helper.solve_from_features(
            feat_t_proj,
            feat_v,
            self.device,
            detach_basis=True,
        )

        Cxy, Cyx = system['Cxy'], system['Cyx']
        Pxy, Pyx = self._soft_correspondence(
            Cxy,
            Cyx,
            system['x_basis'],
            system['y_basis'],
        )
        Cxy_target, Cyx_target = self._proper_targets(
            Pxy,
            Pyx,
            system['x_basis'],
            system['y_basis'],
        )

        fm_terms = self.fm_loss(Cxy, Cyx, system['x_vals'], system['y_vals'])
        fmap_reg = sum(fm_terms.values()) if fm_terms else torch.zeros((), device=self.device)
        proper = self.proper_loss(Cxy, Cxy_target) + self.proper_loss(Cyx, Cyx_target)
        ot = self.ot_loss(system['feat_x'], system['feat_y'], Pxy, Pyx)

        total = (
            fmap_reg
            + self.w_proper * proper
            + self.w_ot * ot
            + self.w_structure * structure
        )

        total.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.projector.parameters(), self.grad_clip)
        self.optimizer.step()

        self._structure_step += 1
        if self._structure_step % self.log_every == 0:
            print(
                f"[structure step {self._structure_step:05d}] "
                f"raw={structure.item():.6f} "
                f"weighted={(self.w_structure * structure).item():.6f}"
            )

        return {
            'loss': float(total.detach().cpu()),
            'fmap': float(fmap_reg.detach().cpu()),
            'proper': float(proper.detach().cpu()),
            'ot': float(ot.detach().cpu()),
            'structure': float(structure.detach().cpu()),
        }
