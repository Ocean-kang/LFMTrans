import os
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from loss.fmap_loss import SURFMNetLoss, SquaredFrobeniusLoss
from loss.ot_loss import SW
from model.Encoder import build_text_projector
from src.l2ipcombinemapping import LFMapIpL2Combination
from utils.fmap_retrieval import accrucy_fn
from utils.fmap_util import fmap2pointmap
from utils.load_feature import maybe_mean_pool_features


class ProjectorFMEvaluator:
    """Shared evaluator for train-time and final multi-dataset evaluation."""

    def __init__(self, cfg, device, fm_helper: LFMapIpL2Combination):
        self.cfg = cfg
        self.device = device
        self.fm_helper = fm_helper
        self.eval_cfg = getattr(cfg, 'validation', None)
        self.shuffle_before_eval = bool(getattr(self.eval_cfg, 'shuffle_before_eval', False))
        self.shuffle_trials = int(getattr(self.eval_cfg, 'shuffle_trials', 1))
        self.shuffle_seed = int(getattr(self.eval_cfg, 'shuffle_seed', getattr(cfg, 'seed', 0)))

    @staticmethod
    def _align_lengths(feat_t: torch.Tensor, feat_v: torch.Tensor):
        if feat_t.shape[0] == feat_v.shape[0]:
            return feat_t, feat_v
        n = min(feat_t.shape[0], feat_v.shape[0])
        return feat_t[:n], feat_v[:n]

    def _shuffle_pair(self, feat_t: torch.Tensor, feat_v: torch.Tensor, seed: int):
        feat_t, feat_v = self._align_lengths(feat_t, feat_v)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        perm = torch.randperm(feat_t.shape[0], generator=generator)
        return feat_t[perm], feat_v[perm]

    @staticmethod
    def _predict_accuracy(system):
        Cxy = system['Cxy'].squeeze(0)
        Cyx = system['Cyx'].squeeze(0)
        phi_t = system['x_basis'].squeeze(0).transpose(0, 1)
        phi_v = system['y_basis'].squeeze(0).transpose(0, 1)
        pred_v_to_t = fmap2pointmap(Cxy, phi_t, phi_v)
        pred_t_to_v = fmap2pointmap(Cyx, phi_v, phi_t)
        gt = torch.arange(min(phi_t.shape[0], phi_v.shape[0]), device=phi_t.device)
        acc_v_to_t = accrucy_fn(gt, pred_v_to_t[:gt.shape[0]])
        acc_t_to_v = accrucy_fn(gt, pred_t_to_v[:gt.shape[0]])
        return acc_v_to_t, acc_t_to_v

    @torch.no_grad()
    def evaluate_pair(self, feat_t: torch.Tensor, feat_v: torch.Tensor, projector=None, dataset_name: str = 'dataset'):
        feat_t = maybe_mean_pool_features(feat_t.float()).to(self.device)
        feat_v = maybe_mean_pool_features(feat_v.float()).to(self.device)
        feat_t, feat_v = self._align_lengths(feat_t, feat_v)

        shuffle_enabled = self.shuffle_before_eval
        trials = self.shuffle_trials if shuffle_enabled else 1
        metrics = []
        last_system = None

        if projector is not None:
            was_training = projector.training
            projector.eval()
        else:
            was_training = False

        for trial_idx in range(trials):
            if shuffle_enabled:
                feat_t_trial, feat_v_trial = self._shuffle_pair(
                    feat_t,
                    feat_v,
                    seed=self.shuffle_seed + trial_idx,
                )
            else:
                feat_t_trial, feat_v_trial = feat_t, feat_v

            if projector is not None:
                feat_t_proj = projector(feat_t_trial)
            else:
                feat_t_proj = feat_t_trial

            feat_v_trial = F.normalize(feat_v_trial, dim=-1)
            system = self.fm_helper.solve_from_features(feat_t_proj, feat_v_trial, self.device, detach_basis=True)
            acc_v_to_t, acc_t_to_v = self._predict_accuracy(system)
            metrics.append((acc_v_to_t, acc_t_to_v))
            last_system = system

        if projector is not None and was_training:
            projector.train()

        avg_v_to_t = sum(m[0] for m in metrics) / len(metrics)
        avg_t_to_v = sum(m[1] for m in metrics) / len(metrics)

        return {
            'dataset': dataset_name,
            'acc_v_to_t': avg_v_to_t,
            'acc_t_to_v': avg_t_to_v,
            'shuffle_enabled': shuffle_enabled,
            'shuffle_trials': trials,
            'Cxy': last_system['Cxy'].squeeze(0),
            'Cyx': last_system['Cyx'].squeeze(0),
            'x_basis': last_system['x_basis'].squeeze(0),
            'y_basis': last_system['y_basis'].squeeze(0),
        }

    @torch.no_grad()
    def evaluate_feature_dict(self, feature_dict_eval: Dict[str, Dict[str, torch.Tensor]], projector, text_key: str, vision_key: str,
                              datasets: Iterable[str]):
        results = {}
        for dataset in datasets:
            if dataset not in feature_dict_eval:
                raise KeyError(f'Dataset {dataset} not found in feature_dict_eval')
            feat_t = feature_dict_eval[dataset][text_key]
            feat_v = feature_dict_eval[dataset][vision_key]
            results[dataset] = self.evaluate_pair(feat_t, feat_v, projector=projector, dataset_name=dataset)
        return results


class ProjectorFMTrainer:
    """Train a single text->vision projector with FM + OT."""

    def __init__(self, cfg, device, text_dim: int, vision_dim: int):
        self.cfg = cfg
        self.device = device
        self.projector = build_text_projector(cfg, text_dim, vision_dim).to(device)
        self.fm_helper = LFMapIpL2Combination(cfg)
        self.evaluator = ProjectorFMEvaluator(cfg, device, self.fm_helper)

        loss_cfg = getattr(cfg, 'loss', None)
        self.fm_loss = SURFMNetLoss(
            w_bij=float(getattr(loss_cfg, 'w_bij', 1.0)),
            w_orth=float(getattr(loss_cfg, 'w_orth', 1.0)),
            w_lap=float(getattr(loss_cfg, 'w_lap', 1e-3)),
        )
        self.proper_loss = SquaredFrobeniusLoss(loss_weight=1.0)
        self.ot_loss = SW(
            L=int(getattr(loss_cfg, 'ot_projections', 64)),
            p=int(getattr(loss_cfg, 'ot_p', 2)),
            loss_weight=1.0,
            bidirectional=True,
        )
        self.temperature = float(getattr(loss_cfg, 'temperature', 0.07))
        self.w_ot = float(getattr(loss_cfg, 'w_ot', 1.0))
        self.w_proper = float(getattr(loss_cfg, 'w_proper', 1.0))

        train_cfg = getattr(cfg, 'projector_train', None)
        self.epochs = int(getattr(train_cfg, 'epochs', 30))
        self.steps_per_epoch = int(getattr(train_cfg, 'steps_per_epoch', 50))
        self.classes_per_step = int(getattr(train_cfg, 'classes_per_step', 32))
        self.samples_per_class = int(getattr(train_cfg, 'samples_per_class', 4))
        self.grad_clip = float(getattr(train_cfg, 'grad_clip', 1.0))
        self.log_every = int(getattr(train_cfg, 'log_every', 10))
        self.ckpt_dir = getattr(train_cfg, 'ckpt_dir', './checkpoints')
        self.report_train_each_epoch = bool(getattr(train_cfg, 'report_train_each_epoch', True))

        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=float(getattr(train_cfg, 'lr', 1e-4)),
            weight_decay=float(getattr(train_cfg, 'weight_decay', 1e-4)),
        )

    @staticmethod
    def _sample_from_bank(bank: torch.Tensor, class_ids: torch.Tensor, samples_per_class: int) -> torch.Tensor:
        _, num_per_class, feat_dim = bank.shape
        if class_ids.numel() == 0:
            raise ValueError('class_ids cannot be empty')
        if samples_per_class > num_per_class:
            raise ValueError(f'samples_per_class={samples_per_class} exceeds available samples {num_per_class}')
        idx = torch.stack([torch.randperm(num_per_class)[:samples_per_class] for _ in range(class_ids.numel())], dim=0)
        selected = bank[class_ids.unsqueeze(1), idx]
        batch = selected.reshape(-1, feat_dim)
        perm = torch.randperm(batch.shape[0])
        return batch[perm]

    def _sample_batch(self, train_bank: Dict[str, torch.Tensor]):
        text_bank = train_bank['text']
        vision_bank = train_bank['vision']
        num_classes = min(text_bank.shape[0], vision_bank.shape[0])
        class_count = min(self.classes_per_step, num_classes)
        class_ids = torch.randperm(num_classes)[:class_count]
        feat_t = self._sample_from_bank(text_bank, class_ids, self.samples_per_class).to(self.device, non_blocking=True)
        feat_v = self._sample_from_bank(vision_bank, class_ids, self.samples_per_class).to(self.device, non_blocking=True)
        return feat_t, feat_v

    def _soft_correspondence(self, Cxy, Cyx, x_basis, y_basis):
        phi_x = x_basis.transpose(1, 2)
        phi_y = y_basis.transpose(1, 2)
        x_embed_in_y = torch.bmm(phi_x, Cxy.transpose(1, 2))
        y_embed_in_x = torch.bmm(phi_y, Cyx.transpose(1, 2))

        x_embed_in_y = F.normalize(x_embed_in_y, dim=-1)
        y_embed_in_x = F.normalize(y_embed_in_x, dim=-1)
        phi_x = F.normalize(phi_x, dim=-1)
        phi_y = F.normalize(phi_y, dim=-1)

        sim_xy = torch.bmm(x_embed_in_y, phi_y.transpose(1, 2)) / self.temperature
        sim_yx = torch.bmm(y_embed_in_x, phi_x.transpose(1, 2)) / self.temperature
        Pxy = torch.softmax(sim_xy, dim=-1)
        Pyx = torch.softmax(sim_yx, dim=-1)
        return Pxy, Pyx

    def _proper_targets(self, Pxy, Pyx, x_basis, y_basis):
        Cxy_target = torch.bmm(torch.bmm(y_basis, Pxy.transpose(1, 2)), x_basis.transpose(1, 2))
        Cyx_target = torch.bmm(torch.bmm(x_basis, Pyx.transpose(1, 2)), y_basis.transpose(1, 2))
        return Cxy_target, Cyx_target

    def _train_step(self, feat_t: torch.Tensor, feat_v: torch.Tensor):
        self.projector.train()
        self.optimizer.zero_grad(set_to_none=True)

        feat_t_proj = self.projector(feat_t)
        feat_v = F.normalize(feat_v, dim=-1)

        system = self.fm_helper.solve_from_features(feat_t_proj, feat_v, self.device, detach_basis=True)
        Cxy, Cyx = system['Cxy'], system['Cyx']
        Pxy, Pyx = self._soft_correspondence(Cxy, Cyx, system['x_basis'], system['y_basis'])
        Cxy_target, Cyx_target = self._proper_targets(Pxy, Pyx, system['x_basis'], system['y_basis'])

        fm_terms = self.fm_loss(Cxy, Cyx, system['x_vals'], system['y_vals'])
        fmap_reg = sum(fm_terms.values()) if fm_terms else torch.zeros((), device=self.device)
        proper = self.proper_loss(Cxy, Cxy_target) + self.proper_loss(Cyx, Cyx_target)
        ot = self.ot_loss(system['feat_x'], system['feat_y'], Pxy, Pyx)
        total = fmap_reg + self.w_proper * proper + self.w_ot * ot

        total.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.projector.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            'loss': float(total.detach().cpu()),
            'fmap': float(fmap_reg.detach().cpu()),
            'proper': float(proper.detach().cpu()),
            'ot': float(ot.detach().cpu()),
        }

    @staticmethod
    def _build_train_eval_pair(train_bank: Dict[str, torch.Tensor]):
        return {
            'text': maybe_mean_pool_features(train_bank['text'].float().cpu()),
            'vision': maybe_mean_pool_features(train_bank['vision'].float().cpu()),
        }

    def _save_checkpoint(self, text_dim: int, vision_dim: int):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self.ckpt_dir, 'text_to_vision_projector.pt')
        torch.save(
            {
                'state_dict': self.projector.state_dict(),
                'text_dim': int(text_dim),
                'vision_dim': int(vision_dim),
            },
            ckpt_path,
        )
        return ckpt_path

    def fit(self, feature_dict_train, feature_dict_eval, final_eval_datasets):
        dataset = self.cfg.train.dataset
        train_bank = {
            'text': feature_dict_train[dataset][self.cfg.train.text_model].float().cpu(),
            'vision': feature_dict_train[dataset][self.cfg.train.type].float().cpu(),
        }
        train_eval_pair = self._build_train_eval_pair(train_bank)
        text_dim = train_bank['text'].shape[-1]
        vision_dim = train_bank['vision'].shape[-1]

        history = []
        train_eval_history = []
        for epoch in range(1, self.epochs + 1):
            running = {'loss': 0.0, 'fmap': 0.0, 'proper': 0.0, 'ot': 0.0}
            for step in range(1, self.steps_per_epoch + 1):
                feat_t, feat_v = self._sample_batch(train_bank)
                step_stats = self._train_step(feat_t, feat_v)
                for key in running:
                    running[key] += step_stats[key]

                if step % self.log_every == 0 or step == self.steps_per_epoch:
                    scale = 1.0 / step
                    print(
                        f"[epoch {epoch:03d} step {step:03d}] "
                        f"loss={running['loss'] * scale:.4f} "
                        f"fmap={running['fmap'] * scale:.4f} "
                        f"proper={running['proper'] * scale:.4f} "
                        f"ot={running['ot'] * scale:.4f}"
                    )

            epoch_scale = 1.0 / self.steps_per_epoch
            epoch_stats = {k: running[k] * epoch_scale for k in running}
            history.append(epoch_stats)

            if self.report_train_each_epoch:
                train_eval = self.evaluator.evaluate_pair(
                    train_eval_pair['text'],
                    train_eval_pair['vision'],
                    projector=self.projector,
                    dataset_name=f'{dataset}_train',
                )
                train_eval_history.append({
                    'epoch': epoch,
                    'acc_v_to_t': train_eval['acc_v_to_t'],
                    'acc_t_to_v': train_eval['acc_t_to_v'],
                })
                print(
                    f"[epoch {epoch:03d} eval {dataset}] "
                    f"vision->text={train_eval['acc_v_to_t']:.4f} "
                    f"text->vision={train_eval['acc_t_to_v']:.4f}"
                )

        final_eval_results = self.evaluator.evaluate_feature_dict(
            feature_dict_eval=feature_dict_eval,
            projector=self.projector,
            text_key=self.cfg.validation.text_model,
            vision_key=self.cfg.validation.type,
            datasets=final_eval_datasets,
        )
        ckpt_path = self._save_checkpoint(text_dim, vision_dim)

        first_dataset = final_eval_datasets[0]
        result = dict(final_eval_results[first_dataset])
        result['checkpoint_path'] = ckpt_path
        result['history'] = history
        result['train_eval_history'] = train_eval_history
        result['final_eval_results'] = final_eval_results
        return result
