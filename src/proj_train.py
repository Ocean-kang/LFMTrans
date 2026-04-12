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
from utils.itsamatch_cluster import (
    assign_to_ip_centers,
    assign_to_centers,
    build_cluster_training_tensor,
    flatten_classwise_features,
)
from utils.cluster_metrics import (
    evaluate_cluster_predictions,
    hungarian_match_from_similarity,
)


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
        # general training hyperparameters
        self.epochs = int(getattr(train_cfg, 'epochs', 30))
        self.steps_per_epoch = int(getattr(train_cfg, 'steps_per_epoch', 50))
        self.classes_per_step = int(getattr(train_cfg, 'classes_per_step', 32))
        self.samples_per_class = int(getattr(train_cfg, 'samples_per_class', 4))
        self.grad_clip = float(getattr(train_cfg, 'grad_clip', 1.0))
        self.log_every = int(getattr(train_cfg, 'log_every', 10))
        self.ckpt_dir = getattr(train_cfg, 'ckpt_dir', './checkpoints')
        self.report_train_each_epoch = bool(getattr(train_cfg, 'report_train_each_epoch', True))
        # clustering-related hyperparameters
        self.mask = int(getattr(train_cfg, 'mask', 1))
        self.cluster_subsample_ratio = float(getattr(train_cfg, 'cluster_subsample_ratio', 0.5))
        self.cluster_kmeans_n_init = int(getattr(train_cfg, 'cluster_kmeans_n_init', 100))
        self.cluster_remove_zero_padding = bool(getattr(train_cfg, 'cluster_remove_zero_padding', True))
        self.cluster_rebuild_each_epoch = bool(getattr(train_cfg, 'cluster_rebuild_each_epoch', False))
        self.report_cluster_each_epoch = bool(getattr(train_cfg, 'report_cluster_each_epoch', True))

        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=float(getattr(train_cfg, 'lr', 1e-4)),
            weight_decay=float(getattr(train_cfg, 'weight_decay', 1e-4)),
        )

    @staticmethod
    def _ensure_3d_bank(bank: torch.Tensor, name: str) -> torch.Tensor:
        if bank.ndim == 2:
            return bank.unsqueeze(1)
        if bank.ndim == 3:
            return bank
        raise ValueError(f'{name} bank must be [C, D] or [C, S, D], got shape={tuple(bank.shape)}')

    @staticmethod
    def _sample_from_bank(bank: torch.Tensor, class_ids: torch.Tensor, samples_per_class: int) -> torch.Tensor:
        if class_ids.numel() == 0:
            raise ValueError('class_ids cannot be empty')

        bank = bank.float().cpu()
        outputs = []

        for cls_id in class_ids.tolist():
            cls_bank = bank[cls_id]
            if cls_bank.ndim == 1:
                cls_bank = cls_bank.unsqueeze(0)

            # 关键修复：cluster 后的 [K, max_count, D] 包含 padding，全0行不能参与采样
            valid_mask = cls_bank.abs().sum(dim=-1) > 0
            valid_rows = cls_bank[valid_mask]

            if valid_rows.shape[0] == 0:
                raise ValueError(
                    f'Class/cluster {cls_id} has no valid non-zero features after filtering. '
                    'This usually means an empty cluster or all-zero padded supports.'
                )

            if valid_rows.shape[0] >= samples_per_class:
                idx = torch.randperm(valid_rows.shape[0])[:samples_per_class]
            else:
                idx = torch.randint(
                    low=0,
                    high=valid_rows.shape[0],
                    size=(samples_per_class,),
                )

            outputs.append(valid_rows[idx])

        return torch.cat(outputs, dim=0)

    def _sample_batch(self, train_bank: Dict[str, torch.Tensor]):
        text_bank = self._ensure_3d_bank(train_bank['text'], 'text')
        vision_bank = self._ensure_3d_bank(train_bank['vision'], 'vision')

        num_classes = min(text_bank.shape[0], vision_bank.shape[0])
        class_count = min(self.classes_per_step, num_classes)
        class_ids = torch.randperm(num_classes)[:class_count]

        feat_t = self._sample_from_bank(
            text_bank, class_ids, self.samples_per_class
        ).to(self.device, non_blocking=True)

        feat_v = self._sample_from_bank(
            vision_bank, class_ids, self.samples_per_class
        ).to(self.device, non_blocking=True)

        return feat_t, feat_v

    def _prepare_cluster_train_bank(self, raw_train_bank: Dict[str, torch.Tensor], seed: int):

        kmeans_metric = str(getattr(self.cfg.projector_train, 'kmeans_metric', 'L2')).lower()
        cluster_device = self.device if kmeans_metric == 'ip' else torch.device('cpu')

        text_bank = self._ensure_3d_bank(raw_train_bank['text'], 'text').float().to(cluster_device, non_blocking=True)
        vision_bank = self._ensure_3d_bank(raw_train_bank['vision'], 'vision').float().to(cluster_device, non_blocking=True)

        n_cls = min(text_bank.shape[0], vision_bank.shape[0])

        cluster_result = build_cluster_training_tensor(
            cfg=self.cfg,
            vision_features=vision_bank[:n_cls],
            n_cls=n_cls,
            sub_seed=seed,
            subsample_ratio=self.cluster_subsample_ratio,
            n_init=self.cluster_kmeans_n_init,
            remove_zero_padding=self.cluster_remove_zero_padding,
        )

        cluster_bank = {
            'text': text_bank[:n_cls],
            'vision': cluster_result.feat_vision_cluster.float().cpu(),  # [K, max_count, D]
        }
        return cluster_bank, cluster_result
    
    @torch.no_grad()
    def _evaluate_cluster_metrics(self, raw_train_bank: Dict[str, torch.Tensor], cluster_result):
        raw_vision = self._ensure_3d_bank(raw_train_bank['vision'], 'vision').float().cpu()
        raw_text = self._ensure_3d_bank(raw_train_bank['text'], 'text').float().cpu()

        eval_features, eval_labels, _, _ = flatten_classwise_features(
            raw_vision,
            remove_zero_padding=self.cluster_remove_zero_padding,
        )

        kmeans_metric = str(getattr(self.cfg.projector_train, 'kmeans_metric', 'L2')).lower()
        if kmeans_metric == 'ip':
            assignments = assign_to_ip_centers(
                eval_features.to(cluster_result.centers.device, non_blocking=True),
                cluster_result.centers,
            ).cpu()
        else:
            assignments = assign_to_centers(eval_features, cluster_result.centers)

        text_proto = maybe_mean_pool_features(raw_text).to(self.device)
        was_training = self.projector.training
        self.projector.eval()
        translated_text = self.projector(text_proto).float().cpu()
        if was_training:
            self.projector.train()

        centers = F.normalize(cluster_result.centers.float().cpu(), dim=-1)
        translated_text = F.normalize(translated_text, dim=-1)

        similarity = centers @ translated_text.T
        permutation = hungarian_match_from_similarity(similarity)

        return evaluate_cluster_predictions(
            assignments=assignments,
            labels=eval_labels,
            predicted_permutation=permutation,
            num_classes=text_proto.shape[0],
        )

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

        raw_train_bank = {
            'text': feature_dict_train[dataset][self.cfg.train.text_model].float().cpu(),
            'vision': feature_dict_train[dataset][self.cfg.train.type].float().cpu(),
        }

        # 训练质量仍然在原始 bank 上看，不在 cluster bank 上看
        train_eval_pair = self._build_train_eval_pair(raw_train_bank)

        text_dim = raw_train_bank['text'].shape[-1]
        vision_dim = raw_train_bank['vision'].shape[-1]

        if self.mask == 1:
            train_bank = {
                'text': self._ensure_3d_bank(raw_train_bank['text'], 'text'),
                'vision': self._ensure_3d_bank(raw_train_bank['vision'], 'vision'),
            }
            cluster_result = None
        elif self.mask == 2:
            train_bank, cluster_result = self._prepare_cluster_train_bank(
                raw_train_bank=raw_train_bank,
                seed=int(getattr(self.cfg, 'seed', 0)),
            )
        else:
            raise ValueError(f'Unsupported projector_train.mask: {self.mask}')

        history = []
        train_eval_history = []
        cluster_eval_history = []

        for epoch in range(1, self.epochs + 1):
            if self.mask == 2 and self.cluster_rebuild_each_epoch:
                train_bank, cluster_result = self._prepare_cluster_train_bank(
                    raw_train_bank=raw_train_bank,
                    seed=int(getattr(self.cfg, 'seed', 0)) + epoch,
                )

            running = {'loss': 0.0, 'fmap': 0.0, 'proper': 0.0, 'ot': 0.0}

            for step in range(1, self.steps_per_epoch + 1):
                feat_t, feat_v = self._sample_batch(train_bank)
                step_stats = self._train_step(feat_t, feat_v)

                for key in running:
                    running[key] += step_stats[key]

                if step % self.log_every == 0 or step == self.steps_per_epoch:
                    avg_loss = running['loss'] / step
                    avg_fmap = running['fmap'] / step
                    avg_proper = running['proper'] / step
                    avg_ot = running['ot'] / step
                    print(
                        f"[epoch {epoch:03d} step {step:04d}] "
                        f"loss={avg_loss:.4f} fmap={avg_fmap:.4f} "
                        f"proper={avg_proper:.4f} ot={avg_ot:.4f}"
                    )

            epoch_stats = {
                'epoch': epoch,
                'loss': running['loss'] / self.steps_per_epoch,
                'fmap': running['fmap'] / self.steps_per_epoch,
                'proper': running['proper'] / self.steps_per_epoch,
                'ot': running['ot'] / self.steps_per_epoch,
            }
            history.append(epoch_stats)

            if self.report_train_each_epoch:
                train_eval = self.evaluator.evaluate_pair(
                    train_eval_pair['text'],
                    train_eval_pair['vision'],
                    projector=self.projector,
                )
                train_eval_record = {'epoch': epoch, **train_eval}
                train_eval_history.append(train_eval_record)
                print(
                    f"[epoch {epoch:03d} train-eval {dataset}] "
                    f"vision->text={train_eval['acc_v_to_t']:.4f} "
                    f"text->vision={train_eval['acc_t_to_v']:.4f}"
                )

            if cluster_result is not None and self.report_cluster_each_epoch:
                cluster_eval = self._evaluate_cluster_metrics(raw_train_bank, cluster_result)
                cluster_eval_record = {
                    'epoch': epoch,
                    'cluster_gt_accuracy': cluster_eval.cluster_gt_accuracy,
                    'cluster_pred_accuracy': cluster_eval.cluster_pred_accuracy,
                    'pred_gt_accuracy': cluster_eval.pred_gt_accuracy,
                }
                cluster_eval_history.append(cluster_eval_record)

                print(
                    f"[epoch {epoch:03d} cluster-eval {dataset}] "
                    f"cluster->gt={cluster_eval.cluster_gt_accuracy:.4f} "
                    f"cluster->pred={cluster_eval.cluster_pred_accuracy:.4f} "
                    f"pred->gt={cluster_eval.pred_gt_accuracy:.4f}"
                )

        os.makedirs(self.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.ckpt_dir,
            f"{dataset}_{self.cfg.train.text_model}_to_{self.cfg.train.type}_projector.pt"
        )
        torch.save(
            {
                'state_dict': self.projector.state_dict(),
                'config': self.cfg,
                'history': history,
                'train_eval_history': train_eval_history,
                'cluster_eval_history': cluster_eval_history,
            },
            ckpt_path,
        )

        final_eval_results = self.evaluator.evaluate_feature_dict(
            feature_dict_eval=feature_dict_eval,
            projector=self.projector,
            text_key=self.cfg.validation.text_model,
            vision_key=self.cfg.validation.type,
            datasets=final_eval_datasets,
        )

        first_dataset = final_eval_datasets[0]
        Cxy = final_eval_results[first_dataset]['Cxy']
        Cyx = final_eval_results[first_dataset]['Cyx']
        x_basis = final_eval_results[first_dataset]['x_basis']
        y_basis = final_eval_results[first_dataset]['y_basis']

        result = dict(final_eval_results[first_dataset])
        result['Cxy'] = Cxy
        result['Cyx'] = Cyx
        result['x_basis'] = x_basis
        result['y_basis'] = y_basis
        result['checkpoint_path'] = ckpt_path
        result['history'] = history
        result['train_eval_history'] = train_eval_history
        result['cluster_eval_history'] = cluster_eval_history
        result['final_cluster_metrics'] = cluster_eval_history[-1] if cluster_eval_history else None
        result['final_eval_results'] = final_eval_results
        return result
