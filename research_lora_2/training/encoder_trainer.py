"""Training loops for the CSPQ preference encoder.

训练循环：train/val 分开、冻结场景特征（h_c 是常量输入）、best/last checkpoint、
train/val 分场景记录 loss、checkpoint 保存模型配置 + manifest hash + 特征索引 hash。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

import torch

from research_lora_2.model.preference_encoder import CSPQPreferenceEncoder
from research_lora_2.training.rnc_loss import encoder_total_loss


def _file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CSPQTrainer:
    """CSPQ 编码器训练器：单步训练、验证与完整 fit 循环。"""

    def __init__(self, model: CSPQPreferenceEncoder, *, device: str, learning_rate: float = 1e-3,
                 lambda_cross: float = 0.1, lambda_r: float = 1.0, lambda_a: float = 1.0,
                 temperature: float = 0.1, grad_clip_norm: float = 5.0,
                 disable_cross_scene: bool = False) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.grad_clip_norm = grad_clip_norm
        self.lambda_cross = 0.0 if disable_cross_scene else lambda_cross
        self.lambda_r = lambda_r
        self.lambda_a = lambda_a
        self.temperature = temperature
        self.step = 0

    @staticmethod
    def _to_scalar(value: Any) -> float:
        """把 Tensor/float/int 统一转成 float（Tensor 先 detach 到 CPU）。"""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _prepare(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """把批次字典移到设备（list 字段如 key 保留）。"""
        return {k: (v if isinstance(v, list) else v.to(self.device)) for k, v in batch.items()}

    def _loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向 + 计算总损失（训练/验证共用）。"""
        b = self._prepare(batch)
        out = self.model(b["trajectory"], b["h_c"])
        return encoder_total_loss(
            z=out["z"], s=out["s"], q_hat=out["q_hat"],
            rank=b["rank"], q_vec=b["q_vec"], valid_mask=b["valid_mask"],
            confidence=b["confidence"], scene_id=b["scene_id"],
            temperature=self.temperature,
            lambda_cross=self.lambda_cross, lambda_r=self.lambda_r, lambda_a=self.lambda_a,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练：损失 -> 反向 -> 梯度裁剪 -> 优化器。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self._loss(batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.step += 1
        return {key: self._to_scalar(value) for key, value in losses.items()}

    def _scene_loss(self, b: Dict[str, Any], scene_id_val: int) -> Dict[str, torch.Tensor]:
        """对单一批内指定场景（scene_id==scene_id_val）的子集单独计算 loss。"""
        mask = b["scene_id"] == scene_id_val
        if mask.sum() == 0:
            return {}
        out = self.model(b["trajectory"][mask], b["h_c"][mask])
        return encoder_total_loss(
            z=out["z"], s=out["s"], q_hat=out["q_hat"],
            rank=b["rank"][mask], q_vec=b["q_vec"][mask], valid_mask=b["valid_mask"][mask],
            confidence=b["confidence"][mask], scene_id=b["scene_id"][mask],
            temperature=self.temperature,
            lambda_cross=self.lambda_cross, lambda_r=self.lambda_r, lambda_a=self.lambda_a,
        )

    @torch.no_grad()
    def validate(self, loader: Iterable[Dict[str, Any]], *, max_batches: int | None = None) -> Dict[str, Any]:
        """验证：各项 loss 按批平均；分场景 loss 按场景子集独立计前向计算。"""
        self.model.eval()
        totals, count = {}, 0
        scene_totals = {"straight_free_drive": {}, "straight_car_follow": {}}
        scene_counts = {"straight_free_drive": 0, "straight_car_follow": 0}
        for batch in loader:
            b = self._prepare(batch)
            out = self.model(b["trajectory"], b["h_c"])
            losses = encoder_total_loss(
                z=out["z"], s=out["s"], q_hat=out["q_hat"],
                rank=b["rank"], q_vec=b["q_vec"], valid_mask=b["valid_mask"],
                confidence=b["confidence"], scene_id=b["scene_id"],
                temperature=self.temperature,
                lambda_cross=self.lambda_cross, lambda_r=self.lambda_r, lambda_a=self.lambda_a,
            )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + self._to_scalar(value)
            # 分场景：对 free/car 子集独立前向计算（一次遍历内统计，不重复遍历 loader）
            for scene_name, sid in (("straight_free_drive", 0), ("straight_car_follow", 1)):
                scene_loss = self._scene_loss(b, sid)
                if scene_loss:
                    scene_counts[scene_name] += 1
                    for key in ("loss", "rnc", "rank_loss", "axis_loss"):
                        scene_totals[scene_name][key] = scene_totals[scene_name].get(key, 0.0) + self._to_scalar(scene_loss[key])
            count += 1
            if max_batches is not None and count >= max_batches:
                break
        if count == 0:
            raise ValueError("Validation loader produced zero batches")
        result = {key: value / count for key, value in totals.items()}
        # 分场景按"出现该场景的批数"平均（避免未出现场景被 0 除）
        for scene_name in scene_counts:
            if scene_counts[scene_name]:
                scene_totals[scene_name] = {k: v / scene_counts[scene_name] for k, v in scene_totals[scene_name].items()}
        result["scene_losses"] = scene_totals
        return result

    def fit(self, train_loader: Iterable[Dict[str, Any]], *, max_steps: int,
            validation_loader: Iterable[Dict[str, Any]] | None = None, validate_every: int = 500,
            checkpoint_callback: Callable[[str, int, Mapping[str, Any]], None] | None = None) -> Dict[str, Any]:
        """单 epoch 训练（遍历 DataLoader 一轮，最多 max_steps 步）。

        同一 epoch 内 sampler 保证样本不重复；若 DataLoader 的批数少于 max_steps，
        本轮在批耗尽后自然结束（绝不重启 loader，避免同一 epoch 重复采样）。
        """
        if max_steps <= 0 or validate_every <= 0:
            raise ValueError("max_steps 和 validate_every 必须为正")
        start_epoch_step = self.step
        start_time, best = time.perf_counter(), {"loss": float("inf")}
        best_state, metrics = None, {}
        batches_seen = 0
        did_final_validate = False
        for batch in train_loader:
            batches_seen += 1
            metrics = self.train_step(batch)
            if checkpoint_callback is not None:
                checkpoint_callback("train", self.step, metrics)
            if validation_loader is not None and (self.step % validate_every == 0 or self.step >= start_epoch_step + max_steps):
                candidate = self.validate(validation_loader)
                if checkpoint_callback is not None:
                    checkpoint_callback("validation", self.step, candidate)
                if candidate["loss"] < best["loss"]:
                    best = candidate
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                did_final_validate = True
            # 达到本 epoch 步数上限即停止（批耗尽也自然停止，不重启 loader）
            if self.step >= start_epoch_step + max_steps:
                break
        # epoch 末：若 DataLoader 耗尽时没有任何验证执行（validate_every > epoch 长度），
        # 强制补一次验证，保证每 epoch 至少更新一次 best/last 候选；
        # 若最后一步刚验证过（did_final_validate），则不重复执行。
        if validation_loader is not None and batches_seen > 0 and not did_final_validate:
            candidate = self.validate(validation_loader)
            if checkpoint_callback is not None:
                checkpoint_callback("validation", self.step, candidate)
            if candidate["loss"] < best["loss"]:
                best = candidate
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return {
            "step": self.step - start_epoch_step,
            "batches_seen": batches_seen,
            "seconds": time.perf_counter() - start_time,
            "best_validation": best,
            "last_train": metrics,
            "best_state": best_state,
        }

    def save_checkpoint(self, path: str | Path, *, manifest_path: str, feature_index_path: str,
                        training_config: Mapping[str, Any], best_validation: Mapping[str, Any]) -> None:
        """保存编码器 checkpoint（含模型配置、manifest hash、特征索引 hash）。"""
        payload = {
            "model_config": {
                "trajectory_dim": self.model.temporal.proj.in_features,
                "hc_dim": self.model.hc_dim,  # 恒为 int，即使 hc_proj=Identity 也不为 None
                "d_model": self.model.d_model,
                "heads": self.model.cross_attn.num_heads,
                "z_dim": self.model.z_dim,
                "query_rank": self.model.query_cond_V.out_features,
            },
            "model_state": self.model.state_dict(),
            "manifest_sha256": _file_sha256(Path(manifest_path)),
            "feature_index_sha256": _file_sha256(Path(feature_index_path)),
            "training_config": dict(training_config),
            "best_validation": best_validation,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)