"""Step-oriented, AMP-safe LoRA training loop with frozen-base checks and small checkpoints.

面向单步训练（step-oriented）、AMP 安全的 LoRA 训练循环：
- 冻结基座校验：训练前后保证只有 LoRA 参数可训练、基座权重零改动；
- 小体积 checkpoint：只保存最优验证损失时激活风格分支的 LoRA 适配器参数（A/B 矩阵），
  而不是整个模型，从而产出极小体积的适配器权重文件。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

import torch
from torch import nn

from stylelora.lora.model.checkpoint import save_adapter_checkpoint
from stylelora.lora.model.injector import assert_only_lora_trainable
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.training.losses import style_diffusion_loss
from stylelora.lora.training.style_prototypes import SceneStylePrototypeTable


def _unpack_prepared_batch(prepared):
    """兼容旧的 ``(inputs, futures)`` 与原型训练的三元 batch 返回格式。"""
    if not isinstance(prepared, tuple):
        raise TypeError("prepare_batch must return a tuple")
    if len(prepared) == 2:
        inputs, futures = prepared
        return inputs, futures, None
    if len(prepared) == 3:
        return prepared
    raise ValueError(f"Unexpected prepare_batch return length {len(prepared)}")


class StyleLoRATrainer:
    """风格 LoRA 训练器：封装单步训练、验证与完整 fit 循环。

    支持 AMP（混合精度）混合精度缩放（GradScaler），但默认全精度：
    由于 DiffPlanner 冻结编码器在 autocast FP16 下存在不合法的内存操作，
    除非基线编码器已被改造为 AMP-safe，否则不启用 AMP（详见 __init__ 注释）。
    """

    def __init__(self, planner: StyleLoRAPlanner, *, normalizer, device: str, learning_rate: float = 1e-4,
                 grad_clip_norm: float = 5.0, amp: bool = False,
                 prototype_table: SceneStylePrototypeTable | None = None,
                 prototype_weight: float = 0.0, prototype_margin_weight: float = 0.0,
                 prototype_margin: float = 0.20) -> None:
        """初始化训练器：校验可训练参数、创建优化器与 AMP 缩放器。

        Args:
            planner: 已注入 LoRA 的风格规划器（StyleLoRAPlanner）。
            normalizer: 观测归一化器（与 losses 中使用的同一实例）。
            device: 设备字符串（如 "cuda:0" / "cpu"）。
            learning_rate: AdamW 学习率。
            grad_clip_norm: 全局梯度裁剪范数上限。
            amp: 是否启用混合精度训练。
            prototype_table: 训练集 scene × style 三轴原型；None 时退化为原有纯 MSE。
            prototype_weight/prototype_margin_weight: 原型距离和方向间隔的辅助权重。
            prototype_margin: 正确原型需要优于相反原型的最小距离间隔。
        """
        # 保存核心组件；device 解析为 torch.device
        self.planner, self.normalizer, self.device = planner, normalizer, torch.device(device)
        self.prototype_table = prototype_table
        self.prototype_weight = float(prototype_weight)
        self.prototype_margin_weight = float(prototype_margin_weight)
        self.prototype_margin = float(prototype_margin)
        if (self.prototype_weight or self.prototype_margin_weight) and self.prototype_table is None:
            raise ValueError("Non-zero prototype weights require a train-split style-prototype artifact")
        if self.prototype_weight < 0 or self.prototype_margin_weight < 0 or self.prototype_margin < 0:
            raise ValueError("Prototype weights and prototype margin must be non-negative")
        # 冻结基座不变量校验：确保只有 LoRA A/B 矩阵可训练
        assert_only_lora_trainable(planner)
        # 优化器只作用于 requires_grad=True 的参数（即各风格分支的 LoRA 参数）
        self.optimizer = torch.optim.AdamW([p for p in planner.parameters() if p.requires_grad], lr=learning_rate)
        self.grad_clip_norm = grad_clip_norm
        # DiffPlanner's frozen encoder writes autocast FP16 activations into
        # preallocated FP32 tensors via indexed assignment.  That operation is
        # invalid in PyTorch, so LoRA training must default to full precision
        # unless the baseline encoder itself is made AMP-safe.
        # DiffPlanner 的冻结编码器会用"索引赋值"把 autocast FP16 激活写入预分配的
        # FP32 张量——这在 PyTorch 中是非法操作。因此 LoRA 训练默认全精度，
        # 除非基线编码器本身被改造成 AMP 安全。
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp and self.device.type == "cuda")
        # 已完成的训练步数
        self.step = 0

    def train_step(self, inputs, futures, *, style_context=None, neighbor_weight=1.0, lora_reg_weight=0.0) -> Dict[str, float]:
        """执行单个训练步：前向计算损失 → 反向传播 → 梯度裁剪 → 优化器步进。

        Args:
            inputs: 归一化后的输入字典。
            futures: (ego_future, neighbors_future, neighbor_future_mask) 真值三元组。
            neighbor_weight: 邻居保持损失权重。
            lora_reg_weight: LoRA 正则项权重。

        Returns:
            本步损失各项的标量字典（已 detach 并移到 CPU，便于日志记录）。
        """
        # 训练模式 + 清空梯度（set_to_none 比 zero_ 更快）
        self.planner.train(); self.optimizer.zero_grad(set_to_none=True)
        # 在 autocast 上下文内计算损失（自动混合精度，若 GradScaler 启用）
        with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            # 注意：model 与 base_model 都传 self.planner——
            # style_diffusion_loss 内部会对 base_model 禁用适配器后前向，
            # 因此等效于"自适应模型 + 冻结基线"的两次前向。
            losses = style_diffusion_loss(self.planner, self.planner, inputs, futures, self.planner.sde.marginal_prob,
                                          self.normalizer, neighbor_weight=neighbor_weight, lora_reg_weight=lora_reg_weight,
                                          style_context=style_context, prototype_table=self.prototype_table,
                                          prototype_weight=self.prototype_weight,
                                          prototype_margin_weight=self.prototype_margin_weight,
                                          prototype_margin=self.prototype_margin)
        # AMP：损失先缩放再反向，防止低精度下梯度下溢
        self.scaler.scale(losses["loss"]).backward()
        # AMP：unscale 梯度以便正确裁剪
        self.scaler.unscale_(self.optimizer)
        # 梯度裁剪：只裁剪可训练（LoRA）参数，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_([p for p in self.planner.parameters() if p.requires_grad], self.grad_clip_norm)
        # AMP：更新优化器并更新缩放因子；训练步数 +1
        self.scaler.step(self.optimizer); self.scaler.update(); self.step += 1
        # 返回各项损失（标量、CPU、断开计算图）
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    @torch.no_grad()
    def validate(self, loader, prepare_batch: Callable, *, max_batches: int | None = None) -> Dict[str, float]:
        """在验证集上评估当前模型：累加各项损失并求平均。

        Args:
            loader: 验证数据加载器。
            prepare_batch: 把原始 batch 转换为 (inputs, futures) 的函数。
            max_batches: 最多评估的批次数（None 表示全部）。

        Returns:
            各项损失的均值字典（batch 维度上的平均，而非样本维度）。

        Raises:
            ValueError: 验证加载器没有产生任何批次。
        """
        # 评估模式：关闭 dropout / 归一化统计更新
        self.planner.eval()
        totals, count = {}, 0
        for batch in loader:
            # 数据搬到设备并归一化
            prepared = prepare_batch(batch, self.device)
            inputs, futures, style_context = _unpack_prepared_batch(prepared)
            # 使用与训练相同的原型项；邻居和 LoRA 正则沿用既有验证口径（1.0 / 0.0）。
            losses = style_diffusion_loss(self.planner, self.planner, inputs, futures, self.planner.sde.marginal_prob,
                                          self.normalizer, style_context=style_context, prototype_table=self.prototype_table,
                                          prototype_weight=self.prototype_weight,
                                          prototype_margin_weight=self.prototype_margin_weight,
                                          prototype_margin=self.prototype_margin)
            # 累加各项损失
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            # 达到上限批次则提前结束
            if max_batches is not None and count >= max_batches:
                break
        if not count:
            raise ValueError("Validation loader produced zero batches")
        # 按批次数求平均（batch 平均）
        return {key: value / count for key, value in totals.items()}

    def fit(self, train_loader, prepare_batch: Callable, *, max_steps: int, validation_loader=None,
            validate_every: int = 500, neighbor_weight: float = 1.0, lora_reg_weight: float = 0.0,
            metrics_callback: Callable[[str, int, Mapping[str, float]], None] | None = None) -> Dict[str, Any]:
        """执行完整训练：逐 epoch 迭代训练，周期性验证并保存最优适配器。

        Args:
            train_loader: 训练数据加载器（其 sampler 可提供 set_epoch 以实现可复现采样）。
            prepare_batch: 把原始 batch 转换为 (inputs, futures) 的函数。
            max_steps: 训练总步数上限。
            validation_loader: 验证加载器（None 表示不验证）。
            validate_every: 每多少步做一次验证。
            neighbor_weight: 邻居保持损失权重（透传给 train_step）。
            lora_reg_weight: LoRA 正则权重（透传给 train_step）。
            metrics_callback: 可选指标回调，参数依次为阶段名、全局 step 和指标字典。

        Returns:
            Dict 包含训练统计：
            - steps / epochs: 实际训练步数与经历 epoch 数
            - seconds / steps_per_second: 耗时与吞吐
            - peak_cuda_bytes: GPU 峰值显存（CPU 时为 0）
            - best_validation: 最优验证损失各项
            - last_train: 最后一步的训练损失各项
            - best_adapter_state: 最优验证时"当前激活风格分支"的 LoRA 适配器参数

        Raises:
            ValueError: max_steps/validate_every 非正，或训练加载器产生零批次。
        """
        # 参数合法性校验
        if max_steps <= 0 or validate_every <= 0:
            raise ValueError("max_steps and validate_every must be positive")
        # 计时开始；best_loss 初始为无穷大
        start, best = time.perf_counter(), {"loss": float("inf")}
        # best_adapter_state：最优验证时的适配器参数；metrics：最后一步训练损失
        best_adapter_state, epoch, metrics = None, 0, {}
        # 主循环：按总步数上限控制训练
        while self.step < max_steps:
            # 若 sampler 支持 set_epoch（如 SceneBalancedSampler），
            # 每个 epoch 重设随机种子以产生可复现的不同采样
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            batch_count = 0
            for batch in train_loader:
                # 数据准备 + 单步训练
                prepared = prepare_batch(batch, self.device)
                inputs, futures, style_context = _unpack_prepared_batch(prepared)
                metrics = self.train_step(inputs, futures, style_context=style_context,
                                          neighbor_weight=neighbor_weight, lora_reg_weight=lora_reg_weight)
                batch_count += 1
                if metrics_callback is not None:
                    metrics_callback("train", self.step, metrics)
                # 周期性验证：每 validate_every 步或到达 max_steps 时评估一次
                if validation_loader is not None and (self.step % validate_every == 0 or self.step == max_steps):
                    candidate = self.validate(validation_loader, prepare_batch)
                    if metrics_callback is not None:
                        metrics_callback("validation", self.step, candidate)
                    # 验证损失更优时：记录最优结果并保存当前激活风格分支的适配器参数
                    if candidate["loss"] < best["loss"]:
                        best = candidate
                        active_style = "aggressive" if self.planner._style == "aggr" else "conservative"
                        best_adapter_state = self.planner.adapter_state_dict(active_style)
                # 达到总步数上限则提前结束本 epoch（外层 while 也会退出）
                if self.step >= max_steps:
                    break
            if not batch_count:
                raise ValueError("Training loader produced zero batches; reduce batch size or inspect manifest/cache data")
            epoch += 1
        # 训练结束：统计耗时与显存峰值
        elapsed = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        # 汇总结果字典
        result = {"steps": self.step, "seconds": elapsed, "steps_per_second": self.step / max(elapsed, 1e-9),
                  "peak_cuda_bytes": memory, "best_validation": best, "last_train": metrics}
        result["epochs"] = epoch
        result["best_adapter_state"] = best_adapter_state
        return result


