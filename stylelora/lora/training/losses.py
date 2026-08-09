"""Fixed-noise x_start loss: expert ego supervision plus frozen-baseline neighbour preservation.

固定噪声 x_start 损失：自车（ego）跟随专家轨迹监督 + 冻结基线的邻居（neighbour）行为保持。
核心思路：
1. 用"自车对准专家目标、邻居对准冻结基线预测"的方式训练 LoRA；
2. 自适应模型与冻结基线模型共享同一个加噪输入 x_t 和时间 t，
   保证两者差异完全来自 LoRA 增量，而不是扩散采样噪声。
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import nn

from stylelora.lora.evaluation.style_metrics import differentiable_scene_style_vector
from stylelora.lora.training.style_prototypes import SceneStylePrototypeTable


def _prediction(output: Dict[str, torch.Tensor], future_steps: int) -> torch.Tensor:
    """从解码器输出字典中提取未来轨迹预测张量。

    Args:
        output: 解码器输出字典，需包含 "x_start"（预测原始数据）或 "score"（预测噪声/分数）。
        future_steps: 期望的未来预测步数。

    Returns:
        预测轨迹张量，形状为 [B, P, future_steps, 4]。

    Raises:
        KeyError: 输出字典中既没有 x_start 也没有 score。
        ValueError: 预测的时间范围与 future_steps 不匹配。
    """
    # 优先取 x_start（预测原始数据），否则取 score（预测噪声）
    value = output.get("x_start", output.get("score"))
    if value is None:
        raise KeyError(f"Decoder output has neither x_start nor score: {sorted(output)}")
    # 若包含当前帧（未来步数 + 1 帧），则去掉当前帧，只保留未来轨迹
    if value.shape[2] == future_steps + 1:
        value = value[:, :, 1:]
    # 最终必须精确等于 future_steps，否则说明预测范围与目标不一致
    if value.shape[2] != future_steps:
        raise ValueError(f"Unexpected prediction horizon {value.shape[2]}, expected {future_steps}")
    return value


def build_noisy_inputs(inputs: Dict[str, torch.Tensor], futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                       marginal_prob, normalizer, *, time: torch.Tensor | None = None,
                       noise: torch.Tensor | None = None, eps: float = 1e-3):
    """用固定（或随机生成的）时间 t 与噪声构造扩散前向加噪输入。

    这是"固定噪声 x_start 损失"的关键：时间与噪声要么传入（可复现/对齐），
    要么随机生成，保证自适应与基线模型吃到的加噪样本完全一致。

    Args:
        inputs: 原始输入字典（含 ego_current_state、neighbor_agents_past 等）。
        futures: (ego_future, neighbors_future, neighbor_future_mask) 三元组，
            - ego_future: 自车未来真值
            - neighbors_future: 邻居未来真值
            - neighbor_future_mask: 邻居是否有有效未来帧的掩码
        marginal_prob: SDE 的边缘概率函数，返回 (mean, std)。
        normalizer: 观测归一化器（对未来真值做归一化）。
        time: 扩散时间 t，未提供则随机采样（落在 [eps, 1] 保证数值稳定）。
        noise: 高斯噪声，未提供则 randn_like 生成。
        eps: 时间下限，避免 t=0 导致数值不稳定。

    Returns:
        (加噪后的输入字典, 归一化后的未来目标, 邻居有效掩码) 三元组。
    """
    # 解包：自车未来、邻居未来、邻居未来有效掩码
    ego_future, neighbors_future, neighbor_future_mask = futures
    # 有效邻居 = 掩码取反（该掩码 True 表示无效帧）
    neighbors_valid = ~neighbor_future_mask.bool()
    bsz, neighbor_count, _, _ = neighbors_future.shape
    # 当前状态：自车取输入的当前前 4 维，邻居取过去序列最后一帧前 4 维
    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :neighbor_count, -1, :4]
    # 当前帧无效掩码：邻居当前位置全 0 视为无效
    current_mask = torch.sum(neighbors_current != 0, dim=-1) == 0
    # 拼接"当前帧掩码 + 未来帧掩码"，覆盖整条时间序列（第 0 帧为当前帧，P 维度从 1 开始是邻居）
    full_mask = torch.cat((current_mask.unsqueeze(-1), neighbor_future_mask.bool()), dim=-1)
    # 拼接自车与邻居的未来/当前状态，形成 [B, P, T, D]
    future = torch.cat((ego_future[:, None], neighbors_future), dim=1)
    current = torch.cat((ego_current[:, None], neighbors_current), dim=1)
    # 目标序列 = 当前帧 + 归一化后的未来帧（[B, P, T+1, D]）
    target = torch.cat((current[:, :, None], normalizer(future)), dim=2)
    # 无效的邻居时间步全部置 0（对齐基线的 padding 约定）
    target[:, 1:][full_mask] = 0.0
    # 时间 t：默认在 [eps, 1] 内均匀采样（避免 t=0 的奇异性）
    time = time if time is not None else torch.rand(bsz, device=future.device) * (1 - eps) + eps
    # 高斯噪声：默认按 future 形状生成标准正态噪声
    noise = noise if noise is not None else torch.randn_like(future)
    # 用 marginal_prob 计算加噪目标的均值与标准差（只对未来帧加噪，保留当前帧）
    mean, std = marginal_prob(target[:, :, 1:], time)
    # 把 std 广播成与 future 相同的形状
    std = std.view(-1, *([1] * (future.ndim - 1)))
    # 加噪序列 = 干净当前帧 + (mean + std * noise) 的未来帧
    noisy = torch.cat((target[:, :, :1], mean + std * noise), dim=2)
    # 返回：带加噪样本的输入字典（供模型前向）、归一化未来目标、邻居有效掩码
    return {**inputs, "sampled_trajectories": noisy, "diffusion_time": time}, target[:, :, 1:], neighbors_valid


def _active_style(model: nn.Module) -> str:
    """将当前路由状态转换为原型文件使用的完整风格名。"""
    style = getattr(model, "_style", "")
    if style == "aggr":
        return "aggressive"
    if style == "cons":
        return "conservative"
    raise RuntimeError("Style-prototype loss requires the aggressive or conservative LoRA branch")


def _style_prototype_terms(*, adaptive: torch.Tensor, normalizer, style_context: Mapping[str, object],
                           prototype_table: SceneStylePrototypeTable, target_style: str,
                           margin: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算场景内三轴原型距离和正确/相反风格间隔，且全程保留预测轨迹梯度。"""
    required = {
        "ego_current_state", "ego_future_gt", "neighbor_agents_past", "neighbors_future_gt",
        "neighbors_future_valid_mask", "route_lanes_speed_limit", "route_lanes_has_speed_limit", "lanes_speed_limit",
        "lanes_has_speed_limit", "scene_types",
    }
    missing = required.difference(style_context)
    if missing:
        raise KeyError(f"Style context is missing {sorted(missing)}")

    # adaptive 是 state-normalized 的 x_start。inverse 不会 detach，因此三轴损失可反传到 LoRA。
    physical_prediction = normalizer.inverse(adaptive)[:, 0]
    scene_types = style_context["scene_types"]
    if not isinstance(scene_types, tuple) or len(scene_types) != physical_prediction.shape[0]:
        raise ValueError("scene_types must match the batch size")
    distances, margins = [], []
    valid_samples = 0
    opposite_style = prototype_table.opposite(target_style)
    for index, scene in enumerate(scene_types):
        vector, valid = differentiable_scene_style_vector(
            scene=scene,
            ego_future=physical_prediction[index],
            ego_current=style_context["ego_current_state"][index],
            neighbors_past=style_context["neighbor_agents_past"][index],
            neighbors_future=style_context["neighbors_future_gt"][index],
            route_limits=style_context["route_lanes_speed_limit"][index],
            route_has_limits=style_context["route_lanes_has_speed_limit"][index],
            lane_limits=style_context["lanes_speed_limit"][index],
            lane_has_limits=style_context["lanes_has_speed_limit"][index],
            lead_reference_future=style_context["ego_future_gt"][index],
            neighbor_future_valid_mask=style_context["neighbors_future_valid_mask"][index],
        )
        # car-follow 没有有效前车时 valid 为全 False：该样本只保留原有去噪和邻车损失。
        if not bool(valid.all()):
            continue
        standardized = prototype_table.standardize(scene, vector)
        correct = (standardized - prototype_table.prototype(scene, target_style, like=standardized)).square().mean()
        opposite = (standardized - prototype_table.prototype(scene, opposite_style, like=standardized)).square().mean()
        distances.append(correct)
        margins.append(torch.relu(correct - opposite + float(margin)))
        valid_samples += 1
    if not distances:
        zero = adaptive.new_zeros(())
        return zero, zero, zero, zero
    count = adaptive.new_tensor(float(valid_samples))
    fraction = count / float(physical_prediction.shape[0])
    return torch.stack(distances).mean(), torch.stack(margins).mean(), count, fraction


def style_diffusion_loss(model: nn.Module, base_model: nn.Module, inputs: Dict[str, torch.Tensor],
                         futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], marginal_prob, normalizer,
                         *, neighbor_weight: float = 1.0, lora_reg_weight: float = 0.0,
                         style_context: Mapping[str, object] | None = None,
                         prototype_table: SceneStylePrototypeTable | None = None,
                         prototype_weight: float = 0.0, prototype_margin_weight: float = 0.0,
                         prototype_margin: float = 0.20, time: torch.Tensor | None = None,
                         noise: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Adaptive/base forwards share *the same* x_t and t by construction.

    风格扩散损失：自适应模型与冻结基线模型构造上共享同一个 x_t 与 t。
    损失由三部分主损失和一个可选的训练集原型辅助项组成：
    1. ego_target_loss —— 自车轨迹逼近专家真值（风格化目标）；
    2. neighbor_preserve_loss —— 邻居轨迹保持与冻结基线一致（避免风格化破坏他人行为）；
    3. lora_regularization —— 当前激活风格分支 LoRA 参数的 L2 平均（可选正则）；
    4. style_prototype_loss / style_margin_loss —— 反归一化物理轨迹在同场景训练集
       三轴原型空间中靠近正确风格、并比相反风格更近的辅助约束。

    Args:
        model: 风格 LoRA 规划器（自适应模型，其 LoRA 参数需要训练）。
        base_model: 冻结基线模型（同一个模型引用或等价模型，禁用适配器后前向）。
        inputs: 原始输入字典。
        futures: (ego_future, neighbors_future, neighbor_future_mask) 真值三元组。
        marginal_prob: SDE 的边缘概率函数 (mean, std)。
        normalizer: 观测归一化器。
        neighbor_weight: 邻居保持损失的权重系数。
        lora_reg_weight: LoRA 正则项权重（0 表示不启用）。
        style_context: 未归一化的道路、邻车、专家轨迹和每条样本的场景名。
        prototype_table: 训练集生成且冻结的 scene × style 三轴原型；None 表示纯 MSE。
        prototype_weight: 正确原型距离的辅助权重。
        prototype_margin_weight: 正确/相反原型间隔的辅助权重。
        prototype_margin: 要求正确原型距离至少比相反原型距离小的间隔。
        time: 可复现的扩散时间 t（默认随机）。
        noise: 可复现的高斯噪声（默认随机）。

    Returns:
        Dict 包含：
        - loss: 加权后的总损失（反传目标）。
        - ego_target_loss: 自车专家监督损失（MSE，仅自车 token 0）。
        - neighbor_preserve_loss: 邻居保持损失（MSE，仅有效邻居）。
        - lora_regularization: 当前风格分支 LoRA 参数的 L2 均值。
        - style_prototype_loss/style_margin_loss: 未加权的风格辅助项，便于 SwanLab 对比。
    """
    # 1) 构造固定噪声/时间的加噪输入：此步保证两次前向（自适应与基线）输入完全一致
    noisy_inputs, target, neighbors_valid = build_noisy_inputs(
        inputs, futures, marginal_prob, normalizer, time=time, noise=noise
    )
    # 2) 自适应模型前向：预测未来轨迹
    _, adaptive_output = model(noisy_inputs)
    adaptive = _prediction(adaptive_output, target.shape[2])
    # 3) 冻结基线前向：在 no_grad 下禁用适配器，得到"未风格化"的基线预测
    with torch.no_grad():
        # 记忆原开关状态，前向后恢复
        was_enabled = getattr(base_model, "_enabled", None)
        if hasattr(base_model, "disable_adapter"):
            base_model.disable_adapter()
        _, base_output = base_model(noisy_inputs)
        if was_enabled is not None and was_enabled:
            base_model.enable_adapter()
        base = _prediction(base_output, target.shape[2])
    # 4) 自车专家监督：只对自车 token（索引 0）计算与真值的 MSE
    ego = ((adaptive[:, 0] - target[:, 0]) ** 2).sum(dim=-1).mean()
    # 5) 邻居保持：只对"有效邻居"计算自适应与基线预测的差异（邻居行为不能被风格化破坏）
    if neighbors_valid.any():
        neighbour = ((adaptive[:, 1:] - base[:, 1:]) ** 2).sum(dim=-1)[neighbors_valid].mean()
    else:
        # 没有有效邻居时返回标量 0（仍是一个 0 维张量，便于求和）
        neighbour = adaptive.new_zeros(())
    # 6) 原型辅助项：只在传入训练集原型时启用；纯 MSE 配置保持逐项完全不变。
    if prototype_table is None:
        prototype, margin_loss = adaptive.new_zeros(()), adaptive.new_zeros(())
        prototype_valid_samples, prototype_valid_fraction = adaptive.new_zeros(()), adaptive.new_zeros(())
    else:
        if style_context is None:
            raise ValueError("style_context is required when prototype_table is enabled")
        prototype, margin_loss, prototype_valid_samples, prototype_valid_fraction = _style_prototype_terms(
            adaptive=adaptive, normalizer=normalizer, style_context=style_context,
            prototype_table=prototype_table, target_style=_active_style(model), margin=prototype_margin,
        )
    # 7) 正则化：只对"当前激活风格分支"的 LoRA A/B 参数做 L2 均值惩罚。
    active_branch = "aggressive" if getattr(model, "_style", "") == "aggr" else "conservative"
    regularizer = sum((p ** 2).mean() for name, p in model.named_parameters() if f".{active_branch}.lora_" in name)
    # 8) 原有 MSE/邻车项仍为主目标；三轴原型仅以小权重作为方向性辅助。
    total = (ego + float(neighbor_weight) * neighbour + float(lora_reg_weight) * regularizer
             + float(prototype_weight) * prototype + float(prototype_margin_weight) * margin_loss)
    # 返回总损失与各分项（便于训练日志记录与监控）
    return {"loss": total, "ego_target_loss": ego, "neighbor_preserve_loss": neighbour,
            "lora_regularization": regularizer, "style_prototype_loss": prototype,
            "style_margin_loss": margin_loss, "style_valid_samples": prototype_valid_samples,
            "style_valid_fraction": prototype_valid_fraction}


