"""
diff_loss.py

Diffusion Loss Function
Author: Shangwen Li
Date: 2026-03-14
Description:
    扩散规划训练损失函数实现，支持两种训练目标：
    1) 'score': 预测归一化噪声（Score Matching）
    2) 'x_start': 直接预测去噪后的轨迹 x0
"""

from typing import Any, Callable, Dict, Tuple
import torch
import torch.nn as nn

from baseline.utils.normalizer import StateNormalizer


def _extract_diffusion_prediction(
        decoder_output: Dict[str, torch.Tensor],
        model_type: str,
        future_steps: int,
) -> torch.Tensor:
    candidate_keys = ["x_start", "score"] if model_type == "x_start" else ["score"]

    prediction = None
    selected_key = None
    for key in candidate_keys:
        if key in decoder_output:
            prediction = decoder_output[key]
            selected_key = key
            break

    if prediction is None:
        raise KeyError(
            "Diffusion decoder output does not contain a supported prediction key. "
            f"model_type={model_type}, available_keys={sorted(decoder_output.keys())}"
        )

    if prediction.ndim != 4:
        raise ValueError(
            "Diffusion decoder output must be a 4D tensor [B, P, T, D]. "
            f"Got key={selected_key}, shape={tuple(prediction.shape)}"
        )

    if prediction.shape[2] == future_steps + 1:
        prediction = prediction[:, :, 1:, :]
    elif prediction.shape[2] != future_steps:
        raise ValueError(
            "Diffusion decoder output time dimension does not match supervision. "
            f"key={selected_key}, pred_shape={tuple(prediction.shape)}, expected_future_steps={future_steps}"
        )

    return prediction


def diffusion_loss_func(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm: StateNormalizer,
        loss: Dict[str, Any],
        model_type: str,
        eps: float = 1e-3,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    计算扩散模型训练损失。

    Args:
        model: 扩散模型（通常为 Diffusion_Planner）
        inputs: 模型输入字典（ego 状态、邻车历史、地图上下文等）
        marginal_prob: SDE 边缘分布函数，返回 x_t 的 (mean, std)
        futures: (ego_future, neighbors_future, neighbor_future_mask)
        norm: 状态归一化器（用于 future 部分）
        loss: 损失字典（累积输出）
        model_type: 'score' 或 'x_start'
        eps: 扩散时间下界，避免 t=0 数值不稳定

    Returns:
        loss: 更新后的损失字典
        decoder_output: 模型解码器原始输出
    """

    # ===== 1. 解包 GT future =====
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask

    B, Pn, T, _ = neighbors_future.shape

    # ===== 2. 构造当前状态与 mask =====
    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

    # 合并当前与未来的 mask: [B, Pn, 1 + T]
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    # ===== 3. 拼接完整轨迹（current + future）=====
    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
    P = gt_future.shape[1]

    # 仅归一化 future，current 作为条件保持原值
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    # ===== 4. 前向扩散：采样 x_t =====
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps
    z = torch.randn_like(gt_future, device=gt_future.device)

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))
    xT_future = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT_future], dim=2)

    # ===== 5. 模型前向 =====
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    encoder_output, decoder_output = model(merged_inputs)
    # Reuse the exact same noised sample and frozen scene encoding for the
    # optional semantic-normal-vs-empty consistency pass. These private fields
    # are consumed only by the StylePlanner research loss and are not model
    # outputs or checkpoint parameters.
    decoder_output["_training_sampled_trajectories"] = xT.detach()
    decoder_output["_training_diffusion_time"] = t.detach()
    # Stage-A3.2 can reconstruct the same-noise x_t at a different, terminal
    # diffusion time for the paired preference branch. This private clean
    # trajectory is training-only and is never exported by the planner API.
    decoder_output["_training_clean_trajectories"] = all_gt.detach()
    if isinstance(encoder_output, dict) and "encoding" in encoder_output:
        decoder_output["_training_context_encoding"] = encoder_output["encoding"].detach()
    prediction = _extract_diffusion_prediction(
        decoder_output,
        model_type=model_type,
        future_steps=all_gt.shape[2] - 1,
    )

    # ===== 6. 损失计算 =====
    if model_type == "score":
        # 目标：预测归一化噪声
        dpm_loss = torch.sum((prediction * std + z) ** 2, dim=-1)
    elif model_type == "x_start":
        # 目标：预测去噪后的 future
        dpm_loss = torch.sum((prediction - all_gt[:, :, 1:, :]) ** 2, dim=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 邻车损失（仅有效 future 帧）
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]
    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Ego 损失（ego 始终有效）
    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    # 安全检查
    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output
