"""Continuous-preference LoRA loss and trainer.

连续偏好 LoRA 训练器：
- 复用 StyleLoRAPlanner（双分支 LoRA + rho 路由），内部 aggressive/conservative 槽位兼容
  research_lora checkpoint，对外方向语义为 high（=aggressive 槽） / low（=conservative 槽）；
- 复用固定噪声去噪（ego 真值 MSE）+ 邻车保持（冻结 baseline 预测）；
- 预测轨迹送入冻结 CSPQ（保留梯度，偏好损失回传 LoRA），MMD(z_pred,z_target) + rank Huber(s,r)；
- 总损失 L = ego_denoise + λn·neighbor + λz·MMD + λs·rank_huber
  + λdyn·dynamics + λq·factor_huber + λlat·lateral + λict·ICT。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from stylelora.lora.training.losses import build_noisy_inputs
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner

from stylelora.model.preference_encoder import CSPQPreferenceEncoder


def _prediction(output: dict, future_steps: int) -> torch.Tensor:
    value = output.get("x_start", output.get("score"))
    if value is None:
        raise KeyError(f"Decoder output has neither x_start nor score: {sorted(output)}")
    if value.shape[2] == future_steps + 1:
        value = value[:, :, 1:]
    if value.shape[2] != future_steps:
        raise ValueError(f"Unexpected prediction horizon {value.shape[2]}, expected {future_steps}")
    return value


def _mmd_rbf_biased(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """有偏 RBF MMD（含核矩阵对角线；论文表述需注明 biased）。"""
    def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(a, b).square() / (2 * sigma * sigma))
    return _kernel(x, x).mean() + _kernel(y, y).mean() - 2 * _kernel(x, y).mean()


def _build_pred_tokens(physical_ego: torch.Tensor) -> torch.Tensor:
    """物理 ego 轨迹 [B,T,4] -> CSPQ 轨迹 token [B,T,6]（pos + delta + cos/sin）。

    解码器/数据管线的物理状态约定为 [x, y, cosθ, sinθ]：第 2/3 维已经是
    归一化方向向量，不是 heading 弧度。因此这里直接做一次 L2 归一化后拼接，
    绝不能再把第 3 维当成 heading 去算 cos/sin（会产生错误的二次编码）。
    """
    pos = physical_ego[:, :, :2]
    delta = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        delta[:, 1:] = pos[:, 1:] - pos[:, :-1]
    heading_vec = F.normalize(physical_ego[:, :, 2:4], dim=-1, eps=1e-6)
    return torch.cat((pos, delta, heading_vec), dim=-1)


def _encode_preference_latent(cspq: CSPQPreferenceEncoder, physical_ego: torch.Tensor,
                              h_c: torch.Tensor) -> torch.Tensor:
    """将物理 ego 轨迹编码为冻结 CSPQ 偏好 latent，并保留输入轨迹的梯度。"""
    return cspq(_build_pred_tokens(physical_ego), h_c)["z"]


def _interpolation_consistency_loss(z_mid: torch.Tensor, z_base: torch.Tensor,
                                    z_endpoint: torch.Tensor,
                                    ratio: torch.Tensor) -> torch.Tensor:
    """ICT：中间强度 latent 对齐 baseline 与完整强度端点的线性插值。

    baseline 和端点只作为停止梯度的插值目标；梯度仅通过 ``z_mid`` 回传。
    ``ratio`` 可为 batch 共享标量，或与 batch 等长的一维张量。
    """
    if z_mid.shape != z_base.shape or z_mid.shape != z_endpoint.shape:
        raise ValueError("z_mid、z_base 与 z_endpoint 必须具有相同形状")
    if z_mid.ndim < 2:
        raise ValueError("偏好 latent 必须至少为 [B,D] 二维张量")

    ratio = torch.as_tensor(ratio, dtype=z_mid.dtype, device=z_mid.device)
    if ratio.ndim == 0:
        broadcast_ratio = ratio
    elif ratio.ndim == 1 and ratio.shape[0] == z_mid.shape[0]:
        broadcast_ratio = ratio.reshape(ratio.shape[0], *([1] * (z_mid.ndim - 1)))
    else:
        raise ValueError("ratio 必须是标量或与 batch 等长的一维张量")
    if not bool(torch.isfinite(ratio).all()) or bool(((ratio < 0) | (ratio > 1)).any()):
        raise ValueError("ratio 必须是位于 [0,1] 的有限数值")

    target = (1.0 - broadcast_ratio) * z_base.detach() + broadcast_ratio * z_endpoint.detach()
    return F.mse_loss(z_mid, target)


def _finite_diff(x: torch.Tensor, order: int) -> torch.Tensor:
    """沿时间维依次做 order 次一阶差分（每次长度 -1）。"""
    for _ in range(order):
        x = x[:, 1:] - x[:, :-1]
    return x


def _masked_factor_huber(q_hat: torch.Tensor, q_vec: torch.Tensor,
                         valid_mask: torch.Tensor) -> torch.Tensor:
    """掩码加权三因子 Huber（q_hat 对齐专家 axis_percentiles）。

    不用 rank_confidence 加权——轴间冲突本身就是需要保留的高维信息；
    分母 clamp_min(1.0) 保证全无效掩码时安全返回 0。
    """
    loss = F.huber_loss(q_hat, q_vec, reduction="none")
    mask = valid_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _dynamics_consistency_loss(pred_xy: torch.Tensor, gt_xy: torch.Tensor, *,
                               dt: float = 0.1, accel_scale: float = 3.0,
                               jerk_scale: float = 20.0) -> dict:
    """Ego 动力学一致性损失（归一化 Huber）。

    时间尺度（物理量纲）：
    - 速度：一阶差分 / dt；
    - 加速度：二阶差分 / dt^2；
    - jerk：三阶差分 / dt^3。
    为避免数值尺度过大，加速度按 /3、jerk 按 /20 归一化后再取 Huber。

    Returns:
        {"dynamics": accel Huber + jerk Huber, "acceleration": accel Huber,
         "jerk": jerk Huber}，全部为标量张量。
    """
    # 有限差分只给出离散帧差，必须除以 dt 的对应幂次才能得到物理量纲：
    # 一阶差分 -> 速度 /dt；二阶差分 -> 加速度 /dt^2；三阶差分 -> jerk /dt^3。
    # 若只除一次 dt，加速度尺度实际是 a·dt、jerk 尺度是 j·dt^2，动力学约束会被严重弱化。
    pred_a = _finite_diff(pred_xy, 2) / (dt ** 2)
    gt_a = _finite_diff(gt_xy, 2) / (dt ** 2)
    pred_j = _finite_diff(pred_xy, 3) / (dt ** 3)
    gt_j = _finite_diff(gt_xy, 3) / (dt ** 3)
    accel = F.huber_loss(pred_a / accel_scale, gt_a / accel_scale)
    jerk = F.huber_loss(pred_j / jerk_scale, gt_j / jerk_scale)
    return {"dynamics": accel + jerk, "acceleration": accel, "jerk": jerk}


def _lateral_residual_loss(adaptive_ego: torch.Tensor, base_ego: torch.Tensor,
                           expert_ego: torch.Tensor, current_xy: torch.Tensor, *,
                           tolerance: float = 0.3, topk_ratio: float = 0.2,
                           smooth_weight: float = 0.1, eps: float = 1e-6) -> dict:
    """专家感知的基线保持横向约束。"""
    if adaptive_ego.shape != base_ego.shape or adaptive_ego.ndim != 3:
        raise ValueError("adaptive_ego 与 base_ego 必须是相同形状的 [B,T,D] 张量")
    if adaptive_ego.shape[-1] < 4:
        raise ValueError("轨迹末维必须至少包含 [x,y,cos,sin]")
    if expert_ego.ndim != 3 or expert_ego.shape[:2] != adaptive_ego.shape[:2] or expert_ego.shape[-1] < 2:
        raise ValueError("expert_ego 必须是与预测轨迹批次和时域一致的 [B,T,D] 张量")
    if current_xy.shape != adaptive_ego.shape[:1] + (1, 2):
        raise ValueError("current_xy 必须是 [B,1,2]")
    if tolerance <= 0:
        raise ValueError("lateral tolerance 必须大于 0")
    if not 0 < topk_ratio <= 1:
        raise ValueError("lateral topk ratio 必须位于 (0,1] 内")
    if smooth_weight < 0:
        raise ValueError("lateral smooth weight 不能小于 0")

    adaptive_xy = adaptive_ego[..., :2]
    base_xy = base_ego[..., :2]
    expert_xy = expert_ego[..., :2].to(adaptive_xy)
    expert_path = torch.cat((current_xy.to(expert_xy), expert_xy), dim=1)
    expert_step = expert_path[:, 1:] - expert_path[:, :-1]

    # 专家低速帧无法稳定确定切向时，回退到 baseline 输出的朝向。
    step_norm = torch.linalg.vector_norm(expert_step, dim=-1, keepdim=True)
    step_tangent = expert_step / step_norm.clamp_min(eps)
    heading = base_ego[..., 2:4]
    heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    heading_tangent = heading / heading_norm.clamp_min(eps)
    fallback = torch.zeros_like(expert_step)
    fallback[..., 0] = 1.0
    tangent = torch.where(
        step_norm > eps,
        step_tangent,
        torch.where(heading_norm > eps, heading_tangent, fallback),
    )
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)

    adaptive_error = ((adaptive_xy - expert_xy) * normal).sum(dim=-1).abs()
    base_error = ((base_xy - expert_xy) * normal).sum(dim=-1).abs()
    excess = F.relu(adaptive_error - base_error - tolerance).square()
    topk = max(1, math.ceil(excess.shape[1] * topk_ratio))
    lateral_offset = torch.topk(excess, topk, dim=1).values.mean()

    lateral_displacement = ((adaptive_xy - base_xy) * normal).sum(dim=-1)
    if lateral_displacement.shape[1] >= 3:
        lateral_second_diff = _finite_diff(lateral_displacement.unsqueeze(-1), 2)
        lateral_smoothness = F.huber_loss(
            lateral_second_diff, torch.zeros_like(lateral_second_diff))
    else:
        # 保留与预测轨迹相连的零值，短预测时 backward 仍然安全。
        lateral_smoothness = lateral_displacement.sum() * 0.0
    lateral = lateral_offset + smooth_weight * lateral_smoothness
    return {
        "lateral": lateral,
        "lateral_offset": lateral_offset,
        "lateral_smoothness": lateral_smoothness,
        "lateral_max_abs": lateral_displacement.abs().max(),
    }


def preference_lora_loss(*, planner: StyleLoRAPlanner, cspq: CSPQPreferenceEncoder,
                         inputs: dict, futures: tuple, marginal_prob, state_normalizer,
                         batch: dict, direction: str,
                         lambda_n: float = 1.0, lambda_z: float = 1.0, lambda_s: float = 1.0,
                         lambda_dyn: float = 0.0, lambda_q: float = 0.0,
                         lambda_lat: float = 0.0, lateral_tolerance: float = 0.3,
                         lateral_topk_ratio: float = 0.2,
                         lateral_smooth_weight: float = 0.1,
                         lambda_ict: float = 0.0, ict_alpha: float = 1.0) -> dict:
    """计算连续偏好 LoRA 总损失。"""
    if lambda_ict < 0:
        raise ValueError("lambda_ict 不能小于 0")
    if ict_alpha <= 0:
        raise ValueError("ict_alpha 必须大于 0")
    if lambda_ict > 0:
        if direction not in {"high", "low"}:
            raise ValueError("启用 ICT 时 direction 必须是 high 或 low")
        # ICT 端点始终使用完整强度；lambda_ict=0 时不触碰原有路由状态。
        planner.set_strength(1.0 if direction == "high" else -1.0)

    # 1) 固定噪声构造（自适应与冻结 baseline 共享 x_t/t）
    noisy_inputs, target, neighbors_valid = build_noisy_inputs(
        inputs, futures, marginal_prob, state_normalizer,
    )
    # 2) 自适应模型（当前方向分支）前向
    _, adaptive_output = planner(noisy_inputs)
    adaptive = _prediction(adaptive_output, target.shape[2])
    # 3) 冻结 baseline 前向（no_grad，得到邻居保持基准）
    with torch.no_grad():
        was_enabled = getattr(planner, "_enabled", None)
        if hasattr(planner, "disable_adapter"):
            planner.disable_adapter()
        _, base_output = planner(noisy_inputs)
        if was_enabled is not None and was_enabled:
            planner.enable_adapter()
        base = _prediction(base_output, target.shape[2])
    # 4) ego 去噪
    ego_denoise = ((adaptive[:, 0] - target[:, 0]) ** 2).sum(dim=-1).mean()
    # 5) 邻车保持
    neighbor = ((adaptive[:, 1:] - base[:, 1:]) ** 2).sum(dim=-1)[neighbors_valid].mean() if neighbors_valid.any() else adaptive.new_zeros(())
    # 6) 偏好对齐：物理预测轨迹 -> 冻结 CSPQ（保留梯度）-> s/z
    # state_normalizer 的均值/方差形状是 [agent, 1, D]（agent 维度在前），
    # 必须先对完整 agent 维度 inverse，再取 ego token[:, 0]；
    # 不能先取 adaptive[:, 0] 再 inverse，否则 [A,1,D] 无法与 [B,T,D] 广播。
    physical_ego = state_normalizer.inverse(adaptive)[:, 0]  # [B, T, 4]: x, y, cos, sin
    with torch.no_grad():
        physical_base_ego = state_normalizer.inverse(base)[:, 0]
    pred_tokens = _build_pred_tokens(physical_ego)
    out = cspq(pred_tokens, batch["h_c"])
    z_pred, s_pred = out["z"], out["s"].squeeze(-1)
    # 置信度加权 rank Huber + MMD
    conf_w = batch["confidence"].clamp_min(0.0)
    rank_huber = (F.huber_loss(s_pred, batch["rank"], reduction="none") * conf_w).sum() / conf_w.sum().clamp_min(1e-12)
    mmd_loss = _mmd_rbf_biased(z_pred, batch["z_target"])

    # 6b) CSPQ 三因子对齐：q_hat[3] 对齐 axis_percentiles（掩码加权 Huber）。
    factor_huber = _masked_factor_huber(out["q_hat"], batch["q_vec"], batch["valid_mask"])

    # 6c) 插值一致性训练：相同扩散噪声下，中间 rho 的偏好 latent 应位于
    # baseline 与完整强度端点之间。关闭时不增加任何额外前向，保持旧训练行为。
    ict_loss = adaptive.new_zeros(())
    ict_rho = adaptive.new_zeros(())
    if lambda_ict > 0:
        concentration = adaptive.new_tensor(float(ict_alpha))
        ratio = torch.distributions.Beta(concentration, concentration).sample()
        endpoint_strength = 1.0 if direction == "high" else -1.0
        ict_rho = ratio if direction == "high" else -ratio
        try:
            planner.set_strength(float(ict_rho.detach().cpu()))
            _, mid_output = planner(noisy_inputs)
            mid = _prediction(mid_output, target.shape[2])
        finally:
            # 中间前向结束或异常退出时都恢复完整端点，避免污染后续训练与验证。
            planner.set_strength(endpoint_strength)

        physical_mid_ego = state_normalizer.inverse(mid)[:, 0]
        z_mid = _encode_preference_latent(cspq, physical_mid_ego, batch["h_c"])
        with torch.no_grad():
            z_base = _encode_preference_latent(cspq, physical_base_ego, batch["h_c"])
        ict_loss = _interpolation_consistency_loss(z_mid, z_base, z_pred, ratio)

    # 7) Ego 动力学一致性：位置 xy [B,T,2] + 当前帧 [B,1,2] -> [B,T+1,2]
    batch_device = next(iter(batch["tensors"].values())).device
    current_xy = batch["tensors"]["ego_current_state"][:, None, :2].to(
        physical_ego.device if physical_ego.device != batch_device else batch_device)
    if current_xy.device != physical_ego.device:
        current_xy = current_xy.to(physical_ego.device)
    pred_xy = torch.cat((current_xy, physical_ego[..., :2]), dim=1)
    expert_ego = futures[0].to(physical_ego.device)
    gt_xy = torch.cat((current_xy, expert_ego[..., :2]), dim=1)
    dynamics = _dynamics_consistency_loss(pred_xy, gt_xy)

    # 8) 专家感知的横向基线保持：只惩罚相对 baseline 增加的专家横向误差。
    lateral = _lateral_residual_loss(
        physical_ego, physical_base_ego, expert_ego, current_xy,
        tolerance=lateral_tolerance, topk_ratio=lateral_topk_ratio,
        smooth_weight=lateral_smooth_weight)

    total = (ego_denoise + lambda_n * neighbor + lambda_z * mmd_loss + lambda_s * rank_huber
             + lambda_dyn * dynamics["dynamics"] + lambda_q * factor_huber
             + lambda_lat * lateral["lateral"])
    if lambda_ict > 0:
        total = total + lambda_ict * ict_loss
    return {
        "loss": total, "ego_denoise": ego_denoise, "neighbor": neighbor,
        "mmd_z": mmd_loss, "rank_huber": rank_huber,
        "dynamics": dynamics["dynamics"], "acceleration": dynamics["acceleration"],
        "jerk": dynamics["jerk"], "factor_huber": factor_huber,
        "lateral": lateral["lateral"], "lateral_offset": lateral["lateral_offset"],
        "lateral_smoothness": lateral["lateral_smoothness"],
        "lateral_max_abs": lateral["lateral_max_abs"],
        "ict": ict_loss, "ict_rho": ict_rho,
        "s_mean": s_pred.mean(), "s_std": s_pred.std(),
    }


class PreferenceLoRATrainer:
    """连续偏好 LoRA 训练器。"""

    def __init__(self, planner: StyleLoRAPlanner, cspq: CSPQPreferenceEncoder, *,
                 observation_normalizer, state_normalizer, device: str, direction: str,
                 learning_rate: float = 1e-4, lambda_n: float = 1.0,
                 lambda_z: float = 1.0, lambda_s: float = 1.0, lambda_dyn: float = 0.0,
                 lambda_q: float = 0.0, lambda_lat: float = 0.0,
                 lateral_tolerance: float = 0.3, lateral_topk_ratio: float = 0.2,
                 lateral_smooth_weight: float = 0.1, lambda_ict: float = 0.0,
                 ict_alpha: float = 1.0, grad_clip_norm: float = 5.0) -> None:
        self.planner = planner
        self.cspq = cspq.eval()
        self.observation_normalizer = observation_normalizer
        self.state_normalizer = state_normalizer
        self.device = torch.device(device)
        self.direction = direction
        self.lambda_n, self.lambda_z, self.lambda_s, self.lambda_dyn, self.lambda_q = (
            lambda_n, lambda_z, lambda_s, lambda_dyn, lambda_q)
        self.lambda_lat = lambda_lat
        self.lateral_tolerance = lateral_tolerance
        self.lateral_topk_ratio = lateral_topk_ratio
        self.lateral_smooth_weight = lateral_smooth_weight
        self.lambda_ict = lambda_ict
        self.ict_alpha = ict_alpha
        self.grad_clip_norm = grad_clip_norm
        self.optimizer = torch.optim.AdamW(
            [p for p in planner.parameters() if p.requires_grad], lr=learning_rate)
        self.style = "aggressive" if direction == "high" else "conservative"
        planner.set_style(self.style).set_strength(1.0 if direction == "high" else -1.0)
        self.step = 0

    def _prepare_batch(self, batch: dict) -> dict:
        # tensors 是嵌套 dict，先跳过；其余字段（张量/列表）按类型移动
        prepared = {}
        for key, value in batch.items():
            if key == "tensors":
                continue
            prepared[key] = value if isinstance(value, list) else value.to(self.device)
        prepared["tensors"] = {k: v.to(self.device) for k, v in batch["tensors"].items()}
        return prepared

    def _loss_from_batch(self, b: dict) -> dict:
        from stylelora.lora.runtime import prepare_diffusion_batch
        metadata = [{"scene_type": ("straight_free_drive" if sid == 0 else "straight_car_follow"),
                     "cache_path": "", "log_name": "", "token": ""}
                    for sid in b["scene_id"].cpu().tolist()]
        prepared_inputs, futures = prepare_diffusion_batch(
            {"tensors": b["tensors"], "metadata": metadata},
            self.device, self.observation_normalizer, return_style_context=False)
        return preference_lora_loss(
            planner=self.planner, cspq=self.cspq, inputs=prepared_inputs, futures=futures,
            marginal_prob=self.planner.sde.marginal_prob, state_normalizer=self.state_normalizer,
            batch=b, direction=self.direction,
            lambda_n=self.lambda_n, lambda_z=self.lambda_z, lambda_s=self.lambda_s,
            lambda_dyn=self.lambda_dyn, lambda_q=self.lambda_q,
            lambda_lat=self.lambda_lat, lateral_tolerance=self.lateral_tolerance,
            lateral_topk_ratio=self.lateral_topk_ratio,
            lateral_smooth_weight=self.lateral_smooth_weight,
            lambda_ict=self.lambda_ict, ict_alpha=self.ict_alpha,
        )

    @torch.no_grad()
    def validate(self, loader, *, max_batches: int | None = None, seed: int | None = None) -> dict:
        """验证集平均损失。seed 给定则按 batch 固定扩散随机（time/noise），保证多次验证可比。"""
        self.planner.eval()
        totals, count = {}, 0
        fork_devices = [
            self.device.index if self.device.index is not None else torch.cuda.current_device()
        ] if self.device.type == "cuda" else []
        for batch in loader:
            b = self._prepare_batch(batch)
            if seed is not None:
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(seed + count)
                    losses = self._loss_from_batch(b)
            else:
                losses = self._loss_from_batch(b)
            for key in ("loss", "ego_denoise", "neighbor", "mmd_z", "rank_huber",
                        "dynamics", "acceleration", "jerk", "factor_huber", "lateral",
                        "lateral_offset", "lateral_smoothness", "lateral_max_abs",
                        "ict", "ict_rho"):
                totals[key] = totals.get(key, 0.0) + float(losses[key].detach().cpu())
            count += 1
            if max_batches is not None and count >= max_batches:
                break
        if count == 0:
            raise ValueError("Validation loader produced zero batches")
        return {k: v / count for k, v in totals.items()}

    def train_step(self, batch: dict) -> dict:
        self.planner.train()
        self.optimizer.zero_grad(set_to_none=True)
        b = self._prepare_batch(batch)
        losses = self._loss_from_batch(b)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.planner.parameters() if p.requires_grad], self.grad_clip_norm)
        self.optimizer.step()
        self.step += 1
        return {k: float(v.detach().cpu()) for k, v in losses.items()}


def load_frozen_cspq(checkpoint_path: str, device: str) -> CSPQPreferenceEncoder:
    """从编码器 checkpoint 重建冻结 CSPQ。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    model = CSPQPreferenceEncoder(
        trajectory_dim=config["trajectory_dim"], hc_dim=config["hc_dim"], d_model=config["d_model"],
        heads=config["heads"], z_dim=config["z_dim"], query_rank=config["query_rank"],
    )
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()
