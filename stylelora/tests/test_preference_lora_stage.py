"""Self-contained tests for the continuous-preference LoRA stage.

覆盖专家验收：
- rank 区间过滤 + latent/h_c 对齐；
- 每批两场景严格 1:1；
- collate 形状（含真实 .npz cache 加载）；
- CSPQ 梯度能回传到当前方向 LoRA 分支（预测轨迹过冻结 CSPQ backward）；
- 仅当前方向分支有梯度（另一分支 grad 为 None）；
- 真正跑一次 rho=0 identity 数值对比（适配器关闭输出 == 冻结 baseline 输出）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from stylelora.lora.model.injector import iter_style_layers
from stylelora.lora.model.lora_layers import EgoMaskedLoRALinear
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner

from stylelora.data.preference_lora_dataset import (
    PreferenceLoRADataset,
    SceneBalancedLoRASampler,
    preference_lora_collate,
)
from stylelora.model.preference_encoder import CSPQPreferenceEncoder
from stylelora.training.preference_lora import (
    _build_pred_tokens,
    _dynamics_consistency_loss,
    _finite_diff,
    _masked_factor_huber,
    preference_lora_loss,
)


# ---------------------------------------------------------------------------
# 轻量 dummy 基线：可被 inject_style_lora 按默认目标后缀注入，forward 输出物理预测
# ---------------------------------------------------------------------------
class _Mlp(nn.Module):
    """timm 风格 Mlp(linear) 结构：模块名匹配 mlp1.fc1 等注入目标后缀。"""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _DummyPlanner(nn.Module):
    """最小 DiffPlanner 替身：含可注入目标层，forward 契约与真实 Diffusion_Planner
    完全一致：返回 (encoder_outputs, decoder_outputs) 二元组，decoder_outputs["x_start"]
    形状为 [B, P, T+1, D]（第 0 帧为当前帧，_prediction 按 future_steps 截掉）。

    - to_d 把 [B,T,6] 轨迹 token 聚合为 [B,1,d]；
    - 注入层只作用于 token 0（与 EgoMaskedLoRALinear 语义一致）；
    - 输出 [B, P, T+1, 4] 物理轨迹（x,y,cos,sin）。
    """

    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.to_d = nn.Linear(6, d)
        self.mlp1 = _Mlp(d)
        self.mlp2 = _Mlp(d)
        # 与真实 DiffPlanner FinalLayer 结构一致：proj = Sequential(LN, Linear, GELU, LN, Linear)，
        # 保证模块名是 final_layer.proj.1 / final_layer.proj.4，能被 DEFAULT_TARGETS 命中。
        self.final_layer = nn.Module()
        self.final_layer.proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d),
        )
        # 模拟冻结基座：所有原始参数不可训练（注入后只有 LoRA A/B 可训练）
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: dict) -> dict:
        traj = inputs["traj"]                                   # [B, T, 6]
        x = self.to_d(traj.mean(dim=1, keepdim=True))           # [B, 1, d]
        x = self.mlp1(x)                                        # [B, 1, d]
        x = self.mlp2(x)                                        # [B, 1, d]
        x = self.final_layer.proj(x)                            # [B, 1, d]
        pos = x[..., :2]
        heading = x[..., 2:3].squeeze(-1)                       # [B, 1] 弧度
        phys = torch.cat((pos, torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)),
                         dim=-1)                                 # [B, 1, 4]
        # 真实 DiffPlanner 的 x_start 形状为 [B, P, T+1, D]（第 0 帧为当前帧，
        # _prediction 会按 future_steps 截掉）。P = 1（ego）+ 邻居数；无邻居输入时 P=1。
        if "neighbor_agents_past" in inputs:
            agent_total = 1 + int(inputs["neighbor_agents_past"].shape[1])
        else:
            agent_total = 1
        phys = phys.unsqueeze(1).expand(-1, agent_total, traj.shape[1] + 1, -1).contiguous()
        # 与真实 Diffusion_Planner.forward 契约一致：返回 (encoder_outputs, decoder_outputs)，
        # 其中 decoder_outputs 是含 "x_start" 的字典。
        return None, {"x_start": phys}


def _featurize_ego(pred_phys: torch.Tensor) -> torch.Tensor:
    """测试侧与训练侧共用同一 token 构造：直接使用 cos/sin 列（真实管线形态）。"""
    if pred_phys.ndim == 4:
        pred_phys = pred_phys[:, 0]  # [B, P, T, 4] -> ego [B, T, 4]
    return _build_pred_tokens(pred_phys)


def _make_planner_and_frozen_cspq(seed: int = 7):
    """构造注入 LoRA 的 dummy planner + 冻结最小 CSPQ。"""
    torch.manual_seed(seed)
    dummy = _DummyPlanner()
    planner = StyleLoRAPlanner(dummy, rank=2, dropout=0.0)
    cspq = CSPQPreferenceEncoder(trajectory_dim=6, hc_dim=8, d_model=16, heads=2, z_dim=4, query_rank=2)
    for parameter in cspq.parameters():
        parameter.requires_grad_(False)
    cspq.eval()
    return planner, cspq


def test_cspq_gradient_flows_to_active_lora_branch() -> None:
    """预测轨迹过冻结 CSPQ 后 backward，当前方向（aggr/high）所有注入层 lora_A/B 有梯度。"""
    planner, cspq = _make_planner_and_frozen_cspq()
    planner.set_style("aggr").set_strength(1.0)  # high 方向 -> aggressive 分支激活
    # 模拟训练后的状态：lora_B 已非零（初始化 B=0 时 ∂loss/∂A = x^T·B 恒为零，
    # 无法验证 A 的梯度链路，因此先把 B 扰动为非零再验证 A/B 都有非零梯度）。
    with torch.no_grad():
        for _, layer in iter_style_layers(planner.baseline):
            layer.aggressive.lora_B.normal_(0.0, 0.1)
    inputs = {"traj": torch.randn(3, 12, 6)}     # [B, T, 6]（真实 token 形态）
    _, decoder_outputs = planner(inputs)         # 与真实 DiffPlanner 契约一致：二元组
    out = decoder_outputs["x_start"]             # [B, P, T+1, 4] 物理预测
    pred_tokens = _featurize_ego(out)
    h_c = torch.randn(3, 8)
    pref = cspq(pred_tokens, h_c)
    loss = pref["s"].mean() + pref["z"].sum()    # 同时覆盖 s 与 z 两条回传路径
    loss.backward()
    for name, layer in iter_style_layers(planner.baseline):
        assert layer.aggressive.lora_A.grad is not None, f"{name} aggressive lora_A 无梯度"
        assert layer.aggressive.lora_B.grad is not None, f"{name} aggressive lora_B 无梯度"
        assert layer.aggressive.lora_A.grad.abs().sum() > 0, f"{name} aggressive lora_A 梯度为零"
        assert layer.aggressive.lora_B.grad.abs().sum() > 0, f"{name} aggressive lora_B 梯度为零"


def test_inactive_lora_branch_has_no_gradient() -> None:
    """当前方向分支（aggr）激活时，另一分支（cons）lora_A/B 的 grad 必须为 None。"""
    planner, cspq = _make_planner_and_frozen_cspq()
    planner.set_style("aggr").set_strength(1.0)
    inputs = {"traj": torch.randn(3, 12, 6)}
    _, decoder_outputs = planner(inputs)
    out = decoder_outputs["x_start"]
    loss = cspq(_featurize_ego(out), torch.randn(3, 8))["s"].mean()
    loss.backward()
    for name, layer in iter_style_layers(planner.baseline):
        assert layer.conservative.lora_A.grad is None, f"{name} conservative lora_A 不应有梯度"
        assert layer.conservative.lora_B.grad is None, f"{name} conservative lora_B 不应有梯度"
    # 冻结基座（非 lora_ 参数）同样无梯度
    for name, parameter in planner.named_parameters():
        if ".aggressive.lora_" not in name and ".conservative.lora_" not in name:
            assert parameter.grad is None, f"冻结基座参数 {name} 不应有梯度"


def test_rho_zero_identity_equals_frozen_baseline() -> None:
    """真正数值对比：adapter 关闭的包装器输出 == 同一冻结 baseline 输出（rho=0 identity）。"""
    torch.manual_seed(0)
    dummy = _DummyPlanner()
    inputs = {"traj": torch.randn(2, 10, 6)}
    with torch.no_grad():
        y_before = dummy(inputs)[1]["x_start"]   # 注入前（纯冻结 baseline，decoder 输出）
    planner = StyleLoRAPlanner(dummy, rank=2, dropout=0.0)
    planner.disable_adapter()                     # rho=0 路由（normal/关闭）
    with torch.no_grad():
        y_after = planner(inputs)[1]["x_start"]  # 注入后 + adapter 关闭（decoder 输出）
    assert y_before.shape == y_after.shape
    max_diff = float((y_before - y_after).abs().max())
    assert max_diff < 1e-6, f"rho=0 identity 失效：max|Δ|={max_diff}"


class _StateNormalizerMock:
    """模拟 state_normalizer：均值形状为 [agent, 1, D]（agent 维度在前）。

    复现真实 normalizer 的广播语义：若 preference_lora_loss 错误地先取
    adaptive[:, 0] 再 inverse，[agent,1,D] 均值无法与 [B,T,D] 广播而报错；
    正确实现必须先对完整 agent 维度 inverse 再取 [:, 0]。
    """

    def __init__(self, agent: int = 11, dim: int = 4) -> None:
        self._mean = torch.zeros(agent, 1, dim)
        self._std = torch.ones(agent, 1, dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std + self._mean


def _loss_fixture(seed: int = 3):
    """构造真实 preference_lora_loss 所需的全部项（符合 build_noisy_inputs 契约）。

    N 取 10、normalizer agent 取 11（1 ego + 10 neighbor），与真实管线一致：
    state_normalizer 均值形状 [11, 1, D] 沿 P=1+N=11 维广播；neighbors_valid 为
    [B, 10, T] 与 adaptive[:, 1:] 的 [B, 10, T, D] 索引对齐。
    """
    torch.manual_seed(seed)
    planner, cspq = _make_planner_and_frozen_cspq()
    planner.set_style("aggr").set_strength(1.0)
    bsz, T, D, N = 4, 12, 4, 10
    inputs = {
        "traj": torch.randn(bsz, T, 6),
        "ego_current_state": torch.randn(bsz, 8),
        "neighbor_agents_past": torch.randn(bsz, N, 10, 8),
    }
    ego_future = torch.randn(bsz, T, D)
    neighbors_future = torch.randn(bsz, N, T, D)
    neighbor_mask = torch.zeros(bsz, N, T, dtype=torch.bool)  # 全有效
    futures = (ego_future, neighbors_future, neighbor_mask)
    norm = _StateNormalizerMock(agent=11, dim=D)

    class _Marginal:
        def __call__(self, target, t):
            return target, target.new_ones(target.shape[:1] + (1,) * (target.ndim - 1)) * 0.1

    batch = {
        "h_c": torch.randn(bsz, 8),
        "z_target": torch.randn(bsz, 4),
        "rank": torch.full((bsz,), 0.9),
        "confidence": torch.ones(bsz),
        "q_vec": torch.rand(bsz, 3),            # CSPQ 三因子对齐目标（0~1）
        "valid_mask": torch.ones(bsz, 3, dtype=torch.bool),  # 全有效
        "tensors": {"ego_current_state": torch.randn(bsz, 8)},  # 动力学一致性损失需要当前帧 xy
    }
    return planner, cspq, inputs, futures, norm, _Marginal(), batch


def test_preference_lora_loss_inverse_on_full_agent_dim() -> None:
    """真实 preference_lora_loss：inverse 必须作用于完整 agent 维度，再取 ego（不抛错）。"""
    planner, cspq, inputs, futures, norm, marginal, batch = _loss_fixture()
    result = preference_lora_loss(
        planner=planner, cspq=cspq, inputs=inputs, futures=futures,
        marginal_prob=marginal, state_normalizer=norm, batch=batch, direction="high",
    )
    assert torch.isfinite(result["loss"])


def test_build_pred_tokens_uses_cos_sin_columns_directly() -> None:
    """_build_pred_tokens 不应把第 3 维当作 heading 重算 cos/sin。"""
    phys = torch.zeros(2, 5, 4)
    phys[:, :, 0] = 1.0          # x
    phys[:, :, 1] = 2.0          # y
    phys[:, :, 2] = 0.0          # cos
    phys[:, :, 3] = 1.0          # sin（heading=0 时重算会得到 cos=1,sin=0，必须不相同）
    tokens = _build_pred_tokens(phys)
    assert tokens.shape == (2, 5, 6)
    assert torch.allclose(tokens[:, :, 4], torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(tokens[:, :, 5], torch.ones(2, 5), atol=1e-5)


def _smooth_xy(bsz: int = 2, length: int = 8) -> torch.Tensor:
    """线性匀速 xy 序列 [B, T, 2]（速度恒定 -> 加速度/jerk 全 0）。"""
    t = torch.arange(length, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    base = torch.randn(bsz, 1, 2)
    v = torch.full((bsz, 1, 2), 2.0)
    return base + v * t


def test_dynamics_loss_near_zero_when_pred_equals_gt() -> None:
    """pred == gt（且当前帧一致）时动力学损失接近 0。"""
    gt = _smooth_xy()
    pred = gt.clone()
    out = _dynamics_consistency_loss(pred, gt)
    assert float(out["dynamics"]) < 1e-6
    assert float(out["acceleration"]) < 1e-6
    assert float(out["jerk"]) < 1e-6


def test_dynamics_loss_jerk_increases_with_alternating_jitter() -> None:
    """给预测位置加入交替抖动后，jerk 明显大于平滑预测的 jerk。"""
    gt = _smooth_xy()
    pred_jitter = gt.clone()
    pred_jitter[:, ::2, 0] += 0.5    # 交替符号抖动 -> 三阶差分（jerk）显著增大
    smooth = _dynamics_consistency_loss(gt, gt)["jerk"]
    jittered = _dynamics_consistency_loss(pred_jitter, gt)["jerk"]
    assert float(jittered) > float(smooth) + 1e-3


def test_dynamics_loss_backpropagates_to_prediction() -> None:
    """动力学损失能向预测轨迹反传梯度。

    注意：pred == gt 时损失为零、梯度也必然为零，无法验证回传链路；
    必须先给 pred 加小扰动使 loss 非零，再 backward 检查非零梯度。
    """
    gt = _smooth_xy()
    pred = gt.clone()
    pred[:, ::2, 0] += 0.05
    pred = pred.requires_grad_(True)
    out = _dynamics_consistency_loss(pred, gt)
    out["dynamics"].backward()
    assert pred.grad is not None
    assert float(pred.grad.abs().sum()) > 0


def test_dynamics_acceleration_scale_is_dt_squared() -> None:
    """解析检查：匀加速轨迹二阶差分 / dt^2 必须等于指定加速度（锁定 dt**2 / dt**3）。

    注意：t 必须乘 dt，否则构造轨迹的采样间隔是 1（而非 0.1s），
    二阶差分原始值 = a·(1s)² = 3，再除以 dt²=0.01 会得到 300 而非 3。
    """
    dt = 0.1
    # x(t) = 2 + 1*t + 0.5*3*t^2：x0=2, v0=1, a=3；y 恒 0
    # float64 保证解析差分的数值误差远小于容差
    t = torch.arange(10, dtype=torch.float64) * dt
    x = 2.0 + 1.0 * t + 0.5 * 3.0 * t * t
    xy = torch.stack((x, torch.zeros_like(x)), dim=-1).unsqueeze(0)  # [1, T, 2]
    accel = _finite_diff(xy, 2) / (dt ** 2)
    # 每帧二阶差分 / dt^2 都应等于 3.0（x 维）；y 维为 0
    assert torch.allclose(accel[0, :, 0], torch.full_like(accel[0, :, 0], 3.0), atol=1e-9)
    assert torch.allclose(accel[0, :, 1], torch.zeros_like(accel[0, :, 1]), atol=1e-9)
    # 匀加速 -> 三阶差分 / dt^3 恒为 0（jerk = 0）
    jerk = _finite_diff(xy, 3) / (dt ** 3)
    assert float(jerk.abs().max()) < 1e-8


def test_lambda_q_zero_recovers_loss_without_factor() -> None:
    """lambda_q=0 时总损失与旧公式（无三因子项）数值一致。"""
    planner, cspq, inputs, futures, norm, marginal, batch = _loss_fixture()
    result = preference_lora_loss(
        planner=planner, cspq=cspq, inputs=inputs, futures=futures,
        marginal_prob=marginal, state_normalizer=norm, batch=batch, direction="high",
        lambda_n=1.0, lambda_z=1.0, lambda_s=1.0, lambda_dyn=0.0, lambda_q=0.0,
    )
    expected = (result["ego_denoise"] + result["neighbor"]
                + result["mmd_z"] + result["rank_huber"])
    assert torch.allclose(result["loss"], expected.detach(), atol=1e-5)


def test_factor_huber_zero_when_all_masked() -> None:
    """valid_mask 全 0 时 factor_huber 安全返回 0（分母 clamp_min(1.0)）。"""
    planner, cspq, inputs, futures, norm, marginal, batch = _loss_fixture()
    batch["valid_mask"] = torch.zeros_like(batch["valid_mask"])
    result = preference_lora_loss(
        planner=planner, cspq=cspq, inputs=inputs, futures=futures,
        marginal_prob=marginal, state_normalizer=norm, batch=batch, direction="high",
    )
    assert float(result["factor_huber"].detach().cpu()) == 0.0
    assert torch.isfinite(result["loss"])


def test_factor_huber_near_zero_when_q_hat_matches_q_vec() -> None:
    """直接调用 _masked_factor_huber：q_hat 与 q_vec 完全一致时等于 0。"""
    rng = torch.Generator().manual_seed(0)
    q_hat = torch.rand(4, 3, generator=rng)
    q_vec = q_hat.clone()
    valid_mask = torch.ones(4, 3, dtype=torch.bool)
    value = _masked_factor_huber(q_hat, q_vec, valid_mask)
    assert float(value) < 1e-6


def test_lambda_q_backpropagates_to_lora_not_cspq() -> None:
    """lambda_q>0 时 factor_huber 能向激活 LoRA 分支反传梯度，冻结 CSPQ 无梯度。"""
    planner, cspq = _make_planner_and_frozen_cspq()
    planner.set_style("aggr").set_strength(1.0)
    with torch.no_grad():
        for _, layer in iter_style_layers(planner.baseline):
            layer.aggressive.lora_B.normal_(0.0, 0.1)  # 非零 B 保证 A 梯度可达
    # 最小化：直接构造 batch，使 factor_huber 经 q_hat -> 预测轨迹回传
    bsz, T, N = 3, 12, 2
    inputs = {"traj": torch.randn(bsz, T, 6), "ego_current_state": torch.randn(bsz, 8),
              "neighbor_agents_past": torch.randn(bsz, N, 10, 8)}
    ego_future = torch.randn(bsz, T, 4)
    futures_full = (ego_future, torch.randn(bsz, N, T, 4), torch.zeros(bsz, N, T, dtype=torch.bool))
    norm = _StateNormalizerMock(agent=1 + N, dim=4)
    class _Marginal2:
        def __call__(self, target, t):
            return target, target.new_ones(target.shape[:1] + (1,) * (target.ndim - 1)) * 0.1
    batch = {"h_c": torch.randn(bsz, 8), "z_target": torch.randn(bsz, 4),
             "rank": torch.full((bsz,), 0.9), "confidence": torch.ones(bsz),
             "q_vec": torch.rand(bsz, 3), "valid_mask": torch.ones(bsz, 3, dtype=torch.bool),
             "tensors": {"ego_current_state": torch.randn(bsz, 8)}}
    result = preference_lora_loss(
        planner=planner, cspq=cspq, inputs=inputs, futures=futures_full,
        marginal_prob=_Marginal2(), state_normalizer=norm, batch=batch, direction="high",
        lambda_q=1.0,
    )
    # 只对 factor_huber 反传：梯度必须来自因子损失本身，
    # 而不是被 ego/rank/MMD 的梯度掩盖
    result["factor_huber"].backward()
    # 激活分支有梯度
    any_lora_grad = False
    for _, layer in iter_style_layers(planner.baseline):
        if layer.aggressive.lora_A.grad is not None and layer.aggressive.lora_A.grad.abs().sum() > 0:
            any_lora_grad = True
        if layer.aggressive.lora_B.grad is not None and layer.aggressive.lora_B.grad.abs().sum() > 0:
            any_lora_grad = True
    assert any_lora_grad, "factor_huber 未向激活 LoRA 分支回传梯度"
    # 冻结 CSPQ 参数无梯度
    for name, parameter in cspq.named_parameters():
        assert parameter.grad is None, f"冻结 CSPQ 参数 {name} 不应有梯度"


def test_lambda_dyn_zero_recovers_original_loss() -> None:
    """lambda_dyn=0 时总损失与旧公式（无动力学项）数值一致。"""
    planner, cspq, inputs, futures, norm, marginal, batch = _loss_fixture()
    result = preference_lora_loss(
        planner=planner, cspq=cspq, inputs=inputs, futures=futures,
        marginal_prob=marginal, state_normalizer=norm, batch=batch, direction="high",
        lambda_n=1.0, lambda_z=1.0, lambda_s=1.0, lambda_dyn=0.0,
    )
    # 旧公式：loss = ego_denoise + neighbor + mmd_z + rank_huber（λ 均 =1）
    expected = (result["ego_denoise"] + result["neighbor"]
                + result["mmd_z"] + result["rank_huber"])
    assert torch.allclose(result["loss"], expected.detach(), atol=1e-5)


# ---------------------------------------------------------------------------
# 数据侧（原有覆盖保持）
# ---------------------------------------------------------------------------
def _make_cache(path: Path, idx: int) -> None:
    """生成一个最小 DiffPlanner 缓存 npz（含基线字段 + ego/neighbor future）。"""
    from stylelora.lora.data.dataset import BASELINE_TENSOR_KEYS
    payload = {}
    for key in BASELINE_TENSOR_KEYS:
        if key == "neighbor_agents_past":
            payload[key] = np.zeros((10, 10, 4), dtype=np.float32)
        elif key == "static_objects":
            payload[key] = np.zeros((4, 8), dtype=np.float32)
        elif key.endswith("lanes") or "lanes" in key:
            payload[key] = np.zeros((20, 5, 4), dtype=np.float32)
        elif key.endswith("speed_limit") or "speed_limit" in key:
            payload[key] = np.zeros((20, 5, 1), dtype=np.float32)
        elif key.endswith("has_speed_limit"):
            payload[key] = np.zeros((20, 5, 1), dtype=np.float32)
        else:
            payload[key] = np.zeros((4, 10, 4), dtype=np.float32)
    payload["ego_future_gt"] = np.zeros((20, 4), dtype=np.float32)
    payload["neighbors_future_gt"] = np.zeros((10, 20, 4), dtype=np.float32)
    payload["neighbor_agents_future_mask"] = np.ones((10, 20), dtype=bool)
    np.savez(path / f"cache_{idx}.npz", **payload)


def _make_manifest_and_banks(path: Path, n_per_scene: int = 8) -> Path:
    """rank 两场景各自覆盖完整 0~1（保证 high/low 窗口两场景都有样本）。"""
    rows, num = [], 0
    for s in range(2):
        scene = "straight_free_drive" if s == 0 else "straight_car_follow"
        for i in range(n_per_scene):
            rank = i / max(n_per_scene - 1, 1)  # 每个场景 0~1
            idx = s * n_per_scene + i
            _make_cache(path, idx)
            rows.append({
                "scene_type": scene,
                "axis_names": ["speed_preference", "longitudinal_intensity", "smoothness"],
                "axis_raw": [rank, 1.0, 1.0],
                "axis_valid": [True, True, True],
                "axis_percentiles": [rank, 0.5, 0.5],
                "preference_rank": float(rank),
                "rank_confidence": 1.0,
                "cache_path": f"cache_{idx}.npz",
                "log_name": "log",
                "token": str(idx),
            })
            num += 1
    manifest = path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    np.save(path / "latent.npy", np.random.RandomState(0).randn(num, 8).astype(np.float32))
    with (path / "latent_index.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(num):
            handle.write(json.dumps({"fid": i, "key": f"log:{i}"}, ensure_ascii=False) + "\n")
    np.save(path / "features.npy", np.random.RandomState(1).randn(num, 192).astype(np.float32))
    with (path / "features_index.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(num):
            handle.write(json.dumps({"fid": i, "log_name": "log", "token": str(i)}, ensure_ascii=False) + "\n")
    return manifest


def _build_ds(root: Path, direction="high", lo=0.8, hi=1.0):
    return PreferenceLoRADataset(
        root / "manifest.jsonl", root, root / "latent.npy", root / "latent_index.jsonl",
        root / "features.npy", root / "features_index.jsonl",
        direction=direction, rank_low=lo, rank_high=hi,
    )


def test_dataset_alignment_and_window() -> None:
    """high 方向只取 rank 窗口内样本，且 latent/h_c 全部对齐（key=log:i）。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_manifest_and_banks(root)
        ds = _build_ds(root, direction="high", lo=0.8, hi=1.0)
        assert ds.missing == 0
        assert all(s.preference_rank >= 0.8 for s in ds.samples)
        assert len(ds) == len(ds.latent_rows) == len(ds.fids)
        # 每场景 rank 覆盖 0~1，high 窗口两场景都应非空
        assert len(ds.free_indices) > 0 and len(ds.car_indices) > 0


def test_balanced_sampler_1to1() -> None:
    """每批两场景严格各半。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_manifest_and_banks(root)
        ds = _build_ds(root, direction="low", lo=0.0, hi=0.6)
        assert len(ds.free_indices) > 0 and len(ds.car_indices) > 0, "窗口过窄导致采样空循环"
        sampler = SceneBalancedLoRASampler(ds, batch_size=4)
        batches = list(sampler)
        assert batches, "采样器不应空"
        for batch in batches:
            scenes = [ds.samples[i].scene_type for i in batch]
            assert scenes.count("straight_free_drive") == scenes.count("straight_car_follow") == 2


def test_collate_with_real_cache() -> None:
    """collate 会触发 __getitem__ 加载真实 .npz cache，形状正确。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_manifest_and_banks(root)
        ds = _build_ds(root, direction="high", lo=0.8, hi=1.0)
        indices = list(ds.free_indices[:2] + ds.car_indices[:2])
        collated = preference_lora_collate([ds[i] for i in indices])
        assert collated["h_c"].shape == (4, 192)
        assert collated["z_target"].shape == (4, 8)
        assert collated["rank"].shape == (4,)
        assert "tensors" in collated and "ego_current_state" in collated["tensors"]


def test_lora_zero_strength_identity_layer_level() -> None:
    """层级补充：EgoMaskedLoRALinear 在 strength=0 时输出与基座 Linear 完全一致。"""
    linear = nn.Linear(3, 2)
    adapter = EgoMaskedLoRALinear(linear, rank=2)
    adapter.strength = 0.0
    x = torch.randn(2, 4, 3)
    assert torch.allclose(adapter(x), linear(x), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_dataset_alignment_and_window()
    test_balanced_sampler_1to1()
    test_collate_with_real_cache()
    test_cspq_gradient_flows_to_active_lora_branch()
    test_inactive_lora_branch_has_no_gradient()
    test_rho_zero_identity_equals_frozen_baseline()
    test_preference_lora_loss_inverse_on_full_agent_dim()
    test_build_pred_tokens_uses_cos_sin_columns_directly()
    test_dynamics_loss_near_zero_when_pred_equals_gt()
    test_dynamics_loss_jerk_increases_with_alternating_jitter()
    test_dynamics_loss_backpropagates_to_prediction()
    test_dynamics_acceleration_scale_is_dt_squared()
    test_lambda_q_zero_recovers_loss_without_factor()
    test_factor_huber_zero_when_all_masked()
    test_factor_huber_near_zero_when_q_hat_matches_q_vec()
    test_lambda_q_backpropagates_to_lora_not_cspq()
    test_lambda_dyn_zero_recovers_original_loss()
    test_lora_zero_strength_identity_layer_level()
    print("All preference-lora stage tests passed.")


