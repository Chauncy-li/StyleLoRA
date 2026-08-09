"""Rank-N-Contrast (RNC) losses for the CSPQ preference encoder.

RNC 用于连续标签的对比表示学习：
- 标准连续标签 RNC：对 anchor i，正对 j 的单项损失按原始定义：

      L_ij = -log[ exp(sim_ij / tau) /  Σ_{k: d(r_i,r_k) >= d(r_i,r_j)} exp(sim_ik / tau) ]

  即分母只包含与 i 的排名距离 >= 与 j 的排名距离的样本（排除自身、排除
  与 i 排名更近的样本），这是原 RNC 论文的排序分母。

- cross-scene RNC：只让跨场景（free<->car）的对参与 RNC；同场景对不参与。
  这是"强制两场景同排名样本靠近"的关键监督（避免形成两个独立隐空间）。

- 置信度加权：所有配对权重为 w_ij = c_i * c_j；c=0 的样本被显式排除，
  不会通过 clamp_min 残留参与。

数值稳定：exp 前减 per-anchor log-max 偏移（max 提取），空配对保护。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _gather_pair_losses(sim: torch.Tensor, rank_dist: torch.Tensor, pair_weights: torch.Tensor,
                        valid_pair: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """标准 RNC 逐 anchor 汇聚。

    Args:
        sim: [B, B] 相似度（含 / tau）。
        rank_dist: [B, B] |rank_i - rank_k|。
        pair_weights: [B, B] 配对权重（c_i * c_j），已排除 c=0。
        valid_pair: [B, B] bool，是否可作为 anchor i 的正对 j。

    Returns:
        (loss, 有效对数)。
    """
    # 张量化 RNC：一次构造三维候选掩码 [B, B, B]（i, j, k），完全消除逐 pair 的
    # Python 循环与 GPU→CPU 同步。数学定义与原循环实现完全一致：
    #   L_ij = -sim[i,j] + logsumexp_k(sim[i,k] · M_{ijk})
    # 其中 M_{ijk} = valid_pair[i,k] & (k!=i) & (|r_i-r_k| >= |r_i-r_j|)
    b = sim.shape[0]
    eye = torch.eye(b, device=sim.device, dtype=torch.bool)

    # den_cond[i,j,k] = valid[i,k] AND (i!=k) AND (rankdist[i,k] >= rankdist[i,j])
    den_cond = (
        valid_pair[:, None, :]
        & ~eye[:, None, :]
        & (rank_dist[:, None, :] >= rank_dist[:, :, None])
    )
    den_logits = sim[:, None, :].masked_fill(~den_cond, -torch.inf)  # [B, B, B]
    log_den = torch.logsumexp(den_logits, dim=-1)  # [B, B]
    pair_loss = -sim + log_den  # [B, B]; 无效/空分母位置为 inf 但权重为 0

    positive_mask = valid_pair & ~eye
    # 分母为空的 pair：pair_loss = -inf（logsumexp(-inf)=-inf）。原循环实现对该 pair
    # 直接 continue（完全不参与损失）；这里必须把 -inf pair 的权重也置 0，
    # 使 total_w 与贡献 loss 的 pair 一致，避免稀释梯度。
    finite = positive_mask & torch.isfinite(pair_loss)
    weights = pair_weights * finite.float()
    total_w = weights.sum()
    if total_w <= 0:
        return sim.new_zeros(()), 0.0
    safe_loss = torch.where(finite, pair_loss, sim.new_zeros(()))
    return (safe_loss * weights).sum() / total_w, float(finite.sum())


def rnc_loss(z: torch.Tensor, rank: torch.Tensor, confidence: torch.Tensor,
             *, temperature: float = 0.1, scene_id: torch.Tensor | None = None,
             cross_scene: bool = False) -> Tuple[torch.Tensor, float]:
    """连续标签 RNC（同场景）；cross_scene=True 时只保留跨场景配对。

    Args:
        z: [B, D] 归一化隐向量。
        rank: [B] 连续偏好排名（[0,1]）。
        confidence: [B] 样本置信度（[0,1]）；c=0 样本完全排除。
        temperature: 对比温度。
        scene_id: [B] 场景 id。
        cross_scene: True=只保留跨场景配对（free<->car）；False=只保留同场景配对。

    Returns:
        (loss, 有效配对样本数)。
    """
    if z.shape[0] < 2:
        return z.new_zeros(()), 0.0
    sim = z @ z.T / temperature  # [B, B]
    rank_dist = (rank.unsqueeze(1) - rank.unsqueeze(0)).abs()  # [B, B]

    # 有效性：置信度 > 0 的样本才可参与（显式排除 c=0，不用 clamp）
    active = confidence > 0
    pair_active = active.unsqueeze(1) & active.unsqueeze(0)  # [B, B]
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)

    # 场景配对：cross_scene=True 只保留异场景；False 只保留同场景
    if scene_id is not None:
        same_scene = scene_id.unsqueeze(0) == scene_id.unsqueeze(1)
        if cross_scene:
            scene_mask = ~same_scene
        else:
            scene_mask = same_scene
    else:
        scene_mask = torch.ones_like(eye)

    valid_pair = pair_active & scene_mask & ~eye

    # 配对权重：主动样本的置信度乘积（c=0 已在 active 中排除）
    conf = confidence.clamp_min(0.0)
    pair_weights = conf.unsqueeze(1) * conf.unsqueeze(0)

    if not valid_pair.any():
        return sim.new_zeros(()), 0.0
    return _gather_pair_losses(sim, rank_dist, pair_weights, valid_pair)


def cross_scene_rnc_loss(z: torch.Tensor, rank: torch.Tensor, confidence: torch.Tensor,
                         scene_id: torch.Tensor, *, temperature: float = 0.1) -> Tuple[torch.Tensor, float]:
    """跨场景 RNC：只让 free<->car 的配对参与，强制同排名样本跨场景靠近。"""
    return rnc_loss(z, rank, confidence, temperature=temperature, scene_id=scene_id, cross_scene=True)


def rank_huber_loss(s: torch.Tensor, rank: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    """置信度加权的标量排序 Huber：Σ c·Huber(s,rank) / Σc（除权重和，非 batch size）。"""
    delta = F.smooth_l1_loss(s.squeeze(-1), rank, reduction="none")
    total_w = confidence.sum()
    if total_w <= 0:
        return delta.new_zeros(())
    return (delta * confidence).sum() / total_w


def axis_loss(q_hat: torch.Tensor, q_vec: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """三因子百分位回归：先清理目标 NaN，再按 valid_mask 监督。

    无效轴的 q_vec 为 NaN：若先算 (q_hat-q_vec)^2，NaN 会进入计算图，
    前向虽被 where 屏蔽，但反向仍可能产生 NaN 梯度。因此先把无效目标的
    梯度路径切断（safe_q 在无效位置为 0），再按掩码求和。

    不使用置信度 c 降权——轴冲突本身是高维表示要保留的信息。
    """
    safe_q = torch.where(valid_mask, q_vec, torch.zeros_like(q_vec))
    diff = (q_hat - safe_q) ** 2
    diff = torch.where(valid_mask, diff, torch.zeros_like(diff))
    denom = valid_mask.float().sum()
    if denom <= 0:
        return q_hat.new_zeros(())
    return diff.sum() / denom


def encoder_total_loss(*, z: torch.Tensor, s: torch.Tensor, q_hat: torch.Tensor,
                       rank: torch.Tensor, q_vec: torch.Tensor, valid_mask: torch.Tensor,
                       confidence: torch.Tensor, scene_id: torch.Tensor,
                       temperature: float = 0.1, lambda_cross: float = 0.1,
                       lambda_r: float = 1.0, lambda_a: float = 1.0) -> dict:
    """CSPQ 总损失：L = RNC + lambda_x*cross-scene-RNC + lambda_r*c*Huber(s,r) + lambda_a*L_axis。

    Returns dict with keys: loss / rnc / cross_rnc / rank_loss / axis_loss / rnc_pairs / cross_pairs。
    """
    rnc, rnc_pairs = rnc_loss(z, rank, confidence, temperature=temperature, scene_id=scene_id, cross_scene=False)
    cross, cross_pairs = cross_scene_rnc_loss(z, rank, confidence, scene_id, temperature=temperature)
    rank_l = rank_huber_loss(s, rank, confidence)
    axis_l = axis_loss(q_hat, q_vec, valid_mask)
    total = rnc + lambda_cross * cross + lambda_r * rank_l + lambda_a * axis_l
    return {
        "loss": total,
        "rnc": rnc,
        "cross_rnc": cross,
        "rank_loss": rank_l,
        "axis_loss": axis_l,
        "rnc_pairs": rnc_pairs,
        "cross_pairs": cross_pairs,
    }

