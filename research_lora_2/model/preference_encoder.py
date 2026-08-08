"""CSPQ (Cross-Scene Preference Query Encoder) — lightweight preference encoder.

CSPQ 是本工作提出的组合架构（非已有方法名称）：
- 专家轨迹 → 时序 token 编码（1 层轻量 Transformer/注意力）；
- 4 个共享可学习 Query（全局激进度 + 三因子）通过 cross-attention 从轨迹提取偏好特征；
- 冻结场景特征 h_c 通过低秩映射轻微调制 Query（Q_c = Q_0 + U·σ(V·h_c)，rank=4），
  让同一 Query 在不同场景关注不同物理指标，但仍投影到共同偏好因子；
- 输出：s ∈ [0,1]（统一连续偏好坐标）、q_hat ∈ [0,1]^3（三因子百分位）、z ∈ R^8（归一化隐向量）。

所有参数完全共享，无场景独立 head；不使用 aggr/norm/cons。
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class TemporalTokenEncoder(nn.Module):
    """轻量时序 token 编码：线性投影 + 固定正弦位置编码 + 单层自注意力。"""

    def __init__(self, input_dim: int, d_model: int, heads: int, dropout: float = 0.1,
                 max_len: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 固定正弦位置编码（不参与训练），保证注意力能区分轨迹发生顺序
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else pe[:, 1::2]
        # 修正奇偶 d_model：末维若为奇数则补零（pe[:, 1::2] 已覆盖 0 长度时无副作用）
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1]) if div_term.shape[0] > 1 else 0.0
        self.register_buffer("pos_enc", pe.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T, input_dim] -> [B, T, d_model]。"""
        x = self.dropout(self.proj(tokens))
        x = x + self.pos_enc[:, :x.shape[1]]
        x = x + self.dropout(self.attn(x, x, x, need_weights=False)[0])
        return self.norm(x)


class CSPQPreferenceEncoder(nn.Module):
    """CSPQ 偏好编码器：可学习偏好 Query + 低秩场景条件化 + 共享偏好头。

    Args:
        trajectory_dim: 轨迹 token 输入维度（默认 6: x,y,dx,dy,cos,sin）。
        hc_dim: 冻结场景特征 h_c 维度（自动投影到 d_model）。
        d_model: 隐藏维度（默认 128）。
        heads: 注意力头数（默认 4）。
        z_dim: 隐向量维度（默认 8）。
        query_rank: 低秩场景条件化的秩（默认 4）。
        dropout: 丢弃率（默认 0.1）。
    """

    def __init__(self, *, trajectory_dim: int = 6, hc_dim: int = 192, d_model: int = 128,
                 heads: int = 4, z_dim: int = 8, query_rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.hc_dim = int(hc_dim)  # 保存实际输入维度，供 checkpoint 重建（即使 hc_proj=Identity）
        self.num_queries = 4  # 全局 + 三因子

        # h_c 投影：hc_dim != d_model 时线性投影对齐
        self.hc_proj = nn.Linear(hc_dim, d_model) if hc_dim != d_model else nn.Identity()

        self.temporal = TemporalTokenEncoder(trajectory_dim, d_model, heads, dropout=dropout)

        # 可学习初始 Query
        self.query_init = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.02)
        # 低秩场景条件化：Q_c = Q_0 + U·σ(V·h_c)，rank=query_rank
        self.query_cond_V = nn.Linear(d_model, query_rank, bias=False)
        self.query_cond_U = nn.Linear(query_rank, self.num_queries * d_model, bias=False)
        # U 零初始化：保证训练开始时从未调制 Query（Q_c = Q_0）出发，避免场景条件化破坏预训练语义
        nn.init.zeros_(self.query_cond_U.weight)

        # cross-attention：query 从轨迹 token 中提取偏好特征
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        # 输出头：全局 s、每因子 q_hat（共享参数，逐因子应用）
        self.s_head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        self.q_hat_head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        # z：拼接 4 个 query 特征
        self.z_mlp = nn.Sequential(nn.Linear(self.num_queries * d_model, 64), nn.GELU(), nn.Linear(64, z_dim))

    def forward(self, trajectory: torch.Tensor, h_c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向。

        Args:
            trajectory: [B, T, 6] 轨迹 token。
            h_c: [B, D_hc] 冻结场景特征。

        Returns:
            {"s": [B, 1], "q_hat": [B, 3], "z": [B, z_dim], "query_feats": [B, 4, d_model]}。
        """
        # 1) 时序编码
        tokens = self.temporal(trajectory)  # [B, T, d_model]

        # 2) 场景条件化 Query
        h_proj = self.hc_proj(h_c)          # [B, d_model]
        cond = self.query_cond_U(torch.relu(self.query_cond_V(h_proj)))  # [B, 4*d_model]
        cond = cond.view(-1, self.num_queries, self.d_model)
        queries = self.query_init.unsqueeze(0) + cond  # [B, 4, d_model]

        # 3) cross-attention：query 主动抽取偏好
        attn_out, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        query_feats = self.cross_norm(queries + attn_out)  # [B, 4, d_model]

        # 4) 输出：s（全局）、q_hat（三因子逐因子共享 head）
        g = query_feats[:, 0]                      # [B, d_model]
        f = query_feats[:, 1:]                     # [B, 3, d_model]
        s = torch.sigmoid(self.s_head(g))          # [B, 1]
        q_hat = torch.sigmoid(self.q_hat_head(f))  # [B, 3, 1]
        q_hat = q_hat.squeeze(-1)                  # [B, 3]

        # 5) 归一化隐向量 z
        flat = query_feats.reshape(-1, self.num_queries * self.d_model)
        z = F.normalize(self.z_mlp(flat), dim=-1)  # [B, z_dim]

        return {"s": s, "q_hat": q_hat, "z": z, "query_feats": query_feats}