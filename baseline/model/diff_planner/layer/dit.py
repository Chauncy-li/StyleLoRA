"""
DiT（Diffusion Transformer）基础层模块。

主要包含：
1. 时间步嵌入（TimestepEmbedder）；
2. DiTBlock 与最终输出层；
3. 条件调制函数（modulate / scale）。
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp


def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    扩散 Transformer 块（DiT Block）：基于 adaLN-Zero 的条件化模块

    这是 DiT（Diffusion Transformer）的核心构建块，采用 adaptive Layer Normalization Zero
    （adaLN-Zero）机制进行条件化，融合了自注意力和交叉注意力机制。

    架构特点：
    1. AdaLN-Zero 条件化：通过仿射变换（shift, scale, gate）动态调整归一化层
    2. 自注意力（Self-Attention）：处理多智能体之间的交互
    3. 交叉注意力（Cross-Attention）：融合场景上下文信息
    4. 双 MLP 结构：每个子层后都有独立的 MLP 进行特征变换

    数据流向：
    Input -> [Norm1 + AdaLN + SelfAttn + Gate] -> [Norm2 + AdaLN + MLP1 + Gate]
          -> [Norm3 + CrossAttn] -> [Norm4 + MLP2] -> Output

    Attributes:
        norm1 (LayerNorm): 第一个归一化层（自注意力前）
        attn (MultiheadAttention): 多头自注意力层
        norm2 (LayerNorm): 第二个归一化层（MLP1 前）
        mlp1 (Mlp): 第一个多层感知机
        adaLN_modulation (Sequential): 自适应层归一化调制网络
            - SiLU 激活函数
            - Linear 层：将条件 y 映射到 6*dim 维
            - 输出分解为 6 个参数：shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        norm3 (LayerNorm): 第三个归一化层（交叉注意力前）
        cross_attn (MultiheadAttention): 多头交叉注意力层
        norm4 (LayerNorm): 第四个归一化层（MLP2 前）
        mlp2 (Mlp): 第二个多层感知机
    """

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        """
        初始化 DiT Block

        Args:
            dim (int): 输入/输出特征维度
            heads (int): 注意力头数
            dropout (float): Dropout 比率
            mlp_ratio (float): MLP 隐藏层维度扩展比例（通常为 4.0）
        """
        super().__init__()

        # ========== 自注意力子层 ==========
        self.norm1 = nn.LayerNorm(dim)
        # 批量优先的自注意力（batch_first=True）
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        # ========== 第一个 MLP 子层 ==========
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 通常隐藏层是输入的 4 倍
        # 使用近似 tanh 的 GELU 激活函数，更稳定
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ========== AdaLN-Zero 调制网络 ==========
        # 生成 6 个调制参数：shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # 激活函数
            nn.Linear(dim, 6 * dim, bias=True)  # 映射到 6*dim 维
        )

        # ========== 交叉注意力子层 ==========
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        # ========== 第二个 MLP 子层 ==========
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, cross_c, y, attn_mask):
        """
        DiT Block 的前向传播

        Args:
            x (torch.Tensor): 输入特征 [B, P, D]
                - B: 批次大小
                - P: 智能体数量（1 个自车 + N 个邻居）
                - D: 特征维度（hidden_dim）
                - 包含轨迹嵌入和时间嵌入的融合特征

            cross_c (torch.Tensor): 交叉注意力上下文 [B, N_context, D]
                - 场景上下文特征（来自 Encoder 的输出）
                - 用于融合交通场景的全局信息
                - N_context: context token 数量（智能体 + 静态物体 + 车道线）

            y (torch.Tensor): 条件特征 [B, D]
                - 融合路线编码和时间嵌入的条件向量
                - 通过 adaLN_modulation 生成 6 个调制参数
                - 用于动态调整网络的归一化和门控

            attn_mask (torch.Tensor): 自注意力掩码 [B, P]
                - bool 类型，True 表示需要 mask 的位置
                - 用于屏蔽无效的邻居智能体
                - 形状为 [B, P]，会自动广播到注意力权重

        Returns:
            torch.Tensor: 输出特征 [B, P, D]
                - 经过自注意力、交叉注意力和 MLP 处理后的特征
                - 融合了轨迹信息、场景上下文和条件信息

        Note:
            AdaLN-Zero 的核心思想：
            1. 通过条件 y 生成 shift（平移）、scale（缩放）、gate（门控）参数
            2. 对归一化后的特征进行仿射变换：x * (1 + scale) + shift
            3. 通过 gate 控制残差连接的强度（初始为 0，训练中学会控制）
            这种设计使得网络能够根据条件动态调整特征，提升表达能力。
        """
        # ========== AdaLN-Zero 调制参数生成 ==========
        # 将条件 y 通过调制网络，生成 6 个参数
        # 前 3 个用于自注意力子层，后 3 个用于 MLP1 子层
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)
        # 每个参数的形状都是 [B, D]

        # ========== 自注意力子层 ==========
        # 1. 归一化
        # 2. AdaLN 调制：x * (1 + scale) + shift
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # 3. 自注意力 + 门控残差连接：x + gate * Attn(modulated_x)
        # key_padding_mask 用于屏蔽无效的邻居智能体
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]
        # gate_msa.unsqueeze(1) 的形状变为 [B, 1, D]，可以广播到所有智能体

        # ========== MLP1 子层 ==========
        # 1. 归一化
        # 2. AdaLN 调制
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        # 3. MLP + 门控残差连接：x + gate * MLP(modulated_x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        # ========== 交叉注意力子层 ==========
        # 1. 归一化
        # 2. 交叉注意力：Query 来自 x，Key 和 Value 来自场景上下文 cross_c
        #    这使得轨迹特征能够关注场景中的重要信息
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        # 注意：这里没有 gate 控制，直接输出注意力结果

        # ========== MLP2 子层 ==========
        # 1. 归一化
        # 2. MLP 变换（没有 AdaLN 调制，也没有残差连接）
        x = self.mlp2(self.norm4(x))

        return x
    
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape
        
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x
    
