# mixer.py
# 构建 MLP Mixer 网络
#
# Author: Shangwen Li
# Date: 2026-03-07
# Description: 实现 MLP Mixer 架构的核心组件，包含 MixerBlock 类
#              通过 Token Mixing 和 Channel Mixing 两个 MLP 层进行特征提取
#              适用于序列数据处理和时空特征建模
# License: MIT

import torch.nn as nn
from timm.models.layers import Mlp


class MixerBlock(nn.Module):
    """
    MLP Mixer 基础模块

    采用纯 MLP 架构替代注意力机制，通过以下方式处理输入：
    1. Token Mixing: 在序列维度上混合信息，捕捉全局依赖关系
    2. Channel Mixing: 在特征维度上混合信息，增强特征表示能力
    """

    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        """
        初始化 MixerBlock

        Args:
            tokens_mlp_dim (int): Token MLP 的隐藏层维度
            channels_mlp_dim (int): Channel MLP 的输入输出维度
            drop_path_rate (float): Dropout 比率，用于正则化
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU,
                                drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU,
                              drop=drop_path_rate)

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_tokens, channels_mlp_dim)

        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)