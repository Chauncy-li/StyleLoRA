"""
Diffusion Planner 主模型装配模块。

职责说明：
1. 组装 Encoder / Decoder，形成完整的规划模型；
2. 对外暴露统一 forward 接口；
3. 在子模块中完成初始化策略，保证训练稳定性。

备注：
- 本文件仅负责模型结构拼装，不包含训练循环与数据读取逻辑；
"""

import torch
import torch.nn as nn

# 导入底层的 Encoder 和 Decoder 实现
from baseline.model.style_planner.layer.encoder import Encoder
from baseline.model.style_planner.layer.decoder import Decoder
from baseline.model.style_planner.layer.preference_axis_router import (
    PreferenceAxisRouter,
    SignedPreferenceAxisRouter,
)


class Diffusion_Planner(nn.Module):
    """
    Diffusion Planner 主模型类。

    该类整合了编码器 (Encoder) 和解码器 (Decoder)，构成了完整的扩散规划模型。
    它负责接收环境输入，提取特征，并通过扩散过程生成轨迹。
    """

    def __init__(self, config):
        """
        初始化 Diffusion Planner 模型。

        Args:
            config: 配置对象，包含模型结构参数（如隐藏层维度、层数等）。
        """
        super().__init__()

        # 实例化编码器和解码器
        # 这里使用了封装后的类 (Diffusion_Planner_Encoder/Decoder) 以包含特定的权重初始化逻辑
        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)

    @property
    def sde(self):
        """
        属性装饰器：获取随机微分方程 (SDE) 对象。

        SDE 定义了扩散过程的前向加噪和反向去噪逻辑。
        直接代理访问解码器内部的 sde 属性。
        """
        return self.decoder.decoder.sde

    def forward(self, inputs):
        """
        模型的前向传播函数。

        Args:
            inputs (dict): 包含自车状态、环境信息、地图信息等的输入字典。

        Returns:
            encoder_outputs: 编码器的输出特征。
            decoder_outputs: 解码器的输出（通常包含预测的轨迹或 Score）。
        """
        # 1. 编码阶段：提取场景上下文特征
        encoder_outputs = self.encoder(inputs)

        # 2. 解码阶段：结合上下文特征进行扩散去噪/轨迹生成
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs


class Diffusion_Planner_Encoder(nn.Module):
    """
    Diffusion Planner 编码器封装类。

    主要作用是包装底层的 Encoder 模块，并应用特定的权重初始化策略。
    """

    def __init__(self, config):
        super().__init__()

        # 实例化底层的 Encoder 网络
        self.encoder = Encoder(config)

        # 执行权重初始化
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化编码器权重的具体逻辑。
        """

        # 定义基础的层初始化函数
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                # 线性层使用 Xavier 均匀分布初始化，偏置置为 0
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm 初始化：偏置为 0，权重（缩放因子）为 1
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                # Embedding 层使用正态分布初始化
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # 对所有子模块应用基础初始化
        self.apply(_basic_init)

        # 针对特定 Embedding 层的初始化 (覆盖上面的通用初始化)
        # 位置编码 (Position Embedding)
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        # 邻居车辆类型编码 (Type Embedding)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        # 车道限速编码 (Speed Limit Embedding)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        # 交通信号编码 (Traffic Light Embedding)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):
        """
        编码器前向传播。
        直接调用内部 self.encoder 的 forward 方法。
        """
        encoder_outputs = self.encoder(inputs)

        return encoder_outputs


class Diffusion_Planner_Decoder(nn.Module):
    """
    Diffusion Planner 解码器封装类。

    主要作用是包装底层的 Decoder (通常基于 DiT 架构)，并应用 DiT 特有的权重初始化策略。
    """

    def __init__(self, config):
        super().__init__()

        # 实例化底层的 Decoder 网络
        self.decoder = Decoder(config)

        # 执行权重初始化
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化解码器权重的具体逻辑。
        包含了 DiT (Diffusion Transformer) 特有的 "零初始化" (Zero-Initialization) 策略。
        """

        # 定义基础的层初始化函数 (与 Encoder 相同)
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # 对所有子模块应用基础初始化
        self.apply(_basic_init)

        # 初始化时间步嵌入 (Timestep Embedding) 的 MLP 层
        # 这些层用于将时间 t 映射为向量，对生成质量很关键
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # DiT 特有的初始化策略：
        # 将 DiT Block 中 adaLN (Adaptive Layer Norm) 调制层的输出初始化为 0。
        for block in self.decoder.dit.blocks:
            # 最后一个 Linear 层负责生成 shift, scale, gate 参数
            # 初始化为 0 意味着初始状态下 block 近似为恒等映射 (Identity mapping)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 初始化最终输出层：
        # 同样将 adaLN 调制层和最终投影层初始化为 0
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)
        if isinstance(
            self.decoder.dit.style_condition_proj,
            (PreferenceAxisRouter, SignedPreferenceAxisRouter),
        ):
            # The generic Linear initialization above intentionally initializes
            # the router internals, then this final projection is reset so a
            # pretrained base planner starts with exactly zero style residual.
            self.decoder.dit.style_condition_proj.reset_output_projection()

    def forward(self, encoder_outputs, inputs):
        """
        解码器前向传播。

        Args:
            encoder_outputs: 编码器输出的上下文特征。
            inputs: 原始输入字典，通常包含采样的时间步 t 和加噪后的轨迹 x_t。
        """
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return decoder_outputs
