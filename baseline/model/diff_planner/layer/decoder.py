# decoder.py
# 轨迹扩散解码器模块
#
# Author: Shangwen Li
# Date: 2026-03-09
# Description: 实现基于扩散模型的轨迹解码器，包括 DiT (Diffusion Transformer) 架构
#              使用 DPM-Solver 进行采样，支持 Classifier-Free Guidance
#              包含 Decoder、RouteEncoder 和 DiT 三个核心组件
# License: MIT

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from baseline.model.diff_planner.library.sampling import dpm_sampler
from baseline.model.diff_planner.library.sde import SDE, VPSDE_linear
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer
from baseline.model.diff_planner.layer.mixer import MixerBlock
from baseline.model.diff_planner.layer.dit import TimestepEmbedder, DiTBlock, FinalLayer


class Decoder(nn.Module):
    """
    扩散模型解码器：基于条件扩散模型的轨迹生成器

    该模块使用扩散概率模型（Diffusion Probabilistic Model）生成自车和周围智能体的未来轨迹：
    1. 训练阶段：通过去噪过程学习轨迹分布，预测噪声或原始数据
    2. 推理阶段：使用 DPM-Solver 从噪声中生成高质量轨迹样本
    3. 条件机制：融合场景编码、路线信息和当前状态作为生成条件
    4. 引导增强：可选的 classifier guidance 提升生成质量

    核心组件：
    - DiT (Diffusion Transformer): 基于 Transformer 的去噪网络
    - RouteEncoder: 路线特征编码器
    - VPSDE_linear: 线性方差保持随机微分方程
    - DPM-Solver: 高效的扩散模型采样器

    Attributes:
        _predicted_neighbor_num (int): 需要预测的邻居智能体数量
        _future_len (int): 预测的未来时间步长度
        _sde (VPSDE_linear): 前向扩散过程的 SDE 定义
        dit (DiT): 扩散 Transformer 主网络
        _state_normalizer (StateNormalizer): 状态归一化器
        _observation_normalizer (ObservationNormalizer): 观测归一化器
        _guidance_fn: 引导函数（用于 classifier guidance）
    """
    def __init__(self, config):
        """
        初始化解码器

        Args:
            config (Config): 配置对象，包含以下关键参数：
                - predicted_neighbor_num: 预测的邻居数量
                - future_len: 未来轨迹长度
                - decoder_drop_path_rate: Decoder 的 DropPath 比率
                - route_num: 路线数量
                - lane_len: 车道长度
                - encoder_drop_path_rate: Encoder 的 DropPath 比率
                - hidden_dim: 隐藏层维度
                - decoder_depth: Decoder 层数
                - num_heads: 注意力头数
                - diffusion_model_type: 扩散模型类型 ("score" 或 "x_start")
                - state_normalizer: 状态归一化器
                - observation_normalizer: 观测归一化器
                - guidance_fn: 引导函数
        """
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()
        self._style_value_dim = int(getattr(config, "style_value_dim", 0))
        self._cfg_guidance_scale = float(getattr(config, "cfg_guidance_scale", 1.0))
        self._use_style_condition = bool(getattr(config, "use_style_condition", self._style_value_dim > 0))

        self.dit = DiT(
            sde=self._sde, 
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= (config.future_len + 1) * 4, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type,
            style_condition_dim=self._style_value_dim,
            use_style_condition=self._use_style_condition,
        )
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        
        self._guidance_fn = config.guidance_fn
        
    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }
        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        is_diffusion_loss_pass = ("sampled_trajectories" in inputs) and ("diffusion_time" in inputs)

        if is_diffusion_loss_pass:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            diffusion_time = inputs['diffusion_time']

            denoised = self.dit(
                sampled_trajectories,
                diffusion_time,
                style_condition=self._resolve_style_condition(inputs, B),
                cross_c=ego_neighbor_encoding,
                route_lanes=route_lanes,
                neighbor_current_mask=neighbor_current_mask,
            ).reshape(B, P, -1, 4)
            if self.dit.model_type == "x_start":
                return {
                    "x_start": denoised,
                    "score": denoised,
                }
            return {
                "score": denoised
            }
        else:
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = torch.cat([current_states[:, :, None], torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5], dim=2).reshape(B, P, -1)

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)
            
            style_condition = self._resolve_style_condition(inputs, B)
            if style_condition is not None:
                model_wrapper_params = {
                    "condition": style_condition,
                    "unconditional_condition": torch.zeros_like(style_condition),
                    "guidance_scale": float(inputs.get("cfg_guidance_scale", self._cfg_guidance_scale)),
                    "guidance_type": "classifier-free",
                }
            else:
                model_wrapper_params = {
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask                            
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                }

            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask                            
                        },
                        dpm_solver_params={
                            "correcting_xt_fn":initial_state_constraint,
                        },
                        model_wrapper_params=model_wrapper_params,
                )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            return {
                    "prediction": x0
                }

    def _resolve_style_condition(self, inputs, batch_size: int):
        if (not self._use_style_condition) or self._style_value_dim <= 0:
            return None
        style_condition = inputs.get("style_value_condition")
        if style_condition is None:
            return None
        if style_condition.dim() == 1:
            style_condition = style_condition.unsqueeze(0)
        if style_condition.shape[0] == 1 and batch_size > 1:
            style_condition = style_condition.expand(batch_size, -1)
        return style_condition

        
class RouteEncoder(nn.Module):
    """
    路线编码器：处理规划路线的特征表示

    负责编码车辆的行驶路线信息，用于引导轨迹生成：
    1. 使用 MLP Mixer 架构处理路线点序列
    2. 提取路线的几何特征和方向信息
    3. 为轨迹生成提供全局路径指引

    Attributes:
        _channel (int): 通道维度
        channel_pre_project (Mlp): 通道维度预投影层
        token_pre_project (Mlp): Token 维度预投影层
        Mixer (MixerBlock): MLP Mixer 块
        norm (LayerNorm): 归一化层
        emb_project (Mlp): 特征投影层
    """
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        """
       初始化路线编码器

       Args:
           route_num (int): 路线数量
           lane_len (int): 每条车道的长度（点数）
           drop_path_rate (float): DropPath 比率，用于正则化
           hidden_dim (int): 隐藏层维度
           tokens_mlp_dim (int): Token MLP 的中间维度
           channels_mlp_dim (int): Channel MLP 的中间维度
       """
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        """
        前向传播：编码路线特征

        Args:
            x (torch.Tensor): 路线数据 [B, P_route, V_points, D]
                - B: 批次大小
                - P_route: 路线数量
                - V_points: 每条路线的点数
                - D: 特征维度（只使用前 4 维：x, y, cos, sin）

        Returns:
            torch.Tensor: 编码后的路线特征 [B, hidden_dim]
                - 如果存在有效路线，返回编码后的特征
                - 如果所有路线都无效，返回全零张量

        Note:
            该编码器只处理位置和方向信息（x, y, cos, sin），
            不包含边界、速度限制或交通灯信息。
        """
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        # 只保留前 4 维：x, y, cos, sin（位置和方向信息）
        x = x[..., :4]

        B, P, V, _ = x.shape
        # 生成掩码：判断哪些点是无效的（全 0）
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        # 判断哪些路线完全无效
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        # 判断批次中哪些样本完全无效
        mask_b = torch.sum(~mask_p, dim=-1) == 0

        # 展平为 (B, P*V, D) 便于处理
        x = x.view(B, P * V, -1)

        # 记录有效样本的索引
        valid_indices = ~mask_b.view(-1)
        # 只处理有效数据
        x = x[valid_indices]

        # ========== MLP Mixer 编码 ==========
        # 通道混合：在每个点上进行特征变换
        x = self.channel_pre_project(x)
        # 转置为 (B, D, P*V) 以便在空间维度上进行混合
        x = x.permute(0, 2, 1)
        # 空间混合：跨路线点交换信息
        x = self.token_pre_project(x)
        # 转置回 (B, P*V, D)
        x = x.permute(0, 2, 1)
        # 使用 MixerBlock 进行深度特征提取
        x = self.Mixer(x)

        # ========== 空间维度池化 ==========
        # 平均池化：聚合所有路线点的信息
        x = torch.mean(x, dim=1)

        # ========== 特征投影 ==========
        x = self.emb_project(self.norm(x))

        # ========== 填充有效部分 ==========
        # 创建全零结果张量
        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        # 填充有效部分
        x_result[valid_indices] = x

        return x_result.view(B, -1)


class DiT(nn.Module):
    """
    扩散 Transformer（Diffusion Transformer）：去噪网络核心

    基于 Transformer 架构的扩散模型去噪网络，负责预测噪声或原始数据：
    1. 使用自注意力机制处理多智能体轨迹
    2. 使用交叉注意力融合场景上下文信息
    3. 使用时间嵌入编码扩散过程的进度
    4. 使用路线编码作为全局条件

    架构设计：
    - 输入嵌入：将轨迹数据投影到隐藏维度 + 智能体类型嵌入
    - 时间嵌入：通过 MLP 编码扩散时间步
    - 路由编码：通过 RouteEncoder 提取路线特征
    - DiT Blocks：多层 Transformer 进行特征提取
    - 输出层：预测去噪结果（分数函数或原始数据）

    Attributes:
        _model_type (str): 模型类型 ("score" 预测噪声 / "x_start" 预测原始数据)
        route_encoder (RouteEncoder): 路线编码器
        agent_embedding (Embedding): 智能体类型嵌入（自车 vs 邻居）
        preproj (Mlp): 输入预投影层
        t_embedder (TimestepEmbedder): 时间步嵌入器
        blocks (ModuleList): DiTBlock 序列
        final_layer (FinalLayer): 输出层
        _sde (SDE): 随机微分方程对象
        marginal_prob_std: 边缘概率的标准差函数
    """
    def __init__(
        self,
        sde: SDE,
        route_encoder: nn.Module,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        model_type="x_start",
        style_condition_dim: int = 0,
        use_style_condition: bool = False,
    ):
        """
        初始化 DiT

        Args:
            sde (SDE): 随机微分方程对象，定义前向扩散过程
            route_encoder (RouteEncoder): 路线编码器实例
            depth (int): Transformer 层数
            output_dim (int): 输出维度（轨迹维度）
            hidden_dim (int): 隐藏层维度
            heads (int): 注意力头数
            dropout (float): Dropout 比率
            mlp_ratio (float): MLP 隐藏层扩展比例
            model_type (str): 模型类型
                - "score": 预测噪声 ε（分数匹配）
                - "x_start": 预测原始数据 x_0
        """
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder

        # 智能体类型嵌入：区分自车（index=0）和邻居（index=1）
        self.agent_embedding = nn.Embedding(2, hidden_dim)

        # 输入预投影：将轨迹数据投影到隐藏维度
        self.preproj = Mlp(
            in_features=output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.
        )

        # 时间步嵌入器：将连续时间 t 编码为高维特征
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.use_style_condition = bool(use_style_condition and style_condition_dim > 0)
        if self.use_style_condition:
            self.style_condition_proj = nn.Sequential(
                nn.LayerNorm(style_condition_dim),
                nn.Linear(style_condition_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.style_condition_proj = None

        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])

        # 输出层：预测去噪结果
        self.final_layer = FinalLayer(hidden_dim, output_dim)

        # 保存 SDE 对象及其边缘概率函数
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        """获取模型类型"""
        return self._model_type

    def forward(self, x, t, style_condition=None, cross_c=None, route_lanes=None, neighbor_current_mask=None):
        """
        DiT 前向传播

        Args:
            x (torch.Tensor): 输入轨迹（带噪或纯净）[B, P, output_dim]
                - B: 批次大小
                - P: 智能体数量（1 个自车 + P_pred 个邻居）
                - output_dim: (1+V_future) * 4，包含当前帧和未来帧的 (x, y, cos, sin)

            t (torch.Tensor): 扩散时间步 [B]
                - t ∈ [0, 1]，0 表示无噪声，1 表示纯噪声

            cross_c (torch.Tensor): 交叉注意力上下文 [B, N, D]
                - 场景上下文特征（来自 Encoder）
                - 用于融合交通场景信息

            route_lanes (torch.Tensor): 路线车道数据 [B, M, V, D]
                - 用于条件生成的全局路径指引

            neighbor_current_mask (torch.Tensor): 邻居掩码 [B, P_pred]
                - True 表示无效邻居，False 表示有效
                - 用于自注意力掩码，避免关注无效智能体

        Returns:
            torch.Tensor: 去噪预测结果 [B, P, output_dim]
                - 如果 model_type="score"：返回 ε/σ(t)（归一化噪声）
                - 如果 model_type="x_start"：返回 x_0（原始数据）

        Note:
            对于 score 类型，输出需要除以边际标准差 σ(t)，
            这是为了匹配 SDE 的分数函数 ∇_x log p_t(x)。
        """
        B, P, _ = x.shape

        # ========== 输入投影 ==========
        # 将轨迹数据投影到隐藏维度
        x = self.preproj(x)

        # ========== 智能体类型嵌入 ==========
        # 自车（index=0）和邻居（index=1）使用不同的嵌入
        # x_embedding: [P, D] = [1 个自车 + (P-1) 个邻居]
        x_embedding = torch.cat([
            self.agent_embedding.weight[0][None, :],  # [1, D] - 自车
            self.agent_embedding.weight[1][None, :].expand(P - 1, -1)  # [P-1, D] - 邻居
        ], dim=0)
        # 扩展到 batch 维度：[B, P, D]
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)
        # 添加到输入特征
        x = x + x_embedding

        # ========== 路线编码 + 时间嵌入 ==========
        # 编码路线特征：[B, D]
        route_encoding = self.route_encoder(route_lanes)
        # 作为条件特征 y
        y = route_encoding
        # 添加时间嵌入：[B, D]
        y = y + self.t_embedder(t)
        if self.style_condition_proj is not None and style_condition is not None:
            if style_condition.dim() == 1:
                style_condition = style_condition.unsqueeze(0)
            y = y + self.style_condition_proj(style_condition.to(device=y.device, dtype=y.dtype))

        # ========== 构建注意力掩码 ==========
        # attn_mask: [B, P] - True 表示需要 mask 的位置
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        # 对邻居应用掩码（自车 index=0 永远不 mask），注意这里的是平均注意力
        attn_mask[:, 1:] = neighbor_current_mask

        # ========== DiT Blocks ==========
        # 通过多层 Transformer 进行特征提取
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        # ========== 输出层 ==========
        # 预测去噪结果
        x = self.final_layer(x, y)

        # ========== 模型类型处理 ==========
        if self._model_type == "score":
            # 分数匹配：返回归一化的噪声预测
            # ε/σ(t) 用于匹配分数函数 ∇_x log p_t(x)
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            # 直接预测原始数据
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
