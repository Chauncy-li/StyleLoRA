import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath


class FusionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim

        # Ego(1) + Neighbors + Static + Lanes
        self.token_num = 1 + config.agent_num + config.static_objects_num + config.lane_num

        # Ego 编码器 (带时间编码)
        self.ego_encoder = EgoFusionEncoder(
            config.time_len,
            drop_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )

        # Neighbor 编码器(带时间编码)
        self.neighbor_encoder = AgentFusionEncoder(
            config.time_len,
            drop_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )

        # Static Object 编码器 (无时间)
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim
        )

        # Lane 编码器 (点集聚合)
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )

        # 位置编码: [x, y, cos, sin, type_1, type_2, type_3] -> 7维
        self.pos_emb = nn.Linear(7, config.hidden_dim)

        # 初始化权重
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, inputs):
        """
        Returns:
            mixed_features: [B, Total_Tokens, Hidden_Dim]
            mixed_mask: [B, Total_Tokens] (True = Padding/Invalid)
        """

        # 1. 提取输入
        ego_past = inputs['ego_agent_past']  # [B, T, 7]
        neighbors = inputs['neighbor_agents_past']  # [B, P, T, 11]
        static = inputs['static_objects']  # [B, S, D]
        lanes = inputs['lanes']  # [B, L, V, D]
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = ego_past.shape[0]

        # 2. 分别编码
        encoding_ego, ego_mask, ego_pos = self.ego_encoder(ego_past)
        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        # 3. 拼接所有特征 (Concatenate All)
        encoding_input = torch.cat([encoding_ego, encoding_neighbors, encoding_static, encoding_lanes], dim=1)
        encoding_mask = torch.cat([ego_mask, neighbors_mask, static_mask, lanes_mask], dim=1)
        encoding_pos_raw = torch.cat([ego_pos, neighbor_pos, static_pos, lane_pos], dim=1)

        # 4. 应用位置编码 (Apply Geometry Positional Embedding)
        total_tokens = encoding_input.shape[1]
        flat_pos = encoding_pos_raw.view(-1, 7)
        flat_mask = encoding_mask.view(-1)

        pos_embedding = torch.zeros((B * total_tokens, self.hidden_dim), device=encoding_input.device)
        valid_indices = ~flat_mask  # False is valid -> ~False is True

        if valid_indices.any():
            pos_embedding[valid_indices] = self.pos_emb(flat_pos[valid_indices])

        # 5. 将位置编码加到特征上
        pos_embedding = pos_embedding.view(B, total_tokens, -1)
        mixed_features = encoding_input + pos_embedding

        # --- [修改核心] ---
        # 提取 Neighbor 特征用于辅助预测任务
        # 拼接顺序是 [Ego(index 0), Neighbors(index 1...P), ...]
        num_agents = self.config.agent_num
        # 切片范围: [1 : 1 + agent_num]
        feature_neighbors = mixed_features[:, 1:1 + num_agents, :] # 给新增的 AgentPredictor 用

        return mixed_features, encoding_mask, feature_neighbors


class EgoFusionEncoder(nn.Module):
    """
    Ego 编码器: Linear -> Add Time Emb -> MLP -> MaxPool
    """

    def __init__(self, time_len, hidden_dim=256, drop_rate=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 基础特征投影: 7 -> hidden_dim
        self.input_proj = nn.Linear(7, hidden_dim)

        # [新增] 可学习的时间位置编码
        self.temporal_emb = nn.Parameter(torch.zeros(1, time_len, hidden_dim))
        nn.init.normal_(self.temporal_emb, std=0.02)

        # 特征提取 MLP
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * 2, out_features=hidden_dim,
                       act_layer=nn.GELU, drop=drop_rate)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        '''
        x: [B, T, 7]
        '''
        B, T, C = x.shape

        # 1. 提取用于后续几何位置编码的 Current State
        current_state = x[:, -1, :4].clone()
        pos_vec = torch.zeros((B, 7), device=x.device)
        pos_vec[:, 0] = current_state[:, 0]  # x
        pos_vec[:, 1] = current_state[:, 1]  # y
        pos_vec[:, 2] = torch.cos(current_state[:, 2])  # cos(h)
        pos_vec[:, 3] = torch.sin(current_state[:, 2])  # sin(h)
        pos_vec[:, 4] = 1.0  # Type: Ego

        # 2. 特征投影
        x = self.input_proj(x)  # [B, T, D]

        # 3. 加上时间编码
        x = x + self.temporal_emb[:, :T, :]

        # 4. MLP 处理
        x = self.mlp(x)  # [B, T, D]

        # 5. 时序聚合 (MaxPooling)
        # MaxPool 通常比 MeanPool 能保留更显著的特征（如急刹车、急转弯）
        x = torch.max(x, dim=1)[0]  # [B, D]

        x = self.norm(x)

        # 格式对齐: [B, 1, D]
        x_result = x.unsqueeze(1)
        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        pos_result = pos_vec.unsqueeze(1)

        return x_result, mask, pos_result


class AgentFusionEncoder(nn.Module):
    """
    Neighbor 编码器: Linear -> Add Time Emb -> MLP -> MaxPool
    """

    def __init__(self, time_len, hidden_dim=256, drop_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 输入维度: 8 (state) + 3 (type one-hot) = 11
        self.input_proj = nn.Linear(11, hidden_dim)

        # [新增] 可学习的时间位置编码
        self.temporal_emb = nn.Parameter(torch.zeros(1, time_len, hidden_dim))
        nn.init.normal_(self.temporal_emb, std=0.02)

        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * 2, out_features=hidden_dim,
                       act_layer=nn.GELU, drop=drop_rate)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        '''
        x: [B, P, T, 11] (state + type)
        '''
        # 1. 准备几何位置编码
        x_states = x[..., :8]
        pos = x_states[:, :, -1, :7].clone()  # 取最后一帧的位置
        pos[..., -3:] = 0.0  # 清除原来的 velocity/size 占位
        pos[..., -3] = 1.0  # Type bit for Dynamic Agent (与 Ego 共享 1.0)

        B, P, T, C = x.shape

        # 计算 Mask (如果整个时间轴都是0，则该Agent无效)
        # [B, P, T] -> [B, P]
        mask_v = torch.sum(torch.ne(x_states, 0), dim=-1).to(x.device) == 0  # 某一帧是否无效
        mask_p = torch.all(mask_v, dim=-1)  # 整个 Agent 是否无效

        # 2. 扁平化处理 [B*P, T, C]
        x_flat = x.view(B * P, T, -1)
        valid_agents_mask = ~mask_p.view(-1)  # [B*P]

        # 只处理有效的 Agent 以节省计算
        x_valid = x_flat[valid_agents_mask]  # [N_valid, T, C]

        if x_valid.shape[0] > 0:
            # 3. 特征投影
            x_valid = self.input_proj(x_valid)  # [N_valid, T, D]

            # 4. 加上时间编码
            x_valid = x_valid + self.temporal_emb[:, :T, :]

            # 5. MLP + MaxPool
            x_valid = self.mlp(x_valid)
            x_valid = torch.max(x_valid, dim=1)[0]  # [N_valid, D]
            x_valid = self.norm(x_valid)
        else:
            x_valid = torch.zeros((0, self.hidden_dim), device=x.device)

        # 6. 填回结果
        x_result = torch.zeros((B * P, self.hidden_dim), device=x.device)
        x_result[valid_agents_mask] = x_valid

        return x_result.view(B, P, -1), mask_p, pos.view(B, P, -1)


class StaticFusionEncoder(nn.Module):
    """
    Static Object 编码器: Linear Only
    """

    def __init__(self, dim, hidden_dim=256, drop_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 简单投影
        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim,
                              act_layer=nn.GELU, drop=drop_rate)

    def forward(self, x):
        '''
        x: [B, P, D]
        '''
        B, P, _ = x.shape

        # 1. 几何位置编码
        pos = torch.zeros((B, P, 7), device=x.device)
        pos[..., :4] = x[..., :4]  # x, y, cos, sin
        pos[..., 4] = 0.0
        pos[..., 5] = 1.0  # Type bit for static
        pos[..., 6] = 0.0

        # 2. Mask 计算
        mask_p = torch.sum(torch.ne(x[..., :6], 0), dim=-1).to(x.device) == 0
        valid_indices = ~mask_p.view(-1)

        x_result = torch.zeros((B * P, self.hidden_dim), device=x.device)

        if valid_indices.any():
            x_flat = x.view(B * P, -1)
            x_valid = x_flat[valid_indices]

            x_valid = self.projection(x_valid)
            x_result[valid_indices] = x_valid

        return x_result.view(B, P, -1), mask_p, pos


class LaneFusionEncoder(nn.Module):
    """
    Lane 编码器: Linear -> MLP -> MaxPool (PointNet style)
    """

    def __init__(self, lane_len, hidden_dim=256, drop_rate=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self._lane_len = lane_len

        # 属性 Embedding
        self.speed_limit_emb = nn.Linear(1, hidden_dim)
        self.unknown_speed_emb = nn.Embedding(1, hidden_dim)
        self.traffic_emb = nn.Linear(4, hidden_dim)

        # 几何特征投影: 8 -> hidden_dim
        self.input_proj = nn.Linear(8, hidden_dim)

        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * 2, out_features=hidden_dim,
                       act_layer=nn.GELU, drop=drop_rate)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: [B, L, V, 8] (V points per lane)
        '''
        traffic = x[:, :, 0, 8:]  # [B, L, 4] 取第一个点的交通灯状态(假设整条路一致)
        x_geo = x[..., :8]

        # 1. 几何位置编码 (取中点)
        mid_idx = int(self._lane_len / 2)
        pos_raw = x_geo[:, :, mid_idx, :4].clone()
        heading = torch.atan2(pos_raw[..., 3], pos_raw[..., 2])

        pos = torch.zeros((x.shape[0], x.shape[1], 7), device=x.device)
        pos[..., 0] = pos_raw[..., 0]
        pos[..., 1] = pos_raw[..., 1]
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos[..., 6] = 1.0  # Type bit for Lane

        B, L, V, _ = x_geo.shape

        # Mask 计算
        mask_v = torch.sum(torch.ne(x_geo, 0), dim=-1).to(x.device) == 0
        mask_l = torch.all(mask_v, dim=-1)  # 整条 Lane 是否无效

        # 扁平化
        x_flat = x_geo.view(B * L, V, -1)
        valid_indices = ~mask_l.view(-1)
        x_valid = x_flat[valid_indices]  # [N_valid, V, 8]

        if x_valid.shape[0] > 0:
            # 2. 点特征投影
            x_valid = self.input_proj(x_valid)  # [N_valid, V, D]

            # 3. 聚合 (MaxPooling over V points)
            x_valid = torch.max(x_valid, dim=1)[0]  # [N_valid, D]

            # 4. 处理额外属性 (Speed Limit, Traffic Light)
            # 提取对应有效 Lane 的属性
            sl_valid = speed_limit.view(B * L, 1)[valid_indices]
            has_sl_valid = has_speed_limit.view(B * L, 1)[valid_indices].squeeze(-1)
            traffic_valid = traffic.view(B * L, 4)[valid_indices]

            # 叠加属性
            attr_emb = torch.zeros_like(x_valid)

            # Speed Limit
            if has_sl_valid.sum() > 0:
                attr_emb[has_sl_valid] += self.speed_limit_emb(sl_valid[has_sl_valid])
            if (~has_sl_valid).sum() > 0:
                attr_emb[~has_sl_valid] += self.unknown_speed_emb.weight.expand((~has_sl_valid).sum(), -1)

            # Traffic Light
            attr_emb += self.traffic_emb(traffic_valid)

            x_valid = x_valid + attr_emb

            # 5. MLP
            x_valid = self.mlp(x_valid)
            x_valid = self.norm(x_valid)

        else:
            x_valid = torch.zeros((0, self.hidden_dim), device=x.device)

        # 6. 填回结果
        x_result = torch.zeros((B * L, self.hidden_dim), device=x.device)
        x_result[valid_indices] = x_valid

        return x_result.view(B, L, -1), mask_l, pos