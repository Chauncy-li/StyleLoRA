#
# Wayformer 模型定义（含训练损失与仿真输出适配）。
#
# 兼容性说明：
# - 历史代码中该类名为 `Alpha_Planner`；
# - 为了与 simulation / train 注册制统一，文件末尾提供 `WayFormer = Alpha_Planner` 别名。

import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入你现有的强力 Encoder
from baseline.model.wayformer.layer.fusion_encoder import FusionEncoder as Encoder
# 引入 Wayformer 的核心解码组件
from baseline.model.wayformer.layer.wayformer_layers import PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider

from baseline.model.wayformer.loss.gmm_loss import nll_loss_gmm_direct
from baseline.common.safety_shield import SafetyShield


class AgentPredictor(nn.Module):
    """
    简单的 MLP 用于预测他车轨迹
    Input: [B, N, D] (Agent Features)
    Output: [B, N, T, 2] (Agent Trajectory x,y)
    """
    def __init__(self, hidden_dim, future_steps):
        super().__init__()
        self.future_steps = future_steps
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2) # Output x, y per step
        )

    def forward(self, x):
        B, N, D = x.shape
        out = self.net(x)
        return out.view(B, N, self.future_steps, 2)


class Alpha_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.future_len = config.future_len
        self.num_modes = getattr(config, 'num_modes', 6)
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.output_dim = 5  # (x, y, log_sig_x, log_sig_y, rho)
        self.log_std_range = config.log_std_range

        # 特征提取器 (Feature Extractor) 负责提取特征并拼接，不进行融合
        self.encoder = Encoder(config)

        # 感知编码器 (Perceiver Encoder) - Wayformer 的核心
        # 作用：用少量的 Latent Queries (e.g. 256) 去“读”成千上万个输入 Token，从而实现高效的场景理解和信息压缩
        self.num_queries_enc = getattr(config, 'num_queries_enc', 192)

        self.perceiver_encoder = PerceiverEncoder(
            num_latents=self.num_queries_enc,
            num_latent_channels=self.hidden_dim,  # Latent 维度通常等于 hidden_dim
            # num_cross_attention_heads=config.num_heads,
            num_cross_attention_qk_channels=self.hidden_dim,
            num_cross_attention_v_channels=self.hidden_dim,
            # num_cross_attention_layers=getattr(config, 'encoder_depth', 2),  # Cross-Attn 层数
            # num_self_attention_heads=config.num_heads,
            num_self_attention_qk_channels=self.hidden_dim,
            num_self_attention_v_channels=self.hidden_dim,
            # num_self_attention_layers_per_block=getattr(config, 'encoder_self_attn_layers', 2),
            # 每个 Block 里的 Self-Attn 层数
            # num_self_attention_blocks=encoder_depth,  # 通常 1 个 Block 就够了
            # dropout=config.encoder_drop_path_rate
        )

        # 意图查询供应器 (Learned Queries)
        # 这些是可以学习的“种子”，代表了 6 种不同的潜在驾驶意图
        self.query_provider = TrainableQueryProvider(
            num_queries=getattr(config, 'num_queries_dec', 64),
            num_query_channels=self.hidden_dim,
            init_scale=0.1
        )

        # 轨迹解码器 (Perceiver Decoder)
        # 让意图查询 (Queries) 去关注感知编码器输出的 Context (Latents)
        self.decoder = PerceiverDecoder(
            output_query_provider=self.query_provider,
            num_latent_channels=self.hidden_dim,  # 这里对应 PerceiverEncoder 输出的维度
            # num_cross_attention_heads=config.num_heads,
            # num_cross_attention_layers=config.decoder_depth,
            # cross_attention_widening_factor=2,
            # dropout=config.decoder_drop_path_rate
        )

        # 输出头 (Prediction Heads)
        self.prob_predictor = nn.Linear(self.hidden_dim, 1)
        self.output_model = nn.Linear(self.hidden_dim, 5 * self.future_len)

        # Agent Predictor (辅助任务)
        self.agent_predictor = AgentPredictor(self.hidden_dim, self.future_len)

        # 7. === [修改] Safety Shield ===
        # 直接使用 config 中的车辆参数，确保与你的 safety_shield.py 兼容
        # 假设 config 里有 vehicle_width 等参数，没有则使用默认
        ego_width = getattr(config, 'vehicle_width', 2.297)
        ego_front_length = getattr(config, 'vehicle_front_length', 4.049)
        ego_rear_length = getattr(config, 'vehicle_rear_length', 1.127)

        self.safety_shield = SafetyShield(
            num_circles=3,
            ego_width=ego_width,
            ego_front_length=ego_front_length,
            ego_rear_length=ego_rear_length,
            safety_threshold=0.1  # 这里的阈值用于 Inference 时的硬约束
        )

        self.initialize_weights()


    def initialize_weights(self):
        # 简单的 Xavier 初始化
        nn.init.xavier_uniform_(self.prob_predictor.weight)
        nn.init.constant_(self.prob_predictor.bias, 0)
        nn.init.xavier_uniform_(self.output_model.weight)
        nn.init.constant_(self.output_model.bias, 0)


    def forward(self, inputs):
        # 1. 特征提取
        input_features, input_masks, feature_neighbors = self.encoder(inputs)

        # 2. 感知编码
        context = self.perceiver_encoder(input_features, pad_mask=input_masks)

        # 3. 解码
        out_seq = self.decoder(context)

        # 4. 预测
        B = out_seq.shape[0]
        relevant_queries = out_seq[:, :self.num_modes]
        mode_probs = self.prob_predictor(relevant_queries).squeeze(-1)
        out_dists = self.output_model(relevant_queries).reshape(B, self.num_modes, self.future_len, 5)

        neighbor_preds = self.agent_predictor(feature_neighbors)

        result = {
            "predicted_probability": mode_probs,
            "predicted_trajectory": out_dists,
            "predicted_neighbor_trajectory": neighbor_preds
        }

        # ==========================================================
        # 仿真接口适配 (Simulation Adapter)
        # ==========================================================
        if not self.training:

            # 你的模型输出 out_dists 是 Meters (Relative to Ego).
            # inputs['ego_current_state'] 是 Normalized (Sigma).
            # 不能把 Sigma 赋值给 Meters。在 Local Frame 下，T=0 时刻 Ego 就在 (0,0)。
            out_dists[:, :, 0, :2] = 0.0

            # 简单的平滑处理，防止 T=0 和 T=1 跳变
            out_dists[:, :, 1, :2] = 0.5 * out_dists[:, :, 0, :2] + 0.5 * out_dists[:, :, 2, :2]

            # [Safety Shielding]
            # 优先使用 planner.py 传来的原始米制数据
            if 'safety_shield_neighbors' in inputs:
                raw_neighbors = inputs['safety_shield_neighbors']  # [B, P, T, 11]
            elif 'neighbor_agents_past' in inputs:
                # Fallback: 哪怕是 Normalized 的也比没有强，但在 log 里警告
                raw_neighbors = inputs['neighbor_agents_past']
            else:
                raw_neighbors = None

            if raw_neighbors is not None:
                # 使用原始数据取最后一帧
                neighbors = raw_neighbors[:, :, -1, :]  # [B, P, 11] (x, y, heading, vx, vy...)
                
                # 计算有效性 mask (全0为无效)
                # 注意：如果 DataProcessor 处理后的 padding 是 0，这里判断没问题
                neighbor_mask = (torch.sum(torch.abs(neighbors[..., :2]), dim=-1) > 0.1)

                # 构造用于检测的轨迹：需要包含 cos/sin
                pos_xy = out_dists[..., :2] # 这是模型输出，假设是 Meters (因为 FDE 正常)
                
                # 快速差分计算 heading
                dx = pos_xy[..., 1:, 0] - pos_xy[..., :-1, 0]
                dy = pos_xy[..., 1:, 1] - pos_xy[..., :-1, 1]
                yaw = torch.atan2(dy, dx)
                yaw = torch.cat([yaw, yaw[..., -1:]], dim=-1)  # [B, Modes, T]

                # 构造 Shield 需要的输入 [B, Modes, T, 4]
                traj_for_check = torch.cat([
                    pos_xy,
                    torch.cos(yaw).unsqueeze(-1),
                    torch.sin(yaw).unsqueeze(-1)
                ], dim=-1)

                # 调用 Shield (现在 traj 和 neighbors 都是米制了！)
                is_safe_mask = self.safety_shield.check_collision(
                    traj_for_check,
                    neighbors,
                    neighbor_mask
                )  # [B, Modes]

                # 轨迹选择逻辑
                # 将不安全轨迹的概率设为极小值 (-inf)
                safe_probs = mode_probs.clone()
                safe_probs[~is_safe_mask] = -1e9

                # 重新选择 best mode
                best_mode_idx = torch.argmax(safe_probs, dim=1)

                # [Fallback] 如果所有轨迹都不安全
                all_unsafe = (~is_safe_mask).all(dim=1)
                if all_unsafe.any():
                    # 方案 1: 维持原判 (现在的逻辑，也是导致撞车的原因)
                    # best_mode_idx[all_unsafe] = torch.argmax(mode_probs[all_unsafe], dim=1)

                    # 方案 2 (推荐): 强制刹车逻辑
                    # 我们找不到不撞的轨迹，但我们可以选择"撞得最慢"的那条，或者手动构造刹车
                    # 既然没法改变已生成的轨迹形状，我们可以在后续构造 vel 时动手脚

                    # 暂时先选概率最高的，但在下面打个标记
                    fallback_idx = torch.argmax(mode_probs[all_unsafe], dim=1)
                    best_mode_idx[all_unsafe] = fallback_idx

                    # 标记这些 batch 需要急刹车
                    emergency_brake_mask = all_unsafe

            else:
                best_mode_idx = torch.argmax(mode_probs, dim=1)

            # [Fix 3] 关键修复: arange 必须在正确的 device 上
            batch_idx = torch.arange(B, device=out_dists.device)
            best_traj = out_dists[batch_idx, best_mode_idx]  # [B, T, 5]
            pos_xy = best_traj[..., :2]

            # 差分计算航向角和速度
            dt = 0.1
            dx = pos_xy[:, 1:, 0] - pos_xy[:, :-1, 0]
            dy = pos_xy[:, 1:, 1] - pos_xy[:, :-1, 1]
            heading = torch.atan2(dy, dx)
            heading = torch.cat([heading, heading[:, -1:]], dim=1).unsqueeze(-1)

            dist = torch.sqrt(dx ** 2 + dy ** 2)
            vel = dist / dt
            vel = torch.cat([vel, vel[:, -1:]], dim=1).unsqueeze(-1)

            # [新增] 紧急制动覆盖
            if 'emergency_brake_mask' in locals() and emergency_brake_mask.any():
                # 强制将这些车辆未来几帧的速度设为 0
                # 这会让 NuPlan 的控制器认为你要停车
                vel[emergency_brake_mask] = 0.0
                # 或者做一个线性衰减模拟急刹

            # 静止处理
            stop_mask = vel < 0.1
            vel[stop_mask] = 0.0

            # [Fix Heading Override]
            # 如果车停了，保持当前的真实航向 (从 inputs 中取原始值)
            if 'safety_shield_ego_state' in inputs:
                # 使用原始的 Ego State (Meters/Radians)
                current_h = inputs['safety_shield_ego_state'][:, 2].view(B, 1, 1)
                heading[stop_mask] = current_h.repeat(1, self.future_len, 1)[stop_mask]
            elif 'ego_current_state' in inputs:
                # Fallback 到 normalized 的 input (可能会有点误差，但好过没有)
                # 注意：Normalized Heading 也是 Heading，只要 Normalizer 没对 Heading 做 scale
                pass

            ego_prediction = torch.cat([pos_xy, heading, vel], dim=-1)  # [B, T, 4]

            # 构造 [B, P, T, 4] 输出
            # 注意：这里需要知道 P (Agent数量) 才能正确构造形状
            # 如果 inputs 里有 neighbor_agents_past，可以用它的 shape
            if 'neighbor_agents_past' in inputs:
                P_total = 1 + inputs['neighbor_agents_past'].shape[1]  # 1(Ego) + Neighbors
                # 或者直接取 config 中的设定，这里为了安全起见，我们构造一个足够大的或者根据需求构造
                # 通常仿真器只取 index 0 (Ego)，但为了格式对齐：
                final_prediction = torch.zeros((B, P_total, self.future_len, 4), device=pos_xy.device)
                final_prediction[:, 0] = ego_prediction
                result["prediction"] = final_prediction
            else:
                # Fallback
                result["prediction"] = ego_prediction.unsqueeze(1)

        return result


    def _inference_post_process(self, result, inputs, mode_probs, out_dists, B):
        # 1. 轨迹平滑与归零
        out_dists[:, :, 0, :2] = 0.0
        out_dists[:, :, 1, :2] = 0.5 * out_dists[:, :, 0, :2] + 0.5 * out_dists[:, :, 2, :2]

        # 2. === [调用] Safety Shield Logic ===
        best_mode_idx = torch.argmax(mode_probs, dim=1)  # 默认

        if 'neighbor_agents_past' in inputs and 'neighbor_agents_past_mask' in inputs:
            # 准备数据: SafetyShield 需要 [B, P, D] 的障碍物
            # input 里的 neighbor_agents_past 是 [B, P, T, 11]
            # 我们取最后一帧 (当前时刻)
            neighbors_current = inputs['neighbor_agents_past'][:, :, -1, :]  # [B, P, 11]

            # Mask: [B, P, T] -> [B, P] (当前帧有效的)
            neighbors_mask = inputs['neighbor_agents_past_mask'][:, :, -1]

            # 准备 Ego Trajectory: 需要 [B, Modes, T, 4] (x, y, cos, sin)
            # 模型输出是 [x, y, log_sig, log_sig, rho]，我们需要计算 heading -> cos/sin
            pred_pos = out_dists[..., :2]  # Meters

            # 差分计算航向
            dx = pred_pos[..., 1:, 0] - pred_pos[..., :-1, 0]
            dy = pred_pos[..., 1:, 1] - pred_pos[..., :-1, 1]
            yaw = torch.atan2(dy, dx)
            yaw = torch.cat([yaw, yaw[..., -1:]], dim=-1)  # [B, Modes, T]

            ego_traj_for_shield = torch.cat([
                pred_pos,
                torch.cos(yaw).unsqueeze(-1),
                torch.sin(yaw).unsqueeze(-1)
            ], dim=-1)

            # === [关键] 直接调用 check_collision ===
            # is_safe_mask: [B, Modes] (True=Safe, False=Collision)
            is_safe_mask = self.safety_shield.check_collision(
                ego_traj_for_shield,
                neighbors_current,
                neighbors_mask
            )

            # 过滤逻辑：将碰撞轨迹概率置为 -inf
            safe_probs = mode_probs.clone()
            safe_probs[~is_safe_mask] = -1e9

            # 重新选择 Best Mode
            best_mode_idx = torch.argmax(safe_probs, dim=1)

            # Fallback: 如果全部撞车，选撞得最轻的（或者概率最大的）
            all_unsafe = (~is_safe_mask).all(dim=1)
            if all_unsafe.any():
                # 回退到原始概率最大值，后续 velocity 处理会刹停
                fallback_idx = torch.argmax(mode_probs[all_unsafe], dim=1)
                best_mode_idx[all_unsafe] = fallback_idx
                # 标记需要急刹车 (后续处理)
                emergency_brake_mask = all_unsafe

        # 3. 生成最终轨迹 (计算速度、航向等)
        batch_idx = torch.arange(B, device=out_dists.device)
        best_traj = out_dists[batch_idx, best_mode_idx]  # [B, T, 5]
        pos_xy = best_traj[..., :2]

        dt = 0.1
        dx = pos_xy[:, 1:, 0] - pos_xy[:, :-1, 0]
        dy = pos_xy[:, 1:, 1] - pos_xy[:, :-1, 1]
        heading = torch.atan2(dy, dx)
        heading = torch.cat([heading, heading[:, -1:]], dim=1).unsqueeze(-1)

        dist = torch.sqrt(dx ** 2 + dy ** 2)
        vel = dist / dt
        vel = torch.cat([vel, vel[:, -1:]], dim=1).unsqueeze(-1)

        # 紧急制动处理
        if 'emergency_brake_mask' in locals() and emergency_brake_mask.any():
            vel[emergency_brake_mask] = 0.0

        ego_prediction = torch.cat([pos_xy, heading, vel], dim=-1)  # [B, T, 4]
        result["prediction"] = ego_prediction.unsqueeze(1)  # [B, 1, T, 4]


    def compute_loss(self, model_output, inputs):
        """
        综合 Loss 计算:
        1. Planning Loss (GMM NLL): 模仿专家轨迹
        2. Prediction Loss (Smooth L1): 预测周边车辆轨迹 (辅助任务)
        3. Collision Loss (Safety): 惩罚 Ego 最佳轨迹与真实障碍物 (GT) 的重叠
        """
        loss_dict = {}

        # ====================================================
        # 1. Planning Loss (GMM Negative Log Likelihood)
        # ====================================================
        gt_trajs = inputs['ego_future_gt'][:, :self.future_len]  # [B, T, 3]
        gt_xy = gt_trajs[..., :2]

        # 获取有效性 Mask (如果数据中没有提供显式 mask，则默认全有效或根据坐标是否为0判断)
        if 'ego_future_mask' in inputs:
            valid_mask = inputs['ego_future_mask'][:, :self.future_len]  # [B, T]
        else:
            # 简易 fallback: 如果坐标全为0则无效
            valid_mask = (torch.abs(gt_xy).sum(dim=-1) > 1e-4).float()

        # 构造 _nll_loss_gmm_direct 需要的格式: [B, T, 3] -> (x, y, mask)
        gt_xy_mask = torch.cat([gt_xy, valid_mask.unsqueeze(-1)], dim=-1)

        loss_planning = nll_loss_gmm_direct(
            pred_scores=model_output['predicted_probability'],
            pred_trajs=model_output['predicted_trajectory'],
            gt_trajs=gt_xy_mask,
            log_std_range=self.log_std_range
        )
        loss_dict['loss_planning'] = loss_planning

        # ====================================================
        # 2. Prediction Loss (Smooth L1 - Auxiliary Task)
        # ====================================================
        # 只有在存在邻居数据时计算
        if 'neighbors_future_gt' in inputs and 'neighbor_agents_future_mask' in inputs:
            # GT: [B, N, T, 3] -> (x, y, h)
            neighbor_gt = inputs['neighbors_future_gt'][..., :2]  # [B, N, T, 2]
            neighbor_mask = inputs['neighbor_agents_future_mask']  # [B, N, T]

            # Pred: [B, N, T, 2]
            neighbor_pred = model_output['predicted_neighbor_trajectory']

            loss_pred = F.smooth_l1_loss(neighbor_pred, neighbor_gt, reduction='none')

            # Masking: [B, N, T, 2] * [B, N, T, 1]
            loss_pred = (loss_pred * neighbor_mask.unsqueeze(-1)).sum()

            # Normalize
            valid_count = neighbor_mask.sum() * 2 + 1e-6
            loss_dict['loss_prediction'] = loss_pred / valid_count
        else:
            loss_dict['loss_prediction'] = torch.tensor(0.0, device=gt_xy.device)

        # ====================================================
        # 3. Collision Loss (Differentiable Safety Layer)
        # ====================================================
        # 逻辑：找出 Ego 预测最好的一条轨迹 (Best Mode)，计算它与 GT 障碍物的距离
        # 如果距离 < 安全阈值，则产生 Loss
        if 'neighbor_agents_future' in inputs and 'neighbor_agents_future_mask' in inputs:
            neighbor_gt = inputs['neighbor_agents_future'][..., :2]  # [B, N, T, 2]
            neighbor_mask = inputs['neighbor_agents_future_mask']  # [B, N, T]

            # 3.1 找到 Ego 的 Best Trajectory (Winner-Takes-All based on Planning GT)
            # 我们希望优化的是"最可能被执行"的那条轨迹的安全性
            pred_trajs = model_output['predicted_trajectory']  # [B, M, T, 5]

            # 计算所有 Mode 与 Ego GT 的距离
            dist_to_gt = torch.norm(pred_trajs[..., :2] - gt_xy.unsqueeze(1), dim=-1)  # [B, M, T]
            dist_to_gt = (dist_to_gt * valid_mask.unsqueeze(1)).sum(dim=-1)  # [B, M]

            best_mode_idx = torch.argmin(dist_to_gt, dim=1)  # [B]

            # 提取 Best Mode 轨迹
            batch_idx = torch.arange(len(best_mode_idx), device=pred_trajs.device)
            best_ego_traj = pred_trajs[batch_idx, best_mode_idx]  # [B, T, 5] (x, y, params...)

            # 3.2 构造 SafetyShield 需要的几何输入
            # 需要: [B, 1, T, 4] -> (x, y, cos, sin)
            pos = best_ego_traj[..., :2]  # [B, T, 2]

            # 差分计算 heading (保证可微性)
            # Pad 一位保持长度 T
            vel_vec = pos[:, 1:] - pos[:, :-1]
            vel_vec = torch.cat([vel_vec, vel_vec[:, -1:]], dim=1)

            # 计算 cos, sin
            # 避免 vel_vec 为 0 导致 nan (虽然极少见)
            yaw = torch.atan2(vel_vec[..., 1], vel_vec[..., 0])

            ego_geom_input = torch.cat([
                pos,
                torch.cos(yaw).unsqueeze(-1),
                torch.sin(yaw).unsqueeze(-1)
            ], dim=-1).unsqueeze(1)  # [B, 1, T, 4]

            # 3.3 调用 SafetyShield 获取圆心
            # 这一步利用了你提供的 SafetyShield 类中的 buffer 和逻辑
            ego_circles = self.safety_shield.get_ego_circles(ego_geom_input).squeeze(1)  # [B, T, N_circles, 2]

            # 3.4 计算距离 (Broadcasting)
            # Ego: [B, 1,        T, N_circles, 2]
            # Obs: [B, N_agents, T, 1,         2]
            ego_c = ego_circles.unsqueeze(1)
            obs_p = neighbor_gt.unsqueeze(-2)

            # [B, N_agents, T, N_circles]
            dists = torch.norm(ego_c - obs_p, dim=-1)

            # 3.5 计算 Loss
            # 安全阈值 = Ego Radius (from shield) + Obs Radius (Estimated ~2.0m) + Margin
            safe_dist = self.safety_shield.radius + 2.0

            # Penalty = ReLU(Safe - Dist)
            collision_penalty = F.relu(safe_dist - dists)

            # Apply Masks
            # neighbor_mask: [B, N_agents, T] -> [B, N_agents, T, 1]
            mask_expanded = neighbor_mask.unsqueeze(-1)
            loss_coll = (collision_penalty * mask_expanded).sum()

            # Normalize
            denom = mask_expanded.sum() * self.safety_shield.N + 1e-6
            loss_dict['loss_collision'] = loss_coll / denom

        else:
            loss_dict['loss_collision'] = torch.tensor(0.0, device=gt_xy.device)

        # ====================================================
        # 4. Total Loss Aggregation
        # ====================================================
        # 权重建议:
        # Planning: 1.0 (核心)
        # Prediction: 0.5 (辅助特征学习)
        # Collision: 0.1 ~ 0.2 (强约束，值通常较大，权重给小一点防止梯度不稳定)
        loss_dict['loss'] = (
                1.0 * loss_dict['loss_planning'] +
                0.5 * loss_dict['loss_prediction'] +
                0.1 * loss_dict['loss_collision']
        )

        return loss_dict


    @torch.no_grad()
    def compute_metrics(self, model_output, inputs):
        """
        移植自 base_model.py 的开环指标计算，使用纯 PyTorch 实现加速
        返回: minADE, minFDE, miss_rate, brier_fde
        """
        # [B, K, T, 5]
        pred_trajs = model_output['predicted_trajectory'][..., :2]
        # [B, K]
        pred_probs = F.softmax(model_output['predicted_probability'], dim=-1)

        # GT: [B, T, 3] -> (x, y, h) or similar. We need x, y.
        gt_trajs = inputs['ego_future_gt'][..., :2]

        # Valid Mask: [B, T]
        if 'ego_future_mask' in inputs:
            valid_mask = inputs['ego_future_mask']
        else:
            valid_mask = torch.ones(gt_trajs.shape[0], gt_trajs.shape[1], device=gt_trajs.device)

        # 扩展维度以进行广播: GT [B, 1, T, 2], Pred [B, K, T, 2]
        gt_trajs_exp = gt_trajs.unsqueeze(1)
        valid_mask_exp = valid_mask.unsqueeze(1)  # [B, 1, T]

        # 计算距离 [B, K, T]
        l2_dist = torch.norm(pred_trajs - gt_trajs_exp, dim=-1)

        # === ADE ===
        # 仅计算 valid 帧的距离和
        # [B, K]
        ade_sum = (l2_dist * valid_mask_exp).sum(dim=-1)
        valid_count = valid_mask_exp.sum(dim=-1).clamp(min=1.0)  # 防止除零
        ade = ade_sum / valid_count

        min_ade, _ = torch.min(ade, dim=-1)  # [B]

        # === FDE ===
        # 找到每个 batch 最后一个有效帧的索引
        # 注意：如果数据已经规整为固定长度且最后时刻有效，可以直接取 -1
        # 这里为了严谨，取最后一个 mask 为 1 的位置
        # (简化处理：假设最后一个时刻也是我们要评估的目标时刻)
        fde = l2_dist[..., -1]  # [B, K]

        min_fde, best_mode_idx = torch.min(fde, dim=-1)  # [B], [B]

        # === Miss Rate (MR) ===
        # 阈值 2.0 米
        miss_rate = (min_fde > 2.0).float()

        # === Brier-FDE ===
        # 公式: minFDE + (1 - prob(best_fde_trajectory))^2
        # 获取 FDE 最小的那条轨迹对应的概率
        # best_mode_idx: [B]
        best_mode_prob = pred_probs.gather(1, best_mode_idx.unsqueeze(-1)).squeeze(-1)  # [B]

        brier_fde = min_fde + torch.square(1.0 - best_mode_prob)

        metrics = {
            'minADE': min_ade.mean(),
            'minFDE': min_fde.mean(),
            'miss_rate': miss_rate.mean(),
            'brier_fde': brier_fde.mean()
        }

        return metrics


# -----------------------------------------------------------------------------
# 向后兼容别名
# -----------------------------------------------------------------------------
# simulation/planner.py 和 train.py 统一使用 `WayFormer` 名称，
# 这里保留别名以兼容历史类名 `Alpha_Planner`。
WayFormer = Alpha_Planner
