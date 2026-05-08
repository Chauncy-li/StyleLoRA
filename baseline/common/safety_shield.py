import torch
import torch.nn as nn
import math


class SafetyShield(nn.Module):
    """
    优化后的 Safety Shield
    1. 修正了障碍物半径过大的问题 (2.5m -> 动态读取 or 1.1m)
    2. 增加了障碍物速度噪声过滤 (防止静止车辆漂移)
    3. 优化了广播内存占用
    """

    def __init__(
            self,
            num_circles=3,
            ego_width=2.297,
            ego_front_length=4.049,
            ego_rear_length=1.127,
            safety_threshold=0.1  # 适当的安全缓冲，单位米
    ):
        super().__init__()

        # --- 自车几何参数 ---
        ego_length = ego_front_length + ego_rear_length
        interval = ego_length / num_circles

        self.N = num_circles
        self.width = ego_width
        self.length = ego_length
        
        # 优化：自车半径计算
        # 使用勾股定理计算覆盖车身矩形的最小圆半径
        self.radius = math.sqrt((ego_width / 2) ** 2 + (interval / 2) ** 2)

        # 计算圆心偏移量 (相对于后轴)
        # 3个圆通常分布在：后部、中部、前部
        self.register_buffer(
            'offset',
            torch.Tensor([-ego_rear_length + interval / 2 * (2 * i + 1) for i in range(num_circles)])
        )

        self.safety_threshold = safety_threshold

    def get_ego_circles(self, trajectory):
        """
        trajectory: [B, Modes, T, 4] (x, y, heading/cos, sin)
        """
        # 兼容性处理：检查输入是 heading 还是 cos, sin
        if trajectory.shape[-1] == 3:  # x, y, heading
            cos = torch.cos(trajectory[..., 2])
            sin = torch.sin(trajectory[..., 2])
        elif trajectory.shape[-1] == 4: # x, y, cos, sin
            cos = trajectory[..., 2]
            sin = trajectory[..., 3]
        elif trajectory.shape[-1] == 5: # x, y, heading, vel ...
            cos = torch.cos(trajectory[..., 2])
            sin = torch.sin(trajectory[..., 2])
        else:
            raise ValueError(f"Unknown trajectory shape: {trajectory.shape}")

        # [B, Modes, T, 1]
        x = trajectory[..., 0:1]
        y = trajectory[..., 1:2]

        # Offset: [1, 1, 1, N]
        offset = self.offset.view(1, 1, 1, self.N).to(trajectory.device)

        # 计算圆心: [B, Modes, T, N]
        c_x = x + offset * cos.unsqueeze(-1)
        c_y = y + offset * sin.unsqueeze(-1)

        # [B, Modes, T, N, 2]
        centers = torch.stack([c_x, c_y], dim=-1)
        return centers

    def check_collision(self, ego_traj, obstacles, obstacle_valid_mask):
        """
        ego_traj: [B, Modes, T, 4]
        obstacles: [B, P, D] (D >= 8) Expecting x, y, heading, vx, vy, ax, ay, width, length...
        obstacle_valid_mask: [B, P]
        """
        B, Modes, T, _ = ego_traj.shape
        _, P, D = obstacles.shape

        # 1. 获取自车圆心 [B, Modes, T, N_circles, 2]
        ego_circles = self.get_ego_circles(ego_traj)

        # 2. 解析障碍物信息
        obs_pos = obstacles[:, :, :2].unsqueeze(2)  # [B, P, 1, 2]

        # [关键修复 1] 速度噪声过滤
        raw_vel = obstacles[:, :, 4:6]  # [B, P, 2]
        speed = torch.norm(raw_vel, dim=-1, keepdim=True)  # [B, P, 1]
        is_static = speed < 0.2  # 0.2 m/s 阈值
        filtered_vel = raw_vel.clone()
        filtered_vel[is_static.squeeze(-1)] = 0.0
        obs_vel = filtered_vel.unsqueeze(2)  # [B, P, 1, 2]

        # 3. 障碍物轨迹预测 (Constant Velocity)
        time_steps = torch.arange(1, T + 1, device=ego_traj.device).view(1, 1, T, 1) * 0.1
        obs_future_pos = obs_pos + obs_vel * time_steps  # [B, P, T, 2]

        # 4. [关键修复 2] 动态计算障碍物半径
        if D >= 8:
            w = obstacles[:, :, 6]
            l = obstacles[:, :, 7]
            obs_radius_metric = 0.5 * torch.sqrt(w ** 2 + l ** 2)

            # [Clip] 防止某些异常数据导致半径过大或过小
            # 原始形状: [B, P] -> [B, P, 1, 1]
            obs_radius = torch.clamp(obs_radius_metric, min=0.5, max=4.0).unsqueeze(-1).unsqueeze(-1)

            # === [关键修复 3] 增加 Modes 维度 ===
            # [B, P, 1, 1] -> [B, 1, P, 1, 1] 以匹配 [B, Modes, P, T, N]
            obs_radius = obs_radius.unsqueeze(1)
        else:
            # Fallback
            # [1, 1, 1, 1] -> 广播时会自动适应
            obs_radius = torch.tensor(1.2, device=ego_traj.device).view(1, 1, 1, 1, 1)

        # 5. 计算距离 (利用 PyTorch 广播机制)
        # Ego: [B, Modes, 1, T, N_circles, 2]
        # Obs: [B, 1,     P, T, 1,         2]
        ego_c = ego_circles.unsqueeze(2)
        obs_p = obs_future_pos.unsqueeze(1).unsqueeze(-2)

        # Dist Sq: [B, Modes, P, T, N_circles]
        dist_sq = torch.sum((ego_c - obs_p) ** 2, dim=-1)

        # 碰撞阈值判定
        # threshold_dist shape: [B, 1, P, 1, 1]
        threshold_dist = self.radius + obs_radius + self.safety_threshold
        collision_dist_sq = threshold_dist ** 2

        # 6. 判定碰撞
        # [B, Modes, P, T, N_circles] < [B, 1, P, 1, 1]
        # 此时广播机制将正确工作: Modes 对齐 1, P 对齐 P
        has_collision = dist_sq < collision_dist_sq

        # 7. 应用 Mask
        # obstacle_valid_mask: [B, P] -> [B, 1, P, 1, 1]
        valid_mask = obstacle_valid_mask.view(B, 1, P, 1, 1)
        has_collision = has_collision & valid_mask

        # 8. 聚合结果
        # 只要碰到任何一个障碍物(P)，在任何时间(T)，任何圆心(N) -> 该 Mode 不安全
        mode_collision = has_collision.view(B, Modes, -1).any(dim=-1)  # [B, Modes]

        return ~mode_collision