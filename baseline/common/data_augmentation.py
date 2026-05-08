# data_augmentation.py
# 该模块用于自动驾驶决策规划模型的训练数据增强。
# 核心功能：通过扰动当前自车状态，并生成一条平滑、可行的轨迹使其在指定时间内回归到原始未来轨迹上。
# 这解决了开环训练中，模型一旦偏离专家轨迹（Expert Trajectory）就不知如何“重回正轨”的问题，
# 让模型在训练阶段就学习如何从偏差状态中恢复，从而提升在闭环仿真或实车中的鲁棒性。

import torch
import numpy as np
from typing import List, Optional, Tuple, Union, cast
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

# 插值相关的常数定义
NUM_REFINE = 20      # 使用样条插值生成的轨迹点数
REFINE_HORIZON = 2.0 # 插值的时间长度（秒），即从扰动状态回归到原轨迹所需的时间
TIME_INTERVAL = 0.1  # 轨迹点的时间间隔（秒）


def vector_transform(vector, transform_mat, bias=None):
    """
    对二维向量（坐标或速度等）进行批量坐标系变换（旋转+平移）。
    通常用于将全局坐标系下的量转换到以自车为中心的局部坐标系。

    Args:
        vector: 形状为 (B, ..., 2) 的张量，表示待转换的坐标或向量。
        transform_mat: 形状为 (B, 2, 2) 的张量，表示批次中每个样本的旋转矩阵。
        bias: 可选，形状为 (B, ..., 2) 的张量，表示平移量（通常是原坐标系的原点在目标系中的坐标）。
              如果提供，则先执行 `vector - bias`。

    Returns:
        变换后的向量，形状与输入 `vector` 相同。
    """
    shape = vector.shape
    B = vector.shape[0]
    nexpand = vector.ndim - 2  # 计算除批次和最后二维（x,y）之外的维度数
    if bias is not None:
        # 执行平移：减去偏移量，例如将全局坐标转换为以某点为原点的坐标
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)
    # 重塑向量以进行批量矩阵乘法: (B, N, 2) -> (B, 2, N)
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1)
    # 应用旋转变换: (B, 2, 2) @ (B, 2, N) -> (B, 2, N)，然后恢复原始形状
    return torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape)


def heading_transform(heading, transform_mat):
    """
    根据坐标系的旋转矩阵，批量更新朝向角（Heading）。

    Args:
        heading: 形状为 (B, ...) 的张量，表示弧度制的朝向角。
        transform_mat: 形状为 (B, 2, 2) 的张量，表示旋转矩阵。

    Returns:
        变换后的朝向角，形状与输入 `heading` 相同。
    """
    B = heading.shape[0]
    shape = heading.shape
    nexpand = heading.ndim - 1
    heading = heading.reshape(B, -1)
    # 将旋转矩阵扩展到合适维度以便广播计算
    transform_mat = transform_mat.reshape(B, 1, 2, 2)
    # 通过旋转矩阵对朝向角的 sin 和 cos 分量进行变换，然后用 atan2 计算新角度
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0] + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0] + torch.sin(heading) * transform_mat[..., 0, 1]
    ).reshape(*shape)


class StatePerturbation():
    """
    状态扰动数据增强类。
    通过对当前自车状态（位置、朝向、速度等）添加随机扰动，并利用五次样条插值生成一条
    从扰动状态平滑连接到原始未来轨迹的可行轨迹，从而创造更多的训练样本。
    """

    def __init__(
        self,
        low: List[float] = [-0., -0.75, -0.35, -1, -0.5, -0.2, -0.1, 0., -0.],
        high: List[float] = [0., 0.75, 0.35, 1, 0.5, 0.2, 0.1, 0., 0.],
        augment_prob: float = 0.5,
        normalize=True,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        """
        初始化增强器。

        Args:
            low: 均匀噪声的下界向量，对应扰动的9个维度：[x, y, yaw, vx, vy, ax, ay, steering_angle, yaw_rate]。
            high: 均匀噪声的上界向量，顺序与 `low` 相同。
            augment_prob: 应用数据增强的概率，介于0和1之间。
            normalize: 是否进行归一化（代码中未直接使用，可能为预留参数）。
            device: 计算设备。
        """
        self._augment_prob = augment_prob
        self._normalize = normalize
        self._device = torch.device(device)
        # 将噪声边界转换为张量
        self._low = torch.tensor(low).to(self._device)
        self._high = torch.tensor(high).to(self._device)
        # 获取车辆轴距参数，用于运动学计算
        self._wheel_base = get_pacifica_parameters().wheel_base

        # 插值参数
        self.refine_horizon = REFINE_HORIZON
        self.num_refine = NUM_REFINE
        self.time_interval = TIME_INTERVAL

        T = REFINE_HORIZON + TIME_INTERVAL  # 插值的总时间范围

        # 预计算五次样条插值的系数矩阵的逆。
        # 五次样条公式：p(t) = a*t^5 + b*t^4 + c*t^3 + d*t^2 + e*t + f
        # 该矩阵用于根据起始和终止点的状态（位置、速度、加速度）求解多项式系数 [a, b, c, d, e, f]^T。
        # 边界条件矩阵的行对应: p(0), p'(0), p''(0), p(T), p'(T), p''(T)
        self.coeff_matrix = torch.linalg.inv(torch.tensor([
            [1, 0, 0, 0, 0, 0],       # t^0 @ t=0
            [0, 1, 0, 0, 0, 0],       # t^1 @ t=0
            [0, 0, 2, 0, 0, 0],       # t^2 @ t=0 (二阶导系数为2!)
            [1, T, T**2, T**3, T**4, T**5], # t^0 @ t=T
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4], # t^1 @ t=T
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]  # t^2 @ t=T
        ], device=device, dtype=torch.float32))
        # 预计算时间幂矩阵，用于快速计算插值点。 shape: (NUM_REFINE, 6)
        # 每一行是 [t^0, t^1, t^2, t^3, t^4, t^5] 对于某一个时间点 t
        self.t_matrix = torch.pow(torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE).unsqueeze(1),
                                  torch.arange(6).unsqueeze(0)).to(device=device)

    def __call__(self, inputs, ego_future, neighbors_future):
        """
        数据增强的主调用函数。
        1. 扰动当前自车状态。
        2. 为扰动后的状态插值生成一条回归到原未来轨迹的平滑轨迹。
        3. 将所有数据转换到以新的自车状态为中心的坐标系。

        Args:
            inputs: 包含当前状态、历史信息、车道线等数据的字典。
            ego_future: 自车原始的未来轨迹，形状 (B, future_len, 3)，最后一维为 (x, y, heading)。
            neighbors_future: 他车未来的轨迹。

        Returns:
            转换到新坐标系后的 inputs, ego_future, neighbors_future。
        """
        # 步骤1: 状态扰动
        aug_flag, aug_ego_current_state = self.augment(inputs)
        # 步骤2: 轨迹插值
        interpolated_ego_future = self.interpolation_future_trajectory(aug_ego_current_state, ego_future)

        # 步骤3: 更新数据
        inputs['ego_current_state'][aug_flag] = aug_ego_current_state[aug_flag]
        ego_future[aug_flag] = interpolated_ego_future[aug_flag]

        # 步骤4: 坐标系变换（将所有信息转换到以扰动后自车为中心的坐标系）
        return self.centric_transform(inputs, ego_future, neighbors_future)

    def augment(
        self,
        inputs
    ):
        """
        对当前自车状态施加随机扰动。

        Args:
            inputs: 输入数据字典。

        Returns:
            aug_flag: 布尔张量，标记哪些批次样本被增强了。
            ego_current_state: 扰动后的自车当前状态。
        """
        ego_current_state = inputs['ego_current_state'].clone()
        B = ego_current_state.shape[0]

        # 决定哪些样本进行增强：随机概率 + 排除速度过慢的样本（避免数值问题）
        aug_flag = (torch.rand(B) >= self._augment_prob).bool().to(self._device) & ~(abs(ego_current_state[:, 4]) < 2.0)

        # 生成均匀分布的噪声张量，并缩放到 [low, high] 区间
        random_tensor = torch.rand(B, len(self._low)).to(self._device)
        scaled_random_tensor = self._low + (self._high - self._low) * random_tensor

        # 创建新状态，初始化时将位置(x,y,yaw)置零（因为后续会做中心化变换），保留并扰动其他状态
        new_state = torch.zeros((B, 9), dtype=torch.float32).to(self._device)
        new_state[:, 3:] = ego_current_state[:, 4:10]  # 复制原始状态的 vx, vy, ax, ay, steering, yaw_rate
        new_state = new_state + scaled_random_tensor  # 添加噪声
        # 施加物理约束
        new_state[:, 3] = torch.max(new_state[:, 3], torch.tensor(0.0, device=new_state.device))  # 纵向速度非负
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)  # 限制横摆角速度

        # 将扰动后的状态写回原状态张量的对应位置
        # 注意：x, y, yaw 的扰动值在这里被写入，但后续 centric_transform 会将其置零并转换其他物体坐标
        ego_current_state[:, :2] = new_state[:, :2]  # x, y
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])  # cos(yaw)
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])  # sin(yaw)
        ego_current_state[:, 4:8] = new_state[:, 3:7]  # vx, vy, ax, ay
        ego_current_state[:, 8:10] = new_state[:, -2:]  # steering_angle, yaw_rate

        # 根据运动学模型，用速度和横摆角速度重新计算方向盘转角，确保状态间的一致性
        cur_velocity = ego_current_state[:, 4]  # 纵向速度 vx
        yaw_rate = ego_current_state[:, 9]

        steering_angle = torch.zeros_like(cur_velocity)
        new_yaw_rate = torch.zeros_like(yaw_rate)

        mask = torch.abs(cur_velocity) < 0.2  # 低速 mask
        not_mask = ~mask

        # 运动学模型：tan(steering_angle) = (yaw_rate * wheel_base) / velocity
        steering_angle[not_mask] = torch.atan(yaw_rate[not_mask] * self._wheel_base / torch.abs(cur_velocity[not_mask]))
        steering_angle[not_mask] = torch.clamp(steering_angle[not_mask], -2 / 3 * np.pi, 2 / 3 * np.pi)
        new_yaw_rate[not_mask] = yaw_rate[not_mask]
        # 低速时，保持方向盘和横摆角速度为0（或原噪声值），避免除以零
        steering_angle[mask] = new_state[mask, -2]
        new_yaw_rate[mask] = new_state[mask, -1]

        ego_current_state[:, 8] = steering_angle
        ego_current_state[:, 9] = new_yaw_rate

        return aug_flag, ego_current_state


    def normalize_angle(self, angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """将角度归一化到 [-pi, pi] 区间。"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def get_transform_matrix_batch(self, cur_state):
        """
        根据当前自车状态的朝向（cos, sin），批量生成从全局坐标系到自车坐标系的旋转矩阵。

        Args:
            cur_state: 自车状态，包含朝向的 cos 和 sin 值。

        Returns:
            transform_matrix: 旋转矩阵，形状 (B, 2, 2)。
        """
        # 提取 cos 和 sin
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )
        # 这个固定矩阵用于将 (cos, sin) 对转换为旋转矩阵。
        # 旋转矩阵 R = [[cosθ, -sinθ], [sinθ, cosθ]]。通过矩阵乘法实现批量计算。
        reshaping_tensor = torch.tensor(
            [
                [1, 0, 0, 1],   # 这一行计算第一列：取第一个cos，第一个sin
                [0, 1, -1, 0],  # 这一行计算第二列：取第二个sin，第二个cos的负值
            ], dtype=torch.float32
        ).to(processed_input.device)
        # (B, 2) @ (2, 4) -> (B, 4)，然后重塑为 (B, 2, 2)
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)

    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor,
        neighbors_future: torch.Tensor,
    ):
        """
        核心函数：将所有物体的坐标、速度等信息，从全局坐标系转换到以（扰动后的）当前自车状态为中心的局部坐标系。
        这是自动驾驶模型训练中标准的“自我中心（Ego-centric）”表示方法。

        Args:
            inputs: 包含各种输入特征的字典。
            ego_future: 自车未来轨迹。
            neighbors_future: 他车未来轨迹。

        Returns:
            转换后的 inputs, ego_future, neighbors_future。
        """
        cur_state = inputs['ego_current_state'].clone()
        center_xy = cur_state[:, :2]  # 新的坐标系原点：扰动后的自车位置
        transform_matrix = self.get_transform_matrix_batch(cur_state)  # 旋转矩阵

        # --- 转换自车当前状态 ---
        # xy 坐标：减去原点，再旋转
        inputs["ego_current_state"][..., :2] = vector_transform(inputs["ego_current_state"][..., :2], transform_matrix,
                                                                center_xy)
        # 朝向向量 (cos, sin)：仅旋转
        inputs["ego_current_state"][..., 2:4] = vector_transform(inputs["ego_current_state"][..., 2:4], transform_matrix)
        # 速度 (vx, vy)：仅旋转
        inputs["ego_current_state"][..., 4:6] = vector_transform(inputs["ego_current_state"][..., 4:6], transform_matrix)
        # 加速度 (ax, ay)：仅旋转
        inputs["ego_current_state"][..., 6:8] = vector_transform(inputs["ego_current_state"][..., 6:8], transform_matrix)
        # 注意：方向盘和横摆角速度是标量，不随坐标系旋转而改变。

        # --- 转换自车未来轨迹 (x, y, heading) ---
        ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)  # 坐标
        ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)  # 朝向角

        # --- 转换他车过去轨迹 ---
        # 先创建一个mask，标记哪些是他车轨迹的无效填充点（全为0）
        mask = torch.sum(torch.ne(inputs["neighbor_agents_past"][..., :6], 0), dim=-1) == 0
        inputs["neighbor_agents_past"][..., :2] = vector_transform(inputs["neighbor_agents_past"][..., :2],
                                                                   transform_matrix, center_xy)
        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(inputs["neighbor_agents_past"][..., 2:4],
                                                                    transform_matrix)
        inputs["neighbor_agents_past"][..., 4:6] = vector_transform(inputs["neighbor_agents_past"][..., 4:6],
                                                                    transform_matrix)
        inputs["neighbor_agents_past"][mask] = 0.  # 将无效点重置为0

        # --- 转换他车未来轨迹 ---
        mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0
        neighbors_future[..., :2] = vector_transform(neighbors_future[..., :2], transform_matrix, center_xy)
        neighbors_future[..., 2] = heading_transform(neighbors_future[..., 2], transform_matrix)
        neighbors_future[mask] = 0.

        # --- 转换车道线信息（通常用多个点表示）---
        # 车道线数据可能包含起点、终点、方向等多种向量，需要分别转换。
        mask = torch.sum(torch.ne(inputs["lanes"][..., :8], 0), dim=-1) == 0
        inputs["lanes"][..., :2] = vector_transform(inputs["lanes"][..., :2], transform_matrix, center_xy)
        inputs["lanes"][..., 2:4] = vector_transform(inputs["lanes"][..., 2:4], transform_matrix)
        inputs["lanes"][..., 4:6] = vector_transform(inputs["lanes"][..., 4:6], transform_matrix)
        inputs["lanes"][..., 6:8] = vector_transform(inputs["lanes"][..., 6:8], transform_matrix)
        inputs["lanes"][mask] = 0.

        # --- 转换路由车道线信息（转换方式与普通车道线相同）---
        mask = torch.sum(torch.ne(inputs["route_lanes"][..., :8], 0), dim=-1) == 0
        inputs["route_lanes"][..., :2] = vector_transform(inputs["route_lanes"][..., :2], transform_matrix, center_xy)
        inputs["route_lanes"][..., 2:4] = vector_transform(inputs["route_lanes"][..., 2:4], transform_matrix)
        inputs["route_lanes"][..., 4:6] = vector_transform(inputs["route_lanes"][..., 4:6], transform_matrix)
        inputs["route_lanes"][..., 6:8] = vector_transform(inputs["route_lanes"][..., 6:8], transform_matrix)
        inputs["route_lanes"][mask] = 0.

        # --- 转换静态障碍物信息 ---
        mask = torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1) == 0
        inputs["static_objects"][..., :2] = vector_transform(inputs["static_objects"][..., :2], transform_matrix,
                                                             center_xy)
        inputs["static_objects"][..., 2:4] = vector_transform(inputs["static_objects"][..., 2:4], transform_matrix)
        inputs["static_objects"][mask] = 0.

        # --- [新增] 转换 Ego 历史轨迹 (x, y, heading, vx, vy, ax, ay) ---
        # 必须判断 key 是否存在，兼容可能没有该数据的情况
        if "ego_agent_past" in inputs:
            # 1. 转换位置 (x, y) [Indices 0, 1]
            inputs["ego_agent_past"][..., :2] = vector_transform(
                inputs["ego_agent_past"][..., :2], transform_matrix, center_xy
            )

            # 2. 转换朝向 (heading) [Index 2]
            # 注意: ego_agent_past 中存的是弧度 heading，不是 cos/sin，所以用 heading_transform
            inputs["ego_agent_past"][..., 2] = heading_transform(
                inputs["ego_agent_past"][..., 2], transform_matrix
            )

            # 3. 转换速度 (vx, vy) [Indices 3, 4]
            inputs["ego_agent_past"][..., 3:5] = vector_transform(
                inputs["ego_agent_past"][..., 3:5], transform_matrix
            )

            # 4. 转换加速度 (ax, ay) [Indices 5, 6]
            inputs["ego_agent_past"][..., 5:7] = vector_transform(
                inputs["ego_agent_past"][..., 5:7], transform_matrix
            )

        return inputs, ego_future, neighbors_future

    def interpolation_future_trajectory(self, aug_current_state, ego_future):
        """
        核心函数：使用五次样条插值，生成一条从扰动后的当前状态平滑过渡到原始未来轨迹的路径。

        Args:
            aug_current_state: 增强后的自车当前状态，形状 (B, 16)。
            ego_future: 自车原始的未来轨迹，形状 (B, 80, 3)。前3维是 (x, y, heading)。

        Returns:
            修正后的未来轨迹，形状 (B, 80, 3)。前 `NUM_REFINE` 个点是新插值的，后面的点沿用原始轨迹。
        """
        P = self.num_refine  # 插值点数
        dt = self.time_interval
        T = self.refine_horizon
        B = aug_current_state.shape[0]
        # 扩展预计算的时间幂矩阵和系数逆矩阵以适应批次大小
        M_t = self.t_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, P, 6)
        A = self.coeff_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, 6, 6)

        # --- 提取起始点 (t=0) 的状态：即扰动后的当前状态 ---
        x0 = aug_current_state[:, 0]  # x
        y0 = aug_current_state[:, 1]  # y
        # 起始朝向：指向原未来轨迹的第 P/2 个点，这样插值轨迹初始方向更合理。
        theta0 = torch.atan2((ego_future[:, int(P / 2), 1] - aug_current_state[:, 1]),
                             (ego_future[:, int(P / 2), 0] - aug_current_state[:, 0]))
        v0 = torch.norm(aug_current_state[:, 4:6], dim=-1)  # 合速度
        a0 = torch.norm(aug_current_state[:, 6:8], dim=-1)  # 合加速度
        omega0 = aug_current_state[:, 9]  # 横摆角速度

        # --- 提取终止点 (t=T) 的状态：即原始未来轨迹上的第 P 个点 ---
        xT = ego_future[:, P, 0]
        yT = ego_future[:, P, 1]
        thetaT = ego_future[:, P, 2]
        # 通过差分近似计算终止点的速度和加速度
        vT = torch.norm(ego_future[:, P, :2] - ego_future[:, P - 1, :2], dim=-1) / dt
        aT = torch.norm(ego_future[:, P, :2] - 2 * ego_future[:, P - 1, :2] + ego_future[:, P - 2, :2],
                        dim=-1) / dt ** 2
        omegaT = self.normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2]) / dt

        # --- 构建五次样条的边界条件向量 ---
        # 对于 x 方向：
        # 位置, 速度*cosθ, 加速度*cosθ - 速度*sinθ*ω (考虑向心加速度)
        sx = torch.stack([
            x0,
            v0 * torch.cos(theta0),
            a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
            xT,
            vT * torch.cos(thetaT),
            aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT
        ], dim=-1)  # (B, 6)

        # 对于 y 方向：同理
        sy = torch.stack([
            y0,
            v0 * torch.sin(theta0),
            a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
            yT,
            vT * torch.sin(thetaT),
            aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT
        ], dim=-1)  # (B, 6)

        # 求解五次多项式系数： coeff = A @ boundary_conditions
        ax = A @ sx[:, :, None]  # (B, 6, 1)
        ay = A @ sy[:, :, None]  # (B, 6, 1)

        # 计算插值点在 T 时刻内的位置： p(t) = [t^0, t^1, ..., t^5] @ coeff
        traj_x = M_t @ ax  # (B, P, 1)
        traj_y = M_t @ ay  # (B, P, 1)

        # 计算插值点的朝向：通过相邻点的位置差计算方向角
        # 第一个点的朝向用从起始点指向它的向量计算
        traj_heading = torch.cat([
            torch.atan2(traj_y[:, :1, 0] - y0.unsqueeze(-1), traj_x[:, :1, 0] - x0.unsqueeze(-1)),
            # 后续点的朝向用前一个点指向当前点的向量计算
            torch.atan2(traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0])
        ], dim=1)

        # 将新插值的前 P 个点与原始轨迹 P 点之后的部分拼接起来，形成完整的修正后未来轨迹。
        new_traj = torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1)
        return torch.concatenate([new_traj, ego_future[:, P:, :]], axis=1)