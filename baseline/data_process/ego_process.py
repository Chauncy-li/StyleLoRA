"""
Module: Ego Agent Data Preprocessing

此模块负责处理自车 (Ego Vehicle) 的历史和未来轨迹数据。
主要功能包括：
- 从 NuPlan Scenario 对象中提取历史/未来轨迹
- 将 EgoState 对象列表转换为 NumPy 数组
- 执行绝对坐标到相对坐标的转换
- 基于运动学模型计算额外的车辆状态（如转向角、横摆角速度）
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


# ==============================================================================
# *** Scenario Data Extraction
# ==============================================================================

def get_ego_past_array_from_scenario(scenario, num_past_poses: int, past_time_horizon: float) -> Tuple[
    npt.NDArray, npt.NDArray]:
    """
    从 Scenario 中提取自车历史轨迹和对应的时间戳。
    结果包含当前时刻 (Current State)。

    Args:
        scenario: NuPlan Scenario 对象
        num_past_poses: 历史采样点数量 (不包含当前帧)
        past_time_horizon: 历史时间跨度 [seconds]

    Returns:
        past_ego_states_array: 历史轨迹数组 [num_past + 1, 7] (x, y, heading, vx, vy, ax, ay)
        past_time_stamps_array: 对应的时间戳数组 [num_past + 1] (int64, microseconds)
    """
    current_ego_state = scenario.initial_ego_state

    # 获取纯历史部分
    past_ego_states = scenario.get_ego_past_trajectory(
        iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
    )

    # 拼接: Past + Current
    sampled_past_ego_states = list(past_ego_states) + [current_ego_state]

    # 转换为数组
    past_ego_states_array = sampled_past_ego_states_to_array(sampled_past_ego_states)

    # 获取时间戳
    past_time_stamps = list(
        scenario.get_past_timestamps(
            iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
        )
    ) + [scenario.start_time]

    # 内部辅助函数：展平时间戳列表
    def sampled_past_timestamps_to_array(past_time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:
        flat = [t.time_us for t in past_time_stamps]
        return np.array(flat, dtype=np.int64)

    past_time_stamps_array = sampled_past_timestamps_to_array(past_time_stamps)

    return past_ego_states_array, past_time_stamps_array


def get_ego_future_array_from_scenario(scenario, current_ego_state: EgoState, num_future_poses: int,
                                       future_time_horizon: float) -> npt.NDArray:
    """
    提取自车未来轨迹 (Ground Truth)，并将其转换为相对于当前自车的坐标系。

    Args:
        scenario: NuPlan Scenario 对象
        current_ego_state: 当前自车状态 (用于坐标系原点)
        num_future_poses: 未来采样点数量
        future_time_horizon: 未来时间跨度 [seconds]

    Returns:
        future_trajectory_relative_poses: 相对坐标下的未来轨迹 [T, 3] (x, y, heading)
    """
    future_trajectory_absolute_states = scenario.get_ego_future_trajectory(
        iteration=0, num_samples=num_future_poses, time_horizon=future_time_horizon
    )

    # 坐标转换: Absolute (Global) -> Relative (Ego-Centric)
    # 仅保留位置和航向 (x, y, heading)
    future_trajectory_relative_poses = convert_absolute_to_relative_poses(
        current_ego_state.rear_axle, [state.rear_axle for state in future_trajectory_absolute_states]
    )

    return future_trajectory_relative_poses


# ==============================================================================
# *** Array Conversion Utilities
# ==============================================================================

def sampled_past_ego_states_to_array(past_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    将 EgoState 对象列表转换为 NumPy 数组。
    提取特征: [x, y, heading, vx, vy, ax, ay]

    Args:
        past_ego_states: EgoState 列表

    Returns:
        output: [N, 7] 浮点数组
    """
    output = np.zeros((len(past_ego_states), 7), dtype=np.float64)

    for i in range(0, len(past_ego_states), 1):
        output[i, EgoInternalIndex.x()] = past_ego_states[i].rear_axle.x
        output[i, EgoInternalIndex.y()] = past_ego_states[i].rear_axle.y
        output[i, EgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading
        output[i, EgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
        output[i, EgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
        output[i, EgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
        output[i, EgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y

    return output


# ==============================================================================
# *** Kinematic State Calculation
# ==============================================================================

def calculate_additional_ego_states(ego_agent_past: npt.NDArray, time_stamp: npt.NDArray) -> npt.NDArray:
    """
    计算当前时刻的扩展自车状态向量。
    基于历史轨迹的最后两帧进行差分，计算 Yaw Rate 和 Steering Angle。

    Input Format (ego_agent_past): [T, 7] -> (x, y, heading, vx, vy, ax, ay)
    Output Format: [10] -> (x, y, cos_h, sin_h, vx, vy, ax, ay, steering_angle, yaw_rate)

    Args:
        ego_agent_past: 自车历史轨迹数组
        time_stamp: 对应的时间戳数组 (microseconds)

    Returns:
        current: 包含扩展特征的当前状态向量 (1D array)
    """

    # 提取当前帧 (T) 和上一帧 (T-1)
    current_state = ego_agent_past[-1]
    prev_state = ego_agent_past[-2]

    # 计算时间差 (Microseconds -> Seconds)
    dt = (time_stamp[-1] - time_stamp[-2]) * 1e-6

    # 提取当前合速度
    # 注意: current_state[3] 这里假设是 vx?
    # 原代码中使用 index 3，对应 EgoInternalIndex.vx()。这通常是纵向速度或 X 轴分量。
    # 如果是 Body Frame，vx 即为车速。如果是 Global Frame，应为 sqrt(vx^2 + vy^2)。
    # 保持原逻辑不变：使用 index 3。
    cur_velocity = current_state[3]

    # 计算航向角变化 (Yaw Rate)
    angle_diff = current_state[2] - prev_state[2]
    # 归一化角度差到 [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = angle_diff / dt

    # 计算转向角 (Steering Angle) - 基于阿克曼转向几何 (Kinematic Bicycle Model)
    # Formula: tan(delta) = (yaw_rate * L) / v
    if abs(cur_velocity) < 0.2:
        # 低速停车状态下，Yaw Rate 不可靠，置零
        steering_angle = 0.0
        yaw_rate = 0.0
    else:
        steering_angle = np.arctan(
            yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity)
        )
        # 物理限制截断 (Clip)
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

    # 构建输出向量 [10]
    # Input dims: 7
    # Output dims: 7 + 3 = 10 (原代码逻辑: shape[1] + 3)
    current = np.zeros((ego_agent_past.shape[1] + 3), dtype=np.float32)

    current[:2] = current_state[:2]  # x, y
    current[2] = np.cos(current_state[2])  # cos(heading)
    current[3] = np.sin(current_state[2])  # sin(heading)
    current[4:8] = current_state[3:7]  # vx, vy, ax, ay
    current[8] = steering_angle  # steering
    current[9] = yaw_rate  # yaw_rate

    return current