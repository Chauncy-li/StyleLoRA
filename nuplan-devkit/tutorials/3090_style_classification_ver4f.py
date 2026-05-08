'''
style_classification_ver4f.py 继承了 style_classification_ver4j.py，采用一种 “分阶段处理”（Map-Reduce）的策略。
本版本为最终内存与空间优化版，采用 "Just-in-Time (JIT) 重计算" 策略。
style_classification_ver4f进一步利用计算机多核心的特性，并引入 faiss 库代替 sklearn，提升文件并行处理速度与数据聚类速度

核心优化点：
1.  中间文件极小化：第一阶段的并行处理只提取并保存用于聚类的特征和用于重建场景的“配方”（log, token, start_frame），
    不再保存庞大的轨迹和地图数据，大幅降低了磁盘空间占用。
2.  引入JIT重计算阶段：在聚类和筛选完成后，针对被选中的核心样本“配方”，启动新的进程池按需、即时地重新计算生成
    完整的 ScenarioDescription 数据。
3.  以总运行时间为代价，换取了极低的内存和磁盘空间占用，确保了在任何规模的数据集上都能流畅稳定地运行。

从nuPlan数据集中提取驾驶轨迹数据，并根据驾驶风格（激进、正常、保守）对数据进行分类和筛选，最终生成标准化的风格化驾驶数据集。

具体实现流程如下：
数据加载：从指定的nuPlan数据库文件中加载驾驶场景数据
轨迹切片：将连续的驾驶轨迹分割成固定长度的片段（每个81帧）
特征提取：为每个轨迹片段计算描述驾驶风格的特征指标，如加速度、 jerk、车头时距等
风格聚类：使用K-means算法对轨迹片段进行聚类，识别出激进、正常、保守三种驾驶风格
核心样本筛选：根据设定规则（按数量或比例）筛选出每种风格最具代表性的核心样本
数据可视化：生成风格分布饼图和各特征的箱型图，并输出统计报告
格式转换与保存：将筛选后的数据转换为ScenarioNet格式并保存，便于后续用于自动驾驶模型训练

该脚本支持多进程处理以提高效率，并提供了丰富的参数配置选项用于调试和控制数据处理过程
存在不足：处理步骤为串行处理，给内存带来极大压力
'''
import os
import sys
import copy
import pickle
import torch
import argparse
import hydra
import tempfile
from collections import defaultdict
import concurrent.futures
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- (Import部分保持不变) ---
# 添加项目根目录到 PYTHONPATH，用于服务器训练
project_root = "/home/lisw/programs/mdsn/scenarionet"
if project_root not in sys.path:
    sys.path.append(project_root)

from metadrive.type import MetaDriveType
from metadrive.scenario import ScenarioDescription as SD

project_root = "/home/lisw/programs/mdsn/TrafficDataSetSource/nuplan-devkit"
if project_root not in sys.path:
    sys.path.append(project_root)

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType, TrafficLightStatusType
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

import nuplan
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.utils import set_up_common_builder

sys.path.append('/home/lisw/programs/mdsn/scenarionet')
from style_process.scenarionet_utils.utils import write_to_directory

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
# 确保环境变量已设置或在此处硬编码
if 'NUPLAN_DATA_ROOT' not in os.environ:
    os.environ['NUPLAN_DATA_ROOT'] = "/mnt/mydata/lishangwen/TrafficDataSetSource/nuplan/dataset"
if 'NUPLAN_MAPS_ROOT' not in os.environ:
    os.environ['NUPLAN_MAPS_ROOT'] = os.path.join(os.environ['NUPLAN_DATA_ROOT'], "maps")

# ===========================================================================
# ===== A. 文件读取和nuplan场景加载处理函数 ================================
# ===========================================================================
@dataclass
class ConfigPaths:
    """存储Hydra框架所需的各种配置文件路径"""
    common_dir: str  # 指向包含共享配置文件的目录
    config_name: str  # 要使用的主配置文件的名称
    config_path: str  # 主配置文件所在的目录路径
    experiment_dir: str  # 指向包含实验配置文件的目录


def construct_simulation_hydra_paths(base_config_path: str) -> ConfigPaths:
    """
    将一个基础路径，转换成 Hydra 框架能理解的、用于查找不同类型配置文件
    (.yaml 文件)的多个精确路径。
    """
    common_dir = "file://" + os.path.join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = os.path.join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + os.path.join(base_config_path, 'experiments')
    return ConfigPaths(common_dir, config_name, config_path, experiment_dir)


def get_nuplan_scenarios_from_db(
        data_root: str,
        map_root: str,
        log_db_file: str,
        builder: str = "nuplan"
) -> List[NuPlanScenario]:
    """
    使用 nuPlan SDK 和 Hydra 配置，从指定的 .db 日志文件中加载并筛选场景。

    Args:
        data_root: nuPlan 数据集根目录。
        map_root: nuPlan 地图数据根目录。
        log_db_file: 要处理的 .db 文件名。
        builder: 使用的场景构建器配置名。

    Returns:
        从该日志文件中提取的 NuPlanScenario 对象列表。
    """
    if not NUPLAN_PACKAGE_PATH:
        print("错误: nuPlan SDK 未加载，无法获取场景。")
        return []

    log_name = os.path.splitext(log_db_file)[0]

    # nuPlan SDK 的数据集参数
    # scenario_builder: 定义如何从连续日志中切分出离散场景。
    # subsample_ratio_override: 数据降采样率，0.5 表示从 20Hz 降到 10Hz。
    # remove_invalid_goals: 移除任务目标点无效的场景。
    # expand_scenarios: False 表示一个驾驶事件只生成一个标准场景。
    # shuffle: False 保证场景按时间顺序排列。
    dataset_parameters = [
        f"scenario_builder={builder}",
        "scenario_builder.scenario_mapping.subsample_ratio_override=0.5",
        f"scenario_builder.data_root={data_root}",
        f"scenario_builder.map_root={map_root}",
        "scenario_filter=all_scenarios",
        "scenario_filter.remove_invalid_goals=true",
        "scenario_filter.expand_scenarios=false",
        "scenario_filter.shuffle=false",
        f"scenario_filter.log_names=[{log_name}]",
        "scenario_filter.timestamp_threshold_s=1",
    ]
    base_config_path = os.path.join(NUPLAN_PACKAGE_PATH, "planning", "script")
    simulation_hydra_paths = construct_simulation_hydra_paths(base_config_path)

    # 初始化 Hydra 配置
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=simulation_hydra_paths.config_path)

    # 创建一个临时的、唯一的文件夹，用于存放本次运行可能产生的日志或文件
    save_dir = tempfile.mkdtemp()

    # 覆盖默认配置的参数列表
    overrides = [
        f'group={save_dir}',    # 设置一个名为 group 的配置变量，它的值是刚刚创建的临时目录路径
        'worker=sequential',    # 指定工作模式为“顺序执行”，即一个一个地处理场景，而不是并行处理
        'ego_controller=perfect_tracking_controller',
        'observation=box_observation',
        f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
        'output_dir=${group}/${experiment}',
        'metric_dir=${group}/${experiment}',
        *dataset_parameters,
        'job_name=action_classification',
        'experiment=${experiment_name}/${job_name}',
        'experiment_name=data_classification'
    ]

    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)
    profiler_name = 'action_classification_profiler'

    # set_up_common_builder: 负责处理所有模块都需要的通用服务（如日志、缓存等）。
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)
    # build_scenario_builder: 创建场景构建器，负责从 .db 文件读取并组装场景。
    scenario_builder = build_scenario_builder(cfg=cfg)
    # build_scenario_filter: 创建场景过滤器，封装所有筛选逻辑。
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    print(f"正在从日志 '{log_name}' (文件: {log_db_file}) 中加载场景...")
    scenarios = scenario_builder.get_scenarios(scenario_filter, common_builder.worker)
    print(f"从日志 '{log_name}' 中成功加载了 {len(scenarios)} 个场景。")
    return scenarios


def extract_ego_trajectory_from_scenario(scenario: NuPlanScenario) -> np.ndarray:
    """从场景中提取自车轨迹，并返回一个 NumPy 数组 [N_frames, 5]。"""
    parsed_states_list = []
    for i in range(scenario.get_number_of_iterations()):
        ego_state = scenario.get_ego_state_at_iteration(i)
        if ego_state is None:
            continue

        waypoint = ego_state.waypoint
        dynamic_state = ego_state.dynamic_car_state
        parsed_states_list.append([
            waypoint.x, waypoint.y, waypoint.heading,
            dynamic_state.center_velocity_2d.x,
            dynamic_state.center_velocity_2d.y
        ])

    if not parsed_states_list:
        return np.array([])

    return np.array(parsed_states_list, dtype=np.float32)


# ===========================================================================
# ===== B. 特征计算函数 =====================================================
# ===========================================================================
def _get_robust_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """健壮地计算两个角度或角度数组之间的最小差值 (弧度)。"""
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi


def get_speed_limit_for_position(map_api, position: np.ndarray) -> float:
    """
    根据一个二维位置点，从nuPlan地图中获取最接近车道的限速。

    Args:
        map_api: NuPlanMapApi 对象。
        position: 自车的位置 [x, y]。

    Returns:
        限速值 (m/s)，如果找不到则返回默认值。
    """
    point = Point2D(x=position[0], y=position[1])
    try:
        # 查询该点附近1米内的车道和车道连接器
        proximal_map_objects = map_api.get_proximal_map_objects(point, 1.0, [SemanticMapLayer.LANE,
                                                                             SemanticMapLayer.LANE_CONNECTOR])
        all_lanes = proximal_map_objects.get(SemanticMapLayer.LANE, []) + proximal_map_objects.get(
            SemanticMapLayer.LANE_CONNECTOR, [])
    except Exception:
        # 在某些边缘情况下，地图查询可能会失败
        all_lanes = []

    if not all_lanes:
        return 15.0  # 如果找不到车道，返回一个合理的默认值 (约 54 km/h)

    # 简单起见，直接取找到的第一个车道的限速
    closest_lane = all_lanes[0]
    speed_limit = closest_lane.speed_limit_mps

    return speed_limit if speed_limit is not None else 15.0


def _get_distance_to_lead_vehicle(ego_state: EgoState, all_tracked_objects: List[TrackedObject]) -> float:
    """
    严谨地计算自车与正前方领头车的纵向距离。

    Args:
        ego_state: 当前帧的自车状态。
        all_tracked_objects: 同一帧的所有被探测到的交通参与者。

    Returns:
        与前方领头车的距离(米)。如果没有领头车，则返回极大值。
    """
    min_longitudinal_distance = float('inf')
    ego_pos = np.array([ego_state.center.x, ego_state.center.y])
    ego_heading = ego_state.center.heading

    # 将自车朝向转换为旋转矩阵所需的角度（逆时针为正，因此使用-heading进行坐标系旋转）
    cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)

    for agent in all_tracked_objects:
        if agent.tracked_object_type != TrackedObjectType.VEHICLE:
            continue

        # 将其他车辆的位置转换到自车的局部坐标系下
        delta_pos = np.array([agent.center.x, agent.center.y]) - ego_pos
        local_x = delta_pos[0] * cos_h - delta_pos[1] * sin_h  # 纵向距离
        local_y = delta_pos[0] * sin_h + delta_pos[1] * cos_h  # 横向距离

        # 筛选条件: 1. 在自车前方; 2. 在自车行驶路径上 (横向距离小于2米)
        if local_x > 0 and abs(local_y) < 2.0:
            min_longitudinal_distance = min(min_longitudinal_distance, local_x)

    return min_longitudinal_distance


def calculate_driving_style_features(
        ego_trajectory: np.ndarray, scenario: NuPlanScenario, iteration: int
) -> Dict[str, float]:
    """
    为给定的轨迹切片计算一组全面、鲁棒的驾驶风格特征。

    Args:
        ego_trajectory: 自车轨迹切片 [T, 5]。
        scenario: 对应的nuPlan场景对象。
        iteration: 切片在场景中的起始帧索引。

    Returns:
        一个包含计算出的特征值的字典。
    """
    WINDOW_SIZE = 81
    if ego_trajectory.shape[0] < 20:
        return {}
    dt = 0.1

    # --- 1. 基础运动学 ---
    speed = np.linalg.norm(ego_trajectory[:, 3:5], axis=1)
    accel = np.diff(speed) / dt
    jerk = np.diff(accel) / dt

    # --- 2. 局部坐标系动态 (更精确) ---
    cos_h = np.cos(ego_trajectory[:, 2])
    sin_h = np.sin(ego_trajectory[:, 2])
    lat_speed = -ego_trajectory[:, 3] * sin_h + ego_trajectory[:, 4] * cos_h
    lat_accel = np.diff(lat_speed) / dt
    yaw_rates = _get_robust_angle_diff(ego_trajectory[1:, 2], ego_trajectory[:-1, 2]) / dt

    # --- 3. 场景与交互上下文 ---
    # a. 限速信息
    speed_limit = get_speed_limit_for_position(scenario.map_api, ego_trajectory[0, :2])
    # b. 最小车头时距 (Min Time Headway)
    min_thw = float('inf')
    expert_traj_list = list(scenario.get_expert_ego_trajectory())
    for i in range(ego_trajectory.shape[0]):
        frame_idx = iteration + i
        if frame_idx < len(expert_traj_list):
            ego_state_expert = expert_traj_list[frame_idx]
            all_other_objects = scenario.get_tracked_objects_at_iteration(frame_idx).tracked_objects
            dist_to_front = _get_distance_to_lead_vehicle(ego_state_expert, all_other_objects)
            ego_speed = ego_state_expert.dynamic_car_state.speed
            if ego_speed > 1.0 and dist_to_front != float('inf'):
                min_thw = min(min_thw, dist_to_front / ego_speed)

    # --- 4. 整合计算最终的特征向量 ---
    features = {
        # 效率类: 速度与限速的比值
        'speed_to_limit_ratio': float(np.mean(speed[:WINDOW_SIZE]) / (speed_limit + 1e-6)),
        # 激进类: 最大加速度和减速度
        'max_acceleration': float(np.max(accel[:WINDOW_SIZE])) if len(accel[:WINDOW_SIZE]) > 0 else 0.0,
        'max_deceleration': float(np.abs(np.min(accel[:WINDOW_SIZE]))) if len(accel[:WINDOW_SIZE]) > 0 else 0.0,
        # 舒适性类 (值越小越舒适): 纵向加加速度和横向加速度的均方根
        'longitudinal_jerk_rms': float(np.sqrt(np.mean(jerk[:WINDOW_SIZE] ** 2))) if len(
            jerk[:WINDOW_SIZE]) > 0 else 0.0,
        'lateral_acceleration_rms': float(np.sqrt(np.mean(lat_accel[:WINDOW_SIZE] ** 2))) if len(
            lat_accel[:WINDOW_SIZE]) > 0 else 0.0,
        # 稳定性类: 偏航率和速度的标准差
        'yaw_rate_rms': float(np.sqrt(np.mean(yaw_rates[:WINDOW_SIZE] ** 2))) if len(
            yaw_rates[:WINDOW_SIZE]) > 0 else 0.0,
        'speed_std_dev': float(np.std(speed[:WINDOW_SIZE])),
        # 交互风险类 (值越小越有风险): 最小车头时距
        'min_time_headway': float(min_thw) if min_thw != float('inf') else 8.0,
    }

    # 替换所有nan/inf为0，确保数据有效性
    return {k: v if np.isfinite(v) else 0.0 for k, v in features.items()}


def set_light_status(status: TrafficLightStatusType) -> MetaDriveType:
    """将 nuPlan 的交通灯状态映射到 MetaDrive 的类型。"""
    status_map = {
        TrafficLightStatusType.GREEN: MetaDriveType.LIGHT_GREEN,
        TrafficLightStatusType.RED: MetaDriveType.LIGHT_RED,
        TrafficLightStatusType.YELLOW: MetaDriveType.LIGHT_YELLOW,
    }
    return status_map.get(status, MetaDriveType.LIGHT_UNKNOWN)


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    """将 nuPlan 坐标向量平移到以 nuplan_center 为原点的坐标系。"""
    return np.array(vector) - np.asarray(nuplan_center)


def get_traffic_obj_type(nuplan_type: TrackedObjectType) -> str:
    """将 nuPlan 的对象类型映射到 MetaDrive 的类型。"""
    type_map = {
        TrackedObjectType.VEHICLE: MetaDriveType.VEHICLE,
        TrackedObjectType.PEDESTRIAN: MetaDriveType.PEDESTRIAN,
        TrackedObjectType.BICYCLE: MetaDriveType.CYCLIST,
        TrackedObjectType.TRAFFIC_CONE: MetaDriveType.TRAFFIC_CONE,
        TrackedObjectType.BARRIER: MetaDriveType.TRAFFIC_BARRIER,
    }
    return type_map.get(nuplan_type, MetaDriveType.UNSET)


def _extract_traffic_for_slice(scenario: NuPlanScenario, center: List[float], start_index: int, slice_len: int) -> Dict:
    """为轨迹切片提取所有交通参与者的轨迹，并进行平移归一化。"""
    EGO_ID = "ego"
    all_objs_in_slice = {EGO_ID}
    for i in range(slice_len):
        frame_idx = start_index + i
        if frame_idx < scenario.get_number_of_iterations():
            for obj in scenario.get_tracked_objects_at_iteration(frame_idx).tracked_objects:
                all_objs_in_slice.add(obj.track_token)

    tracks = {
        k: {
            SD.TYPE: MetaDriveType.UNSET,
            SD.STATE: {"position": np.zeros((slice_len, 3)), "heading": np.zeros(slice_len),
                       "velocity": np.zeros((slice_len, 2)), "valid": np.zeros(slice_len),
                       "length": np.zeros((slice_len, 1)), "width": np.zeros((slice_len, 1)),
                       "height": np.zeros((slice_len, 1))},
            SD.METADATA: {"track_length": slice_len, "object_id": k, "original_id": k, "type": MetaDriveType.UNSET}
        } for k in all_objs_in_slice
    }

    for i in range(slice_len):
        frame_idx = start_index + i
        if frame_idx >= scenario.get_number_of_iterations(): continue

        # 处理邻车
        for obj in scenario.get_tracked_objects_at_iteration(frame_idx).tracked_objects:
            obj_id, obj_type = obj.track_token, get_traffic_obj_type(obj.tracked_object_type)
            if obj_id in tracks and obj_type != MetaDriveType.UNSET:
                tracks[obj_id][SD.TYPE] = obj_type
                state = tracks[obj_id][SD.STATE]
                state["position"][i, :2] = nuplan_to_metadrive_vector([obj.center.x, obj.center.y], center)
                state["heading"][i], state["velocity"][i] = obj.center.heading, [obj.velocity.x, obj.velocity.y]
                state["length"][i], state["width"][i], state["height"][
                    i] = obj.box.length, obj.box.width, obj.box.height
                state["valid"][i] = 1

        # 处理自车
        ego_state = scenario.get_ego_state_at_iteration(frame_idx)
        tracks[EGO_ID][SD.TYPE] = MetaDriveType.VEHICLE
        tracks[EGO_ID][SD.METADATA][SD.TYPE] = MetaDriveType.VEHICLE
        state = tracks[EGO_ID][SD.STATE]
        state["position"][i, :2] = nuplan_to_metadrive_vector([ego_state.waypoint.x, ego_state.waypoint.y], center)
        state["heading"][i] = ego_state.waypoint.heading
        state["velocity"][i] = [ego_state.dynamic_car_state.center_velocity_2d.x,
                                ego_state.dynamic_car_state.center_velocity_2d.y]
        state["length"][:], state["width"][:], state["height"][
                                               :] = ego_state.agent.box.length, ego_state.agent.box.width, ego_state.agent.box.height
        state["valid"][i] = 1

    # 移除所有类型为 UNSET 的对象，防止后续处理出错
    return {key: value for key, value in tracks.items() if value[SD.TYPE] != MetaDriveType.UNSET}


def _extract_map_features(map_api, center: List[float], radius: int = 100) -> Dict:
    """提取以center为中心的局部地图特征。"""
    ret, center_for_query = {}, Point2D(*center)
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.CROSSWALK]
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)

    all_lanes = nearest_vector_map.get(SemanticMapLayer.LANE, []) + nearest_vector_map.get(
        SemanticMapLayer.LANE_CONNECTOR, [])
    for lane in all_lanes:
        ret[lane.id] = {SD.TYPE: MetaDriveType.LANE_SURFACE_STREET, SD.POLYLINE: nuplan_to_metadrive_vector(
            np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path]), center)}

    for crosswalk in nearest_vector_map.get(SemanticMapLayer.CROSSWALK, []):
        ret[crosswalk.id] = {SD.TYPE: MetaDriveType.CROSSWALK, SD.POLYGON: nuplan_to_metadrive_vector(
            np.array(list(zip(*crosswalk.polygon.exterior.coords.xy))), center)}

    return ret


def set_light_position(scenario, lane_id, center):
    """查找交通灯的物理位置并进行坐标归一化。"""
    lane_connector = scenario.map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
    if lane_connector is None:
        return [0, 0]  # 如果找不到车道，返回默认位置

    # 默认返回车道连接器的最后一个点，这通常是停止线所在位置，更具代表性
    point_to_return = lane_connector.baseline_path.discrete_path[-1]
    return nuplan_to_metadrive_vector([point_to_return.x, point_to_return.y], center)


def extract_traffic_light_for_slice(scenario: NuPlanScenario, center: list, start_index: int, slice_len: int) -> dict:
    """为一个轨迹切片提取所有交通灯的状态和位置。"""
    # 1. 发现此切片内所有出现过的交通灯ID，并缓存每一帧的状态
    all_lights_in_slice = set()
    frames_data = []
    for i in range(slice_len):
        frame_idx = start_index + i
        if frame_idx >= scenario.get_number_of_iterations():
            frames_data.append({})
            continue
        statuses = {str(t.lane_connector_id): t.status for t in
                    scenario.get_traffic_light_status_at_iteration(frame_idx)}
        all_lights_in_slice.update(statuses.keys())
        frames_data.append(statuses)

    if not all_lights_in_slice:
        return {}

    # 2. 初始化数据结构
    lights = {
        light_id: {
            "type": MetaDriveType.TRAFFIC_LIGHT,
            "state": {SD.TRAFFIC_LIGHT_STATUS: [MetaDriveType.LIGHT_UNKNOWN] * slice_len},
            SD.TRAFFIC_LIGHT_POSITION: None,
            SD.TRAFFIC_LIGHT_LANE: light_id,
            "metadata": dict(track_length=slice_len, type=MetaDriveType.TRAFFIC_LIGHT, object_id=light_id)
        } for light_id in all_lights_in_slice
    }

    # 3. 遍历切片内的每一帧，填充状态数据和物理位置
    for i, frame_statuses in enumerate(frames_data):
        for light_id, status in frame_statuses.items():
            lights[light_id]["state"][SD.TRAFFIC_LIGHT_STATUS][i] = set_light_status(status)
            if lights[light_id][SD.TRAFFIC_LIGHT_POSITION] is None:
                lights[light_id][SD.TRAFFIC_LIGHT_POSITION] = set_light_position(scenario, light_id, center)

    return lights


def extract_training_data_for_slice(
        scenario: NuPlanScenario,
        start_iteration: int,
        slice_length: int,
        ego_slice_np: np.ndarray
) -> SD:
    """
    将单个轨迹切片的数据严格转换为 ScenarioDescription 格式。
    此版本直接使用外部传入的ego_slice_np，不再内部重复提取。
    """
    try:
        # 1. 准备工作
        start_state = scenario.get_ego_state_at_iteration(start_iteration)
        if not start_state: return None
        scenario_center = [start_state.waypoint.x, start_state.waypoint.y]
        slice_lidar_token = scenario.get_scenario_tokens()[start_iteration]
        EGO_ID = 'ego'

        # 2. 构建并返回最终的 ScenarioDescription 对象
        sd = SD()
        sd[SD.ID] = slice_lidar_token
        sd[SD.VERSION] = "nuplan_style_v1.1"
        sd[SD.LENGTH] = slice_length
        # 这里形成 ScenarioNet 格式的元数据
        sd[SD.METADATA] = {
            "dataset": "nuplan_style",
            "map": scenario.map_api.map_name,
            "log_name": scenario.log_name,
            "map_version": scenario.map_version,
            "scenario_token": scenario.token,
            "slice_start_lidar_token": slice_lidar_token,
            "scenario_type": scenario.scenario_type,
            "sdc_id": "ego",
            "style_defining_track_id": EGO_ID,  # 明确记录定义该风格的车辆ID
            "sample_rate": scenario.database_interval,
            "scenario_id": scenario.token,
            SD.TIMESTEP: np.round(np.arange(0, slice_length * scenario.database_interval, scenario.database_interval),
                                  2),
            SD.METADRIVE_PROCESSED: True,
            SD.COORDINATE: "right-handed",
        }
        sd[SD.TRACKS] = _extract_traffic_for_slice(scenario, scenario_center, start_iteration, slice_length)
        sd[SD.MAP_FEATURES] = _extract_map_features(scenario.map_api, scenario_center)
        sd[SD.DYNAMIC_MAP_STATES] = extract_traffic_light_for_slice(scenario, scenario_center, start_iteration,
                                                                    slice_length)

        return sd

    except Exception:
        return None


# ===========================================================================
# ===== C. 工作流与数据处理函数 ===============================================
# ===========================================================================
def select_core_samples(all_features: np.ndarray, all_labels: np.ndarray, centers: np.ndarray, style_map: Dict, mode: str, value: float) -> Dict:
    print(f"\n--- 正在为每个风格筛选核心样本 (模式: {mode}, 值: {value}) ---")
    core_samples_indices_by_style = defaultdict(list)
    for label_idx, style_name in style_map.items():
        indices_in_cluster = np.where(all_labels == label_idx)[0]
        if len(indices_in_cluster) == 0: continue
        num_to_select = int(value) if mode == 'count' else int(len(indices_in_cluster) * value)
        num_to_select = min(num_to_select, len(indices_in_cluster))
        if num_to_select == 0:
            print(f"  - 风格 '{style_name}': 根据比例计算需筛选0个样本，跳过。")
            continue
        features_in_cluster = all_features[indices_in_cluster]
        distances = np.linalg.norm(features_in_cluster - centers[label_idx], axis=1)
        top_local_indices = np.argsort(distances)[:num_to_select]
        top_global_indices = indices_in_cluster[top_local_indices]
        core_samples_indices_by_style[style_name] = top_global_indices
        print(f"  - 风格 '{style_name}': 从 {len(indices_in_cluster)} 个总样本中筛选出 {len(top_global_indices)} 个核心样本。")
    return core_samples_indices_by_style
def visualize_style_proportions(all_labels: np.ndarray, style_map: Dict, saved_counts: Dict, output_path: str):
    print("\n--- 正在生成风格总体分布饼图 ---")
    total_counts = {style_map[i]: count for i, count in enumerate(np.bincount(all_labels))}
    labels, sizes = [], []
    for style_name in ["Aggressive", "Normal", "Conservative"]:
        total, saved = total_counts.get(style_name, 0), saved_counts.get(style_name, 0)
        if total > 0:
            labels.append(f"{style_name}\n(Total: {total}, Saved: {saved})")
            sizes.append(total)
    if not sizes: return
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax.axis('equal'); plt.title('Proportion of Each Style in the Full Dataset', fontsize=16)
    plt.savefig(os.path.join(output_path, "style_proportions_pie_chart.png")); plt.close(fig)
    print(f"--- 风格分布饼图已保存至: {os.path.join(output_path, 'style_proportions_pie_chart.png')} ---")
def visualize_feature_distributions(features_df: pd.DataFrame, output_path: str, title_prefix: str):
    print(f"\n--- 正在为 '{title_prefix}' 生成特征分布箱形图 ---")
    df_all_category = features_df.copy(); df_all_category['style'] = 'All Data'
    df_for_plotting = pd.concat([features_df, df_all_category], ignore_index=True)
    style_order = ["Conservative", "Normal", "Aggressive", "All Data"]
    feature_names = [col for col in features_df.columns if col != 'style']
    num_features = len(feature_names)
    fig, axes = plt.subplots(4, 2, figsize=(24, 25)); axes = axes.flatten()
    for i, feature in enumerate(feature_names):
        sns.boxplot(x='style', y=feature, data=df_for_plotting, order=style_order, ax=axes[i], palette="viridis", hue='style')
        axes[i].set_title(f'Distribution of "{feature}"', fontsize=14); axes[i].set_xlabel("Driving Style (Category)", fontsize=12); axes[i].set_ylabel("Feature Value (Raw)", fontsize=12); axes[i].grid(True, linestyle='--', alpha=0.6)
    for i in range(num_features, len(axes)): fig.delaxes(axes[i])
    fig.suptitle(f'Feature Distributions for {title_prefix}', fontsize=22, y=1.0); plt.tight_layout(pad=3.0)
    save_path = os.path.join(output_path, f"style_distributions_{title_prefix.lower().replace(' ', '_')}.png")
    plt.savefig(save_path); plt.close(fig); print(f"--- 特征分布图已保存至: {save_path} ---")
def generate_statistical_report(features_df: pd.DataFrame, output_path: str, title: str):
    print(f"\n--- 正在为 '{title}' 生成详细统计报告 ---")
    report_lines = ["=" * 80, f"          STATISTICAL REPORT FOR: {title}", "=" * 80, "\n"]
    def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
    def coeff_of_variation(x): return x.std() / x.mean() if abs(x.mean()) > 1e-9 else np.nan
    agg_funcs = ['mean', 'median', 'std', 'var', iqr, coeff_of_variation, 'min', 'max']
    feature_names = [col for col in features_df.columns if col != 'style']
    style_stats_df = features_df.groupby('style')[feature_names].agg(agg_funcs).round(4)
    report_lines.extend(["--- Statistics by Style ---", style_stats_df.to_string(), "\n\n"])
    overall_stats_df = features_df[feature_names].agg(agg_funcs).round(4)
    report_lines.extend(["--- Statistics for All Data Combined ---", overall_stats_df.to_string(), "\n\n"])
    with open(os.path.join(output_path, "statistics_report.txt"), "a") as f: f.write("\n".join(report_lines))
    print(f"--- 统计报告已追加至: {os.path.join(output_path, 'statistics_report.txt')} ---")
def identity_convert_func(scenario_dict, version): return scenario_dict


# ===========================================================================
# ===== D. 新的JIT工作流函数 ================================================
# ===========================================================================

def extract_recipe_worker(
        log_db_file: str, raw_data_path: str, map_root: str, max_scenarios: int, intermediate_dir: str
) -> str:
    """
    【阶段一 Worker】处理单个DB文件，提取轻量级“配方”并保存为中间文件。
    返回中间文件路径。
    """
    scenarios = get_nuplan_scenarios_from_db(raw_data_path, map_root, log_db_file)
    if not scenarios: return None

    if max_scenarios > 0 and len(scenarios) > max_scenarios:
        scenarios = scenarios[:max_scenarios]

    WINDOW_SIZE, STRIDE = 81, 30
    recipes = []
    log_name = os.path.splitext(log_db_file)[0]

    for scenario in scenarios:
        scenario_token = scenario.token
        scenario_type = scenario.scenario_type
        if len(scenario.get_scenario_tokens()) < WINDOW_SIZE + 2: continue

        full_ego_trajectory_list = list(scenario.get_expert_ego_trajectory())
        if len(full_ego_trajectory_list) < WINDOW_SIZE + 2: continue

        for start_frame in range(0, len(full_ego_trajectory_list) - WINDOW_SIZE - 2, STRIDE):
            ego_states_slice = full_ego_trajectory_list[start_frame : start_frame + WINDOW_SIZE + 2]
            ego_slice_np = np.array([[s.waypoint.x, s.waypoint.y, s.waypoint.heading, s.dynamic_car_state.center_velocity_2d.x, s.dynamic_car_state.center_velocity_2d.y] for s in ego_states_slice], dtype=np.float32)

            raw_features = calculate_driving_style_features(ego_slice_np, scenario, start_frame)
            if not raw_features: continue

            recipe = {
                "features": raw_features,
                "scenario_type": scenario_type,
                "log_name": log_name,
                "scenario_token": scenario_token,
                "start_frame": start_frame
            }
            recipes.append(recipe)

    if not recipes: return None

    output_path = os.path.join(intermediate_dir, f"{log_name}_recipes.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(recipes, f)

    # print(f"[Worker for {log_db_file}] 已保存 {len(recipes)} 个配方到: {output_path}")
    return output_path


def rehydrate_slice_worker(recipe: Dict[str, Any], raw_data_path: str, map_root: str) -> Dict:
    """
    【阶段三 Worker】接收一个“配方”，重新计算并返回一个完整的ScenarioDescription对象。
    """
    log_db_file = f"{recipe['log_name']}.db"

    # 注意：这里每次调用都会加载一次场景，是“时间换空间”策略的成本所在
    scenarios = get_nuplan_scenarios_from_db(raw_data_path, map_root, log_db_file)

    target_scenario = None
    for s in scenarios:
        if s.token == recipe['scenario_token']:
            target_scenario = s
            break

    if not target_scenario: return None

    WINDOW_SIZE = 81
    start_frame = recipe['start_frame']

    full_ego_trajectory_list = list(target_scenario.get_expert_ego_trajectory())
    ego_states_slice = full_ego_trajectory_list[start_frame : start_frame + WINDOW_SIZE + 2]
    ego_slice_np = np.array([[s.waypoint.x, s.waypoint.y, s.waypoint.heading, s.dynamic_car_state.center_velocity_2d.x, s.dynamic_car_state.center_velocity_2d.y] for s in ego_states_slice], dtype=np.float32)

    # 重新计算完整的训练数据
    training_data = extract_training_data_for_slice(target_scenario, start_frame, WINDOW_SIZE, ego_slice_np)
    if training_data:
        training_data['features'] = recipe['features'] # 从配方中附加回特征
        SD.update_summaries(training_data)

    return training_data


def rehydrate_log_file_worker(
        log_name: str,
        recipes: List[Dict[str, Any]],
        raw_data_path: str,
        map_root: str
) -> List[Dict]:
    """
    【优化版 阶段三 Worker】接收一个log文件对应的所有“配方”，
    一次性加载该log，并计算出所有配方对应的完整数据。
    """
    log_db_file = f"{log_name}.db"

    # ★★★ 核心优化：对于一个log文件，只加载一次！★★★
    all_scenarios_in_log = get_nuplan_scenarios_from_db(raw_data_path, map_root, log_db_file)
    if not all_scenarios_in_log:
        return []

    # 为了快速查找，将场景列表转换为字典
    scenarios_dict = {s.token: s for s in all_scenarios_in_log}

    rehydrated_slices = []
    WINDOW_SIZE = 81

    for recipe in recipes:
        target_scenario = scenarios_dict.get(recipe['scenario_token'])
        if not target_scenario:
            continue

        start_frame = recipe['start_frame']

        full_ego_trajectory_list = list(target_scenario.get_expert_ego_trajectory())
        ego_states_slice = full_ego_trajectory_list[start_frame: start_frame + WINDOW_SIZE + 2]
        ego_slice_np = np.array([[s.waypoint.x, s.waypoint.y, s.waypoint.heading,
                                  s.dynamic_car_state.center_velocity_2d.x, s.dynamic_car_state.center_velocity_2d.y]
                                 for s in ego_states_slice], dtype=np.float32)

        training_data = extract_training_data_for_slice(target_scenario, start_frame, WINDOW_SIZE, ego_slice_np)

        if training_data:
            training_data['features'] = recipe['features']
            training_data['style'] = recipe['style']  # 风格也从配方中获取
            SD.update_summaries(training_data)
            rehydrated_slices.append(training_data)

    return rehydrated_slices


def main_workflow_jit(raw_data_path: str, map_root: str, db_files: List[str], output_path: str,
                      selection_mode: str, top_k: int, top_p: float, max_scenarios_per_db: int,
                      num_workers: int):
    """【JIT优化版】主工作流函数"""

    intermediate_dir = os.path.join(output_path, "intermediate_recipes")
    os.makedirs(intermediate_dir, exist_ok=True)
    print(f"轻量级配方中间文件将保存在: {intermediate_dir}")

    # --- 阶段一: 并行提取“配方”并保存 ---
    print("\n--- 阶段一: 并行提取轻量级“配方” ---")
    recipe_files = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_recipe_worker, db, raw_data_path, map_root, max_scenarios_per_db, intermediate_dir): db for db in db_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="提取配方"):
            try:
                file_path = future.result()
                if file_path: recipe_files.append(file_path)
            except Exception as exc:
                print(f'DB文件 {futures[future]} 生成了异常: {exc}')

    if not recipe_files:
        print("错误：未能生成任何有效的配方文件。程序终止。")
        return

    # --- 阶段二: 聚类、分析和筛选“配方” ---
    print("\n--- 阶段二: 加载配方进行聚类与分析 ---")
    all_recipes = []
    for file_path in tqdm(recipe_files, desc="加载配方"):
        with open(file_path, 'rb') as f:
            all_recipes.extend(pickle.load(f))

    if not all_recipes:
        print("错误：配方文件中不包含任何有效数据。程序终止。")
        return

    # 从配方中提取聚类所需信息
    all_features = [r['features'] for r in all_recipes]
    all_scenario_types = [r['scenario_type'] for r in all_recipes]
    feature_names = list(all_features[0].keys())

    # --- 聚类与分析逻辑 (与之前版本完全相同, 但操作对象是轻量级数据) ---
    features_by_scenario_type = defaultdict(list)
    for r in all_recipes:
        features_by_scenario_type[r['scenario_type']].append(list(r['features'].values()))

    statistical_model = {}
    for scenario_type, features_list in features_by_scenario_type.items():
        if len(features_list) < 20: continue
        scaler = StandardScaler().fit(features_list)
        statistical_model[scenario_type] = {'mean': dict(zip(feature_names, scaler.mean_)), 'std': dict(zip(feature_names, scaler.scale_))}

    normalized_features, valid_indices = [], []
    for i, r in enumerate(all_recipes):
        if r['scenario_type'] in statistical_model:
            model, raw_vector = statistical_model[r['scenario_type']], np.array(list(r['features'].values()))
            mean_vector = np.array([model['mean'][name] for name in feature_names])
            std_vector = np.array([model['std'][name] for name in feature_names])
            std_vector[std_vector < 1e-6] = 1.0
            normalized_features.append((raw_vector - mean_vector) / std_vector)
            valid_indices.append(i)

    # X = np.array(normalized_features)
    # kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X)
    # labels, centers = kmeans.labels_, kmeans.cluster_centers_

    print("\n--- 使用Faiss进行快速聚类 ---")
    X = np.array(normalized_features).astype('float32')  # Faiss 需要 float32
    n_clusters = 3
    d = X.shape[1]  # 特征维度

    kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, nredo=5, gpu=False)  # niter:迭代次数, nredo:随机初始化次数
    kmeans.train(X)

    # 获取聚类结果
    D, labels_faiss = kmeans.index.search(X, 1)
    labels = labels_faiss.ravel()  # 将结果转换为一维数组
    centers = kmeans.centroids  # 获取聚类中心

    agg_idx, thw_idx = feature_names.index('max_acceleration'), feature_names.index('min_time_headway')
    style_map = {np.argmax(centers[:, agg_idx]): "Aggressive", np.argmax(centers[:, thw_idx]): "Conservative"}
    style_map[list(set(range(3)) - set(style_map.keys()))[0]] = "Normal"
    print("\n聚类释义完成:")
    for i, name in style_map.items(): print(f"  Cluster {i} -> '{name}'")

    # --- 可视化与报告 (逻辑不变) ---
    valid_raw_features = [list(all_recipes[i]['features'].values()) for i in valid_indices]
    features_df_full = pd.DataFrame(valid_raw_features, columns=feature_names)
    features_df_full['style'] = [style_map[l] for l in labels]
    report_file_path = os.path.join(output_path, "statistics_report.txt")
    if os.path.exists(report_file_path): os.remove(report_file_path)
    generate_statistical_report(features_df_full, output_path, title="Full Dataset (Before Selection)")
    visualize_feature_distributions(features_df_full, output_path, title_prefix="Full Dataset")

    # 筛选核心样本索引
    selection_value = top_k if selection_mode == 'count' else top_p
    core_samples_indices_by_style = select_core_samples(X, labels, centers, style_map, selection_mode, selection_value)

    # 提取核心“配方”
    core_recipes_to_rehydrate = []
    for style, indices in core_samples_indices_by_style.items():
        for idx in indices:
            original_recipe_idx = valid_indices[idx]
            recipe = all_recipes[original_recipe_idx]
            recipe['style'] = style # 为配方打上最终的风格标签
            core_recipes_to_rehydrate.append(recipe)

    # --- 阶段三: JIT重计算核心样本并保存 ---
    # 聚合配方：将所有核心配方按log_name分组
    recipes_by_log = defaultdict(list)
    for recipe in core_recipes_to_rehydrate:
        recipes_by_log[recipe['log_name']].append(recipe)

    print(
        f"\n--- 阶段三: JIT重计算 {len(core_recipes_to_rehydrate)} 个核心样本 (聚合在 {len(recipes_by_log)} 个日志文件中) ---")
    final_dataset_by_style = defaultdict(list)

    # 创建进程池，为每个log文件提交一个聚合任务
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交聚合后的任务
        futures = {
            executor.submit(rehydrate_log_file_worker, log_name, recipes, raw_data_path, map_root): log_name
            for log_name, recipes in recipes_by_log.items()
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="JIT重计算(聚合)"):
            try:
                # 一个future返回一个log中所有核心样本的完整数据列表
                hydrated_slices_list = future.result()
                for full_slice_data in hydrated_slices_list:
                    style = full_slice_data['style']
                    final_dataset_by_style[style].append(full_slice_data)
            except Exception as exc:
                log_name = futures[future]
                print(f"重计算日志 {log_name} 时发生错误: {exc}")

    # --- 保存最终数据集 (逻辑不变) ---
    print("\n--- 正在将最终的风格化数据保存为 scenarionet 格式 ---")
    for style_name, data_list in final_dataset_by_style.items():
        if not data_list: continue
        style_output_dir = os.path.join(output_path, style_name)
        print(f"\n--- 正在为风格 '{style_name}' 生成数据集... ---")
        write_to_directory(scenarios=data_list, convert_func=identity_convert_func, output_path=style_output_dir, dataset_version="v1.1", dataset_name=f"nuplan_style_{style_name.lower()}", overwrite=True, num_workers=8)
        print(f"  - 风格 '{style_name}' 的数据集已成功保存至: {style_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从nuPlan日志生成风格化的驾驶数据集（JIT内存与空间优化版）")
    # (ArgumentParser 参数定义不变)
    parser.add_argument('--raw_data_path', type=str, default=r'/mnt/mydata/lishangwen/TrafficDataSetSource/nuplan/dataset/nuplan-v1.1/splits/train_boston', help='包含 .db 文件的nuPlan数据目录')
    parser.add_argument('--map_root', type=str, default=r'/mnt/mydata/lishangwen/TrafficDataSetSource/nuplan/dataset/maps', help='nuPlan地图根目录')
    parser.add_argument('--output_path', type=str, default="/mnt/mydata/lishangwen/TrafficDataSetSource/nuplan/stylized_data_boston_splits_0-800_ratio_0.15", help='输出文件和图表的根目录')
    parser.add_argument('--db_files_slice', type=str, default=":800", help='要处理的db文件切片，格式如 ":5", "3:10", "3:"')
    parser.add_argument('--debug_max_scenarios_per_db', type=int, default=-1, help='【调试用】每个DB文件最多处理的场景数量。设为-1表示无限制。')
    parser.add_argument('--selection_mode', type=str, default='proportion', choices=['count', 'proportion'], help='核心样本的筛选模式: "count" (按数量) 或 "proportion" (按比例)')
    parser.add_argument('--top_k', type=int, default=128, help='在 "count" 模式下，每个风格保留的核心样本数量')
    parser.add_argument('--top_p', type=float, default=0.15, help='在 "proportion" 模式下，每个风格保留的核心样本比例 (0.0 到 1.0)')
    parser.add_argument('--num_workers', type=int, default=None, help='并行处理的工作单元数量。默认为系统CPU核心数。')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    try:
        all_db_files = sorted([f for f in os.listdir(args.raw_data_path) if f.endswith(".db")])
        slice_parts = args.db_files_slice.split(':')
        start = int(slice_parts[0]) if slice_parts[0] else None
        end = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None
        DB_FILES_TO_PROCESS = all_db_files[start:end]
        if not DB_FILES_TO_PROCESS: raise FileNotFoundError("切片结果为空")
    except (FileNotFoundError, IndexError):
        print(f"错误: 无法根据切片 '{args.db_files_slice}' 在目录 '{args.raw_data_path}' 中找到任何 .db 文件。")
        sys.exit(1)

    print(f"将要处理以下 {len(DB_FILES_TO_PROCESS)} 个DB文件: {DB_FILES_TO_PROCESS}")

    # 调用新的JIT工作流
    main_workflow_jit(
        raw_data_path=args.raw_data_path, map_root=args.map_root, db_files=DB_FILES_TO_PROCESS,
        output_path=args.output_path, selection_mode=args.selection_mode, top_k=args.top_k,
        top_p=args.top_p, max_scenarios_per_db=args.debug_max_scenarios_per_db,
        num_workers=args.num_workers
    )