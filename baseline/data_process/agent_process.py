"""
Module: Agent Data Preprocessing Functions

此模块负责处理 NuPlan 仿真中的交通参与者 (Agents) 和静态物体 (Static Objects) 数据。
主要功能包括：
- 从原始 TrackedObjects 提取数组化特征
- 执行坐标系转换 (绝对坐标 -> 自车相对坐标)
- 历史轨迹的对齐、填充与过滤
- 基于距离和类型的 Agent 筛选策略 (Top-K Selection)
- 未来轨迹 (Ground Truth) 的提取与对齐
"""

import numpy as np
from typing import Dict

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from baseline.data_process.utils import convert_absolute_quantities_to_relative


# ==============================================================================
# *** Get list of agent array from raw data
# ==============================================================================

def _extract_agent_array(tracked_objects, track_token_ids, object_types):
    """
    将单帧内的 TrackedObjects 转换为符合 AgentInternalIndex 定义的 NumPy 数组。
    仅转换指定类型的对象，并维护 Token 到 Integer ID 的映射。

    Args:
        tracked_objects: 当前帧的跟踪对象列表
        track_token_ids: Token 到 Int ID 的映射字典
        object_types: 需要提取的对象类型列表

    Returns:
        output: Agent 属性数组 [N, Dim]
        track_token_ids: 更新后的映射字典
        agent_types: 对应的对象类型列表
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = np.zeros((len(agents), AgentInternalIndex.dim()), dtype=np.float64)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        # 维护 Token -> Int ID 的一致性
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        # 填充属性
        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y

        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_array_list(past_tracked_objects):
    """
    将一系列历史帧的 Agent 数据转换为数组列表。

    Args:
        past_tracked_objects: 包含 N 个时间步的 TrackedObjects 列表

    Returns:
        output: 长度为 N 的列表，每个元素为 NumPy 数组
        output_types: 对应的类型列表
    """
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        # 兼容 DetectionsTracks 对象和直接的 List
        if type(past_tracked_objects[i]) == DetectionsTracks:
            track_object = past_tracked_objects[i].tracked_objects
        else:
            track_object = past_tracked_objects[i]

        arrayified, track_token_ids, agent_types = _extract_agent_array(track_object, track_token_ids, object_types)
        output.append(arrayified)
        output_types.append(agent_types)

    return output, output_types


def sampled_static_objects_to_array_list(present_tracked_objects):
    """
    提取当前帧的静态物体 (交通锥、路障等) 并转换为数组。
    """
    static_object_types = [
        TrackedObjectType.CZONE_SIGN,
        TrackedObjectType.BARRIER,
        TrackedObjectType.TRAFFIC_CONE,
        TrackedObjectType.GENERIC_OBJECT
    ]

    if type(present_tracked_objects) == DetectionsTracks:
        present_tracked_objects = present_tracked_objects.tracked_objects

    static_obj = present_tracked_objects.get_tracked_objects_of_types(static_object_types)
    agent_types = []

    # 静态物体特征: x, y, heading, width, length
    output = np.zeros((len(static_obj), 5), dtype=np.float64)

    for idx, agent in enumerate(static_obj):
        output[idx, 0] = agent.center.x
        output[idx, 1] = agent.center.y
        output[idx, 2] = agent.center.heading
        output[idx, 3] = agent.box.width
        output[idx, 4] = agent.box.length
        agent_types.append(agent.tracked_object_type)

    return output, agent_types


# ==============================================================================
# *** Get agents array for model input
# ==============================================================================

def _filter_agents_array(agents, reverse: bool = False):
    """
    过滤掉那些没有出现在目标帧（第一帧或最后一帧）的 Agent。
    这确保了我们只处理当前时刻可见，或者起始时刻存在的 Agent。

    Args:
        agents: Agent 历史数组列表
        reverse: 如果为 True，使用最后一帧作为目标帧（通常用于处理 Past History，以 Current Frame 为准）

    Returns:
        filtered_agents: 过滤后的列表
    """
    target_array = agents[-1] if reverse else agents[0]

    for i in range(len(agents)):
        rows = []
        for j in range(agents[i].shape[0]):
            if target_array.shape[0] > 0:
                agent_id: float = float(agents[i][j, int(AgentInternalIndex.track_token())])
                # 检查该 ID 是否存在于目标帧中
                is_in_target_frame: bool = bool(
                    (agent_id == target_array[:, AgentInternalIndex.track_token()]).max()
                )
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        if len(rows) > 0:
            agents[i] = np.stack(rows)
        else:
            agents[i] = np.empty((0, agents[i].shape[1]), dtype=np.float32)

    return agents


def _pad_agent_states(agent_trajectories, reverse: bool):
    """
    对 Agent 轨迹进行填充。如果某帧数据缺失，使用最近的有效帧数据进行填充。
    保持 Agent 的顺序一致。

    Padding Logic (Forward):
     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|

    Padding Logic (Reverse / Backwards from Current):
     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|
    """
    track_id_idx = AgentInternalIndex.track_token()

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    key_frame = agent_trajectories[0]
    id_row_mapping: Dict[int, int] = {}

    # 建立 Key Frame 中 Agent ID 到行号的映射
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val)] = idx

    current_state = np.zeros((key_frame.shape[0], key_frame.shape[1]), dtype=np.float64)

    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]

        # 如果当前帧有数据，更新 current_state
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx])]
            current_state[mapped_row, :] = frame[row_idx, :]

        # 使用 current_state (可能包含刚更新的，也可能包含上一帧的 padding) 覆盖当前帧
        agent_trajectories[idx] = current_state.copy()

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    return agent_trajectories



def _pad_agent_states_with_zeros(agent_trajectories):
    """
    使用全 0 填充缺失的 Agent 状态 (辅助函数)
    """
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()

    pad_agent_trajectories = np.zeros((len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]), dtype=np.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                pad_agent_trajectories[idx, row_idx] = frame[frame[:, track_id_idx]==row_idx]

    return pad_agent_trajectories


# ==============================================================================
# *** Main Processing Logic: Past History
# ==============================================================================

def agent_past_process(past_ego_states, past_tracked_objects, tracked_objects_types, num_agents, static_objects,
                       static_objects_types, num_static, max_ped_bike, anchor_ego_state, raw_tracked_objects):
    """
    处理历史 Agent 数据，生成模型输入特征。

    核心步骤：
    1. 自车坐标系转换 (Ego Coordinate Transform)
    2. 他车历史轨迹过滤与填充 (Filter & Pad Neighbors)
    3. 他车坐标系转换
    4. 静态物体处理
    5. Agent 筛选 (Top-K Selection): 优先保留近距离的行人和非机动车
    6. Token 提取: 用于后续匹配未来轨迹 (Ground Truth)

    Args:
        past_ego_states: 自车历史轨迹
        past_tracked_objects: 他车历史轨迹列表
        tracked_objects_types: 他车类型列表
        num_agents: 最大保留 Agent 数量
        static_objects: 静态物体列表
        static_objects_types: 静态物体类型
        num_static: 最大保留静态物体数量
        max_ped_bike: 最大保留 VRU (行人/自行车) 数量
        anchor_ego_state: 自车当前状态 (作为相对坐标原点)
        raw_tracked_objects: 原始的 TrackedObjects 对象列表 (用于提取 Token)

    Returns:
        ego: 相对坐标下的自车历史
        agents: 筛选并转换后的他车特征 [num_agents, num_frames, 11] (包含 one-hot type)
        past_agents_mask: 有效性 Mask
        selected_indices: 被选中 Agent 的原始索引
        static_objects: 静态物体特征
        selected_tokens: 被选中 Agent 的 Track Tokens
    """
    agents_states_dim = 8  # x, y, cos h, sin h, vx, vy, length, width
    ego_history = past_ego_states
    agents = past_tracked_objects

    # Ego Coordinate Conversion
    if past_ego_states is not None:
        ego = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    else:
        ego = None

    # Filter Neighbors (基于当前帧是否存在)
    agent_history = _filter_agents_array(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    # Process Neighbor Coordinates
    if len(agent_history[-1]) == 0:
        agents_array = np.zeros((len(agent_history), 0, agents_states_dim))
    else:
        local_coords_agent_states = []
        # 使用 Padding 补全历史
        padded_agent_states = _pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(
                convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))

        # 构建特征数组 (计算 cos/sin heading)
        agents_array = np.zeros(
            (len(local_coords_agent_states), local_coords_agent_states[0].shape[0], agents_states_dim)
        )

        for i in range(len(local_coords_agent_states)):
            agents_array[i, :, 0] = local_coords_agent_states[i][:, AgentInternalIndex.x()].squeeze()
            agents_array[i, :, 1] = local_coords_agent_states[i][:, AgentInternalIndex.y()].squeeze()
            agents_array[i, :, 2] = np.cos(local_coords_agent_states[i][:, AgentInternalIndex.heading()].squeeze())
            agents_array[i, :, 3] = np.sin(local_coords_agent_states[i][:, AgentInternalIndex.heading()].squeeze())
            agents_array[i, :, 4] = local_coords_agent_states[i][:, AgentInternalIndex.vx()].squeeze()
            agents_array[i, :, 5] = local_coords_agent_states[i][:, AgentInternalIndex.vy()].squeeze()
            agents_array[i, :, 6] = local_coords_agent_states[i][:, AgentInternalIndex.width()].squeeze()
            agents_array[i, :, 7] = local_coords_agent_states[i][:, AgentInternalIndex.length()].squeeze()

    # Process Static Objects
    static_objects_array = np.zeros((static_objects.shape[0], 6))
    if static_objects.shape[0] != 0:
        local_coords_static_objects_states = convert_absolute_quantities_to_relative(static_objects, anchor_ego_state,
                                                                                     'static')

        static_objects_array[:, 0] = local_coords_static_objects_states[:, 0]
        static_objects_array[:, 1] = local_coords_static_objects_states[:, 1]
        static_objects_array[:, 2] = np.cos(local_coords_static_objects_states[:, 2])
        static_objects_array[:, 3] = np.sin(local_coords_static_objects_states[:, 2])
        static_objects_array[:, 4] = local_coords_static_objects_states[:, 3]
        static_objects_array[:, 5] = local_coords_static_objects_states[:, 4]

    # Agent Selection Strategy (Top-K)
    # 策略：优先保留靠近自车的行人和自行车，然后再填补车辆，最后按距离截断
    agents = np.zeros((num_agents, agents_array.shape[0], agents_array.shape[-1] + 3), dtype=np.float32)

    distance_to_ego = np.linalg.norm(agents_array[-1, :, :2], axis=-1)
    sorted_indices = np.argsort(distance_to_ego)

    ped_bike_indices = [i for i in sorted_indices if
                        agent_types[i] in (TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE)]
    vehicle_indices = [i for i in sorted_indices if agent_types[i] == TrackedObjectType.VEHICLE]

    if len(ped_bike_indices) + len(vehicle_indices) <= num_agents:
        selected_indices = sorted_indices[:num_agents]
    else:
        # 限制 VRU 数量，防止远处的行人挤占了近处车辆的名额
        selected_ped_bike_indices = ped_bike_indices[:max_ped_bike]
        remaining_ped_bike_indices = ped_bike_indices[max_ped_bike:]

        selected_indices = selected_ped_bike_indices + vehicle_indices

        remaining_slots = num_agents - len(selected_indices)
        if remaining_slots > 0:
            selected_indices += remaining_ped_bike_indices[:remaining_slots]

        # 最终按距离重新排序
        selected_indices = sorted(selected_indices, key=lambda idx: distance_to_ego[idx])[:num_agents]

    # Populate Final Array & Add One-Hot Type
    for i, j in enumerate(selected_indices):
        agents[i, :, :agents_array.shape[-1]] = agents_array[:, j, :agents_array.shape[-1]]
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_array.shape[-1]:] = [1, 0, 0]  # Mark as VEHICLE
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_array.shape[-1]:] = [0, 1, 0]  # Mark as PEDESTRIAN
        else:  # TrackedObjectType.BICYCLE
            agents[i, :, agents_array.shape[-1]:] = [0, 0, 1]  # Mark as BICYCLE

    # Token Extraction for Future Alignment
    # 目的：为了确保 Future Ground Truth 与 Past Input 对应的是同一辆车，我们需要提取 Token
    current_frame_container = raw_tracked_objects[-1]

    target_object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    current_agents_list = current_frame_container.get_tracked_objects_of_types(target_object_types)

    # 这里的 j 必须对应 current_agents_list 中的索引
    selected_tokens = [current_agents_list[j].track_token for j in selected_indices]

    # Static Object Selection (Distance-based)
    static_objects = np.zeros((num_static, static_objects_array.shape[-1] + 4), dtype=np.float32)
    static_distance_to_ego = np.linalg.norm(static_objects_array[:, :2], axis=-1)
    static_indices = list(np.argsort(static_distance_to_ego))[:num_static]

    for i, j in enumerate(static_indices):
        static_objects[i, :static_objects_array.shape[-1]] = static_objects_array[j, :static_objects_array.shape[-1]]
        # Static One-Hot Encoding
        if static_objects_types[j] == TrackedObjectType.CZONE_SIGN:
            static_objects[i, static_objects_array.shape[-1]:] = [1, 0, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.BARRIER:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 1, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.TRAFFIC_CONE:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 1, 0]
        else:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 0, 1]

    if ego is not None:
        ego = ego.astype(np.float32)

    # Generate Mask
    # 使用 width > epsilon 判断该帧是否存在有效数据
    past_agents_mask = (agents[:, :, 6] > 1e-6)

    return ego, agents, past_agents_mask, selected_indices, static_objects, selected_tokens


# ==============================================================================
# *** Main Processing Logic: Future GT
# ==============================================================================

def agent_future_process(anchor_ego_state, future_tracked_objects, num_agents, agent_index):
    """
    提取被选中邻居车辆的未来轨迹 (Ground Truth)。
    务必确保 agent_index 与 agent_past_process 输出的 selected_tokens 是一致的，
    从而保证 Past 和 Future 对应的是同一个 Agent。

    Args:
        anchor_ego_state: 当前时刻自车状态 (作为坐标系原点)
        future_tracked_objects: 未来时间步的跟踪对象列表 List[TrackedObjects]
        num_agents: 最大代理数量 (Top-K)
        agent_index: 在过去时间步中被选中的代理的 track_token 列表 (List[str])

    Returns:
        agent_futures: [N, T, 3] (x, y, heading)
        agent_futures_mask: [N, T] Bool Mask
    """
    # Initialize Output
    future_steps = len(future_tracked_objects)
    agent_futures = np.zeros((num_agents, future_steps, 3), dtype=np.float32)

    # Initialize Mask (默认为 False，只有匹配到 Token 才置为 True)
    agent_futures_mask = np.zeros((num_agents, future_steps), dtype=np.bool_)

    # Token Mapping for Fast Lookup
    token_to_idx = {token: i for i, token in enumerate(agent_index) if token is not None}

    # Iterate through Future Frames
    for t, tracked_objects in enumerate(future_tracked_objects):
        # tracked_objects 是该帧所有被跟踪物体的集合
        for obj in tracked_objects:
            # 只处理我们在 Past 步骤中选中的 Top-K 车辆
            if obj.track_token in token_to_idx:
                idx = token_to_idx[obj.track_token]

                # Coordinate Transformation Fix
                # 问题：convert_absolute_quantities_to_relative 期望输入是 (N, Feature_Dim) 的 2D 数组
                # 如果传入 1D 数组或直接取属性，可能会导致维度错误或索引越界。

                # Step 1: 创建符合 Schema 的 2D 单行数组 (1, Dim)
                current_agent_state = np.zeros((1, AgentInternalIndex.dim()), dtype=np.float64)

                # Step 2: 填值 (使用 Index 确保位置正确)
                current_agent_state[0, AgentInternalIndex.x()] = obj.center.x
                current_agent_state[0, AgentInternalIndex.y()] = obj.center.y
                current_agent_state[0, AgentInternalIndex.heading()] = obj.center.heading
                current_agent_state[0, AgentInternalIndex.vx()] = obj.velocity.x
                current_agent_state[0, AgentInternalIndex.vy()] = obj.velocity.y
                current_agent_state[0, AgentInternalIndex.width()] = obj.box.width
                current_agent_state[0, AgentInternalIndex.length()] = obj.box.length

                # Step 3: 坐标转换: World -> Ego Relative
                rel_state = convert_absolute_quantities_to_relative(current_agent_state, anchor_ego_state, 'agent')

                # Step 4: 填充数据 (取第 0 行)
                agent_futures[idx, t, 0] = rel_state[0, AgentInternalIndex.x()]
                agent_futures[idx, t, 1] = rel_state[0, AgentInternalIndex.y()]
                agent_futures[idx, t, 2] = rel_state[0, AgentInternalIndex.heading()]

                # Mark as Valid
                agent_futures_mask[idx, t] = True

    return agent_futures, agent_futures_mask

