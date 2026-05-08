"""
Module: Data Processor

此模块负责将 NuPlan 的原始场景数据 (Scenario/History) 转换为模型可接受的 Tensor 或 NumPy 格式。
它包含两个核心流程：
1. `observation_adapter`: 用于闭环仿真 (Inference)，实时处理当前帧数据。
2. `work`: 用于离线数据预处理 (Training Data Generation)，处理整个场景序列并保存为 .npz 文件。
"""

import os
import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D

from baseline.data_process.roadblock_utils import route_roadblock_correction
from baseline.data_process.agent_process import (
    agent_past_process,
    sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list,
    agent_future_process
)
from baseline.data_process.map_process import get_neighbor_vector_set_map, map_process
from baseline.data_process.ego_process import (
    get_ego_past_array_from_scenario, 
    get_ego_future_array_from_scenario, 
    calculate_additional_ego_states,
    sampled_past_ego_states_to_array
)

from baseline.data_process.utils import convert_to_model_inputs
from baseline.data_process.codebook_labeler import CodebookLabeler


class DataProcessor(object):
    def __init__(self, config):
        """
        初始化数据处理器，配置时间视窗、Agent 数量和地图特征参数。
        """
        self._save_dir = getattr(config, "save_path", None)

        # 时间参数配置
        self.past_time_horizon = 2  # [seconds]  # 历史采样时间，不包括当前时刻点
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        # 场景元素数量限制
        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10  # Limit the number of pedestrians and bicycles in the agent list.

        # 地图查询参数
        self._radius = 100  # [m] query radius scope relative to the current pose.
        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']  # Features to extract

        # 地图元素最大数量限制
        self._max_elements = {
            'LANE': config.lane_num,
            'LEFT_BOUNDARY': config.lane_num,
            'RIGHT_BOUNDARY': config.lane_num,
            'ROUTE_LANES': config.route_num
        }

        # 每个地图元素的最大点数
        self._max_points = {
            'LANE': config.lane_len,
            'LEFT_BOUNDARY': config.lane_len,
            'RIGHT_BOUNDARY': config.lane_len,
            'ROUTE_LANES': config.route_len
        }


    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):
        # ---------------------------------------------------
        # ** Ego Agent Past & Current State
        # ---------------------------------------------------
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
                                    dtype=np.float64)

        # 获取历史状态队列
        all_ego_states = list(history_buffer.ego_states)
        target_frames = self.num_past_poses + 1

        if len(all_ego_states) >= target_frames:
            sampled_ego_states = all_ego_states[-target_frames:]
        else:
            padding_count = target_frames - len(all_ego_states)
            sampled_ego_states = [all_ego_states[0]] * padding_count + all_ego_states

        # 1. 获取原始 Global 历史
        ego_agent_past_global = sampled_past_ego_states_to_array(sampled_ego_states).astype(np.float32)

        # 获取时间戳
        past_time_stamps = [state.time_point.time_us for state in sampled_ego_states]
        past_time_stamps = np.array(past_time_stamps, dtype=np.int64)

        # ---------------------------------------------------
        # ** Neighbor Agents & Static Objects
        # ---------------------------------------------------
        observation_buffer = history_buffer.observation_buffer
        raw_tracked_objects = [obs.tracked_objects for obs in observation_buffer]

        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])

        # 2. 调用 agent_past_process 进行坐标转换
        # ⚠️ [修复核心] 接收第一个返回值 ego_agent_past_relative
        ego_agent_past_relative, neighbor_agents_past, neighbor_agents_past_mask, _, static_objects, _ = \
            agent_past_process(ego_agent_past_global, neighbor_agents_past, neighbor_agents_types, self.num_agents,
                               static_objects, static_objects_types, self.num_static, self.max_ped_bike,
                               anchor_ego_state, raw_tracked_objects)

        # 3. 基于相对历史计算相对 Current State
        # 这样算出来的 x, y 就是 0, 0 (符合模型训练分布)
        ego_current_state_relative = calculate_additional_ego_states(ego_agent_past_relative, past_time_stamps).astype(
            np.float32)

        # ---------------------------------------------------
        # ** Map & Route Features
        # ---------------------------------------------------
        route_roadblock_ids = route_roadblock_correction(ego_state, map_api, route_roadblock_ids)
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit,
                                 lane_route, self._map_features,
                                 self._max_elements, self._max_points)

        # ---------------------------------------------------
        # ** Construct Output
        # ---------------------------------------------------
        data = {
            "neighbor_agents_past": neighbor_agents_past[:, -21:],
            "neighbor_agents_past_mask": neighbor_agents_past_mask[:, -21:],
            "ego_current_state": ego_current_state_relative,  # ✅ 修复：现在是 Relative 的
            "static_objects": static_objects,
            "ego_agent_past": ego_agent_past_relative  # ✅ 修复：现在是 Relative 的
        }

        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data


    def work(self, scenarios):
        """
        [Preprocessing] 离线处理数据集中的 Scenarios，提取特征并保存为 .npz 文件。
        新增：集成 Codebook Label Generation (Step 1)
        """

        # 延迟导入或确保在外部已导入
        from baseline.data_process.codebook_labeler import CodebookLabeler

        # 初始化 Labeler (只需要实例化一次，map_api 在循环中动态更新)
        # 注意：这里我们传入 None 作为 map_api 占位，稍后更新
        labeler = CodebookLabeler(map_api=None, lookahead_dist=50.0, num_bins=8,
                                  custom_bins=[0.40, 0.80, 1.00, 1.35, 2.30, 3.00, 3.50])

        # === [新增 1] 初始化统计计数器 ===
        stats = {
            "total_frames": 0,
            "static_frames": 0,
            "static_valid": 0,  # 静止且 Label != -1
            "moving_frames": 0,
            "moving_valid": 0,  # 运动且 Label != -1
        }
        # ===============================

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api

            # 更新 Labeler 的 Map API
            labeler.map_api = map_api

            # ---------------------------------------------------
            # ** Ego & Agents Past History
            # ---------------------------------------------------
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
                                        dtype=np.float64)

            # 提取自车历史
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses,
                                                                                self.past_time_horizon)

            # 提取他车历史
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]

            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)

            static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

            # 处理历史数据 & 筛选 Top-K Agents
            ego_agent_past, neighbor_agents_past, neighbor_agents_past_mask, neighbor_indices, static_objects, track_tokens = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents,
                                   static_objects, static_objects_types, self.num_static, self.max_ped_bike,
                                   anchor_ego_state, sampled_past_observations)

            # ---------------------------------------------------
            # ** Map Features
            # ---------------------------------------------------
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids
                )

            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius, traffic_light_data
            )

            vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit,
                                     lane_route, self._map_features,
                                     self._max_elements, self._max_points)

            # ---------------------------------------------------
            # ** Ego & Agents Future Ground Truth
            # ---------------------------------------------------
            # 1. 获取相对坐标用于训练 (Relative Frame)
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses,
                                                                  self.future_time_horizon)

            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]
            sampled_future_observations = [present_tracked_objects] + future_tracked_objects
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)

            neighbor_agents_future, neighbor_agents_future_mask = agent_future_process(anchor_ego_state,
                                                                                       future_tracked_objects,
                                                                                       self.num_agents,
                                                                                       track_tokens)

            ego_future_abs_states = list(scenario.get_ego_future_trajectory(
                iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
            ))

            if len(ego_future_abs_states) > 0:
                ego_future_abs_np = np.array([
                    [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    for state in ego_future_abs_states
                ], dtype=np.float32)

                # === FIX: 将 GT future 从全局坐标转为相对坐标（与 vector_map.route_lanes 同一坐标系）===
                ax, ay, ah = float(anchor_ego_state[0]), float(anchor_ego_state[1]), float(anchor_ego_state[2])
                c, s = np.cos(-ah), np.sin(-ah)

                xy = ego_future_abs_np[:, :2].copy()
                xy[:, 0] -= ax
                xy[:, 1] -= ay
                x_rel = xy[:, 0] * c - xy[:, 1] * s
                y_rel = xy[:, 0] * s + xy[:, 1] * c

                h_rel = ego_future_abs_np[:, 2] - ah
                # wrap 到 [-pi, pi]
                h_rel = (h_rel + np.pi) % (2 * np.pi) - np.pi

                ego_future_rel_np = np.stack([x_rel, y_rel, h_rel], axis=1).astype(np.float32)

                current_speed = ego_state.dynamic_car_state.speed

                labels = labeler.get_labels(
                    ego_state,
                    ego_future_rel_np,  # ✅ 注意：这里改成相对坐标
                    current_speed,
                    self.future_time_horizon,
                    route_roadblock_ids=route_roadblock_ids,
                    route_lanes=vector_map.get("route_lanes", None),
                    route_lanes_mask=vector_map.get("route_lanes_mask", None),
                    lat_match_dist_thresh=5.0
                )
            else:
                labels = {"a_lat": -1, "a_lon": -1, "rho": 0.0}

            # ---------------------------------------------------
            # ** Ego Current State (Calculated)
            # ---------------------------------------------------
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # 检查 a_lat 是否超出 route 界限
            route_num = self._max_elements["ROUTE_LANES"]  # 也就是 config.route_num
            a_lat = int(labels.get("a_lat", -1))

            if not (a_lat == -1 or (0 <= a_lat < route_num)):
                mask = vector_map.get("route_lanes_mask", None)
                valid_lanes = 0
                if mask is not None:
                    mask = np.asarray(mask)
                    if mask.ndim == 2:
                        valid_lanes = int((mask.sum(axis=1) > 0).sum())
                    elif mask.ndim == 1:
                        valid_lanes = int(mask.astype(bool).sum())

                if not (a_lat == -1 or (0 <= a_lat < route_num)):
                    print(f"[LatLabelRangeError] token={token} map={map_name} "
                          f"a_lat={a_lat} route_num={route_num} valid_lanes={valid_lanes}")

            data = {
                "map_name": map_name,
                "token": token,
                "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future,
                "ego_agent_past": ego_agent_past,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_past_mask": neighbor_agents_past_mask,
                "neighbor_agents_future": neighbor_agents_future,
                "neighbor_agents_future_mask": neighbor_agents_future_mask,
                "static_objects": static_objects,

                # === [新增] Codebook Labels ===
                "code_lat": np.array(labels["a_lat"], dtype=np.int64),  # 
                "code_lon": np.array(labels["a_lon"], dtype=np.int64),  # 纵向意图标签
                "code_rho": np.array(labels["rho"], dtype=np.float32)   # 具体的纵向连续数值
            }
            data.update(vector_map)

            # # === [新增] 统计探针 & Debug 绘图 ===
            # # =================================================
            # # 1. 统计逻辑
            # is_static = current_speed < 0.1
            # a_lat = labels.get("a_lat", -1)
            #
            # stats["total_frames"] += 1
            # if is_static:
            #     stats["static_frames"] += 1
            #     if a_lat != -1: stats["static_valid"] += 1
            # else:
            #     stats["moving_frames"] += 1
            #     if a_lat != -1: stats["moving_valid"] += 1
            #
            # # 每 20 帧打印一次统计 (仅在单进程调试模式下有效，多进程会乱序但也能看)
            # if stats["total_frames"] % 20 == 0:
            #     s_rate = (stats["static_valid"] / stats["static_frames"] * 100) if stats["static_frames"] > 0 else 0
            #     m_rate = (stats["moving_valid"] / stats["moving_frames"] * 100) if stats["moving_frames"] > 0 else 0
            #     print(
            #         f"\n[PROBE] Total: {stats['total_frames']} | Static Valid: {s_rate:.1f}% ({stats['static_valid']}/{stats['static_frames']}) | Moving Valid: {m_rate:.1f}%")
            #
            # # 2. 绘图逻辑 (仅针对静止车的高概率抽样检查)
            # # 触发条件: 是静止车 AND (Label是-1 或者 随机抽中)
            # # 这里的路径改为你自己的 debug 目录
            # DEBUG_SAVE_DIR = "/mnt/mydata/lishangwen/NuplanBaselinesRecord/debug_plots"
            #
            # if is_static and (a_lat == -1 or np.random.rand() < 0.05):  # 抽 5% 的静止车看，或者失败的静止车必看
            #     try:
            #         import matplotlib.pyplot as plt
            #         import os
            #         os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
            #
            #         plt.figure(figsize=(8, 8))
            #         plt.title(f"Speed:{current_speed:.2f}|Lat:{a_lat}|T:{token[-4:]}")
            #
            #         # 画 Ego
            #         plt.plot(0, 0, 'ro', markersize=8, label='Ego')
            #         plt.arrow(0, 0, 2, 0, head_width=0.5, color='r')
            #
            #         # 画车道
            #         rl = vector_map['route_lanes'][..., :2]
            #         rl_mask = vector_map['route_lanes_mask']
            #         for i in range(len(rl)):
            #             if not rl_mask[i].any(): continue
            #             pts = rl[i][rl_mask[i]]
            #             # 选中为绿色，未选中为灰色
            #             c = 'green' if i == a_lat else 'gray'
            #             w = 2 if i == a_lat else 1
            #             plt.plot(pts[:, 0], pts[:, 1], color=c, alpha=0.5, linewidth=w)
            #             if len(pts) > 0:
            #                 plt.text(pts[0, 0], pts[0, 1], str(i), color=c, fontsize=8)
            #
            #         # 画 GT (相对坐标)
            #         if ego_future_abs_states is not None:
            #             plt.plot(ego_future_rel_np[:, 0], ego_future_rel_np[:, 1], 'b--', label='GT')
            #
            #         plt.legend()
            #         plt.axis('equal')
            #         # 保存文件名带上状态，方便筛选
            #         status = "OK" if a_lat != -1 else "FAIL"
            #         plt.savefig(f"{DEBUG_SAVE_DIR}/debug_{status}_{token}_{stats['total_frames']}.png")
            #         plt.close()
            #     except Exception as e:
            #         print(f"[DEBUG PLOT ERROR] {e}")
            # # =================================================

            self.save_to_disk(self._save_dir, data)


    def save_to_disk(self, dir, data):
        """保存数据到 .npz 文件"""
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)