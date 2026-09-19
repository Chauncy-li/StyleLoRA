"""统一快照抽取器：一次读 scenario，产出三个模型共用的统一 .npz。

- DiffPlanner 字段：复用 DataProcessor.process_scenario（与 work() 落盘口径完全一致）。
- AD-MLP / StageVec 字段：由同一 ego/map/邻居快照按 featurizers.py 派生。

产物 .npz 字段见 comparison/README.md 第 3.2 节。
"""
import numpy as np

from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    convert_absolute_to_relative_poses,
)

from baseline.data_process import featurizers
from baseline.data_process.data_processor import DataProcessor

# AD-MLP / StageVec 共用的目标采样（4s @ 0.5s -> 8 帧）
ADMLP_TIME_HORIZON = 4.0
ADMLP_NUM_POSES = 8


class UnifiedExtractor:
    def __init__(self, config):
        self._processor = DataProcessor(config)

    def extract_scenario(self, scenario):
        data = self._processor.process_scenario(scenario)

        ego = scenario.initial_ego_state
        agents = scenario.initial_tracked_objects.tracked_objects.get_agents()
        map_api = scenario.map_api

        # ---- AD-MLP ----
        data["admlp_x"] = featurizers.build_admlp_input(ego, map_api)
        fut8 = list(scenario.get_ego_future_trajectory(
            iteration=0, time_horizon=ADMLP_TIME_HORIZON, num_samples=ADMLP_NUM_POSES
        ))
        if len(fut8) >= ADMLP_NUM_POSES:
            data["admlp_y"] = convert_absolute_to_relative_poses(
                ego.rear_axle, [s.rear_axle for s in fut8]
            ).astype(np.float32)
        else:
            data["admlp_y"] = np.zeros((ADMLP_NUM_POSES, 3), dtype=np.float32)

        # ---- StageVec ----
        fut1 = list(scenario.get_ego_future_trajectory(iteration=0, time_horizon=0.1, num_samples=1))
        yaw_rate = featurizers.derive_yaw_rate(fut1[0], ego, 0.1) if len(fut1) >= 1 else 0.0

        data["stage_vec_x"] = featurizers.build_vector(ego, agents, map_api, yaw_rate=yaw_rate)
        prefer = featurizers.build_prefer(ego, agents)
        data["stage_vec_prefer"] = prefer
        # AD-MLP 风格标签的连续激进分（规则代理打分，供 train 侧分箱 A/N/C）
        data["admlp_style_score"] = np.array(
            [featurizers.build_admlp_style_score(prefer)], dtype=np.float32
        )

        if len(fut8) >= ADMLP_NUM_POSES:
            traj, steer = featurizers.build_stage_vec_target(ego, fut8, yaw_rate)
        else:
            traj = np.zeros((ADMLP_NUM_POSES, 2), dtype=np.float32)
            steer = np.zeros((1, 2), dtype=np.float32)
        data["stage_vec_y_traj"] = traj
        data["stage_vec_y_steer"] = steer

        return data

    def save(self, out_dir, data):
        # token 全数据集唯一，直接作为文件名，便于 train 脚本按 token 寻址。
        np.savez(f"{out_dir}/{data['token']}.npz", **data)
