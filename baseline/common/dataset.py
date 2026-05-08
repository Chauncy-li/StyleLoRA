"""
Module: Dataset Loader

此模块定义了用于训练和验证的 PyTorch Dataset 类。
它负责读取预处理好的 .npz 文件，并将其解包为模型所需的张量格式。

关键特性：
- 确定性的数据返回顺序 (这对 DataLoader 的 collate_fn 至关重要)
- 动态截取邻居车辆数量 (基于配置)
- 同时返回从 DataProcessor 生成的 Mask 和扩充特征
"""

import os
from torch.utils.data import Dataset

from baseline.utils.io import openjson, opendata


# ==============================================================================
# *** Closed Loop Planner Dataset
# ==============================================================================

class ClosedLoopPlannerData(Dataset):
    """
    闭环仿真规划器数据集类。
    读取由 DataProcessor.work() 生成的 .npz 文件。

    Args:
        data_dir (str): 数据文件(.npz)所在的根目录路径
        data_list (str): 包含训练/验证文件名的 JSON 列表文件路径
        past_neighbor_num (int): 邻居车辆历史轨迹的最大保留数量 (对应 neighbor_agents_past)
        predicted_neighbor_num (int): 邻居车辆未来轨迹的最大预测数量 (对应 neighbors_future_gt)
        future_len (int): 自车未来轨迹的长度 (时间步数)
    """
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        注意：返回值是一个元组 (Tuple)，其顺序严格依赖于内部 `data` 字典的插入顺序。
        训练循环 (train_utils.py) 将通过索引访问这些元素，因此请勿随意更改顺序。

        Batch Data Mapping (Tuple Index -> Content):
            0.  ego_current_state:            [10] 自车当前扩展状态
            1.  ego_future_gt:                [T, 3] 自车未来轨迹真值 (Ground Truth)
            2.  neighbor_agents_past:         [N_past, T_past, 11] 邻居历史轨迹
            3.  neighbors_future_gt:          [N_pred, T_fut, 3] 邻居未来轨迹真值
            4.  lanes:                        [L, V, D] 车道线几何点集
            5.  lanes_speed_limit:            [L, 1] 车道限速
            6.  lanes_has_speed_limit:        [L, 1] 车道是否有限速掩码
            7.  route_lanes:                  [R, V, D] 导航路径几何
            8.  route_lanes_speed_limit:      [R, 1] 导航路径限速
            9.  route_lanes_has_speed_limit:  [R, 1] 导航路径是否有限速
            10. static_objects:               [S, D] 静态障碍物特征
            11. ego_agent_past:               [T_past, 7] 自车历史轨迹 (用于 Encoder)
            12. neighbor_agents_past_mask:    [N_past, T_past] 邻居历史有效性 Mask
            13. neighbor_agents_future_mask:  [N_pred, T_fut] 邻居未来有效性 Mask (用于 Loss 计算)
            14. lanes_mask:                   [L] 车道线有效性 Mask
            15. route_lanes_mask:             [R] 导航路径有效性 Mask

        Returns:
            tuple: 包含上述 16 个元素的元组。
        """

        # Load Data from Disk
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))

        # Extract & Slice Features
        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']

        # 截取固定数量的邻居 (Top-K)
        # 注意：DataProcessor 阶段已经排好序了，这里只需切片
        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        # Map Features
        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']

        # Extra Features & Masks
        ego_agent_past = data['ego_agent_past']
        neighbor_agents_past_mask = data['neighbor_agents_past_mask'][:self._past_neighbor_num]  # 同样需要切片
        neighbor_agents_future_mask = data['neighbor_agents_future_mask'][:self._predicted_neighbor_num]  # 同样需要切片

        # 兼容性处理：如果旧数据没有这些 mask，可能需要做 try-except 或者重新生成数据
        # (用于 Action Head 屏蔽无效车道)
        route_lanes_mask = data['route_lanes_mask']
        lanes_mask = data['lanes_mask']

        # --- [新增] 读取 Codebook Labels ---
        # 务必确保 .npz 里有这些 keys (由 data_processor 生成)
        # 如果是旧数据没有这些 key，给个默认值防止报错 (鲁棒性)
        code_lat = data['code_lat']
        code_lon = data['code_lon']

        # Pack into Dictionary (Order Matters!)
        # 这里的插入顺序决定了 DataLoader collate 后的 list 顺序
        data_dict = {
            "ego_current_state": ego_current_state,  # 0
            "ego_future_gt": ego_agent_future,  # 1
            "neighbor_agents_past": neighbor_agents_past,  # 2
            "neighbors_future_gt": neighbor_agents_future,  # 3
            "lanes": lanes,  # 4
            "lanes_speed_limit": lanes_speed_limit,  # 5
            "lanes_has_speed_limit": lanes_has_speed_limit,  # 6
            "route_lanes": route_lanes,  # 7
            "route_lanes_speed_limit": route_lanes_speed_limit,  # 8
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,  # 9
            "static_objects": static_objects,  # 10

            # 新增字段 (Alpha/Diffusion Planner 必需)
            "ego_agent_past": ego_agent_past,  # 11
            "neighbor_agents_past_mask": neighbor_agents_past_mask,  # 12
            "neighbor_agents_future_mask": neighbor_agents_future_mask,  # 13
            'lanes_mask': lanes_mask,  # 14
            'route_lanes_mask': route_lanes_mask,  # 15

            # --- [新增] 放入 Dictionary (顺序追加) ---
            "code_lat": code_lat,  # 16 (Long/Int64)
            "code_lon": code_lon,  # 17 (Long/Int64)
        }

        # Return as tuple values
        return tuple(data_dict.values())
