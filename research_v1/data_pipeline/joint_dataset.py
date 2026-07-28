"""Joint diffusion dataset with optional encoded style-memory features."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.train.train_utils import openjson, opendata


class RetrievalGuidancePlannerDataset(Dataset):
    """Return planner training tensors plus optional style-memory features."""

    def __init__(
        self,
        planner_cache_dir: str,
        data_list_path: str,
        encoded_memory_cache_dir: str = "",
        style_key_dim: int = 32,
        style_value_dim: int = 32,
        past_neighbor_num: int = 32,
        predicted_neighbor_num: int = 10,
    ) -> None:
        self.planner_cache_dir = planner_cache_dir
        self.encoded_memory_cache_dir = encoded_memory_cache_dir
        self.style_key_dim = int(style_key_dim)
        self.style_value_dim = int(style_value_dim)
        self.past_neighbor_num = int(past_neighbor_num)
        self.predicted_neighbor_num = int(predicted_neighbor_num)
        self.data_list = self._filter_existing(openjson(data_list_path))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        filename = self.data_list[idx]
        planner_path = os.path.join(self.planner_cache_dir, filename)
        planner_data = opendata(planner_path)
        encoded_data = self._open_encoded_feature(filename)

        try:
            sample: Dict[str, object] = {
                "filename": filename,
                "sample_id": os.path.splitext(filename)[0],
                "ego_current_state": self._tensor(planner_data["ego_current_state"]),
                "ego_agent_past": self._tensor(planner_data["ego_agent_past"]),
                "neighbor_agents_past": self._tensor(planner_data["neighbor_agents_past"][: self.past_neighbor_num]),
                "neighbor_agents_past_mask": self._neighbor_past_mask(planner_data),
                "neighbor_agents_future_mask": self._neighbor_future_mask(planner_data),
                "ego_future_gt": self._tensor(planner_data["ego_agent_future"]),
                "neighbors_future_gt": self._tensor(planner_data["neighbor_agents_future"][: self.predicted_neighbor_num]),
                "lanes": self._tensor(planner_data["lanes"]),
                "lanes_speed_limit": self._tensor(planner_data["lanes_speed_limit"]),
                "lanes_has_speed_limit": self._bool_tensor(planner_data["lanes_has_speed_limit"]),
                "route_lanes": self._tensor(planner_data["route_lanes"]),
                "route_lanes_speed_limit": self._tensor(planner_data["route_lanes_speed_limit"]),
                "route_lanes_has_speed_limit": self._bool_tensor(planner_data["route_lanes_has_speed_limit"]),
                "static_objects": self._tensor(planner_data["static_objects"]),
                "lanes_mask": self._lane_mask(planner_data, "lanes_mask", "lanes"),
                "route_lanes_mask": self._lane_mask(planner_data, "route_lanes_mask", "route_lanes"),
            }
            sample.update(self._encoded_feature_payload(encoded_data))
            return sample
        finally:
            if encoded_data is not None:
                encoded_data.close()
            planner_data.close()

    def _filter_existing(self, file_list):
        return [filename for filename in file_list if os.path.exists(os.path.join(self.planner_cache_dir, filename))]

    def _open_encoded_feature(self, filename: str) -> Optional[np.lib.npyio.NpzFile]:
        if not self.encoded_memory_cache_dir:
            return None
        path = os.path.join(self.encoded_memory_cache_dir, filename)
        if not os.path.exists(path):
            return None
        return np.load(path, allow_pickle=False)

    def _encoded_feature_payload(self, encoded_data) -> Dict[str, torch.Tensor]:
        if encoded_data is None:
            return {
                "style_key_feature": torch.zeros((self.style_key_dim,), dtype=torch.float32),
                "style_value_feature": torch.zeros((self.style_value_dim,), dtype=torch.float32),
                "style_feature_valid": torch.as_tensor(False, dtype=torch.bool),
                "behavior_axis_vec": torch.zeros((4,), dtype=torch.float32),
                "style_score_vec": torch.zeros((3,), dtype=torch.float32),
                "scene_bucket_id": torch.as_tensor(-1, dtype=torch.long),
                "style_label_id": torch.as_tensor(-1, dtype=torch.long),
                "subset_id": torch.as_tensor(-1, dtype=torch.long),
            }
        return {
            "style_key_feature": self._tensor(encoded_data["key_feature"]),
            "style_value_feature": self._tensor(encoded_data["value_feature"]),
            "style_feature_valid": torch.as_tensor(True, dtype=torch.bool),
            "behavior_axis_vec": self._tensor(encoded_data["behavior_axis_vec"]),
            "style_score_vec": self._tensor(encoded_data["style_score_vec"]),
            "scene_bucket_id": self._scalar_long(encoded_data, "scene_bucket_id"),
            "style_label_id": self._scalar_long(encoded_data, "style_label_id"),
            "subset_id": self._scalar_long(encoded_data, "subset_id"),
        }

    def _tensor(self, array, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=dtype)

    def _bool_tensor(self, array) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.bool)

    def _optional_tensor(self, npz_data, key: str, dtype=torch.float32) -> torch.Tensor:
        if hasattr(npz_data, "files") and key in npz_data.files:
            return torch.as_tensor(np.asarray(npz_data[key]), dtype=dtype)
        return torch.zeros((0,), dtype=dtype)

    def _neighbor_past_mask(self, npz_data) -> torch.Tensor:
        if hasattr(npz_data, "files") and "neighbor_agents_past_mask" in npz_data.files:
            return torch.as_tensor(
                np.asarray(npz_data["neighbor_agents_past_mask"])[: self.past_neighbor_num],
                dtype=torch.bool,
            )
        neighbor_agents_past = np.asarray(npz_data["neighbor_agents_past"][: self.past_neighbor_num])
        valid = np.any(np.abs(neighbor_agents_past[..., :8]) > 0, axis=-1)
        return torch.as_tensor(~valid, dtype=torch.bool)

    def _neighbor_future_mask(self, npz_data) -> torch.Tensor:
        if hasattr(npz_data, "files") and "neighbor_agents_future_mask" in npz_data.files:
            return torch.as_tensor(
                np.asarray(npz_data["neighbor_agents_future_mask"])[: self.predicted_neighbor_num],
                dtype=torch.bool,
            )
        neighbor_agents_future = np.asarray(npz_data["neighbor_agents_future"][: self.predicted_neighbor_num])
        valid = np.any(np.abs(neighbor_agents_future[..., :3]) > 0, axis=-1)
        return torch.as_tensor(~valid, dtype=torch.bool)

    def _lane_mask(self, npz_data, mask_key: str, value_key: str) -> torch.Tensor:
        if hasattr(npz_data, "files") and mask_key in npz_data.files:
            return torch.as_tensor(np.asarray(npz_data[mask_key]), dtype=torch.bool)
        lane_values = np.asarray(npz_data[value_key])
        lane_feature = np.abs(lane_values[..., :4]) > 0
        if lane_feature.ndim <= 1:
            valid = lane_feature.astype(bool)
        elif lane_feature.ndim == 2:
            valid = np.any(lane_feature, axis=-1)
        else:
            valid = np.any(lane_feature, axis=tuple(range(1, lane_feature.ndim)))
        return torch.as_tensor(~valid, dtype=torch.bool)

    def _scalar_long(self, npz_data, key: str) -> torch.Tensor:
        value = np.asarray(npz_data[key]).reshape(-1)[0]
        return torch.as_tensor(int(value), dtype=torch.long)
