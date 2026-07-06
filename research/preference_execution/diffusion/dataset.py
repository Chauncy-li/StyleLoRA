"""Dataset adapters for preference-conditioned diffusion training."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.utils.io import opendata
from research.preference_execution.conditioning.schema import conditioning_index_path, conditioning_output_dir


def _resolve_index_path(split_root: str, conditioning_index_override: str | None = None) -> str:
    if conditioning_index_override:
        return conditioning_index_override
    return conditioning_index_path(conditioning_output_dir(split_root))


def _tensor_from_array(data: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(data), dtype=dtype)


class PreferenceConditionedPlannerData(Dataset):
    """Read planner cache samples and align them with exported conditioning features."""

    def __init__(
        self,
        cache_dir: str,
        split_root: str,
        *,
        condition_field: str = "effective_preference_global_vec",
        conditioning_index_override: str | None = None,
        start_index: int = 0,
        num_samples: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.split_root = split_root
        self.condition_field = condition_field
        self.conditioning_index_path = _resolve_index_path(split_root, conditioning_index_override)
        if not os.path.exists(self.conditioning_index_path):
            raise FileNotFoundError(f"conditioning index not found: {self.conditioning_index_path}")

        records = self._load_records(self.conditioning_index_path, condition_field)
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        end_index = len(records) if num_samples is None else min(len(records), start_index + int(num_samples))
        self.records = records[start_index:end_index]

    @staticmethod
    def _load_records(index_path: str, condition_field: str) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        with open(index_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if condition_field not in raw:
                    raise KeyError(
                        f"Condition field {condition_field!r} not found in conditioning record. "
                        f"Available keys include: {sorted(raw.keys())[:20]}"
                    )
                style_vec = [float(value) for value in raw[condition_field]]
                records.append(
                    {
                        "sample_id": str(raw.get("sample_id", "")),
                        "filename": str(raw.get("filename", "")),
                        "scene_bucket": str(raw.get("scene_bucket", "none")),
                        "style_label": str(raw.get("style_label", "unknown")),
                        "target_style_label": str(raw.get("target_style_label", "unknown")),
                        "style_value_condition": style_vec,
                        "scene_gate_values": [float(value) for value in raw.get("scene_gate_values", [])],
                        "axis_gate_values": [float(value) for value in raw.get("axis_gate_values", [])],
                        "local_axis_gate_values": [float(value) for value in raw.get("local_axis_gate_values", [])],
                        "target_preference_scene_vec": [
                            float(value) for value in raw.get("target_preference_scene_vec", [])
                        ],
                        "safe_preference_scene_vec": [
                            float(value) for value in raw.get("safe_preference_scene_vec", [])
                        ],
                        "effective_preference_scene_vec": [
                            float(value) for value in raw.get("effective_preference_scene_vec", [])
                        ],
                    }
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        filename = str(record["filename"])
        data_path = filename if os.path.isabs(filename) else os.path.join(self.cache_dir, filename)
        npz_data = opendata(data_path)

        style_value_condition = np.asarray(record["style_value_condition"], dtype=np.float32)
        style_feature_valid = bool(np.any(np.abs(style_value_condition) > 1e-6))

        sample = {
            "sample_id": str(record["sample_id"]),
            "filename": filename,
            "scene_bucket": str(record["scene_bucket"]),
            "style_label": str(record["style_label"]),
            "target_style_label": str(record["target_style_label"]),
            "ego_current_state": _tensor_from_array(npz_data["ego_current_state"]),
            "ego_future_gt": _tensor_from_array(npz_data["ego_agent_future"]),
            "neighbor_agents_past": _tensor_from_array(npz_data["neighbor_agents_past"]),
            "neighbors_future_gt": _tensor_from_array(npz_data["neighbor_agents_future"]),
            "lanes": _tensor_from_array(npz_data["lanes"]),
            "lanes_speed_limit": _tensor_from_array(npz_data["lanes_speed_limit"]),
            "lanes_has_speed_limit": _tensor_from_array(npz_data["lanes_has_speed_limit"], dtype=torch.bool),
            "route_lanes": _tensor_from_array(npz_data["route_lanes"]),
            "route_lanes_speed_limit": _tensor_from_array(npz_data["route_lanes_speed_limit"]),
            "route_lanes_has_speed_limit": _tensor_from_array(npz_data["route_lanes_has_speed_limit"], dtype=torch.bool),
            "static_objects": _tensor_from_array(npz_data["static_objects"]),
            "ego_agent_past": _tensor_from_array(npz_data["ego_agent_past"]),
            "neighbor_agents_past_mask": _tensor_from_array(npz_data["neighbor_agents_past_mask"], dtype=torch.bool),
            "neighbor_agents_future_mask": _tensor_from_array(npz_data["neighbor_agents_future_mask"], dtype=torch.bool),
            "lanes_mask": _tensor_from_array(npz_data["lanes_mask"], dtype=torch.bool),
            "route_lanes_mask": _tensor_from_array(npz_data["route_lanes_mask"], dtype=torch.bool),
            "style_value_condition": torch.as_tensor(style_value_condition, dtype=torch.float32),
            "style_feature_valid": torch.as_tensor(style_feature_valid, dtype=torch.bool),
            "scene_gate_values": torch.as_tensor(record["scene_gate_values"], dtype=torch.float32),
            "axis_gate_values": torch.as_tensor(record["axis_gate_values"], dtype=torch.float32),
            "local_axis_gate_values": torch.as_tensor(record["local_axis_gate_values"], dtype=torch.float32),
            "target_preference_scene_vec": torch.as_tensor(
                record["target_preference_scene_vec"],
                dtype=torch.float32,
            ),
            "safe_preference_scene_vec": torch.as_tensor(
                record["safe_preference_scene_vec"],
                dtype=torch.float32,
            ),
            "effective_preference_scene_vec": torch.as_tensor(
                record["effective_preference_scene_vec"],
                dtype=torch.float32,
            ),
        }
        return sample

