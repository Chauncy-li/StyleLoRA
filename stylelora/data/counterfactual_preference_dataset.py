"""同场景反事实轨迹偏好对数据集。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from stylelora.data.preference_lora_dataset import (
    PreferenceLoRADataset,
    preference_lora_collate,
)


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CounterfactualPreferenceDataset(Dataset):
    """把可行 style 轨迹对与原始场景缓存严格按 key 对齐。"""

    def __init__(
        self,
        *,
        pair_bank: str | Path,
        pair_index: str | Path,
        manifest: str | Path,
        cache_root: str | Path,
        latent_bank: str | Path,
        latent_bank_index: str | Path,
        feature_npy: str | Path,
        feature_index: str | Path,
    ) -> None:
        self.base = PreferenceLoRADataset(
            manifest,
            cache_root,
            latent_bank,
            latent_bank_index,
            feature_npy,
            feature_index,
            direction="counterfactual",
            rank_low=0.0,
            rank_high=1.0,
        )
        base_by_key = {sample.key: index for index, sample in enumerate(self.base.samples)}
        if len(base_by_key) != len(self.base):
            raise ValueError("原始偏好数据存在重复 key，无法安全对齐反事实偏好对")

        raw_rows = _read_jsonl(pair_index)
        if not raw_rows:
            raise ValueError("反事实偏好对索引为空")
        self.rows: list[dict] = []
        self.base_rows: list[int] = []
        seen_pair_rows: set[int] = set()
        for row in raw_rows:
            # 兼容读取旧索引时只保留 style 对，constraint 对绝不进入 LoRA 训练。
            if str(row.get("pair_kind", "style")) != "style":
                continue
            key = str(row["key"])
            base_row = base_by_key.get(key)
            if base_row is None:
                continue
            pair_row = int(row["pair_row"])
            if pair_row in seen_pair_rows:
                raise ValueError(f"反事实 pair_row 重复：{pair_row}")
            seen_pair_rows.add(pair_row)
            self.rows.append(row)
            self.base_rows.append(base_row)
        if not self.rows:
            raise ValueError("反事实偏好对与原始数据没有可对齐样本")

        shape = tuple(int(value) for value in self.rows[0]["trajectory_shape"])
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"trajectory_shape 必须为 [T,D]，实际为 {shape}")
        if any(tuple(int(v) for v in row["trajectory_shape"]) != shape for row in self.rows):
            raise ValueError("反事实轨迹形状不一致")
        pair_count = max(int(row["pair_row"]) for row in raw_rows) + 1
        expected_bytes = pair_count * 2 * shape[0] * shape[1] * np.dtype(np.float32).itemsize
        bank_path = Path(pair_bank)
        if bank_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"反事实轨迹库大小不匹配：expected={expected_bytes}, actual={bank_path.stat().st_size}"
            )
        self._pairs = np.memmap(
            bank_path,
            mode="r",
            dtype=np.float32,
            shape=(pair_count, 2, shape[0], shape[1]),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        item = self.base[self.base_rows[index]]
        pair = np.array(self._pairs[int(row["pair_row"])], copy=True)
        item.update({
            "preferred_ego": torch.from_numpy(pair[0]),
            "rejected_ego": torch.from_numpy(pair[1]),
            "requested_rho": torch.tensor(float(row["requested_rho"]), dtype=torch.float32),
            "pair_confidence": torch.tensor(float(row["pair_confidence"]), dtype=torch.float32),
            "direction": str(row["direction"]),
        })
        return item


def counterfactual_preference_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """沿用现有场景张量契约，仅增加偏好胜负轨迹字段。"""
    base = preference_lora_collate(batch)
    for key in (
        "preferred_ego",
        "rejected_ego",
        "requested_rho",
        "pair_confidence",
    ):
        base[key] = torch.stack([item[key] for item in batch])
    base["direction"] = [str(item["direction"]) for item in batch]
    return base
