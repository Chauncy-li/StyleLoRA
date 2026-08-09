"""Continuous-preference LoRA dataset: rank-window sampling + 1:1 scene balance.

从弱偏好 manifest + 冻结 h_c + latent bank 构建 LoRA 训练数据：
- 按 direction 对应的 rank 区间取样本（high 默认 [0.8, 1.0]，low 默认 [0, 0.2]）；
- 每个样本返回 DiffPlanner cache 张量 + 冻结 CSPQ 输入（trajectory/h_c）；
- target_z 从 latent bank 按稳定键对齐拉取；rank/confidence 来自偏好 manifest；
- 两场景严格 1:1 采样。

不再使用 aggr/norm/cons 标签；rank 区间同一套定义覆盖两个场景。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from stylelora.data.encoder_dataset import SCENE_IDS, _stable_key
from stylelora.data.schema import PreferenceSample


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class PreferenceLoRADataset(Dataset):
    """按 rank 区间取样的 LoRA 数据集（连续偏好无规则标签）。"""

    def __init__(self, manifest_path: str | Path, cache_root: str | Path,
                 latent_bank_npy: str | Path, latent_bank_index: str | Path,
                 feature_npy: str | Path, feature_index: str | Path,
                 *, direction: str = "high", rank_low: float = 0.8, rank_high: float = 1.0,
                 predicted_neighbor_num: int = 10) -> None:
        self.cache_root = Path(cache_root)
        self.predicted_neighbor_num = int(predicted_neighbor_num)

        # 1) latent bank：稳定键 -> 行（真实 index 只存 "key": "log:token"，优先取它）
        self._latent = np.load(latent_bank_npy, allow_pickle=False)
        n_latent = self._latent.shape[0]
        key_to_bank: Dict[str, int] = {}
        for row in _iter_jsonl(Path(latent_bank_index)):
            key = row.get("key") or _stable_key(row)
            if key in key_to_bank:
                raise ValueError(f"Latent index duplicate key {key!r}")
            fid = int(row["fid"])
            if fid < 0 or fid >= n_latent:
                raise ValueError(f"Latent fid {fid} out of range [0,{n_latent})")
            key_to_bank[key] = fid
        self._latent_index = key_to_bank

        # 2) h_c 特征
        self._features = np.load(feature_npy, allow_pickle=False)
        n_feat = self._features.shape[0]
        key_to_fid: Dict[str, int] = {}
        for row in _iter_jsonl(Path(feature_index)):
            key = row.get("key") or _stable_key(row)
            if key in key_to_fid:
                raise ValueError(f"Feature index duplicate key {key!r}")
            fid = int(row["fid"])
            if fid < 0 or fid >= n_feat:
                raise ValueError(f"Feature fid {fid} out of range [0,{n_feat})")
            key_to_fid[key] = fid
        self._fid_index = key_to_fid

        # 3) 偏好样本按 rank 区间过滤 + 拉取 latent/h_c fid；缺失则剔除并计数
        self.samples: List[PreferenceSample] = []
        self.latent_rows: List[int] = []
        self.fids: List[int] = []
        self.missing = 0
        for row in _iter_jsonl(Path(manifest_path)):
            sample = PreferenceSample.from_mapping(row)
            if sample.valid_axes == 0:
                continue
            if not (float(rank_low) <= sample.preference_rank <= float(rank_high)):
                continue
            key = _stable_key({
                "log_name": sample.log_name,
                "token": sample.token,
                "cache_path": sample.cache_path,
            })
            bank_row = key_to_bank.get(key)
            fid = key_to_fid.get(key)
            if bank_row is None or fid is None:
                self.missing += 1
                continue
            self.samples.append(sample)
            self.latent_rows.append(bank_row)
            self.fids.append(fid)

        self.free_indices = [i for i, s in enumerate(self.samples)
                             if s.scene_type == "straight_free_drive"]
        self.car_indices = [i for i, s in enumerate(self.samples)
                            if s.scene_type == "straight_car_follow"]
        self.direction = direction

    def __len__(self) -> int:
        return len(self.samples)

    def _load_cache_tensors(self, cache_path: str) -> Dict[str, torch.Tensor]:
        """加载 DiffPlanner 缓存张量（与 StyleManifestDataset 相同字段契约）。"""
        from stylelora.lora.data.dataset import BASELINE_TENSOR_KEYS
        path = Path(cache_path)
        if not path.is_absolute():
            path = self.cache_root / path
        with np.load(path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        item = {key: torch.as_tensor(arrays[key]) for key in BASELINE_TENSOR_KEYS if key in arrays}
        aliases = {"ego_future_gt": ("ego_future_gt", "ego_agent_future"),
                   "neighbors_future_gt": ("neighbors_future_gt", "neighbor_agents_future")}
        for destination, candidates in aliases.items():
            for source in candidates:
                if source in arrays:
                    item[destination] = torch.as_tensor(arrays[source])
                    break
            else:
                raise KeyError(f"Cache {path} missing {candidates} for {destination}")
        for key in ("neighbors_future_gt", "neighbor_agents_future_mask"):
            if key in item:
                item[key] = item[key][:self.predicted_neighbor_num]
        return item

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        tensors = self._load_cache_tensors(sample.cache_path)
        trajectory = tensors["ego_future_gt"].float()
        pos = trajectory[:, :2]
        delta = torch.zeros_like(pos)
        if pos.shape[0] > 1:
            delta[1:] = pos[1:] - pos[:-1]
        heading = trajectory[:, 2]
        cos_sin = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
        tokens = torch.cat((pos, delta, cos_sin), dim=-1)  # [T, 6]
        hc = torch.as_tensor(self._features[self.fids[index]]).float()
        z_target = torch.as_tensor(self._latent[self.latent_rows[index]]).float()
        # CSPQ 三因子对齐监督：axis_percentiles[3]（0~1 或 NaN）+ axis_valid[3]
        # valid_mask 必须同时检查有限值，避免 manifest 中"标记有效但数值 NaN"污染 loss。
        q_vec = torch.as_tensor([float(q) for q in sample.axis_percentiles], dtype=torch.float32)
        axis_valid = torch.as_tensor([bool(v) for v in sample.axis_valid], dtype=torch.bool)
        valid_mask = axis_valid & torch.isfinite(q_vec)
        q_vec = torch.nan_to_num(q_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "tensors": tensors,
            "trajectory": tokens,
            "h_c": hc,
            "z_target": z_target,
            "rank": torch.tensor(sample.preference_rank, dtype=torch.float32),
            "confidence": torch.tensor(sample.rank_confidence, dtype=torch.float32),
            "q_vec": q_vec,
            "valid_mask": valid_mask,
            "scene_id": torch.tensor(SCENE_IDS[sample.scene_type], dtype=torch.long),
            "key": sample.key,
        }


def preference_lora_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate：tensors 子字典单独堆叠，其余统一堆叠；key 保留为列表。"""
    keys = [item["key"] for item in batch]
    collated = {}
    for key in ("trajectory", "h_c", "z_target", "rank", "confidence", "q_vec", "valid_mask", "scene_id"):
        collated[key] = torch.stack([item[key] for item in batch])
    collated["tensors"] = {k: torch.stack([item["tensors"][k] for item in batch]) for k in batch[0]["tensors"]}
    collated["key"] = keys
    return collated


class SceneBalancedLoRASampler(Sampler):
    """每批两场景严格各半，sample with replacement（LoRA 取窗口数据较少）。"""

    def __init__(self, dataset: PreferenceLoRADataset, batch_size: int,
                 generator: torch.Generator | None = None) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.generator = generator or torch.Generator().manual_seed(0)
        if self.batch_size % 2 != 0 or self.batch_size <= 0:
            raise ValueError("batch_size 必须为正偶数（每场景各半）")
        self.half = self.batch_size // 2

    def __iter__(self) -> Iterator[List[int]]:
        free = np.asarray(self.dataset.free_indices, dtype=np.int64)
        car = np.asarray(self.dataset.car_indices, dtype=np.int64)
        n = min(free.shape[0], car.shape[0])
        if n == 0:
            return
        n_batches = max(1, n // self.half)
        for _ in range(n_batches):
            f = free[torch.randperm(free.shape[0], generator=self.generator)[:self.half].numpy()]
            c = car[torch.randperm(car.shape[0], generator=self.generator)[:self.half].numpy()]
            yield [int(i) for i in np.concatenate((f, c))]

    def __len__(self) -> int:
        n = min(len(self.dataset.free_indices), len(self.dataset.car_indices))
        return max(1, n // self.half)

