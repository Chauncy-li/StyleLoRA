"""Dataset and balanced sampler for CSPQ preference encoder training.

CSPQ 偏好编码器的数据集与场景平衡采样器：
- ``PreferenceEncoderDataset``：加载弱偏好 manifest（q/c/valid_mask/rank）+ 冻结 h_c + 专家轨迹，
  过滤全无效样本，按稳定键对齐特征与偏好行；
- ``SceneBalancedBatchSampler``：每个 epoch 使用全部 free-drive，随机下采样等量 car-follow，
  batch 内两场景严格 1:1（不重复扩增少数场景）。

说明：h_c 是预计算输入（冻结 DiffPlanner 产物），只作为常量特征参与前向，不反传到 DiffPlanner。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from research_lora_2.data.schema import PreferenceSample

# 场景编号映射（供数据集返回稳定脚本 id）
SCENE_IDS = {"straight_free_drive": 0, "straight_car_follow": 1}


def _stable_key(row: dict) -> str:
    """稳定键：优先 (log_name, token)，其次 cache_path（与 schema.key 语义一致）。"""
    log_name, token = str(row.get("log_name", "")), str(row.get("token", ""))
    if log_name or token:
        return f"{log_name}:{token}"
    return str(row.get("cache_path", ""))


class PreferenceEncoderDataset(Dataset):
    """加载弱偏好样本 + 冻结 h_c + 专家轨迹，返回编码器训练所需字段。"""

    def __init__(self, manifest_path: str | Path, feature_npy: str | Path,
                 feature_index_path: str | Path, cache_root: str | Path,
                 *, dt: float = 0.1) -> None:
        """初始化。

        Args:
            manifest_path: 弱偏好 manifest JSONL（PreferenceSample 序列化格式）。
            feature_npy: 冻结 h_c 特征 npy（行序与 feature_index 对齐）。
            feature_index_path: 特征索引 JSONL（fid/token/cache_path 等）。
            cache_root: 缓存根目录（相对 cache_path 解析）。
            dt: 轨迹 token 时间步长（未启用，保留接口一致性）。
        """
        self.manifest_path = Path(manifest_path)
        self.feature_npy = Path(feature_npy)
        self.feature_index = Path(feature_index_path)
        self.cache_root = Path(cache_root)

        # 1) 读偏好 manifest，过滤全无效样本（无任何有效轴 -> 三轴无效、rank=0.5 占位）
        self.samples: List[PreferenceSample] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = PreferenceSample.from_mapping(json.loads(line))
                if sample.valid_axes == 0:
                    continue  # 全无效样本不参与编码器训练
                self.samples.append(sample)

        # 2) 读特征索引，建立 稳定键 -> fid 映射；校验重复 key 与 fid 越界
        key_to_fid: Dict[str, int] = {}
        with self.feature_index.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = _stable_key(row)
                fid = int(row["fid"])
                if key in key_to_fid:
                    raise ValueError(f"Feature index has duplicate key {key!r}; refusing silent aliasing")
                key_to_fid[key] = fid
        self._features: np.ndarray = np.load(self.feature_npy, allow_pickle=False)
        total_features = self._features.shape[0]
        bad_fids = [fid for fid in key_to_fid.values() if fid < 0 or fid >= total_features]
        if bad_fids:
            raise ValueError(f"Feature index contains {len(bad_fids)} fids out of range [0, {total_features})")

        # 3) 逐样本解析 -> (fid, 稳定键)；特征缺失的样本从 self.samples 中剔除，
        #    保证 self.samples / self.fids / self.keys 三者在同一索引下严格对应。
        self.fids: List[np.int64] = []
        self.keys: List[str] = []
        filtered_samples: List[PreferenceSample] = []
        self.missing = 0
        for sample in self.samples:
            key = _stable_key({
                "log_name": sample.log_name,
                "token": sample.token,
                "cache_path": sample.cache_path,
            })
            fid = key_to_fid.get(key)
            if fid is None:
                self.missing += 1
                continue
            filtered_samples.append(sample)
            self.fids.append(fid)
            self.keys.append(key)
        self.samples = filtered_samples  # 同步压缩，与 fids 索引一一对应
        # 统一为 numpy 数组便于索引
        self.fids = np.asarray(self.fids, dtype=np.int64)

        # 每场景样本下标（此时 samples/fids 已对齐，索引即自身下标）
        self.free_indices = [i for i, s in enumerate(self.samples)
                             if s.scene_type == "straight_free_drive"]
        self.car_indices = [i for i, s in enumerate(self.samples)
                            if s.scene_type == "straight_car_follow"]

    @staticmethod
    def _load_ego_future(cache_path: str, root: Path) -> torch.Tensor:
        """从缓存 npz 加载自车未来真值（物理坐标），返回 [T, 4] 张量。"""
        path = Path(cache_path)
        if not path.is_absolute():
            path = root / path
        with np.load(path, allow_pickle=False) as cache:
            key = "ego_future_gt" if "ego_future_gt" in cache.files else "ego_agent_future"
            return torch.as_tensor(cache[key]).float()

    def __len__(self) -> int:
        return int(self.fids.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        hc = torch.as_tensor(self._features[self.fids[index]]).float()
        trajectory = self._load_ego_future(sample.cache_path, self.cache_root)
        # 轨迹 token：pos + 增量 + cos/sin（heading 为第 3 维弧度）
        pos = trajectory[:, :2]
        delta = torch.zeros_like(pos)
        if pos.shape[0] > 1:
            delta[1:] = pos[1:] - pos[:-1]
        heading = trajectory[:, 2]
        cos_sin = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
        tokens = torch.cat((pos, delta, cos_sin), dim=-1)  # [T, 6]
        # 偏好监督
        q_vec = torch.as_tensor([float(q) for q in sample.axis_percentiles], dtype=torch.float32)
        valid_mask = torch.as_tensor([bool(v) for v in sample.axis_valid], dtype=torch.bool)
        rank = torch.tensor(float(sample.preference_rank), dtype=torch.float32)
        confidence = torch.tensor(float(sample.rank_confidence), dtype=torch.float32)
        scene_id = torch.tensor(SCENE_IDS[sample.scene_type], dtype=torch.long)
        return {
            "trajectory": tokens,
            "h_c": hc,
            "q_vec": q_vec,
            "valid_mask": valid_mask,
            "rank": rank,
            "confidence": confidence,
            "scene_id": scene_id,
            "key": sample.key,
        }


def encoder_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """把样本堆叠为批；''key'' 保留为字符串列表，其余堆叠。"""
    keys = [item["key"] for item in batch]
    collated = {k: torch.stack([item[k] for item in batch]) for k in batch[0] if k != "key"}
    collated["key"] = keys
    return collated


class SceneBalancedBatchSampler(Sampler):
    """每批两场景各 50%：free-drive 全量，car-follow 随机下采样等量。"""

    def __init__(self, dataset: PreferenceEncoderDataset, batch_size: int,
                 generator: torch.Generator | None = None) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.generator = generator or torch.Generator().manual_seed(0)
        # 每批样本数 = 2 * 少数场景数量下限，且为偶数可均分
        per_scene = min(len(dataset.free_indices), len(dataset.car_indices))
        self.per_scene = per_scene
        if self.batch_size % 2 != 0 or self.batch_size <= 0:
            raise ValueError("batch_size 必须为偶数且为正（每场景各半）")
        self.samples_per_scene_per_batch = self.batch_size // 2

    def __iter__(self) -> Iterator[List[int]]:
        """每 epoch：free 全量 + car 随机等量；每 batch 两场景严格各半且样本不重复。

        每个 epoch 开头分别打乱一次 free/car 下标数组，然后按 half 连续切片；
        同一 epoch 内两场景的样本都不重复（各自被使用约一次），不同 epoch 重新打乱。
        """
        free_all = np.asarray(self.dataset.free_indices, dtype=np.int64)
        car_all = np.asarray(self.dataset.car_indices, dtype=np.int64)
        n = min(free_all.shape[0], car_all.shape[0])
        half = self.samples_per_scene_per_batch
        n_batches = n // half
        if n_batches == 0:
            return
        # 每 epoch 打乱一次：free 全量参与 + car 等量下采样；按 half 切片不再重复抽取
        free_shuffled = torch.randperm(free_all.shape[0], generator=self.generator).numpy()[: n_batches * half]
        car_shuffled = torch.randperm(car_all.shape[0], generator=self.generator).numpy()[: n_batches * half]
        free_sel = free_all[free_shuffled]
        car_sel = car_all[car_shuffled]
        for b in range(n_batches):
            f = free_sel[b * half:(b + 1) * half]
            c = car_sel[b * half:(b + 1) * half]
            yield [int(i) for i in np.concatenate((f, c))]

    def __len__(self) -> int:
        n = min(len(self.dataset.free_indices), len(self.dataset.car_indices))
        return max(0, n // self.samples_per_scene_per_batch)
