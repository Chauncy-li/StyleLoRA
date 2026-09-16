"""Dataset that reads existing DiffPlanner cache files without changing their format.

读取既有 DiffPlanner 缓存文件的数据集，不改变其存储格式：
- 从 manifest（jsonl）逐行读取样本元信息；
- 按 manifest 中的 cache_path 加载对应的 .npz 缓存；
- 在原有张量字段名之上只做别名映射与邻居数量截断，并附加风格元数据；
- 张量字段名保持与基线一致，确保下游可以直接喂给 DiffPlanner。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from stylelora.lora.data.schema import StyleSample

# 基线张量字段：这些键与 DiffPlanner 缓存中的名字完全一致，原样加载
BASELINE_TENSOR_KEYS = (
    "ego_current_state", "neighbor_agents_past", "lanes", "lanes_speed_limit", "lanes_has_speed_limit",
    "route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit", "static_objects",
    "lanes_mask", "route_lanes_mask", "neighbor_agents_past_mask", "neighbor_agents_future_mask",
)


class StyleManifestDataset(Dataset):
    """Loads .npz caches and adds style metadata; tensor fields keep baseline names.

    加载 .npz 缓存并附加风格元数据；张量字段保持基线命名。
    """

    def __init__(self, manifest_path: str | Path, *, root: str | Path | None = None,
                 predicted_neighbor_num: int = 10, style: str | None = None,
                 transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> None:
        """初始化数据集。

        Args:
            manifest_path: manifest 文件路径（jsonl），每行一个样本。
            root: 缓存根目录；为 None 时默认取 manifest 所在目录。
            predicted_neighbor_num: 需要预测（参与扩散 token）的未来邻居数量。
            style: 可选风格过滤（aggr/aggressive、cons/conservative、norm/normal）。
            transform: 可选张量变换函数（用于数据增强或归一化）。
        """
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.transform = transform
        self.predicted_neighbor_num = int(predicted_neighbor_num)
        # 逐行解析 manifest（跳过空行），每行转为 StyleSample
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.samples: List[StyleSample] = [StyleSample.from_mapping(json.loads(line)) for line in handle if line.strip()]
        # 可选风格过滤：先把别名归一化，只保留目标风格的样本
        if style is not None:
            canonical_style = {"aggr": "aggressive", "cons": "conservative", "norm": "normal"}.get(style, style)
            self.samples = [sample for sample in self.samples if sample.style == canonical_style]

    def __len__(self) -> int:
        """返回样本总数。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """加载第 index 个样本的缓存张量，并附加风格元数据。

        Args:
            index: 样本索引。

        Returns:
            {"tensors": 张量字典, "metadata": 风格元数据字典}。

        Raises:
            FileNotFoundError: 对应缓存文件不存在。
            KeyError: 缓存缺少必要字段（如自车/邻居未来真值）。
        """
        sample = self.samples[index]
        # 解析缓存路径：相对路径则以 root 为基准
        cache_path = Path(sample.cache_path)
        if not cache_path.is_absolute():
            cache_path = self.root / cache_path
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache for manifest row {index} does not exist: {cache_path}")
        # 读 .npz（allow_pickle=False 出于安全考虑）
        with np.load(cache_path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        # 字段别名：不同版本缓存可能用不同字段名，这里统一成规范名
        aliases = {"ego_future_gt": ("ego_future_gt", "ego_agent_future"),
                   "neighbors_future_gt": ("neighbors_future_gt", "neighbor_agents_future")}
        # 加载与基线同名的张量字段（转成 torch.Tensor）
        item = {key: torch.as_tensor(arrays[key]) for key in BASELINE_TENSOR_KEYS if key in arrays}
        # 按别名解析自车/邻居未来真值：任一候选字段存在即采用
        for destination, candidates in aliases.items():
            for source in candidates:
                if source in arrays:
                    item[destination] = torch.as_tensor(arrays[source])
                    break
            else:
                raise KeyError(f"Cache {cache_path} is missing {candidates} required for {destination}")
        # DiffPlanner encodes up to agent_num historical neighbours, while only
        # predicted_neighbor_num future targets belong to the diffusion tokens.
        # DiffPlanner 编码的历史邻居最多为 agent_num 个，而属于扩散 token 的
        # 未来目标只有 predicted_neighbor_num 个，因此按预测数量截断未来字段。
        for key in ("neighbors_future_gt", "neighbor_agents_future_mask"):
            if key in item:
                item[key] = item[key][:self.predicted_neighbor_num]
        # 风格元数据：来自 manifest 行的样式/场景/拆分/日志/令牌/风格度量与掩码
        metadata = {
            "style": sample.style, "scene_type": sample.scene_type, "split": sample.split,
            "log_name": sample.log_name, "token": sample.token,
            "style_metrics": sample.metrics, "style_metric_mask": sample.metric_mask,
        }
        # 应用可选变换（如归一化）
        item = self.transform(item) if self.transform else item
        return {"tensors": item, "metadata": metadata}


def style_collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate only homogeneous tensors; keep variable scene metadata as a list.

    只对同构的张量做默认 collate（堆成批），而把长度可变的场景元数据
    （场景/风格/令牌等）保留为列表。

    Args:
        samples: 单个样本字典（"tensors" + "metadata"）的序列。

    Returns:
        {"tensors": 堆叠后的批张量, "metadata": 元数据列表}。

    Raises:
        ValueError: 空批次，或批内样本的张量键集合不一致。
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = set(samples[0]["tensors"])
    if any(set(sample["tensors"]) != tensor_keys for sample in samples):
        raise ValueError("Cache tensor keys differ inside one batch")
    return {"tensors": default_collate([sample["tensors"] for sample in samples]),
            "metadata": [sample["metadata"] for sample in samples]}
