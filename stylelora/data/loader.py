"""Cache-only dataset that reads raw cache rows without requiring style labels.

不依赖规则风格标签的轻量缓存读取器（问题 5 修复）：
- 逐行读取 manifest JSONL，只提取每行的原文字段（cache_path / scene_type / log_name / token ...），
  不做 StyleSample 规范化，因此不会因"缺少 style/无法推断风格"而拒绝任何轨迹；
- 按 cache_path 加载 .npz 缓存张量（字段名保持与 DiffPlanner 一致，可直接喂模型）；
- 输出格式与 research_lora 的 style_collate 兼容：{"tensors": {...}, "metadata": [原生行字段...]}，
  其中 metadata 保留 cache_path 等完整字段便于溯源。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from stylelora.lora.data.dataset import BASELINE_TENSOR_KEYS


class CacheOnlyDataset(Dataset):
    """只读缓存数据集：不要求 split/style 字段，混合/模糊/未分类轨迹均可进入。"""

    def __init__(self, manifest_path: str | Path, *, root: str | Path | None = None,
                 predicted_neighbor_num: int = 10) -> None:
        """初始化。

        Args:
            manifest_path: manifest JSONL 路径（每行一个原始字典）。
            root: 缓存根目录；None 时取 manifest 所在目录。
            predicted_neighbor_num: 需要预测的未来邻居数量（截断 future 字段）。
        """
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.predicted_neighbor_num = int(predicted_neighbor_num)
        # 逐行读取原始字典，不转 StyleSample（不要求 scene/style）
        self.rows: List[Dict[str, Any]] = [json.loads(line) for line in self.manifest_path.open("r", encoding="utf-8") if line.strip()]
        # 优先级字段别名：兼容 research_lora manifest 与偏好 manifest 的 cache_path 命名
        first = self.rows[0] if self.rows else {}
        self._cache_key = "cache_path" if "cache_path" in first else "filename" if "filename" in first else "path"

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_cache(self, index: int) -> Path:
        raw = self.rows[index]
        cache_path = raw.get(self._cache_key, "")
        path = Path(cache_path)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            raise FileNotFoundError(f"Cache for row {index} does not exist: {path}")
        return path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.rows[index]
        cache_path = self._resolve_cache(index)
        with np.load(cache_path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        # 加载与基线同名的张量字段（转成 torch.Tensor）
        item = {key: torch.as_tensor(arrays[key]) for key in BASELINE_TENSOR_KEYS if key in arrays}
        # 自车/邻居未来真值字段（同 StyleManifestDataset 的别名约定）
        aliases = {"ego_future_gt": ("ego_future_gt", "ego_agent_future"),
                   "neighbors_future_gt": ("neighbors_future_gt", "neighbor_agents_future")}
        for destination, candidates in aliases.items():
            for source in candidates:
                if source in arrays:
                    item[destination] = torch.as_tensor(arrays[source])
                    break
            else:
                raise KeyError(f"Cache {cache_path} is missing {candidates} required for {destination}")
        # 按预测邻居数截断未来字段（同 baseline 约定）
        for key in ("neighbors_future_gt", "neighbor_agents_future_mask"):
            if key in item:
                item[key] = item[key][:self.predicted_neighbor_num]
        # metadata 直接保留原始行字段（含 cache_path），供溯源
        return {"tensors": item, "metadata": raw}
