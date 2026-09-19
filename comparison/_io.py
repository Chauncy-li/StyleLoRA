"""comparison 脚本共用的数据 IO 工具。

- load_splits: 读取 splits.json
- load_fields: 从 <npz_dir>/<token>.npz 加载指定字段并堆叠为 float32 数组
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def load_splits(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def config_value(key, default=None):
    """从 stylelora/config/paths.local.json 读一个值；缺省返回 default。"""
    try:
        from stylelora.config.runtime_paths import get_config_value

        value = get_config_value(key, None)
        return value if value else default
    except Exception:
        return default


def load_fields(npz_dir: str, tokens, fields) -> dict:
    """逐 token 加载 .npz 并提取 fields，返回 {field: np.ndarray(float32)}。

    :param npz_dir: 存放 <token>.npz 的目录
    :param tokens: split 的 token 列表
    :param fields: 需要读取的字段名
    """
    arrays = {f: [] for f in fields}
    missing = 0
    for tok in tokens:
        p = os.path.join(npz_dir, f"{tok}.npz")
        if not os.path.exists(p):
            missing += 1
            continue
        with np.load(p, allow_pickle=True) as d:
            for f in fields:
                arrays[f].append(np.asarray(d[f]))
    if missing:
        print(f"[io] {missing}/{len(tokens)} npz 缺失（已跳过）")
    out = {}
    for f in fields:
        if arrays[f]:
            out[f] = np.stack(arrays[f]).astype(np.float32)
        else:
            out[f] = np.zeros((0,), dtype=np.float32)
    return out
