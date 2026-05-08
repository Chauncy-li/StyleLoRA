"""
I/O 通用工具模块。

职责：
1. 读取 JSON 配置或列表文件；
2. 读取 .npz 数据文件（支持 mmengine 的 fileio 后端）。

设计说明：
- 该模块从训练脚本中抽离，避免 `common` / `utils` 反向依赖 `train`；
- 函数行为保持与历史 `train_utils.openjson/opendata` 一致。
"""

from __future__ import annotations

import io
import json

import numpy as np
from mmengine import fileio


def openjson(path: str):
    """读取 JSON 文件并返回解析结果。"""
    value = fileio.get_text(path)
    return json.loads(value)


def opendata(path: str):
    """读取 NPZ 文件并返回 `np.load` 结果对象。"""
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    return np.load(buff)

