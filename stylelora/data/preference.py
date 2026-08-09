"""Weak-ordering preference core algorithms.

弱排序偏好核心算法：经验 CDF、场景内百分位转换、三轴加权中位数稳健聚合、
基于最大绝对偏离的置信度计算。全部纯 numpy/CPU，支持大规模训练集流式处理。

约定：
- 每个 (scene, axis) 单独构建训练集经验 CDF F_{s,k}；
- 正向轴（方向 +1）：q = F(m)；反向轴（方向 -1）：q = 1 - F(m)；
- preference_rank = 三轴百分位加权中位数（默认等权）；
- rank_confidence = 1 - max|q_i - median|，三轴越一致置信度越高，
  并惩罚单轴严重冲突。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Iterable, Mapping, Sequence

import numpy as np

from stylelora.data.axes import AXES_BY_SCENE, direction_signs

# 异常物理值保护：超出 [lo, hi] 的原始值按边界截断后再进 CDF（防离群污染内存排序）
_VALUE_BOUND = (-1e6, 1e6)


class EmpiricalCDF:
    """按 (scene, axis) 构建训练集经验 CDF，支持增量 append 与二分插值查询。

    值存储采用惰性落盘/就地排序：当单轴样本量较大时会把所有值放入内存数组，
    查询时用二分插值（线性插值位于相邻排序值之间）。small-data 时允许流式排序。
    """

    def __init__(self, axis: str) -> None:
        self.axis = axis
        self._values: list[float] = []

    def append(self, value: float) -> None:
        """追加一个有效值（NaN/Inf 会被跳过）。"""
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return
        value = float(np.clip(value, *_VALUE_BOUND))
        self._values.append(value)

    def finalize(self) -> None:
        """排序并转为 numpy 数组，之后不可再 append（查询阶段）。"""
        if self._values:
            self._sorted = np.sort(np.asarray(self._values, dtype=np.float64))
        else:
            self._sorted = np.zeros(0, dtype=np.float64)
        del self._values

    def __len__(self) -> int:
        values = getattr(self, "_sorted", None)
        if values is None:
            return len(self._values)
        return int(values.shape[0])

    def sorted_values(self) -> np.ndarray:
        """返回 finalize 后的排序值数组（用于持久化复用到 val/test）。"""
        if not hasattr(self, "_sorted"):
            raise RuntimeError("EmpiricalCDF must be finalized before it can be serialized")
        return self._sorted

    @classmethod
    def from_sorted(cls, axis: str, sorted_values) -> "EmpiricalCDF":
        """从已排序的值数组直接重建 CDF（用于加载训练集 CDF）。"""
        obj = cls(axis)
        obj._sorted = np.asarray(sorted_values, dtype=np.float64)
        return obj

    def percentile(self, value: float) -> float:
        """返回经验 CDF 的百分位 F(value) ∈ [0,1]，使用平均秩处理重复值。

        训练集中已出现的值使用其并列组的 0-based 平均秩，并除以 ``n - 1``
        映射到 [0,1]。验证值若落在相邻训练值之间，则在两端平均秩之间线性插值；
        超出训练样本区间的值截断为 0 / 1。训练值全部相同时返回 0.5。
        """
        if not hasattr(self, "_sorted"):
            raise RuntimeError("EmpiricalCDF must be finalized before percentile queries")
        n = self._sorted.shape[0]
        if n == 0:
            return 0.5
        if not (isinstance(value, (int, float)) and np.isfinite(value)):
            return 0.5
        value = float(np.clip(value, *_VALUE_BOUND))
        if n == 1:
            return 0.5
        if value < self._sorted[0]:
            return 0.0
        if value > self._sorted[-1]:
            return 1.0

        left = bisect_left(self._sorted, value)
        right = bisect_right(self._sorted, value)
        if left < right:
            # 训练集中已出现的值：并列组共享 0-based 平均秩。
            average_rank = (left + right - 1) / 2.0
            return float(average_rank / (n - 1))

        # 验证值未在训练集中出现：在相邻训练值各自的平均秩之间插值。
        lower_value = self._sorted[left - 1]
        upper_value = self._sorted[left]
        lower_left = bisect_left(self._sorted, lower_value)
        lower_right = bisect_right(self._sorted, lower_value)
        upper_left = bisect_left(self._sorted, upper_value)
        upper_right = bisect_right(self._sorted, upper_value)
        lower_rank = (lower_left + lower_right - 1) / 2.0
        upper_rank = (upper_left + upper_right - 1) / 2.0
        fraction = (value - lower_value) / (upper_value - lower_value)
        interpolated_rank = lower_rank + fraction * (upper_rank - lower_rank)
        return float(interpolated_rank / (n - 1))


class SceneCDFTable:
    """按 (scene, axis) 聚合的经验 CDF 表，负责两遍处理中的训练集汇总。"""

    def __init__(self, scenes: Sequence[str] = tuple(AXES_BY_SCENE)) -> None:
        self._cdfs: dict[tuple[str, str], EmpiricalCDF] = {}
        for scene in scenes:
            signs = direction_signs(scene)
            for axis, sign in zip(AXES_BY_SCENE[scene], signs):
                self._cdfs[(scene, axis)] = EmpiricalCDF(axis)

    def append(self, scene: str, axis: str, value: float) -> None:
        """向 (scene, axis) 的 CDF 追加一个物理指标值。"""
        key = (scene, axis)
        if key not in self._cdfs:
            raise KeyError(f"Unknown (scene, axis) {key!r}")
        self._cdfs[key].append(value)

    def finalize_all(self) -> None:
        """排序所有 CDF，进入查询阶段。"""
        for cdf in self._cdfs.values():
            cdf.finalize()

    def percentile(self, scene: str, axis: str, value: float, *, sign: int = 1) -> float:
        """返回综合方向先验后的场景内百分位（[0,1]，越大越激进）。

        Args:
            scene: 场景名。
            axis: 轴名。
            value: 原始物理指标值。
            sign: 方向符号（由 axes.direction_sign 给出）；-1 时取 q = 1 - F(m)。
        """
        key = (scene, axis)
        if key not in self._cdfs:
            raise KeyError(f"Unknown (scene, axis) {key!r}")
        q = self._cdfs[key].percentile(value)
        return 1.0 - q if sign < 0 else q

    def save(self, path) -> None:
        """把每个 (scene, axis) 的排序值数组保存为 JSON，供 val/test 原样复用。"""
        import json
        from pathlib import Path

        payload = {
            "_version": 1,
            "_keys": [f"{scene}||{axis}" for scene, axis in self._cdfs],
            "_values": [cdf.sorted_values().tolist() for cdf in self._cdfs.values()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SceneCDFTable":
        """从训练集 CDF 保存文件重建表；缺失的 (scene, axis) 键保持为空。"""
        import json
        from pathlib import Path

        table = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key_str, values in zip(payload["_keys"], payload["_values"]):
            scene, axis = key_str.split("||", 1)
            if (scene, axis) not in table._cdfs:
                continue
            table._cdfs[(scene, axis)] = EmpiricalCDF.from_sorted(axis, values)
        return table


def weighted_median(percentiles: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """三轴百分位的加权中位数。

    Args:
        percentiles: 三个轴的有效百分位 [0,1]。
        weights: 与 percentiles 等长的权重；None 表示等权。

    Returns:
        加权中位数（[0,1]）。
    """
    values = np.asarray(percentiles, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    mid = total / 2.0
    for i, cum in enumerate(cumulative):
        if cum >= mid:
            # 在跨越中位权重的样本处做线性插值
            if cum - weights[i] >= mid and i > 0:
                return float((values[i - 1] + values[i]) / 2.0)
            return float(values[i])
    return float(values[-1])


def rank_confidence(percentiles: Sequence[float]) -> float:
    """三轴一致度置信度 c = 1 - max|q_i - median| ∈ [0,1]。

    使用最大绝对偏离（而非 MAD）作为离散度：能惩罚单轴严重冲突。
    例如 [0,0,1] 的 median=0、max|q-median|=1 → c=0（低置信）。
    若三轴均无效返回 0，只有 1 轴有效返回中间置信度 0.5。
    """
    valid = [float(q) for q in percentiles if q is not None and np.isfinite(q) and 0.0 <= q <= 1.0]
    n = len(valid)
    if n == 0:
        return 0.0
    if n < 2:
        return 0.5
    median = float(np.median(valid))
    max_dev = float(max(abs(q - median) for q in valid))
    # 偏离范围最大为 1，1 - max_dev 即为置信度
    return float(np.clip(1.0 - max_dev, 0.0, 1.0))


def aggregate_weak_rank(percentiles: Sequence[float], valid: Sequence[bool],
                        weights: Sequence[float] | None = None) -> tuple[float, float]:
    """把三轴百分位与有效性聚合为 preference_rank 与 rank_confidence。

    Args:
        percentiles: 三个轴的百分位 [0,1]（无效轴可为 NaN）。
        valid: 三个轴是否有效。
        weights: 可选轴权重（None=等权）。

    Returns:
        (preference_rank, rank_confidence)。

    Raises:
        ValueError: 三个轴全部无效。
    """
    active = [float(q) for q, v in zip(percentiles, valid) if v and q is not None and np.isfinite(q)]
    active_weights = list(weights) if weights else None
    if active_weights is not None:
        active_weights = [w for w, v in zip(active_weights, valid) if v]
    if not active:
        raise ValueError("All three axes are invalid; cannot aggregate weak rank")
    rank = weighted_median(active, weights=active_weights)
    confidence = rank_confidence(active)
    return float(np.clip(rank, 0.0, 1.0)), confidence


def stream_chunk(values: Iterable[float], chunk_size: int = 1 << 16) -> Iterable[np.ndarray]:
    """把迭代器按块转成 numpy 数组，降低内存占用。"""
    buffer: list[float] = []
    for value in values:
        buffer.append(value)
        if len(buffer) >= chunk_size:
            yield np.asarray(buffer, dtype=np.float64)
            buffer.clear()
    if buffer:
        yield np.asarray(buffer, dtype=np.float64)
