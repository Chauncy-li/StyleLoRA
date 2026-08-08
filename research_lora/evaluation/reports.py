"""JSON report I/O and evidence-oriented experiment summaries.

JSON 报告读写与"证据导向"的实验结果汇总：
- write_json: 把实验结果写成格式化的 JSON 文件（自动建目录、UTF-8、排序键）；
- summarize: 把一组带 rho（风格强度）的记录聚合为实验结论，
  核心判断依据是"normal（rho=0）是否保持基线与 rho 是否产生单调风格响应"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """把结果字典写成 JSON 文件。

    Args:
        path: 输出文件路径（自动创建父目录）。
        payload: 要序列化的结果字典。
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 便于阅读；ensure_ascii=False 保留中文；
    # sort_keys=True 保证键有序，便于 diff 对比多次实验输出。
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """按 rho 分组聚合实验记录，并给出"是否支持进入下一阶段"的汇总判断。

    以一段 rho 扫描实验（rho = -1.0 ... 1.0）为例，该函数会检查：
    1. normal 档（rho=0.0）是否与基线完全一致（identity_error 全部为 0）；
    2. 风格指标 style_score 是否随 |rho| 单调变化；
    只有当两者同时满足时，才认为实验支持进入 HyperLoRA 下一阶段。

    Args:
        records: 实验记录的可迭代对象，每条应含 rho（风格强度）等字段。

    Returns:
        Dict 汇总：
        - records: 记录总数
        - normal_preserves_baseline: normal 档是否保持基线
        - rho_style_response_monotonic: 风格指标是否随 rho 单调
        - supports_hyperlora_next_stage: 是否支持进入下一阶段
        - by_rho_count: 每个 rho 有多少条记录
    """
    # 转成列表（可重复遍历），并按 rho 字符串分组
    records = list(records)
    by_rho = {}
    for record in records:
        by_rho.setdefault(str(record.get("rho")), []).append(record)
    # normal 档 = rho=0.0 的记录；身份保持 OK = 有记录且所有 identity_error 均为 0
    normal = by_rho.get("0.0", [])
    identity_ok = bool(normal) and all("identity_error" in item and float(item["identity_error"]) == 0.0 for item in normal)
    # 若存在 style_score 字段，则判断它是否随 rho 单调变化
    monotonic = _is_monotonic(records, "style_score") if any("style_score" in item for item in records) else False
    return {"records": len(records), "normal_preserves_baseline": identity_ok,
            "rho_style_response_monotonic": monotonic,
            "supports_hyperlora_next_stage": identity_ok and monotonic,
            "by_rho_count": {rho: len(items) for rho, items in by_rho.items()}}


def _is_monotonic(records: list[Mapping[str, Any]], field: str) -> bool:
    """判断某字段是否随 rho 单调不减（按 rho 分组后取各组的均值比较）。

    Args:
        records: 实验记录列表。
        field: 需要检查单调性的字段名。

    Returns:
        记录不足 2 个 rho 分组时返回 True（数据不足视为无违例）。
    """
    # 按 rho 分组收集有效字段值
    groups = {}
    for record in records:
        if record.get(field) is not None:
            groups.setdefault(float(record["rho"]), []).append(float(record[field]))
    # 按 rho 排序，并取每组均值作为该点的代表值
    points = sorted((rho, sum(values) / len(values)) for rho, values in groups.items())
    # 单调不减：每一对相邻点都满足 b >= a
    return len(points) < 2 or all(b >= a for (_, a), (_, b) in zip(points, points[1:]))