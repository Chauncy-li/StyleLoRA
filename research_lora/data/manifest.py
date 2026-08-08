"""Manifest construction and data-audit utilities independent of research_v2.

与 research_v2 无关的 manifest 构建与数据审计工具：
- 从 JSON/JSONL 索引流式读取样本（支持大文件）；
- 把原始行规范化、校验并转成 StyleSample；
- 提供拆分防泄漏检查、数据审计摘要、manifest 哈希与按拆分写出 JSONL 等能力。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

from research_lora.data.schema import StyleSample


def _read_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    """读取 JSON 数组文件或逐行 JSON（JSONL）文件，产出每行/每元素字典。

    Args:
        path: 输入文件路径。

    Yields:
        逐条样本映射字典。兼容两种格式：以 "[" 开头的 JSON 数组，或逐行 JSON。
    """
    with path.open("r", encoding="utf-8") as handle:
        # 读第一个字符判断是否为 JSON 数组格式
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            # 整个文件是一个 JSON 数组
            yield from json.load(handle)
        else:
            # 每行一条 JSON 记录
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_style_index(path: str | Path, *, split_override: str | None = None,
                     low_threshold: float = 0.33, high_threshold: float = 0.67,
                     normal_half_width: float = 0.08) -> List[StyleSample]:
    """一次性加载风格索引（返回列表）。

    Args:
        path: 索引文件路径。
        split_override: 统一覆盖所有样本的拆分（可选）。
        low_threshold / high_threshold / normal_half_width: 无显式风格标签时，
            根据风格向量分位数推断 aggressive/conservative/normal 的阈值。

    Returns:
        校验通过后的 StyleSample 列表。
    """
    return list(iter_style_index(path, split_override=split_override, low_threshold=low_threshold,
                                 high_threshold=high_threshold, normal_half_width=normal_half_width))


def iter_style_index(path: str | Path, *, split_override: str | None = None,
                     low_threshold: float = 0.33, high_threshold: float = 0.67,
                     normal_half_width: float = 0.08,
                     skipped: MutableMapping[str, int] | None = None) -> Iterator[StyleSample]:
    """Stream a potentially multi-GB index, recording rejected rows instead of aborting the job.

    流式读取可能达到数 GB 的索引文件，把被拒绝的行记录到 ``skipped`` 统计中，
    而不是让整个任务崩溃退出（适合离线大数据量索引构建）。

    Args:
        path: 索引文件路径。
        split_override: 统一覆盖所有样本的拆分（可选）。
        low_threshold / high_threshold / normal_half_width: 无显式风格时推断风格的阈值。
        skipped: 可选计数器字典，按原因累计被跳过（err 类型）的行数。

    Yields:
        合法样本的 StyleSample；非法/不支持的样本会被跳过并（若提供 skipped）计数。
    """
    source = Path(path)
    for row in _read_rows(source):
        try:
            # 注入 source_index（来源索引文件路径）便于溯源，再做规范化与阈值推断
            yield StyleSample.from_mapping({**row, "source_index": str(source)}, split_override=split_override,
                                           low_threshold=low_threshold, high_threshold=high_threshold,
                                           normal_half_width=normal_half_width)
        except (TypeError, ValueError, KeyError) as error:
            # 按错误类型归类跳过原因：不支持的场景/风格、无效拆分、其它格式错误
            if skipped is not None:
                message = str(error)
                if "Unsupported source labels" in message:
                    skipped["unsupported_scene_or_style"] = skipped.get("unsupported_scene_or_style", 0) + 1
                elif "Unknown split" in message:
                    skipped["missing_or_invalid_split"] = skipped.get("missing_or_invalid_split", 0) + 1
                else:
                    skipped["malformed_record"] = skipped.get("malformed_record", 0) + 1


def filter_samples(samples: Iterable[StyleSample], *, min_confidence: float = 0.0) -> List[StyleSample]:
    """按可用性与置信度下限过滤样本。

    Args:
        samples: StyleSample 可迭代对象。
        min_confidence: 最小标签置信度。

    Returns:
        满足 is_usable（无质量告警）且置信度 >= min_confidence 的样本列表。
    """
    return [sample for sample in samples if sample.is_usable and sample.label_confidence >= min_confidence]


def assert_no_split_overlap(samples: Sequence[StyleSample]) -> None:
    """检查不同拆分（train/val/test）之间是否存在样本泄漏（同一 key 出现在多个拆分）。

    Args:
        samples: 全部样本序列。

    Raises:
        ValueError: 发现同一 sample.key 被分配到了不同拆分。
    """
    owners: Dict[str, str] = {}
    conflicts = []
    for sample in samples:
        # 首次登记该 key 所属拆分；若后续出现相同 key 但拆分不同，记为冲突
        previous = owners.setdefault(sample.key, sample.split)
        if previous != sample.split:
            conflicts.append((sample.key, previous, sample.split))
    if conflicts:
        preview = ", ".join(f"{key} ({a}/{b})" for key, a, b in conflicts[:5])
        raise ValueError(f"Manifest split leakage: {len(conflicts)} duplicate sample keys; {preview}")


def audit_samples(samples: Sequence[StyleSample]) -> Dict[str, Any]:
    """生成数据审计摘要：拆分/场景/风格计数、风格度量分布、质量告警统计。

    Args:
        samples: 全部样本序列。

    Returns:
        审计字典：
        - sample_count: 样本总数
        - counts: "split|scene|style" -> 数量
        - metric_distribution: 每个被启用度量的 min/max/mean/count
        - quality_flag_counts: 各质量告警标记的出现次数
    """
    # 先做防泄漏校验
    assert_no_split_overlap(samples)
    # 按 (split, scene_type, style) 计数
    counts = Counter((sample.split, sample.scene_type, sample.style) for sample in samples)
    # 收集所有启用（metric_mask 为 True）的度量值
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for sample in samples:
        for key, value in sample.metrics.items():
            if sample.metric_mask.get(key, True):
                metric_values[key].append(float(value))
    # 逐度量求分布
    distributions = {
        key: {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}
        for key, values in metric_values.items() if values
    }
    return {
        "sample_count": len(samples),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "metric_distribution": distributions,
        "quality_flag_counts": dict(Counter(flag for s in samples for flag, value in s.quality_flags.items() if value)),
    }


def manifest_hash(samples: Sequence[StyleSample]) -> str:
    """计算 manifest 的 SHA-256 哈希，用于追踪数据版本。

    Args:
        samples: 样本序列。

    Returns:
        基于排序键的规范化 JSON 行的 SHA-256 十六进制哈希。
    """
    canonical = "\n".join(json.dumps(s.to_dict(), ensure_ascii=False, sort_keys=True) for s in samples)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifests(samples: Sequence[StyleSample], output_dir: str | Path) -> Dict[str, Path]:
    """按拆分（train/val/test）把样本写出为三个 JSONL 文件。

    Args:
        samples: 样本序列。
        output_dir: 输出目录（自动创建）。

    Returns:
        {split: 对应 JSONL 文件路径}。

    Raises:
        ValueError: 样本在不同拆分间存在泄漏时抛出。
    """
    # 先做防泄漏校验
    assert_no_split_overlap(samples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "val", "test"):
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                if sample.split == split:
                    handle.write(json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        paths[split] = path
    return paths