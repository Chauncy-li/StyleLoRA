"""Read-only split parsing, weak style labels, manifests, and data audits."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from baseline.utils.io import opendata


STYLE_NAMES: Tuple[str, str, str] = ("aggr", "norm", "cons")
STYLE_TO_INDEX = {name: index for index, name in enumerate(STYLE_NAMES)}
FC_SCENES = ("straight_free_drive", "straight_car_follow")
FCL_SCENES = (*FC_SCENES, "straight_lane_change")


@dataclass(frozen=True)
class StyleEntry:
    filename: str
    cache_path: str
    split: str
    scene: str
    style: Optional[str]
    label_source: str
    label_confidence: Optional[float]
    split_valid: bool
    sample_quality_valid: bool
    log_id: Optional[str]
    token: Optional[str]
    label_values: Tuple[float, ...]
    active_axis_count: int
    extra: Dict[str, Any]

    @property
    def trainable(self) -> bool:
        return bool(
            self.split_valid
            and self.sample_quality_valid
            and self.style in STYLE_TO_INDEX
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "valid", "ok"}
    return default


def _float_vector(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty((0,), dtype=np.float64)
    return array[np.isfinite(array)] if array.size else array


def _relative_filename(value: Any, cache_root: Path) -> Optional[str]:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(cache_root)
        except ValueError:
            return None
    if any(part == ".." for part in path.parts):
        return None
    return path.as_posix().lstrip("./")


def _scene(record: Mapping[str, Any]) -> str:
    for key in ("offline_scene_bucket", "scene_bucket", "causal_scene_bucket"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def _explicit_style(record: Mapping[str, Any]) -> Optional[str]:
    mapping = {
        "aggr": "aggr", "aggressive": "aggr", "positive": "aggr", "high": "aggr",
        "norm": "norm", "normal": "norm", "neutral": "norm", "mid": "norm",
        "cons": "cons", "conservative": "cons", "negative": "cons", "low": "cons",
    }
    for key in ("style_class", "style_label", "preference_class", "style_category"):
        value = record.get(key)
        if value is None:
            continue
        normalized = mapping.get(str(value).strip().lower())
        if normalized is not None:
            return normalized
    return None


def weak_style_label(
    record: Mapping[str, Any],
    *,
    low_threshold: float,
    high_threshold: float,
    normal_half_width: float,
) -> Tuple[Optional[str], str, Optional[float], Tuple[float, ...], int]:
    """Return a conservative three-way weak label without averaging axes.

    Every active axis must independently agree on aggr, norm, or cons.  Mixed
    axes are intentionally ``None`` rather than being collapsed into a scalar.
    """
    explicit = _explicit_style(record)
    if explicit is not None:
        return explicit, "explicit_style_class", 1.0, (), 0
    values = _float_vector(record.get("direct_axis_label_percentile_vec"))
    source = "direct_axis_label_percentile_vec"
    if values.size != 3:
        values = _float_vector(record.get("canonical_style_vec"))
        source = "canonical_style_vec"
    if values.size != 3:
        condition = _float_vector(record.get("style_value_condition"))
        values = condition[:3] if condition.size >= 3 else np.empty((0,), dtype=np.float64)
        source = "style_value_condition"
    if values.size != 3:
        return None, "missing_three_axis_label", None, (), 0

    mask = _float_vector(record.get("m_train"))
    if mask.size != 3:
        mask = _float_vector(record.get("m_label"))
    if mask.size != 3:
        condition = _float_vector(record.get("style_value_condition"))
        mask = condition[3:6] if condition.size >= 6 else np.ones((3,), dtype=np.float64)
    active = mask > 0.5
    selected = values[active]
    if selected.size == 0 or not np.isfinite(selected).all():
        return None, f"{source}:no_active_axis", None, tuple(float(x) for x in values), int(active.sum())
    if bool(np.all(selected >= high_threshold)):
        return "aggr", source, float(np.min(selected - high_threshold)), tuple(float(x) for x in selected), int(active.sum())
    if bool(np.all(selected <= low_threshold)):
        return "cons", source, float(np.min(low_threshold - selected)), tuple(float(x) for x in selected), int(active.sum())
    if bool(np.all(np.abs(selected - 0.5) <= normal_half_width)):
        return "norm", source, float(np.min(normal_half_width - np.abs(selected - 0.5))), tuple(float(x) for x in selected), int(active.sum())
    return None, f"{source}:mixed_or_ambiguous", None, tuple(float(x) for x in selected), int(active.sum())


def load_entries(
    index_path: str | Path,
    cache_root: str | Path,
    *,
    split: str,
    allowed_scenes: Sequence[str],
    low_threshold: float,
    high_threshold: float,
    normal_half_width: float,
) -> Tuple[List[StyleEntry], Counter]:
    root = Path(cache_root).expanduser().resolve()
    skipped: Counter = Counter()
    entries: List[StyleEntry] = []
    with Path(index_path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{index_path}:{line_number} invalid JSONL") from error
            if not isinstance(record, Mapping):
                skipped["non_mapping_row"] += 1
                continue
            scene = _scene(record)
            if scene not in allowed_scenes:
                skipped[f"scene:{scene}"] += 1
                continue
            filename = _relative_filename(record.get("filename", record.get("cache_path", "")), root)
            if filename is None:
                skipped["invalid_cache_filename"] += 1
                continue
            cache_path = (root / filename).resolve()
            try:
                cache_path.relative_to(root)
            except ValueError:
                skipped["cache_path_escape"] += 1
                continue
            style, source, confidence, values, active_count = weak_style_label(
                record,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                normal_half_width=normal_half_width,
            )
            log_id = next(
                (str(record[key]) for key in ("log_name", "logfile", "log", "source_log") if record.get(key)),
                None,
            )
            token = next(
                (str(record[key]) for key in ("token", "scenario_token", "sample_token") if record.get(key)),
                None,
            )
            entries.append(
                StyleEntry(
                    filename=filename,
                    cache_path=str(cache_path),
                    split=str(split), scene=scene, style=style, label_source=source,
                    label_confidence=confidence,
                    split_valid=_as_bool(record.get("split_valid"), True),
                    sample_quality_valid=_as_bool(record.get("sample_quality_valid"), True),
                    log_id=log_id, token=token, label_values=values,
                    active_axis_count=active_count,
                    extra={
                        "route_lane_change_intent": record.get("route_lane_change_intent"),
                        "target_lane_known": record.get("target_lane_known"),
                    },
                )
            )
    return entries, skipped


def write_manifest(path: str | Path, entries: Iterable[StyleEntry]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_manifest(path: str | Path) -> List[StyleEntry]:
    rows: List[StyleEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            value["label_values"] = tuple(float(item) for item in value.get("label_values", ()))
            rows.append(StyleEntry(**value))
    return rows


def _trajectory_stats(cache_path: str, dt_seconds: float) -> Dict[str, Optional[float]]:
    """Small cache-only audit; no future value is ever used for deployment."""
    cache = opendata(cache_path)
    try:
        ego = np.asarray(cache["ego_agent_future"], dtype=np.float64)
        xy = ego[:, :2]
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        speed = steps / dt_seconds if steps.size else np.empty((0,))
        neighbor = np.asarray(cache["neighbor_agents_past"])
        last = neighbor[:, -1, :3] if neighbor.ndim >= 3 else np.empty((0, 3))
        density = int(np.sum(np.any(np.abs(last) > 1e-8, axis=-1))) if last.size else 0
        if xy.shape[0] >= 3:
            direction = np.diff(xy, axis=0)
            heading = np.arctan2(direction[:, 1], direction[:, 0])
            curvature = np.abs(np.diff(np.unwrap(heading))) / np.maximum(steps[1:], 1e-4)
        else:
            curvature = np.empty((0,))
        route = np.asarray(cache["route_lanes"])
        route_available = bool(np.any(np.abs(route[..., :2]) > 1e-8))
        return {
            "speed_mean_mps": float(speed.mean()) if speed.size else None,
            "speed_p95_mps": float(np.percentile(speed, 95)) if speed.size else None,
            "speed_max_mps": float(speed.max()) if speed.size else None,
            "traffic_density": density,
            "curvature_mean_1pm": float(curvature.mean()) if curvature.size else 0.0,
            "curvature_p95_1pm": float(np.percentile(curvature, 95)) if curvature.size else 0.0,
            "route_available": route_available,
        }
    finally:
        cache.close()


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size), "mean": float(array.mean()), "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def audit_entries(
    train: Sequence[StyleEntry], val: Sequence[StyleEntry], *, dt_seconds: float, max_cache_audit: int
) -> Dict[str, Any]:
    """Return the required audit without changing splits or silently filtering rows."""
    grouped: Dict[Tuple[str, str, str], List[StyleEntry]] = defaultdict(list)
    for entry in (*train, *val):
        grouped[(entry.split, entry.scene, entry.style or "unknown")].append(entry)
    train_logs = {entry.log_id for entry in train if entry.log_id}
    val_logs = {entry.log_id for entry in val if entry.log_id}
    train_tokens = {entry.token for entry in train if entry.token}
    val_tokens = {entry.token for entry in val if entry.token}
    train_keys = {entry.filename for entry in train}
    val_keys = {entry.filename for entry in val}
    selected = [entry for entry in (*train, *val) if entry.trainable]
    if max_cache_audit > 0:
        selected = sorted(selected, key=lambda entry: hashlib.sha1(entry.filename.encode()).hexdigest())[:max_cache_audit]
    stats: Dict[Tuple[str, str, str], List[Dict[str, Optional[float]]]] = defaultdict(list)
    cache_failures: Counter = Counter()
    for entry in selected:
        try:
            stats[(entry.split, entry.scene, entry.style or "unknown")].append(
                _trajectory_stats(entry.cache_path, dt_seconds)
            )
        except Exception as error:
            cache_failures[type(error).__name__] += 1
    geometry = {}
    for key, rows in stats.items():
        geometry["/".join(key)] = {
            metric: _summary([row.get(metric) for row in rows])
            for metric in ("speed_mean_mps", "speed_p95_mps", "traffic_density", "curvature_mean_1pm", "curvature_p95_1pm")
        }
        geometry["/".join(key)]["route_available_rate"] = (
            None if not rows else float(np.mean([bool(row["route_available"]) for row in rows]))
        )
    lane = [entry for entry in (*train, *val) if entry.scene == "straight_lane_change"]
    return {
        "schema_version": "style_prototype_residual_data_audit_v1",
        "train_count": len(train), "val_count": len(val),
        "trainable_count": sum(entry.trainable for entry in train),
        "val_trainable_count": sum(entry.trainable for entry in val),
        "counts_by_split_scene_style": {"/".join(key): len(value) for key, value in sorted(grouped.items())},
        "label_source_counts": dict(Counter(entry.label_source for entry in (*train, *val))),
        "invalid_counts": {
            "unknown_style": sum(entry.style is None for entry in (*train, *val)),
            "split_invalid": sum(not entry.split_valid for entry in (*train, *val)),
            "quality_invalid": sum(not entry.sample_quality_valid for entry in (*train, *val)),
        },
        "split_overlap": {
            "log_overlap_count": len(train_logs & val_logs), "token_overlap_count": len(train_tokens & val_tokens),
            "cache_key_overlap_count": len(train_keys & val_keys),
            "train_rows_without_log": sum(entry.log_id is None for entry in train),
            "val_rows_without_log": sum(entry.log_id is None for entry in val),
            "log_disjoint": bool(train_logs and val_logs and not (train_logs & val_logs)),
        },
        "geometry_sampled_rows": len(selected), "geometry": geometry,
        "cache_audit_failures": dict(cache_failures),
        "lane_change": {
            "row_count": len(lane),
            "route_intent_available_rate": None if not lane else float(np.mean([entry.extra.get("route_lane_change_intent") is not None for entry in lane])),
            "target_lane_available_rate": None if not lane else float(np.mean([entry.extra.get("target_lane_known") is not None for entry in lane])),
        },
    }


def group_histogram(entries: Sequence[StyleEntry], indices: Optional[Sequence[int]] = None) -> Dict[str, int]:
    selected = entries if indices is None else [entries[int(index)] for index in indices]
    counts: Counter = Counter()
    for entry in selected:
        if entry.trainable and entry.style in STYLE_TO_INDEX:
            counts[f"{entry.scene}/{entry.style}"] += 1
    return dict(sorted(counts.items()))


def balanced_indices(
    entries: Sequence[StyleEntry], count: int, *, seed: int, batch_size: int = 1
) -> List[int]:
    """Deterministic scene-by-style sampling with repeated groups per batch.

    Repeating a group makes a mini-batch prototype term an actual group mean,
    rather than silently degrading it to a one-sample target.
    """
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry.trainable and entry.style in STYLE_TO_INDEX:
            groups[(entry.scene, entry.style)].append(index)
    if not groups:
        raise ValueError("manifest has no trainable scene-by-style rows")
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    if count <= 0 or batch_size <= 0:
        raise ValueError("count and batch_size must be positive")
    selected: List[int] = []
    group_cursor = 0
    while len(selected) < count:
        remaining = min(batch_size, count - len(selected))
        repeats = 2 if remaining >= 2 else 1
        unique_groups = max(1, math.ceil(remaining / repeats))
        batch: List[int] = []
        for group_offset in range(unique_groups):
            key = keys[(group_cursor + group_offset) % len(keys)]
            values = groups[key]
            draw_count = min(repeats, remaining - len(batch))
            batch.extend(int(values[int(rng.integers(0, len(values)))]) for _ in range(draw_count))
        group_cursor = (group_cursor + unique_groups) % len(keys)
        rng.shuffle(batch)
        selected.extend(batch)
    return selected
