"""Versioned, serialisable records used by the LoRA data pipeline.

LoRA 数据管线使用的、可序列化的样本记录定义：
- 合法取值常量（拆分/场景/风格）及其别名映射；
- 核心数据类 StyleSample：一条样本的规范化表示，
  支持从多种历史索引字段名（别名）构造、风格标签推断与质量标记。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


# 合法拆分 / 场景 / 风格取值
VALID_SPLITS = {"train", "val", "test"}
VALID_SCENES = {"straight_free_drive", "straight_car_follow"}
VALID_STYLES = {"aggressive", "normal", "conservative"}

# 场景名别名：兼容不同数据管线/历史索引的写法
SCENE_ALIASES = {
    "straight_free_drive": "straight_free_drive", "free-drive": "straight_free_drive", "free_drive": "straight_free_drive",
    "straight_car_follow": "straight_car_follow", "car-follow": "straight_car_follow", "car_follow": "straight_car_follow",
}
# 风格名别名：aggr -> aggressive、cons -> conservative、norm -> normal 等
STYLE_ALIASES = {
    "aggressive": "aggressive", "aggr": "aggressive", "normal": "normal", "norm": "normal",
    "conservative": "conservative", "cons": "conservative",
}
# 每个场景的三轴风格定义（与评价脚本保持一致）
AXES_BY_SCENE = {
    "straight_free_drive": ("speed_preference", "longitudinal_intensity", "smoothness"),
    "straight_car_follow": ("headway_margin", "response_decisiveness", "response_smoothness"),
}


@dataclass(frozen=True)
class StyleSample:
    """一条规范化后的风格样本记录（不可变）。

    Attributes:
        cache_path: 对应缓存文件路径（相对或绝对）。
        split: train / val / test。
        scene_type: 场景类型（须在 VALID_SCENES 中）。
        style: aggressive / normal / conservative（须在 VALID_STYLES 中）。
        metrics: 风格度量名 -> 数值。
        metric_mask: 度量是否被启用（只统计启用项）。
        log_name / token: 场景日志名与令牌，用于跨拆分泄漏检测。
        label_confidence: 风格标签置信度（[0, 1]）。
        quality_flags: 质量告警标记（True 表示该样本有质量问题）。
        source_index: 来源索引文件路径，便于溯源。
    """

    cache_path: str
    split: str
    scene_type: str
    style: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_mask: Dict[str, bool] = field(default_factory=dict)
    log_name: str = ""
    token: str = ""
    label_confidence: float = 1.0
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    source_index: str = ""

    def __post_init__(self) -> None:
        """构造后校验：拆分/场景/风格合法、缓存路径非空、置信度在 [0,1]。"""
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {self.split!r}; expected one of {sorted(VALID_SPLITS)}")
        if self.scene_type not in VALID_SCENES:
            raise ValueError(f"Unknown scene_type {self.scene_type!r}; expected one of {sorted(VALID_SCENES)}")
        if self.style not in VALID_STYLES:
            raise ValueError(f"Unknown style {self.style!r}; expected one of {sorted(VALID_STYLES)}")
        if not self.cache_path:
            raise ValueError("cache_path must not be empty")
        if not 0.0 <= float(self.label_confidence) <= 1.0:
            raise ValueError("label_confidence must be in [0, 1]")

    @property
    def key(self) -> str:
        """Stable leakage-detection key; cache path is the fallback for legacy indexes.

        稳定的泄漏检测键：优先用 log_name:token 定位同一物理场景；
        旧索引缺少这些字段时回退到 cache_path。
        """
        return f"{self.log_name}:{self.token}" if self.log_name or self.token else self.cache_path

    @property
    def is_usable(self) -> bool:
        """该样本是否可用：没有任何质量告警标记。"""
        return not any(self.quality_flags.values())

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典（用于 JSON 序列化）。"""
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, split_override: str | None = None,
                     low_threshold: float = 0.33, high_threshold: float = 0.67,
                     normal_half_width: float = 0.08) -> "StyleSample":
        """From an arbitrary index row, normalise field names and infer missing style.

        从任意索引行构造 StyleSample：归一化字段名（别名兼容）并在缺少显式风格
        标签时根据风格向量推断风格。

        Args:
            row: 原始索引行（字典）。
            split_override: 若给出，则覆盖行的拆分字段。
            low_threshold: 推断 conservative 的向量值上限。
            high_threshold: 推断 aggressive 的向量值下限。
            normal_half_width: 推断 normal 时允许偏离 0.5 的半宽。

        Returns:
            校验通过的 StyleSample。

        Raises:
            ValueError: 场景/风格无法识别，或拆分/置信度等校验失败。
        """
        # 字段别名表：统一不同管线/历史版本对同一字段的不同命名
        aliases = {
            "cache_path": ("cache_path", "filename", "planner_cache_path", "path", "cache", "file_path"),
            "split": ("split", "partition"),
            "scene_type": ("offline_scene_bucket", "scene_bucket", "scene_bucket_name", "scene_type", "scene", "interaction_type"),
            "style": ("style", "style_label", "style_label_name", "style_class", "preference_class", "style_category", "label"),
            "log_name": ("log_name", "log", "scenario_log"),
            "token": ("token", "scenario_token", "sample_token"),
            "label_confidence": ("label_confidence", "confidence"),
        }
        # 按别名优先级取第一个非空、非 None 的值
        def get(name: str, default: Any = "") -> Any:
            for key in aliases.get(name, (name,)):
                if key in row and row[key] is not None and str(row[key]).strip() != "":
                    return row[key]
            return default

        # 场景/风格名先做小写 + 别名归一化
        scene = SCENE_ALIASES.get(str(get("scene_type")).strip().lower())
        style = STYLE_ALIASES.get(str(get("style")).strip().lower())
        # 若没有显式风格标签，则从风格向量（三轴）推断风格
        label_vector = row.get("direct_axis_label_percentile_vec", row.get("canonical_style_vec", row.get("style_value_condition", [])))
        label_mask = row.get("m_train", row.get("m_label", []))
        if style is None and isinstance(label_vector, (list, tuple)) and len(label_vector) >= 3:
            values = [float(value) for value in label_vector[:3]]
            # 用掩码决定哪些轴参与推断（缺省全部参与）
            mask = [float(value) > 0.5 for value in label_mask[:3]] if isinstance(label_mask, (list, tuple)) and len(label_mask) >= 3 else [True] * 3
            active = [value for value, enabled in zip(values, mask) if enabled]
            # 被启用的轴全部 ≥ high：aggressive；全部 ≤ low：conservative；全部贴近 0.5：normal
            if active and all(value >= high_threshold for value in active): style = "aggressive"
            elif active and all(value <= low_threshold for value in active): style = "conservative"
            elif active and all(abs(value - 0.5) <= normal_half_width for value in active): style = "normal"
        # 场景或风格仍无法识别则报错
        if scene is None or style is None:
            raise ValueError(f"Unsupported source labels scene={get('scene_type')!r}, style={get('style')!r}")

        # 度量字典（优先取元数据 metrics，否则原始 metrics）
        raw_metrics = dict(row.get("metrics", row.get("raw_metrics", {})))
        # 若行内带三轴风格向量且与 AXES_BY_SCENE 对齐，则合并进 metrics 作为三轴值
        vector = row.get("style_performance_vec", row.get("label_values", label_vector))
        axis_names = row.get("style_axis_names", AXES_BY_SCENE.get(scene, ()))
        if isinstance(vector, (list, tuple)) and len(vector) == len(axis_names) == 3:
            raw_metrics.update({str(name): float(value) for name, value in zip(axis_names, vector)})
        # 度量启用掩码；为空时默认全部启用
        raw_mask = dict(row.get("metric_mask", row.get("valid_mask", {})))
        if not raw_mask and raw_metrics:
            raw_mask = {key: True for key in raw_metrics}
        # 质量告警：合并显式质量标志 + 样本/拆分有效性的隐式告警
        quality = {k: bool(v) for k, v in dict(row.get("quality_flags", row.get("quality", {}))).items()}
        if not bool(row.get("sample_quality_valid", True)):
            quality["sample_quality_invalid"] = True
        if not bool(row.get("split_valid", True)):
            quality["split_invalid"] = True
        return cls(
            cache_path=str(get("cache_path")), split=str(split_override or get("split")), scene_type=scene, style=style,
            metrics={str(k): float(v) for k, v in raw_metrics.items()},
            metric_mask={k: bool(v) for k, v in raw_mask.items()},
            log_name=str(get("log_name", row.get("log_id", ""))), token=str(get("token")),
            label_confidence=float(get("label_confidence", 1.0)),
            quality_flags=quality,
            source_index=str(row.get("source_index", "")),
        )
