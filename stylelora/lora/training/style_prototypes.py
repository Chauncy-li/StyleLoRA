"""Frozen train-split scene/style prototypes for the three-axis auxiliary loss.

三轴原型只由训练集专家轨迹生成。它们是训练时的冻结监督目标，不包含
任何 LoRA 参数，也不参与推理接口。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from stylelora.lora.evaluation.style_metrics import AXES_BY_SCENE


SCENES = tuple(AXES_BY_SCENE)
STYLES = ("aggressive", "conservative")
_OPPOSITE_STYLE = {"aggressive": "conservative", "conservative": "aggressive"}


@dataclass(frozen=True)
class SceneStylePrototypeTable:
    """训练集三轴统计量与 ``scene × style`` 原型的只读张量视图。"""

    means: Mapping[str, torch.Tensor]
    stds: Mapping[str, torch.Tensor]
    prototypes: Mapping[tuple[str, str], torch.Tensor]
    source_path: str
    source_manifest_hash: str

    @classmethod
    def from_json(cls, path: str | Path, *, device: torch.device) -> "SceneStylePrototypeTable":
        """读取并严格校验由 ``build_style_prototypes`` 写出的训练集原型文件。"""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != 1:
            raise ValueError("Unsupported style-prototype artifact version")
        if payload.get("metric_space") != "expert_trajectory_proxy_axes":
            raise ValueError("Style prototypes must use expert_trajectory_proxy_axes")
        if payload.get("source", {}).get("training_split_only") is not True:
            raise ValueError("Style-prototype artifact must be built from the train split only")
        source_manifest_hash = payload.get("source", {}).get("manifest_hash")
        if not isinstance(source_manifest_hash, str) or not source_manifest_hash:
            raise ValueError("Style-prototype artifact is missing its source manifest hash")

        means, stds, prototypes = {}, {}, {}
        scene_payload = payload.get("scenes", {})
        for scene in SCENES:
            entry = scene_payload.get(scene)
            if not isinstance(entry, dict) or entry.get("axis_names") != list(AXES_BY_SCENE[scene]):
                raise ValueError(f"Missing or incompatible prototype axes for {scene}")
            statistics = entry.get("statistics", {})
            mean = torch.as_tensor(statistics.get("mean"), dtype=torch.float32, device=device)
            std = torch.as_tensor(statistics.get("std"), dtype=torch.float32, device=device)
            if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(f"Invalid standardization statistics for {scene}")
            if (std <= 0).any():
                raise ValueError(f"Non-positive standard deviation in {scene}")
            means[scene], stds[scene] = mean, std
            for style in STYLES:
                prototype = torch.as_tensor(entry.get("prototypes", {}).get(style, {}).get("standardized_mean"),
                                            dtype=torch.float32, device=device)
                if prototype.shape != (3,) or not torch.isfinite(prototype).all():
                    raise ValueError(f"Missing or invalid {scene} / {style} prototype")
                prototypes[(scene, style)] = prototype
        return cls(means=means, stds=stds, prototypes=prototypes, source_path=str(path.resolve()),
                   source_manifest_hash=source_manifest_hash)

    def standardize(self, scene: str, vector: torch.Tensor) -> torch.Tensor:
        """按该场景的训练集统计量标准化一个三轴向量，保留其梯度。"""
        if scene not in self.means or vector.shape != (3,):
            raise ValueError(f"Expected a three-axis vector for known scene, got scene={scene!r}, shape={tuple(vector.shape)}")
        return (vector - self.means[scene].to(vector)) / self.stds[scene].to(vector)

    def prototype(self, scene: str, style: str, *, like: torch.Tensor) -> torch.Tensor:
        """返回与 ``like`` 同设备、同精度的冻结场景内目标原型。"""
        if style not in STYLES:
            raise ValueError(f"Unknown prototype style {style!r}")
        return self.prototypes[(scene, style)].to(like)

    @staticmethod
    def opposite(style: str) -> str:
        """返回 aggressive/conservative 的相反风格标签。"""
        try:
            return _OPPOSITE_STYLE[style]
        except KeyError as exc:
            raise ValueError(f"Unknown prototype style {style!r}") from exc


