"""Reproducible 1:1 free-drive/car-follow sampling for a single target style."""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from stylelora.lora.data.schema import StyleSample


class SceneBalancedSampler(Sampler[int]):
    def __init__(self, samples: Sequence[StyleSample], style: str, *, seed: int = 0,
                 num_samples: int | None = None, replacement: bool = True) -> None:
        """构造指定风格的跨场景平衡采样器。

        训练默认采用有放回采样，将少数场景补齐到多数场景的数量；固定验证集则传入
        ``replacement=False``，每类取相同数量且不重复的样本，避免验证结果被少数场景
        的重复样本主导。
        """
        if style not in {"aggressive", "conservative"}:
            raise ValueError("SceneBalancedSampler only accepts aggressive or conservative training styles")
        self.style, self.seed, self.epoch = style, int(seed), 0
        self.replacement = bool(replacement)
        self.groups = {scene: [i for i, item in enumerate(samples) if item.style == style and item.scene_type == scene]
                       for scene in ("straight_free_drive", "straight_car_follow")}
        if not all(self.groups.values()):
            missing = [name for name, indices in self.groups.items() if not indices]
            raise ValueError(f"Cannot balance style={style}: no usable samples for {missing}")
        if num_samples is None:
            # 训练补齐少数场景；验证仅使用两类共同拥有的、不重复的样本数量。
            per_scene = max(map(len, self.groups.values())) if self.replacement else min(map(len, self.groups.values()))
            self.num_samples = 2 * per_scene
        else:
            self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not self.replacement:
            if self.num_samples % 2:
                raise ValueError("replacement=False requires an even num_samples for exact 1:1 balancing")
            per_scene = self.num_samples // 2
            if any(per_scene > len(indices) for indices in self.groups.values()):
                raise ValueError("replacement=False cannot draw more samples than either scene contains")
        self.last_epoch_counts: Counter[str] = Counter()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        per_scene = self.num_samples // 2
        selected = []
        for scene, indices in self.groups.items():
            if self.replacement:
                selected.extend(rng.choice(indices) for _ in range(per_scene))
            else:
                selected.extend(rng.sample(indices, per_scene))
        if self.num_samples % 2:
            selected.append(rng.choice(self.groups["straight_free_drive"]))
        rng.shuffle(selected)
        self.last_epoch_counts = Counter("straight_free_drive" if index in self.groups["straight_free_drive"] else "straight_car_follow" for index in selected)
        return iter(selected)

    def epoch_report(self) -> dict[str, float | int]:
        total = sum(self.last_epoch_counts.values())
        return {"samples": total, **{scene: self.last_epoch_counts[scene] / max(total, 1) for scene in self.groups}}


