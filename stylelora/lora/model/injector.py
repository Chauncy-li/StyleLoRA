"""Safe, name-based injection of LoRA layers into an already-loaded DiffPlanner.

安全地、基于模块名（name-based）把 LoRA 层注入到已经加载好的 DiffPlanner 模型中。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from torch import nn

from stylelora.lora.model.lora_layers import StyleEgoMaskedLoRALinear


# 默认注入目标：DiT 解码器中每个 DiTBlock 的 MLP 线性层 + FinalLayer 投影层
# - mlp1.fc1 / mlp1.fc2 / mlp2.fc1 / mlp2.fc2:
#   timm 的 Mlp(fc1 -> 激活 -> fc2) 内部的线性层，位于每个 DiTBlock 中
# - final_layer.proj.1 / final_layer.proj.4:
#   FinalLayer.proj 是 nn.Sequential(LayerNorm, Linear, GELU, LayerNorm, Linear)，
#   其中的索引 1 和 4 正是两个 nn.Linear 层
DEFAULT_TARGETS = ("mlp1.fc1", "mlp1.fc2", "mlp2.fc1", "mlp2.fc2", "final_layer.proj.1", "final_layer.proj.4")


@dataclass(frozen=True)
class InjectionReport:
    """一次 LoRA 注入的统计报告（冻结数据类），记录注入结果与参数量信息。"""

    layers: Tuple[str, ...]           # 实际被替换（注入 LoRA）的模块名称元组
    base_parameters: int              # 注入前基座模型的总参数量
    trainable_parameters: int         # 注入后可训练参数数量（仅 LoRA A/B 矩阵）

    @property
    def trainable_ratio(self) -> float:
        """可训练参数占比，衡量 LoRA 相比完整微调的轻量程度。"""
        return self.trainable_parameters / max(self.base_parameters, 1)


def _parent_and_leaf(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    """按点分模块名（如 'a.b.c'）定位到父模块与最后一个属性名（leaf）。

    Args:
        root: 模型根模块。
        name: 点分的模块路径名称。

    Returns:
        (父模块, 叶子属性名)，用于后续 setattr 替换。
    """
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_baseline(model: nn.Module) -> None:
    """冻结模型全部参数：把所有参数的 requires_grad 置为 False。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _matches(name: str, target_modules: Sequence[str]) -> bool:
    """判断模块名是否以任一目标后缀结尾（后缀式匹配）。"""
    return any(name.endswith(target) for target in target_modules)


def inject_style_lora(model: nn.Module, *, rank: int = 4, alpha: float | None = None,
                      dropout: float = 0.0, target_modules: Sequence[str] = DEFAULT_TARGETS) -> InjectionReport:
    """只替换请求的解码器线性层；基线权重文件与张量保持原样不动。

    Args:
        model: 已加载好权重的 DiffPlanner（或任何含目标 Linear 层的模型）。
        rank: LoRA 低秩矩阵的秩。
        alpha: LoRA 缩放系数，默认取 rank。
        dropout: LoRA 增量的 dropout 比率。
        target_modules: 需要替换的目标模块后缀列表，默认使用 DEFAULT_TARGETS。

    Returns:
        InjectionReport：注入结束后返回层名/参数量统计。

    Raises:
        ValueError: 没有匹配到任何 nn.Linear 模块时抛出，防止静默注入失败。
    """
    # 记录注入前的基座总参数量，作为报告基线
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    # 第一步：冻结整个基座模型
    freeze_baseline(model)
    # 第二步：筛选出所有"是 nn.Linear 且名称后缀匹配目标"的模块
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and _matches(name, target_modules)]
    if not candidates:
        raise ValueError(f"No nn.Linear module matched targets={tuple(target_modules)!r}")
    inserted: List[str] = []
    # 第三步：逐个用 StyleEgoMaskedLoRALinear 替换匹配的 Linear 层
    for name, module in candidates:
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, StyleEgoMaskedLoRALinear(module, rank, alpha, dropout))
        inserted.append(name)
    # 第四步：校验"只有 LoRA 参数可训练"的不变量
    assert_only_lora_trainable(model)
    # 第五步：统计可训练参数量（只会是各风格适配器的 A/B 矩阵）
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return InjectionReport(tuple(inserted), base_parameters, trainable_parameters)


def iter_style_layers(model: nn.Module) -> Iterable[tuple[str, StyleEgoMaskedLoRALinear]]:
    """遍历模型中所有 StyleEgoMaskedLoRALinear 层，产出 (模块名, 模块实例)。"""
    for name, module in model.named_modules():
        if isinstance(module, StyleEgoMaskedLoRALinear):
            yield name, module


def assert_only_lora_trainable(model: nn.Module) -> None:
    """冻结基座不变量校验：确保可训练参数只来自风格适配器的 LoRA A/B 矩阵。

    Raises:
        RuntimeError: 发现非 LoRA 参数也可训练时抛出，防止误训练基座权重。
    """
    invalid = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and not (".aggressive.lora_" in name or ".conservative.lora_" in name)]
    if invalid:
        raise RuntimeError(f"Frozen-base invariant violated; non-LoRA parameters are trainable: {invalid[:8]}")


def frozen_base_hash(model: nn.Module) -> str:
    """只对冻结参数（排除 LoRA 适配器的 A/B 矩阵）计算 SHA-256 哈希。

    用途：训练前后各算一次并比对，可证明基座权重逐字节未被改动，
    发生的仅仅是 LoRA 增量更新。
    """
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".aggressive.lora_" in name or ".conservative.lora_" in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_router(model: nn.Module, style: str, strength: float, enabled: bool = True) -> None:
    """对模型中所有已注入的风格 LoRA 层批量设置路由状态。

    Args:
        model: 已注入 LoRA 的模型。
        style: 风格名称，支持 "aggr"/"aggressive"、"cons"/"conservative"、"normal"。
        strength: 适配器强度（正数-激进，负数-保守，0-关闭）。
        enabled: 是否启用适配器。
    """
    for _, module in iter_style_layers(model):
        module.set_router(style, strength, enabled)

