"""A transparent planner wrapper; its forward contract is the original DiffPlanner contract.

透明的规划器包装层：其 forward 契约与原 DiffPlanner 完全一致。
它负责在基线模型上注入风格化 LoRA 层，并提供以下能力：
- 风格路由切换（aggressive / conservative / normal）与强度调节；
- 冻结基座权重的完整性校验（哈希比对）；
- 单风格分支适配器的保存与加载；
- 与独立加载的未修改基线做身份一致性（identity）误差检测。
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn

from research_lora.model.injector import DEFAULT_TARGETS, InjectionReport, frozen_base_hash, inject_style_lora, set_router


class StyleLoRAPlanner(nn.Module):
    """风格 LoRA 规划器：包装基线 DiffPlanner，注入互斥的激进/保守 LoRA 适配器。

    该包装器的 forward 与原 DiffPlanner 保持一致（透明代理），
    额外提供风格切换、强度调节、冻结基座校验与适配器存取管理等能力。
    """

    def __init__(self, baseline: nn.Module, *, rank: int = 4, alpha: float | None = None,
                 dropout: float = 0.0, target_modules: Sequence[str] = DEFAULT_TARGETS) -> None:
        """初始化：在基线模型上注入风格 LoRA 层。

        Args:
            baseline: 已加载好权重的基线 DiffPlanner 模型。
            rank: LoRA 低秩矩阵的秩。
            alpha: LoRA 缩放系数，默认取 rank。
            dropout: LoRA 增量的 dropout 比率。
            target_modules: 需要替换（注入 LoRA）的目标模块后缀列表。
        """
        super().__init__()
        # 保存原始基线模型引用，后续所有操作都作用于它
        self.baseline = baseline
        # 核心步骤：把风格 LoRA 层注入到基线的指定线性层中，并返回注入报告
        self.report: InjectionReport = inject_style_lora(
            self.baseline, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules
        )
        # 初始路由状态：normal 风格、强度 0（即与基线完全一致）
        self._style, self._strength, self._enabled = "normal", 0.0, True
        # 记录注入后的冻结基座哈希，用于训练前后证明基座权重未被改动
        self._frozen_base_hash = frozen_base_hash(self.baseline)
        # 把初始路由状态批量应用到所有已注入的风格 LoRA 层上
        self._apply_router()

    @property
    def sde(self):
        """透传属性：直接暴露基线的 SDE（随机微分方程）对象。"""
        return self.baseline.sde

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """透明代理：forward 契约与原 DiffPlanner 完全一致，不做任何包装变换。"""
        return self.baseline(inputs)

    def train(self, mode: bool = True):
        """Keep all frozen DiffPlanner modules deterministic; train only adapter dropout/matrices.

        训练模式控制：保持所有冻结的 DiffPlanner 模块处于确定性（eval）状态，
        只允许适配器（aggressive/conservative 分支）的 dropout 与 LoRA 矩阵参与训练。
        """
        super().train(mode)
        # 基线整体始终处于 eval 模式，确保 BatchNorm/Dropout 不改变基座行为
        self.baseline.eval()
        # 逐个遍历基线子模块，找到风格 LoRA 层（含 aggressive/conservative 属性）：
        for _, layer in self.baseline.named_modules():
            if hasattr(layer, "aggressive") and hasattr(layer, "conservative"):
                # 只让两个适配器分支跟随训练/评估开关
                layer.aggressive.train(mode)
                layer.conservative.train(mode)
                # 但适配器内部的共享基座 linear 仍然保持 eval（冻结且确定性）
                layer.aggressive.base.eval()
                layer.conservative.base.eval()
        return self

    def _apply_router(self) -> None:
        """把当前内部的风格/强度/开关状态批量下发给所有风格 LoRA 层。"""
        set_router(self.baseline, self._style, self._strength, self._enabled)

    def set_style(self, style: str) -> "StyleLoRAPlanner":
        """设置风格模式（aggr/aggressive、cons/conservative、normal）。

        - aggr：激进风格，强度取正数（若当前为 0 则默认 1.0）；
        - cons：保守风格，强度取负数（若当前为 0 则默认 -1.0）；
        - normal：正常模式，强度置 0，输出与基线完全一致。
        """
        # 别名映射：aggressive -> aggr，conservative -> cons
        aliases = {"aggressive": "aggr", "conservative": "cons"}
        self._style = aliases.get(style, style)
        # 校验风格合法性
        if self._style not in {"aggr", "cons", "normal"}:
            raise ValueError("style must be aggr/aggressive, cons/conservative, or normal")
        # 根据风格自动决定强度的符号方向
        if self._style == "aggr":
            self._strength = abs(self._strength) or 1.0
        elif self._style == "cons":
            self._strength = -(abs(self._strength) or 1.0)
        else:
            self._strength = 0.0
        self._apply_router()
        return self

    def set_strength(self, rho: float) -> "StyleLoRAPlanner":
        """按强度 rho 的符号设置路由：正数=激进风格，负数=保守风格，0=正常模式。

        Args:
            rho: 适配器强度（rho 越大风格越激进，越小越保守）。
        """
        rho = float(rho)
        # 符号 > 0：切换到激进风格并使用该强度
        if rho > 0:
            self._style, self._strength = "aggr", rho
        # 符号 < 0：切换到保守风格并使用该强度（负值）
        elif rho < 0:
            self._style, self._strength = "cons", rho
        # 等于 0：回到 normal，与基线完全一致
        else:
            self._style, self._strength = "normal", 0.0
        self._apply_router()
        return self

    def enable_adapter(self) -> None:
        """启用 LoRA 适配器（使风格增量生效）。"""
        self._enabled = True
        self._apply_router()

    def disable_adapter(self) -> None:
        """禁用 LoRA 适配器（输出与基线完全一致）。"""
        self._enabled = False
        self._apply_router()

    @torch.no_grad()
    def base_identity_error(self, fixed_noisy_inputs: Dict[str, torch.Tensor], reference_baseline: nn.Module) -> float:
        """在同一固定扩散输入上比较关闭 LoRA 的包装器与独立加载的基线。

        ``fixed_noisy_inputs`` 已包含 ``sampled_trajectories`` 和
        ``diffusion_time``，因此 decoder 只执行单次固定的去噪前向，不会在
        内部重新采样初始噪声。这样测得的误差只用于检验 LoRA 包装是否改变基线。
        """
        # 保存当前路由状态，检测结束后恢复，避免改变调用者的风格设置。
        old_style, old_strength, old_enabled = self._style, self._strength, self._enabled
        try:
            # rho=0 时，每个注入层直接返回其原始 Linear 层的输出。
            self.disable_adapter()
            wrapped = self(fixed_noisy_inputs)
            reference_baseline.eval()
            direct = reference_baseline(fixed_noisy_inputs)
        finally:
            self._style, self._strength, self._enabled = old_style, old_strength, old_enabled
            self._apply_router()
        # 递归计算两次输出（可能是 dict / tuple / list / tensor）的最大绝对误差。
        return _max_nested_abs_error(wrapped, direct)

    def trainable_parameter_report(self) -> Dict[str, Any]:
        """返回 LoRA 注入参数统计报告（层名、基座参数量、可训练参数量与占比）。"""
        return {"layers": list(self.report.layers), "base_parameters": self.report.base_parameters,
                "trainable_parameters": self.report.trainable_parameters,
                "trainable_ratio": self.report.trainable_ratio}

    def assert_frozen_base_unchanged(self) -> None:
        """校验冻结基座是否被改动：重新计算基座哈希并与初始化时记录值比对。

        Raises:
            RuntimeError: 若基座权重在 LoRA 训练期间发生变化，则抛出异常。
        """
        if frozen_base_hash(self.baseline) != self._frozen_base_hash:
            raise RuntimeError("Frozen DiffPlanner weights changed during LoRA training")

    def adapter_state_dict(self, style: str) -> Dict[str, torch.Tensor]:
        """导出指定风格分支的 LoRA 适配器参数（A/B 矩阵）为 CPU 张量字典。

        Args:
            style: 风格名，支持 aggressive/aggr 或 conservative/cons。

        Returns:
            仅包含该风格分支 .lora_ 参数的 state_dict（已移动到 CPU 并 detach）。
        """
        # 把风格别名归一化为完整分支名
        branch = {"aggressive": "aggressive", "aggr": "aggressive", "conservative": "conservative", "cons": "conservative"}.get(style)
        if branch is None:
            raise ValueError("Adapter checkpoints must name aggressive or conservative")
        # 过滤出属于该风格分支的 LoRA 参数（形如 ...<branch>.lora_A/lora_B）
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if f".{branch}.lora_" in name}

    def load_adapter_state_dict(self, state: Dict[str, torch.Tensor], style: str) -> None:
        """严格加载指定风格分支的适配器权重，防止混入其他风格或无关参数。

        Args:
            state: 待加载的适配器 state_dict。
            style: 声明的风格分支名（aggressive/aggr 或 conservative/cons）。

        Raises:
            RuntimeError: 若状态字典为空、包含非本分支的 lora_ 键，或
                存在缺失/多余的键时，均视为不匹配并抛出异常。
        """
        # 别名归一化
        branch = {"aggressive": "aggressive", "aggr": "aggressive", "conservative": "conservative", "cons": "conservative"}.get(style)
        # 状态字典必须非空，且所有键都必须属于声明的分支
        if branch is None or not state or any(f".{branch}.lora_" not in name for name in state):
            raise RuntimeError("Adapter state does not exclusively match its declared style branch")
        # 宽容加载：missing 为本模型有但 state 没有的键，unexpected 反之
        missing, unexpected = self.load_state_dict(state, strict=False)
        # 本分支的 lora_ 键缺失即为校验失败
        invalid_missing = [key for key in missing if f".{branch}.lora_" in key]
        if invalid_missing or unexpected:
            raise RuntimeError(f"Adapter checkpoint mismatch; missing={invalid_missing[:8]}, unexpected={unexpected[:8]}")


def _max_nested_abs_error(a: Any, b: Any) -> float:
    """递归计算嵌套结构（张量 / dict / tuple / list）中所有张量的最大绝对误差。

    Args:
        a: 第一个输出结构（可为 tensor、dict、tuple、list 或其他）。
        b: 与 a 结构对应的第二个输出结构。

    Returns:
        全部张量位置上的最大 |a - b|，无张量时返回 0.0。
    """
    if torch.is_tensor(a):
        # 张量：直接求最大绝对误差
        return float((a - b).abs().max().item())
    if isinstance(a, dict):
        # 字典：递归遍历所有（可能是 tensor 或嵌套容器）的值
        return max((_max_nested_abs_error(a[k], b[k]) for k in a if torch.is_tensor(a[k]) or isinstance(a[k], (dict, tuple, list))), default=0.0)
    if isinstance(a, (tuple, list)):
        # 序列：逐元素递归比较
        return max((_max_nested_abs_error(x, y) for x, y in zip(a, b)), default=0.0)
    # 其他类型（如字符串、标量元信息）不参与误差计算
    return 0.0
