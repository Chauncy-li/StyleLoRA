"""Paired-seed rollout helpers; actual DPM-Solver remains inside the baseline decoder.

成对种子（paired-seed）的 rollout 辅助函数；真正的 DPM-Solver 多步去噪仍由基线
解码器内部完成。
这里提供的核心能力：在**完全相同的随机种子**下强制执行 rollout——
因为推理时解码器会在内部采样初始噪声 xT，只有固定种子才能保证不同风格强度 rho
的 rollout 之间差异完全来自 LoRA 增量，而不是采样随机性。
"""

from __future__ import annotations

import time
from typing import Dict

import torch


@torch.no_grad()
def rollout_with_rho(planner, inputs: Dict[str, torch.Tensor], rho: float, *, seed: int) -> tuple[Dict, float]:
    """在指定风格强度 rho 与固定随机种子下执行一次无梯度 rollout（轨迹生成）。

    Args:
        planner: 风格 LoRA 规划器。
        inputs: 输入字典（推理输入，需含 ego_current_state 等字段）。
        rho: 风格强度（正数=激进，负数=保守，0=正常/基线）。
        seed: 随机种子，用于锁定采样噪声，保证不同 rho 之间的对比公平。

    Returns:
        (输出字典, 耗时秒数) 二元组：
        - 输出字典为解码器推理输出（如 {"prediction": ...}）；
        - 耗时包含 GPU 同步等待（若在 CUDA 上），为真实墙钟时间。
    """
    # 切换风格强度路由
    planner.set_strength(rho)
    # 推理输入所在的设备；若在 CUDA 上收集设备索引用于 fork RNG
    input_device = inputs["ego_current_state"].device
    devices = [input_device.index if input_device.index is not None else torch.cuda.current_device()] if input_device.type == "cuda" else []
    # 在 fork 出的 RNG 环境中设置固定种子，保证这次 rollout 的采样可复现
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        if devices:
            torch.cuda.synchronize(devices[0])
        # 计时开始（同步后会等待排队的内核执行完，确保测得的是真实耗时）
        start = time.perf_counter()
        _, output = planner(inputs)
        # 计时结束前再次同步，确保 GPU 异步执行完成
        if devices:
            torch.cuda.synchronize(devices[0])
    return output, time.perf_counter() - start


def rho_grid() -> tuple[float, ...]:
    """返回标准的 rho 扫描网格：从 -1.0（最保守）到 +1.0（最激进），步长 0.25。"""
    return (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)

