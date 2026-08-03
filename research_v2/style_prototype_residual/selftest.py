"""Pure contract checks for the same-state residual prototype modules.

This deliberately does not load a checkpoint or cache.  Run it before the
server data stages to catch a broken zero-control identity immediately.
"""

from __future__ import annotations

import json

import torch

from research_v2.style_prototype_residual.core import (
    DirectEmbeddingControl,
    NetworkConfig,
    PrototypeControl,
    RawResidualExecutor,
)
from research_v2.style_prototype_residual.data import STYLE_TO_INDEX


def main() -> None:
    torch.manual_seed(3407)
    config = NetworkConfig(future_len=8, scene_dim=16, latent_dim=8, hidden_dim=32)
    executor = RawResidualExecutor(config).eval()
    base = torch.randn(3, config.future_len, 4)
    scene = torch.randn(3, config.scene_dim)
    time = torch.tensor([0.2, 0.5, 0.8])
    zero = torch.zeros(3, config.latent_dim)
    exact = executor.control(base, scene, time, zero)
    direct = DirectEmbeddingControl(config.latent_dim).eval()
    direct_normal = direct.for_labels(torch.tensor([STYLE_TO_INDEX["norm"]]))
    prototypes = torch.randn(3, config.latent_dim)
    prototype = PrototypeControl(prototypes).eval()
    prototype_normal = prototype.for_labels(torch.tensor([STYLE_TO_INDEX["norm"]]))
    control_a = executor.control(base, scene, time, torch.randn_like(zero))
    report = {
        "raw_executor_zero_is_elementwise_exact": bool(torch.equal(exact, torch.zeros_like(exact))),
        "direct_normal_control_is_elementwise_exact": bool(torch.equal(direct_normal, torch.zeros_like(direct_normal))),
        "prototype_normal_control_is_elementwise_exact": bool(torch.equal(prototype_normal, torch.zeros_like(prototype_normal))),
        "nonzero_control_is_finite": bool(torch.isfinite(control_a).all().item()),
        "executor_has_dropout": any(isinstance(module, torch.nn.Dropout) for module in executor.modules()),
    }
    report["passed"] = bool(
        report["raw_executor_zero_is_elementwise_exact"]
        and report["direct_normal_control_is_elementwise_exact"]
        and report["prototype_normal_control_is_elementwise_exact"]
        and report["nonzero_control_is_finite"]
        and not report["executor_has_dropout"]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise AssertionError("style prototype residual selftest failed")


if __name__ == "__main__":
    main()
