from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from stylelora.lora.model.injector import iter_style_layers
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.model.scene_context import masked_mean_scene_context
from stylelora.model.scene_gate import (
    SceneStrengthGate,
    load_scene_gate_checkpoint,
    save_scene_gate_checkpoint,
)
from stylelora.scripts.train_scene_gate import _weighted_gate_loss
from stylelora.scripts.build_scene_gate_targets import _candidate_acceptance


class _Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class _Fusion(nn.Module):
    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 与真实 FusionEncoder 相同：首 token 强制有效。
        mask[:, 0] = False
        return value.masked_fill(mask.unsqueeze(-1), 0.0)


class _InnerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = _Fusion()

    def forward(self, inputs: dict) -> dict:
        return {"encoding": self.fusion(inputs["encoding"], inputs["padding_mask"])}


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _InnerEncoder()

    def forward(self, inputs: dict) -> dict:
        return self.encoder(inputs)


class _Decoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp1 = _Mlp(dim)
        self.mlp2 = _Mlp(dim)
        self.final_layer = nn.Module()
        self.final_layer.proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.LayerNorm(dim), nn.Linear(dim, dim),
        )

    def forward(self, encoder_outputs: dict, inputs: dict) -> dict:
        del inputs
        value = self.mlp1(encoder_outputs["encoding"])
        value = self.mlp2(value)
        return {"prediction": self.final_layer.proj(value)}


class _Baseline(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder(dim)

    def forward(self, inputs: dict):
        encoded = self.encoder(inputs)
        return encoded, self.decoder(encoded, inputs)


def _planner_and_inputs() -> tuple[StyleLoRAPlanner, dict]:
    torch.manual_seed(3)
    planner = StyleLoRAPlanner(_Baseline(), rank=2, dropout=0.0)
    with torch.no_grad():
        for _, layer in iter_style_layers(planner.baseline):
            layer.aggressive.lora_B.normal_(0.0, 0.1)
            layer.conservative.lora_B.normal_(0.0, 0.1)
    inputs = {
        "encoding": torch.randn(3, 5, 8),
        "padding_mask": torch.tensor([
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
        ]),
    }
    return planner.eval(), inputs


def _constant_gate(low: float, high: float) -> SceneStrengthGate:
    gate = SceneStrengthGate(hc_dim=8, hidden_dim=4)
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        gate.network[-1].bias.copy_(torch.tensor((math.log(low / (1 - low)),
                                                   math.log(high / (1 - high)))))
    return gate.eval()


def test_effective_rho_preserves_sign_is_monotonic_and_saturates() -> None:
    caps = torch.tensor([[0.25, 0.6]])
    requested = torch.tensor((-1.0, -0.5, -0.1, 0.0, 0.2, 0.8))
    batch_caps = caps.expand(requested.shape[0], -1)
    effective = SceneStrengthGate.effective_rho(batch_caps, requested)
    assert torch.allclose(effective, torch.tensor((-0.25, -0.25, -0.1, 0.0, 0.2, 0.6)))
    assert torch.all(effective[1:] >= effective[:-1])


def test_masked_mean_scene_context_ignores_padding() -> None:
    encoding = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[False, False, True]])
    assert torch.equal(masked_mean_scene_context(encoding, mask), torch.tensor([[2.0]]))


def test_gate_disabled_keeps_original_lora_forward_exactly() -> None:
    planner, inputs = _planner_and_inputs()
    planner.set_strength(0.8)
    planner.attach_scene_gate(_constant_gate(0.25, 0.6), enabled=False)
    _, wrapped = planner(inputs)
    _, direct = planner.baseline(inputs)
    assert torch.equal(wrapped["prediction"], direct["prediction"])
    assert planner.last_scene_gate is None


@pytest.mark.parametrize("rho,cap", [(0.8, 0.6), (-0.8, 0.25)])
def test_enabled_gate_matches_same_fixed_effective_rho(rho: float, cap: float) -> None:
    planner, inputs = _planner_and_inputs()
    planner.attach_scene_gate(_constant_gate(0.25, 0.6), enabled=True).set_strength(rho)
    _, gated = planner(inputs)
    debug = planner.last_scene_gate
    assert debug is not None
    assert torch.allclose(debug["effective_rho"], torch.full((3,), math.copysign(cap, rho)))

    planner.disable_scene_gate().set_strength(math.copysign(cap, rho))
    _, fixed = planner(inputs)
    assert torch.allclose(gated["prediction"], fixed["prediction"], atol=1e-6, rtol=1e-6)


def test_gate_rho_zero_is_exact_baseline_identity() -> None:
    planner, inputs = _planner_and_inputs()
    planner.attach_scene_gate(_constant_gate(0.25, 0.6), enabled=True).set_strength(0.0)
    _, gated = planner(inputs)
    planner.disable_scene_gate().set_strength(0.0)
    _, baseline = planner(inputs)
    assert torch.equal(gated["prediction"], baseline["prediction"])


def test_scene_gate_checkpoint_round_trip(tmp_path) -> None:
    gate = _constant_gate(0.25, 0.6)
    path = tmp_path / "scene_gate.pt"
    save_scene_gate_checkpoint(path, gate, training_config={"seed": 17}, validation={"loss": 0.1})
    loaded, payload = load_scene_gate_checkpoint(path)
    h_c = torch.randn(4, 8)
    assert torch.equal(gate(h_c), loaded(h_c))
    assert payload["format"] == "stylelora.scene_gate.v1"
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_gate_training_penalizes_unsafe_overestimate_more() -> None:
    target = torch.tensor([[0.5, 0.5]])
    confidence = torch.ones(1)
    over = _weighted_gate_loss(torch.tensor([[0.7, 0.7]]), target, confidence)
    under = _weighted_gate_loss(torch.tensor([[0.3, 0.3]]), target, confidence)
    assert over > under


def _acceptance_args() -> SimpleNamespace:
    return SimpleNamespace(
        style_direction_epsilon=1e-3,
        neighbor_distance_tolerance=0.5,
        min_neighbor_distance=1.5,
        lateral_loss_max=0.25,
        low_imitation_tolerance_multiplier=2.0,
        low_progress_tolerance_multiplier=2.0,
        high_jerk_tolerance_multiplier=0.75,
        ade_degradation=0.5,
        fde_degradation=1.0,
        acceleration_degradation=0.1,
        jerk_degradation=0.1,
        max_mean_accel_degradation=0.5,
        max_mean_jerk_degradation=2.0,
        lateral_soft_loss_scale=0.01,
        max_mean_lateral_deviation_m=1.5,
        max_progress_loss_m=2.0,
        soft_budget=1.0,
    )


def test_low_soft_budget_allows_expected_imitation_and_progress_change() -> None:
    baseline = {
        "ade": torch.zeros(1),
        "fde": torch.zeros(1),
        "acceleration": torch.zeros(1),
        "jerk": torch.zeros(1),
        "min_distance": torch.full((1,), 10.0),
    }
    relative = {
        "mean_accel_degradation": torch.zeros(1),
        "mean_jerk_degradation": torch.zeros(1),
        "progress_loss": torch.tensor([2.0]),
        "mean_path_deviation": torch.zeros(1),
    }
    common = dict(
        s_zero=torch.zeros(1),
        ade=torch.tensor([0.5]),
        fde=torch.tensor([1.0]),
        acceleration=torch.zeros(1),
        jerk=torch.zeros(1),
        lateral=torch.zeros(1),
        min_distance=torch.full((1,), 10.0),
        hard_physical=torch.ones(1, dtype=torch.bool),
        relative_metrics=relative,
        baseline_metrics=baseline,
        args=_acceptance_args(),
    )
    _, _, low_cost = _candidate_acceptance(sign=-1.0, s_value=torch.tensor([-0.1]), **common)
    _, _, high_cost = _candidate_acceptance(sign=1.0, s_value=torch.tensor([0.1]), **common)
    assert low_cost < high_cost
