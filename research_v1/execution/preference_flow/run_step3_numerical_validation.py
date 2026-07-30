"""Deterministic numerical validation for the standalone Step-3 Preference Flow.

This script uses only synthetic latent tensors and analytic/test vector fields.
It neither loads a StylePlanner checkpoint nor calls the DPM sampler.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from baseline.model.style_planner.preference_flow import (
    PreferenceFlowConfig,
    PreferenceFlowContractError,
    PreferenceVectorField,
    integrate,
    integrate_from_neutral,
)


_LATENT_DIM = 8
_CONDITION_DIM = 4
_TOLERANCE = 1e-6


class _ConstantField(nn.Module):
    """Test field V(z, c, q, r)=constant."""

    def __init__(self, constant: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("constant", constant.detach().clone())

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
        preference_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        del condition, diffusion_time, preference_coordinate
        return self.constant.to(device=state.device, dtype=state.dtype).expand_as(state)


class _StateConditionField(nn.Module):
    """Synthetic nonlinear field used to prove integration/state dependence."""

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
        preference_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        condition_term = condition[:, :1].expand_as(state)
        return (
            0.35 * state
            + 0.15 * condition_term
            + 0.05 * diffusion_time.unsqueeze(-1)
            + 0.10 * preference_coordinate.unsqueeze(-1)
        )


class _ExponentialStateField(nn.Module):
    """V(z)=z, whose solution from 0 to rho is z(0)*exp(rho)."""

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
        preference_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        del condition, diffusion_time, preference_coordinate
        return state


def _max_abs_error(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise AssertionError(
            f"{label} shape mismatch: {tuple(left.shape)} versus {tuple(right.shape)}"
        )
    return float((left - right).abs().max().item())


def _assert_close(left: torch.Tensor, right: torch.Tensor, *, label: str, tolerance: float = _TOLERANCE) -> float:
    error = _max_abs_error(left, right, label=label)
    if error > tolerance:
        raise AssertionError(f"{label} error {error} exceeds tolerance {tolerance}")
    return error


def _assert_contract_failure(label: str, callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except PreferenceFlowContractError:
        return True
    except Exception as error:
        raise AssertionError(
            f"{label} raised {type(error).__name__}, expected PreferenceFlowContractError"
        ) from error
    raise AssertionError(f"{label} did not reject the invalid contract")


def _config(*, zero_initialize_output: bool) -> PreferenceFlowConfig:
    return PreferenceFlowConfig(
        latent_dim=_LATENT_DIM,
        condition_dim=_CONDITION_DIM,
        hidden_dim=16,
        num_layers=2,
        zero_initialize_output=zero_initialize_output,
        rho_min=-1.0,
        rho_max=1.0,
        default_method="heun",
        default_num_steps=2,
    )


def _make_input(
    *,
    batch_size: int,
    device: torch.device,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.randn(
        (batch_size, _LATENT_DIM),
        device=device,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    condition = torch.randn(
        (batch_size, _CONDITION_DIM),
        device=device,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    time = torch.linspace(0.1, 0.9, batch_size, device=device, dtype=torch.float32)
    return state, condition, time


def _make_deterministic_nonzero_field(device: torch.device) -> PreferenceVectorField:
    field = PreferenceVectorField(_config(zero_initialize_output=False)).to(device)
    with torch.no_grad():
        for module in field.modules():
            if isinstance(module, nn.Linear):
                module.weight.fill_(0.05)
                module.bias.fill_(0.01)
    return field


def _constant_field_checks(device: torch.device) -> Dict[str, Any]:
    constant = torch.full((1, _LATENT_DIM), 0.125, device=device)
    field = _ConstantField(constant).to(device)
    initial = torch.zeros((2, _LATENT_DIM), device=device)
    condition = torch.zeros((2, _CONDITION_DIM), device=device)
    time = torch.tensor([0.2, 0.8], device=device)
    errors = {"positive": 0.0, "negative": 0.0}
    for method in ("euler", "heun"):
        for steps in (1, 2, 4):
            positive_rho = torch.tensor([0.5, 1.0], device=device)
            positive = integrate_from_neutral(
                field,
                initial,
                condition,
                time,
                positive_rho,
                num_steps=steps,
                method=method,
            )
            expected_positive = initial + positive_rho.unsqueeze(-1) * constant
            errors["positive"] = max(
                errors["positive"],
                _assert_close(
                    positive,
                    expected_positive,
                    label=f"constant positive {method}/{steps}",
                ),
            )
            negative_rho = torch.tensor([-0.5, -1.0], device=device)
            negative = integrate_from_neutral(
                field,
                initial,
                condition,
                time,
                negative_rho,
                num_steps=steps,
                method=method,
            )
            expected_negative = initial + negative_rho.unsqueeze(-1) * constant
            errors["negative"] = max(
                errors["negative"],
                _assert_close(
                    negative,
                    expected_negative,
                    label=f"constant negative {method}/{steps}",
                ),
            )
    return errors


def _zero_and_batch_rho_checks(device: torch.device) -> Dict[str, Any]:
    field = _StateConditionField().to(device)
    initial, condition, time = _make_input(batch_size=5, device=device)
    zero_rho = torch.zeros((5,), device=device)
    zero_output = integrate_from_neutral(
        field,
        initial,
        condition,
        time,
        zero_rho,
        num_steps=2,
        method="heun",
    )
    rho_zero_exact = bool(torch.equal(zero_output, initial))
    if not rho_zero_exact:
        raise AssertionError("rho=0 did not return initial_state exactly")

    constant = torch.full((1, _LATENT_DIM), 0.125, device=device)
    constant_field = _ConstantField(constant).to(device)
    batch_initial = torch.zeros_like(initial)
    mixed_rho = torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0], device=device)
    mixed = integrate_from_neutral(
        constant_field,
        batch_initial,
        condition,
        time,
        mixed_rho,
        num_steps=4,
        method="heun",
    )
    expected = batch_initial + mixed_rho.unsqueeze(-1) * constant
    mixed_error = _assert_close(mixed, expected, label="mixed batch rho")
    zero_row_exact = bool(torch.equal(mixed[2], batch_initial[2]))
    if not zero_row_exact:
        raise AssertionError("rho=0 sample changed inside a mixed batch")
    return {
        "rho_zero_exact": rho_zero_exact,
        "mixed_batch_rho_passed": mixed_error <= _TOLERANCE and zero_row_exact,
        "mixed_batch_rho_error": mixed_error,
    }


def _state_dependence_check(device: torch.device) -> Dict[str, Any]:
    field = _StateConditionField().to(device)
    condition = torch.full((2, _CONDITION_DIM), 0.2, device=device)
    time = torch.tensor([0.4, 0.4], device=device)
    coordinate = torch.tensor([0.0, 0.0], device=device)
    initial_a = torch.zeros((2, _LATENT_DIM), device=device)
    initial_b = torch.full((2, _LATENT_DIM), 0.5, device=device)
    velocity_a = field(initial_a, condition, time, coordinate)
    velocity_b = field(initial_b, condition, time, coordinate)
    velocity_change = _max_abs_error(
        velocity_a, velocity_b, label="state-dependent velocity"
    )
    rho = torch.ones((2,), device=device)
    euler_one = integrate_from_neutral(
        field,
        initial_b,
        condition,
        time,
        rho,
        num_steps=1,
        method="euler",
    )
    euler_two = integrate_from_neutral(
        field,
        initial_b,
        condition,
        time,
        rho,
        num_steps=2,
        method="euler",
    )
    intermediate_state_effect = _max_abs_error(
        euler_one, euler_two, label="intermediate state effect"
    )
    passed = velocity_change > 0.0 and intermediate_state_effect > 0.0
    if not passed:
        raise AssertionError("synthetic field did not demonstrate state dependence")
    return {
        "state_dependence_passed": passed,
        "state_velocity_change": velocity_change,
        "intermediate_state_effect": intermediate_state_effect,
    }


def _composition_check(device: torch.device) -> Dict[str, Any]:
    constant = torch.full((1, _LATENT_DIM), 0.125, device=device)
    constant_field = _ConstantField(constant).to(device)
    initial = torch.zeros((2, _LATENT_DIM), device=device)
    condition = torch.zeros((2, _CONDITION_DIM), device=device)
    time = torch.tensor([0.3, 0.7], device=device)
    rho = torch.tensor([-1.0, 0.5], device=device)
    zero = torch.zeros_like(rho)
    full_constant = integrate(
        constant_field, initial, condition, time, zero, rho, 4, "heun"
    )
    half_constant = integrate(
        constant_field, initial, condition, time, zero, rho / 2.0, 2, "heun"
    )
    composed_constant = integrate(
        constant_field,
        half_constant,
        condition,
        time,
        rho / 2.0,
        rho,
        2,
        "heun",
    )
    constant_exact = bool(torch.equal(full_constant, composed_constant))
    if not constant_exact:
        raise AssertionError("constant-field composition was not exactly equal")

    nonlinear = _ExponentialStateField().to(device)
    nonlinear_initial = torch.full((2, _LATENT_DIM), 0.25, device=device)
    full_nonlinear = integrate(
        nonlinear, nonlinear_initial, condition, time, zero, rho, 4, "heun"
    )
    half_nonlinear = integrate(
        nonlinear,
        nonlinear_initial,
        condition,
        time,
        zero,
        rho / 2.0,
        2,
        "heun",
    )
    composed_nonlinear = integrate(
        nonlinear,
        half_nonlinear,
        condition,
        time,
        rho / 2.0,
        rho,
        2,
        "heun",
    )
    nonlinear_error = _assert_close(
        full_nonlinear,
        composed_nonlinear,
        label="nonlinear composition",
    )
    return {
        "composition_passed": constant_exact and nonlinear_error <= _TOLERANCE,
        "nonlinear_composition_error": nonlinear_error,
    }


def _solver_convergence_check(device: torch.device) -> Dict[str, Any]:
    field = _ExponentialStateField().to(device)
    initial = torch.ones((1, _LATENT_DIM), device=device)
    condition = torch.zeros((1, _CONDITION_DIM), device=device)
    time = torch.tensor([0.5], device=device)
    rho = torch.tensor([1.0], device=device)
    exact = initial * torch.exp(rho).unsqueeze(-1)
    euler_one = integrate_from_neutral(
        field, initial, condition, time, rho, num_steps=1, method="euler"
    )
    heun_two = integrate_from_neutral(
        field, initial, condition, time, rho, num_steps=2, method="heun"
    )
    heun_four = integrate_from_neutral(
        field, initial, condition, time, rho, num_steps=4, method="heun"
    )
    euler_one_error = _max_abs_error(euler_one, exact, label="one-step Euler")
    heun_two_error = _max_abs_error(heun_two, exact, label="two-step Heun")
    heun_four_error = _max_abs_error(heun_four, exact, label="four-step Heun")
    passed = (
        heun_four_error <= heun_two_error + 1e-7
        and heun_two_error <= euler_one_error + 1e-7
    )
    if not passed:
        raise AssertionError(
            "solver convergence order failed: "
            f"Euler1={euler_one_error}, Heun2={heun_two_error}, "
            f"Heun4={heun_four_error}"
        )
    return {
        "solver_convergence_passed": passed,
        "one_step_euler_error": euler_one_error,
        "two_step_heun_error": heun_two_error,
        "four_step_heun_error": heun_four_error,
    }


def _gradient_and_input_checks(device: torch.device) -> Dict[str, Any]:
    field = _make_deterministic_nonzero_field(device)
    state, condition, time = _make_input(
        batch_size=3,
        device=device,
        requires_grad=True,
    )
    coordinate = torch.tensor([0.1, 0.2, 0.3], device=device)
    base_velocity = field(state, condition, time, coordinate)
    state_velocity = field(state + 0.25, condition, time, coordinate)
    condition_velocity = field(state, condition + 0.25, time, coordinate)
    time_velocity = field(state, condition, time + 0.1, coordinate)
    coordinate_velocity = field(state, condition, time, coordinate + 0.1)
    input_changes = {
        "state": _max_abs_error(base_velocity, state_velocity, label="field state input"),
        "condition": _max_abs_error(base_velocity, condition_velocity, label="field condition input"),
        "diffusion_time": _max_abs_error(base_velocity, time_velocity, label="field time input"),
        "preference_coordinate": _max_abs_error(base_velocity, coordinate_velocity, label="field coordinate input"),
    }
    if not all(value > 0.0 for value in input_changes.values()):
        raise AssertionError(f"PreferenceVectorField ignored an input: {input_changes}")

    rho = torch.tensor([-0.5, 0.25, 0.75], device=device)
    output = integrate_from_neutral(
        field,
        state,
        condition,
        time,
        rho,
        num_steps=2,
        method="heun",
    )
    output.square().mean().backward()
    parameter_gradients = [parameter.grad for parameter in field.parameters()]
    parameter_grad_finite = all(
        gradient is not None and bool(torch.isfinite(gradient).all().item())
        for gradient in parameter_gradients
    )
    state_grad_finite = state.grad is not None and bool(torch.isfinite(state.grad).all().item())
    condition_grad_finite = (
        condition.grad is not None and bool(torch.isfinite(condition.grad).all().item())
    )
    gradients_nonzero = (
        any(float(gradient.abs().sum().item()) > 0.0 for gradient in parameter_gradients if gradient is not None)
        and float(state.grad.abs().sum().item()) > 0.0
        and float(condition.grad.abs().sum().item()) > 0.0
    )
    gradient_passed = (
        parameter_grad_finite
        and state_grad_finite
        and condition_grad_finite
        and gradients_nonzero
    )
    if not gradient_passed:
        raise AssertionError("Preference Flow gradients are missing, non-finite, or zero")

    signature = inspect.signature(PreferenceVectorField.forward)
    rho_not_in_field = (
        "rho" not in signature.parameters
        and "preference_coordinate" in signature.parameters
    )
    if not rho_not_in_field:
        raise AssertionError("PreferenceVectorField incorrectly exposes rho")
    return {
        "gradient_passed": gradient_passed,
        "vector_field_uses_all_inputs": True,
        "rho_not_in_vector_field": rho_not_in_field,
        "input_velocity_changes": input_changes,
    }


def _zero_initialization_check(device: torch.device) -> Dict[str, Any]:
    config = _config(zero_initialize_output=True)
    field = PreferenceVectorField(config).to(device)
    state, condition, time = _make_input(batch_size=3, device=device)
    coordinate = torch.tensor([-0.5, 0.0, 0.75], device=device)
    velocity = field(state, condition, time, coordinate)
    velocity_zero = bool(torch.equal(velocity, torch.zeros_like(velocity)))
    rho = torch.tensor([-1.0, 0.0, 1.0], device=device)
    integrated = integrate_from_neutral(field, state, condition, time, rho)
    integration_identity = bool(torch.equal(integrated, state))
    if not velocity_zero or not integration_identity:
        raise AssertionError("zero-initialized PreferenceVectorField is not identity")
    return {"zero_initialization_passed": True}


def _contract_failure_check(device: torch.device) -> Dict[str, Any]:
    config = _config(zero_initialize_output=False)
    field = PreferenceVectorField(config).to(device)
    state, condition, time = _make_input(batch_size=2, device=device)
    rho = torch.tensor([0.25, -0.25], device=device)
    failures = {
        "wrong_latent_dim": _assert_contract_failure(
            "wrong latent dimension",
            lambda: field(state[:, :-1], condition, time, rho),
        ),
        "batch_mismatch": _assert_contract_failure(
            "batch mismatch",
            lambda: field(state, condition[:1], time, rho),
        ),
        "dtype_mismatch": _assert_contract_failure(
            "dtype mismatch",
            lambda: field(state, condition.to(torch.float64), time, rho),
        ),
        "device_mismatch": _assert_contract_failure(
            "device mismatch",
            lambda: field(
                state,
                torch.empty(
                    (2, _CONDITION_DIM), device="meta", dtype=state.dtype
                ),
                time,
                rho,
            ),
        ),
        "nan": _assert_contract_failure(
            "NaN input",
            lambda: field(
                torch.cat(
                    (torch.full_like(state[:1], float("nan")), state[1:]), dim=0
                ),
                condition,
                time,
                rho,
            ),
        ),
        "inf": _assert_contract_failure(
            "Inf input",
            lambda: field(
                torch.cat(
                    (torch.full_like(state[:1], float("inf")), state[1:]), dim=0
                ),
                condition,
                time,
                rho,
            ),
        ),
        "invalid_solver": _assert_contract_failure(
            "invalid solver",
            lambda: integrate_from_neutral(
                field, state, condition, time, rho, method="rk4"
            ),
        ),
        "non_positive_steps": _assert_contract_failure(
            "non-positive steps",
            lambda: integrate_from_neutral(
                field, state, condition, time, rho, num_steps=0
            ),
        ),
        "rho_out_of_range": _assert_contract_failure(
            "rho out of range",
            lambda: integrate_from_neutral(
                field,
                state,
                condition,
                time,
                torch.tensor([1.01, 0.0], device=device),
            ),
        ),
    }
    if not all(failures.values()):
        raise AssertionError(f"contract failure coverage incomplete: {failures}")
    return {"contract_failure_passed": True, "contract_failures": failures}


def run_numerical_validation(
    *,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Run every Step-3 numerical contract and return JSON-safe measurements."""

    torch.manual_seed(int(seed))
    constant = _constant_field_checks(device)
    zero_and_batch = _zero_and_batch_rho_checks(device)
    state_dependence = _state_dependence_check(device)
    composition = _composition_check(device)
    convergence = _solver_convergence_check(device)
    gradient = _gradient_and_input_checks(device)
    zero_initialization = _zero_initialization_check(device)
    contracts = _contract_failure_check(device)
    report: Dict[str, Any] = {
        "schema_version": "preference_flow_step3_numerics_v1",
        "seed": int(seed),
        "device": str(device),
        "latent_dim": _LATENT_DIM,
        "default_solver": "heun",
        "default_steps": 2,
        "integration_sign_convention": "z_end = z_start + integral(V dr)",
        "rho_zero_exact": zero_and_batch["rho_zero_exact"],
        "constant_field_positive_error": constant["positive"],
        "constant_field_negative_error": constant["negative"],
        "mixed_batch_rho_passed": zero_and_batch["mixed_batch_rho_passed"],
        "mixed_batch_rho_error": zero_and_batch["mixed_batch_rho_error"],
        **state_dependence,
        **composition,
        **convergence,
        **gradient,
        **zero_initialization,
        **contracts,
    }
    required_passes = (
        "rho_zero_exact",
        "mixed_batch_rho_passed",
        "state_dependence_passed",
        "composition_passed",
        "solver_convergence_passed",
        "gradient_passed",
        "zero_initialization_passed",
        "contract_failure_passed",
        "rho_not_in_vector_field",
    )
    report["passed"] = all(bool(report[key]) for key in required_passes)
    if not report["passed"]:
        raise AssertionError(f"Step-3 numerical validation failed: {report}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run synthetic Step-3 Preference Flow numerical validation."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_report(path: Path, report: Dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing numerical report: {path}. "
            "Pass --overwrite only when replacement is intended."
        )
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device={device} was requested but CUDA is unavailable")
    report = run_numerical_validation(seed=int(args.seed), device=device)
    path = Path(args.output_dir).expanduser() / "step3_preference_flow_numerics.json"
    _write_report(path, report, overwrite=bool(args.overwrite))
    print(f"Step-3 Preference Flow numerical validation passed: {path}")


if __name__ == "__main__":
    main()
