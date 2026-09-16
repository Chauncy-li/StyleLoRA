"""Non-invasive bridge for an already constructed baseline NuPlan simulation planner."""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch

from baseline.simulation.planner import DiffusionPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from stylelora.lora.model.checkpoint import load_adapter_checkpoint
from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner
from stylelora.lora.safety.trajectory_acceptance import BaselineRelativeCandidateValidator
from stylelora.lora.safety.trajectory_repair import BaselineAnchoredTrajectoryRepair
from stylelora.model.conditional_lora_router import load_conditional_router_checkpoint
from stylelora.model.scene_gate import load_scene_gate_checkpoint
from stylelora.runtime.preference_state import PreferenceState


def _synchronize_model_device(model) -> None:
    """Synchronize only for optional runtime measurements on a CUDA planner."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def attach_lora_to_simulation_planner(simulation_planner, aggressive_adapter: str, conservative_adapter: str, *, baseline_checkpoint: str,
                                      normalization_file: str, rank: int = 4, alpha: float | None = None,
                                      rho: float = 0.0, enable_scene_gate: bool = False,
                                      scene_gate_checkpoint: str | None = None,
                                      enable_conditional_router: bool = False,
                                      conditional_router_checkpoint: str | None = None) -> StyleLoRAPlanner:
    """Replace only the planner instance's in-memory model after its normal baseline loading.

    The caller remains responsible for constructing the existing baseline planner,
    so NuPlan configuration, feature processing, SDE and DPM-Solver are unchanged.
    """
    if not hasattr(simulation_planner, "_planner"):
        raise TypeError("Expected a constructed baseline simulation planner with a `_planner` model")
    baseline = simulation_planner._planner
    try:
        model_device = next(baseline.parameters()).device
    except StopIteration:
        model_device = getattr(simulation_planner, "_device", "cpu")
    wrapped = StyleLoRAPlanner(baseline, rank=rank, alpha=alpha)
    load_adapter_checkpoint(aggressive_adapter, wrapped, baseline_checkpoint=baseline_checkpoint,
                            normalization_file=normalization_file)
    load_adapter_checkpoint(conservative_adapter, wrapped, baseline_checkpoint=baseline_checkpoint,
                            normalization_file=normalization_file)
    # The parent NuPlan planner moves the baseline to CUDA before this wrapper is
    # attached, while newly-created LoRA matrices start on CPU.  Keep every
    # parameter on the baseline device before the first simulation inference.
    wrapped.to(model_device).set_strength(rho)
    if enable_conditional_router:
        if not conditional_router_checkpoint:
            raise ValueError("启用条件 LoRA 路由时必须提供 conditional_router_checkpoint")
        router, prototypes, _ = load_conditional_router_checkpoint(
            conditional_router_checkpoint, model_device
        )
        wrapped.attach_conditional_router(router, prototypes, enabled=True, trainable=False)
    if enable_scene_gate:
        if not scene_gate_checkpoint:
            raise ValueError("启用场景门控时必须提供 scene_gate_checkpoint")
        gate, _ = load_scene_gate_checkpoint(scene_gate_checkpoint, model_device)
        wrapped.attach_scene_gate(gate, enabled=True)
    wrapped.eval()
    simulation_planner._planner = wrapped
    return wrapped


class LoRADiffusionPlanner(DiffusionPlanner):
    """Pickle-safe NuPlan planner that attaches LoRA after baseline loading.

    This class must stay at module scope because NuPlan serializes the planner
    into each simulation log.  A class created inside ``__new__`` cannot be
    pickled and would make an otherwise successful scenario count as failed.
    """
    # 保存场景 token 仅用于一次采集后的严格共同场景筛选，不参与模型推理。
    requires_scenario: bool = True

    def _export_step_bundle(self, current_input, raw_inputs, outputs):
        """Export only arrays required by closed-loop style analysis.

        The baseline debug exporter also stores lanes, route polylines,
        static objects and candidate banks.  A formal multi-rho run can
        contain tens of thousands of steps, so duplicating those arrays
        is unnecessary and expensive.  This override deliberately keeps
        the baseline package untouched.
        """
        prediction = outputs.get("prediction")
        if prediction is None or prediction.ndim != 4 or self._scenario_raw_dir is None:
            return
        step_index = self._step_export_index
        self._step_export_index += 1
        time_us = int(current_input.history.ego_states[-1].time_point.time_us)
        iteration_index = int(current_input.iteration.index)
        file_name = f"step_{step_index:06d}_iter_{iteration_index:04d}_{time_us}.npz"
        prediction_np = prediction[0].detach().cpu().numpy().astype(np.float32)

        def _first(key):
            value = raw_inputs.get(key)
            if value is None:
                return np.zeros((0,), dtype=np.float32)
            return value[0].detach().cpu().numpy().astype(np.float32)

        neighbor_past = _first("neighbor_agents_past")
        neighbor_current = (
            neighbor_past[:, -1]
            if neighbor_past.ndim == 3 and neighbor_past.shape[1]
            else np.zeros((0, 2), dtype=np.float32)
        )
        gate_debug = getattr(self._planner, "last_scene_gate", None)
        router_debug = getattr(self._planner, "last_conditional_router", None)
        bounded_debug = self._last_bounded_style
        repair_debug = self._last_trajectory_repair
        if bounded_debug is not None:
            bounded_effective_rho = float(bounded_debug.get("accepted_rho", 0.0))
        elif gate_debug is not None:
            gate_effective = gate_debug["effective_rho"]
            if hasattr(gate_effective, "reshape"):
                gate_effective = gate_effective.reshape(-1)[0]
            bounded_effective_rho = float(gate_effective)
        else:
            bounded_effective_rho = float(self._lora_rho)

        def _gate_scalar(key: str, default: float) -> np.ndarray:
            if gate_debug is None:
                return np.asarray(default, dtype=np.float32)
            value = gate_debug[key]
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[0]
            return np.asarray(float(value), dtype=np.float32)

        def _router_scalar(key: str, default: float) -> np.ndarray:
            if router_debug is None:
                return np.asarray(default, dtype=np.float32)
            value = router_debug[key]
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[0]
            return np.asarray(float(value), dtype=np.float32)

        np.savez_compressed(
            os.path.join(self._scenario_raw_dir, file_name),
            scenario_token=np.asarray(self._scenario_token),
            step_index=np.asarray(step_index, dtype=np.int64),
            iteration_index=np.asarray(iteration_index, dtype=np.int64),
            time_us=np.asarray(time_us, dtype=np.int64),
            ego_current_state=_first("ego_current_state"),
            neighbor_current_state=neighbor_current,
            generated_ego_future=(
                prediction_np[0] if prediction_np.shape[0] else np.zeros((0, 4), dtype=np.float32)
            ),
            generated_neighbor_future=(
                prediction_np[1:] if prediction_np.shape[0] > 1 else np.zeros((0, 0, 4), dtype=np.float32)
            ),
            scene_gate_enabled=np.asarray(bool(gate_debug is not None), dtype=np.bool_),
            requested_rho=np.asarray(float(self._lora_rho), dtype=np.float32),
            effective_rho=np.asarray(bounded_effective_rho, dtype=np.float32),
            gate_cap_low=_gate_scalar("cap_low", np.nan),
            gate_cap_high=_gate_scalar("cap_high", np.nan),
            conditional_router_enabled=np.asarray(bool(router_debug is not None), dtype=np.bool_),
            conditional_coefficient_mean=_router_scalar("coefficient_mean", np.nan),
            conditional_coefficient_std=_router_scalar("coefficient_std", np.nan),
            bounded_style_enabled=np.asarray(
                bool(bounded_debug is not None), dtype=np.bool_
            ),
            initial_requested_rho=np.asarray(
                float(bounded_debug.get("initial_requested_rho", np.nan))
                if bounded_debug else np.nan,
                dtype=np.float32,
            ),
            accepted_rho=np.asarray(
                float(bounded_debug.get("accepted_rho", np.nan))
                if bounded_debug else np.nan,
                dtype=np.float32,
            ),
            candidate_attempts=np.asarray(
                int(bounded_debug.get("candidate_attempts", 0)) if bounded_debug else 0,
                dtype=np.int64,
            ),
            candidate_accepted=np.asarray(
                bool(bounded_debug.get("candidate_accepted", False)) if bounded_debug else False,
                dtype=np.bool_,
            ),
            baseline_fallback=np.asarray(
                bool(bounded_debug.get("baseline_fallback", False)) if bounded_debug else False,
                dtype=np.bool_,
            ),
            baseline_hard_valid=np.asarray(
                bool(bounded_debug.get("baseline_hard_valid", False)) if bounded_debug else False,
                dtype=np.bool_,
            ),
            candidate_failure_reasons=np.asarray(
                bounded_debug.get("failure_reasons", []) if bounded_debug else [],
                dtype="U64",
            ),
            trajectory_repair_enabled=np.asarray(
                bool(repair_debug is not None), dtype=np.bool_
            ),
            trajectory_repair_applied=np.asarray(
                bool(repair_debug.get("applied", False)) if repair_debug else False,
                dtype=np.bool_,
            ),
            trajectory_repair_longitudinal_scale=np.asarray(
                float(repair_debug.get("longitudinal_scale", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_lateral_scale=np.asarray(
                float(repair_debug.get("lateral_scale", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_baseline_fallback=np.asarray(
                bool(repair_debug.get("baseline_fallback", False)) if repair_debug else False,
                dtype=np.bool_,
            ),
            trajectory_repair_candidate_attempts=np.asarray(
                int(repair_debug.get("candidate_attempts", 0)) if repair_debug else 0,
                dtype=np.int64,
            ),
            trajectory_repair_baseline_hard_valid=np.asarray(
                bool(repair_debug.get("baseline_hard_valid", False)) if repair_debug else False,
                dtype=np.bool_,
            ),
            trajectory_repair_failure_reasons=np.asarray(
                repair_debug.get("failure_reasons", []) if repair_debug else [],
                dtype="U64",
            ),
            trajectory_repair_collision_triggered=np.asarray(
                bool(repair_debug.get("collision_triggered", False)) if repair_debug else False,
                dtype=np.bool_,
            ),
            trajectory_repair_drivable_triggered=np.asarray(
                bool(repair_debug.get("drivable_triggered", False)) if repair_debug else False,
                dtype=np.bool_,
            ),
            trajectory_repair_selected_alpha=np.asarray(
                float(repair_debug.get("selected_alpha", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_min_clearance_m=np.asarray(
                float(repair_debug.get("min_clearance_m", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_style_offroad_fraction=np.asarray(
                float(repair_debug.get("style_offroad_fraction", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_selected_offroad_fraction=np.asarray(
                float(repair_debug.get("selected_offroad_fraction", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_drivable_check_time_ms=np.asarray(
                float(repair_debug.get("drivable_check_time_ms", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_model_inference_time_ms=np.asarray(
                float(repair_debug.get("model_inference_time_ms", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_time_ms=np.asarray(
                float(repair_debug.get("repair_time_ms", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
            trajectory_repair_total_time_ms=np.asarray(
                float(repair_debug.get("total_time_ms", np.nan))
                if repair_debug else np.nan,
                dtype=np.float32,
            ),
        )

    def initialize(self, initialization):
        # Some NuPlan runners reuse one planner object for consecutive
        # scenarios.  Undo the previous wrapper before the parent loads
        # the baseline checkpoint again; otherwise wrappers and LoRA
        # layers would be nested once per scenario.
        if isinstance(getattr(self, "_planner", None), StyleLoRAPlanner):
            self._planner = self._planner.baseline
        super().initialize(initialization)
        # NuPlan 会为每个场景单独构造 planner；按 token 建目录可避免不同场景
        # 都写入 scenario_000001 时发生文件覆盖，并支持事后精确筛选。
        if self.raw_data_save_dir and self._scenario_token:
            self._scenario_raw_dir = os.path.join(self.raw_data_save_dir, self._scenario_token)
            os.makedirs(self._scenario_raw_dir, exist_ok=True)
            self._raw_trace_path = os.path.join(self._scenario_raw_dir, "step_trace.jsonl")
        attach_lora_to_simulation_planner(
            self,
            self._lora_aggressive_adapter,
            self._lora_conservative_adapter,
            baseline_checkpoint=self._ckpt_path,
            normalization_file=self._lora_normalization_file,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
            rho=self._lora_rho,
            enable_scene_gate=self._scene_gate_enabled,
            scene_gate_checkpoint=self._scene_gate_checkpoint,
            enable_conditional_router=self._conditional_router_enabled,
            conditional_router_checkpoint=self._conditional_router_checkpoint,
        )

    def __init__(self, config, *planner_args, scenario=None, **planner_kwargs):
        self._scenario_token = str(getattr(scenario, "token", ""))
        self._lora_aggressive_adapter = str(getattr(config, "lora_aggressive_adapter"))
        self._lora_conservative_adapter = str(getattr(config, "lora_conservative_adapter"))
        normalization_file = getattr(config, "lora_normalization_file", None)
        if not normalization_file:
            normalization_file = getattr(config, "normalization_file_path")
        self._lora_normalization_file = str(normalization_file)
        self._lora_rank = int(getattr(config, "lora_rank", 4))
        alpha = getattr(config, "lora_alpha", None)
        self._lora_alpha = None if alpha is None else float(alpha)
        self._lora_rho = float(getattr(config, "lora_rho", 0.0))
        self._preference_state = PreferenceState(
            rho=self._lora_rho,
            step=float(getattr(config, "feedback_rho_step", 0.25)),
        )
        self._scene_gate_enabled = bool(getattr(config, "scene_gate_enabled", False))
        gate_checkpoint = getattr(config, "scene_gate_checkpoint", None)
        self._scene_gate_checkpoint = str(gate_checkpoint) if gate_checkpoint else None
        self._conditional_router_enabled = bool(
            getattr(config, "conditional_router_enabled", False)
        )
        router_checkpoint = getattr(config, "conditional_router_checkpoint", None)
        self._conditional_router_checkpoint = str(router_checkpoint) if router_checkpoint else None
        self._bounded_style_enabled = bool(getattr(config, "bounded_style_enabled", False))
        self._trajectory_repair_enabled = bool(
            getattr(config, "trajectory_repair_enabled", False)
        )
        if self._bounded_style_enabled and self._trajectory_repair_enabled:
            raise ValueError("bounded style and trajectory repair cannot be enabled together")
        candidate_ratios = getattr(config, "bounded_candidate_ratios", [1.0, 0.75, 0.5, 0.25])
        if isinstance(candidate_ratios, str):
            candidate_ratios = [item for item in candidate_ratios.split(",") if item.strip()]
        self._bounded_candidate_ratios = tuple(
            sorted({float(value) for value in candidate_ratios}, reverse=True)
        )
        if (
            not self._bounded_candidate_ratios
            or self._bounded_candidate_ratios[0] != 1.0
            or any(value <= 0.0 or value > 1.0 for value in self._bounded_candidate_ratios)
        ):
            raise ValueError("bounded_candidate_ratios 必须包含 1.0，且全部位于 (0,1]")
        self._last_bounded_style = None
        self._last_trajectory_repair = None
        super().__init__(config, *planner_args, **planner_kwargs)
        self._bounded_validator = (
            BaselineRelativeCandidateValidator(config, step_interval=self._step_interval)
            if self._bounded_style_enabled
            else None
        )
        self._trajectory_repair = (
            BaselineAnchoredTrajectoryRepair(config, step_interval=self._step_interval)
            if self._trajectory_repair_enabled
            else None
        )

    def apply_user_feedback(self, feedback: str) -> dict[str, object]:
        """Update rho from relative user feedback without retraining the planner.

        Supported commands are ``more_conservative``, ``keep`` and
        ``more_aggressive``.  Existing fixed-``--rho`` evaluation is unchanged
        unless this method is explicitly called.
        """
        update = self._preference_state.update(feedback)
        self._lora_rho = float(update["rho_after"])
        planner = getattr(self, "_planner", None)
        if isinstance(planner, StyleLoRAPlanner):
            planner.set_strength(self._lora_rho)
        return update

    def current_preference_state(self) -> dict[str, float]:
        """Return the current rho and configured feedback increment."""
        return self._preference_state.snapshot()

    def _paired_noise_seed(self, current_input) -> int:
        """为同一场景同一步的所有候选构造相同且可复现的扩散种子。"""
        iteration = int(current_input.iteration.index)
        payload = f"{self._scenario_token}:{iteration}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

    def _decode_with_seed(self, encoder_outputs, h_c, model_inputs, rho: float, seed: int):
        input_device = h_c.device
        devices = (
            [input_device.index if input_device.index is not None else torch.cuda.current_device()]
            if input_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            return self._planner.decode_from_context(
                encoder_outputs,
                h_c,
                model_inputs,
                rho,
                requested=rho,
            )

    def _bounded_outputs(self, current_input, raw_inputs, model_inputs):
        """验证有限强度候选，全部不合格时返回同噪声 baseline。"""
        if self._bounded_validator is None:
            raise RuntimeError("有边界风格执行已启用，但候选验证器没有初始化")
        encoder_outputs, h_c = self._planner.encode_context(model_inputs)
        requested_rho = float(self._lora_rho)
        seed = self._paired_noise_seed(current_input)
        baseline_output = self._decode_with_seed(
            encoder_outputs, h_c, model_inputs, 0.0, seed
        )
        debug = {
            "requested_rho": float(self._lora_rho),
            "initial_requested_rho": requested_rho,
            "accepted_rho": 0.0,
            "candidate_attempts": 0,
            "candidate_accepted": False,
            "baseline_fallback": abs(float(self._lora_rho)) > 1e-12,
            "baseline_hard_valid": False,
            "failure_reasons": [],
        }
        if abs(requested_rho) <= 1e-12:
            self._last_bounded_style = debug
            return baseline_output

        all_reasons: list[str] = []
        ego_state = current_input.history.ego_states[-1]
        for attempt, ratio in enumerate(self._bounded_candidate_ratios):
            candidate_rho = requested_rho * ratio
            candidate_output = self._decode_with_seed(
                encoder_outputs, h_c, model_inputs, candidate_rho, seed
            )
            result = self._bounded_validator.validate(
                candidate_output["prediction"],
                baseline_output["prediction"],
                raw_inputs=raw_inputs,
                ego_state=ego_state,
                map_api=self._map_api,
            )
            debug["candidate_attempts"] = attempt + 1
            debug["baseline_hard_valid"] = result.baseline_hard_valid
            all_reasons.extend(result.failure_reasons)
            if result.accepted:
                debug.update({
                    "accepted_rho": candidate_rho,
                    "candidate_accepted": True,
                    "baseline_fallback": False,
                })
                debug["failure_reasons"] = list(dict.fromkeys(all_reasons))
                self._last_bounded_style = debug
                return candidate_output

        debug["failure_reasons"] = list(dict.fromkeys(all_reasons))
        self._last_bounded_style = debug
        return baseline_output

    def _trajectory_repair_outputs(self, current_input, raw_inputs, model_inputs):
        """Repair the requested-rho trajectory around a paired rho=0 reference."""
        if self._trajectory_repair is None:
            raise RuntimeError("trajectory repair is enabled but was not initialized")
        sync_runtime = self._trajectory_repair.mode == "collision_parallel"
        if sync_runtime:
            _synchronize_model_device(self._planner)
        total_started = time.perf_counter()
        encoder_outputs, h_c = self._planner.encode_context(model_inputs)
        requested_rho = float(self._lora_rho)
        seed = self._paired_noise_seed(current_input)
        baseline_output = self._decode_with_seed(
            encoder_outputs, h_c, model_inputs, 0.0, seed
        )
        if abs(requested_rho) <= 1e-12:
            if sync_runtime:
                _synchronize_model_device(self._planner)
            model_inference_time_ms = (time.perf_counter() - total_started) * 1000.0
            self._last_trajectory_repair = {
                "requested_rho": requested_rho,
                "mode": self._trajectory_repair.mode,
                "applied": False,
                "baseline_fallback": False,
                "longitudinal_scale": 1.0,
                "lateral_scale": 1.0,
                "selected_alpha": 1.0,
                "candidate_attempts": 0,
                "baseline_hard_valid": False,
                "collision_triggered": False,
                "drivable_triggered": False,
                "min_clearance_m": float("nan"),
                "style_offroad_fraction": 0.0,
                "selected_offroad_fraction": 0.0,
                "drivable_check_time_ms": 0.0,
                "model_inference_time_ms": model_inference_time_ms,
                "repair_time_ms": 0.0,
                "total_time_ms": model_inference_time_ms,
                "failure_reasons": [],
            }
            return baseline_output

        styled_output = self._decode_with_seed(
            encoder_outputs, h_c, model_inputs, requested_rho, seed
        )
        if sync_runtime:
            _synchronize_model_device(self._planner)
        model_inference_time_ms = (time.perf_counter() - total_started) * 1000.0
        repair_started = time.perf_counter()
        result = self._trajectory_repair.repair(
            styled_output["prediction"],
            baseline_output["prediction"],
            raw_inputs=raw_inputs,
            ego_state=current_input.history.ego_states[-1],
            map_api=self._map_api,
        )
        if sync_runtime:
            _synchronize_model_device(self._planner)
        repair_time_ms = (time.perf_counter() - repair_started) * 1000.0
        total_time_ms = (time.perf_counter() - total_started) * 1000.0
        outputs = dict(baseline_output if result.baseline_fallback else styled_output)
        outputs["prediction"] = result.prediction
        self._last_trajectory_repair = {
            "requested_rho": requested_rho,
            "mode": self._trajectory_repair.mode,
            "applied": result.applied,
            "baseline_fallback": result.baseline_fallback,
            "longitudinal_scale": result.longitudinal_scale,
            "lateral_scale": result.lateral_scale,
            "selected_alpha": result.longitudinal_scale,
            "candidate_attempts": result.candidate_attempts,
            "baseline_hard_valid": result.baseline_hard_valid,
            "collision_triggered": result.collision_triggered,
            "drivable_triggered": result.drivable_triggered,
            "min_clearance_m": result.selected_metrics.get(
                "min_predicted_clearance_m", float("nan")
            ),
            "style_offroad_fraction": result.style_offroad_fraction,
            "selected_offroad_fraction": result.selected_offroad_fraction,
            "drivable_check_time_ms": result.drivable_check_time_ms,
            "model_inference_time_ms": model_inference_time_ms,
            "repair_time_ms": repair_time_ms,
            "total_time_ms": total_time_ms,
            "failure_reasons": list(result.failure_reasons),
            "selected_metrics": dict(result.selected_metrics),
        }
        return outputs

    def compute_planner_trajectory(self, current_input):
        if not self._bounded_style_enabled and not self._trajectory_repair_enabled:
            self._last_bounded_style = None
            self._last_trajectory_repair = None
            return super().compute_planner_trajectory(current_input)
        with torch.no_grad():
            self._last_runtime_preference_debug = None
            raw_inputs = self.planner_input_to_model_inputs(current_input)
            normalized_inputs = self.observation_normalizer(raw_inputs)
            model_inputs = self._augment_model_inputs(raw_inputs, normalized_inputs)
            if self._trajectory_repair_enabled:
                self._last_bounded_style = None
                outputs = self._trajectory_repair_outputs(
                    current_input, raw_inputs, model_inputs
                )
            else:
                self._last_trajectory_repair = None
                outputs = self._bounded_outputs(current_input, raw_inputs, model_inputs)
            self._update_runtime_preference_debug(outputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )
        if self.renderer is not None:
            self._render_and_save(current_input, trajectory)
        if self._scenario_raw_dir is not None:
            self._export_step_bundle(current_input, raw_inputs, outputs)
        self._export_runtime_preference_trace(current_input)
        return trajectory
