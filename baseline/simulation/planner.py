"""
NuPlan 仿真 planner 封装（注册制）。

包含两个可切换的 planner：
1. `DiffusionPlanner`：diffusion-planner 闭环推理；
2. `Wayformer`：wayformer 闭环推理。

两者都支持：
- 场景视频渲染；
- step 级原始输入与预测结果导出；
- 通过注册表统一查询可用 planner（便于入口脚本动态切换）。
"""

from __future__ import annotations

import json
import os
from typing import Deque, Dict, List, Optional, Type

import cv2
import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from baseline.data_process.data_processor import DataProcessor
from baseline.core.register import Registry
from baseline.model.diff_planner.diffusion_planner import Diffusion_Planner as BaseDiffusionPlannerModel
from baseline.model.style_planner.diffusion_planner import Diffusion_Planner as StyleDiffusionPlannerModel
from baseline.model.wayformer.wayf_planner import WayFormer
from baseline.simulation.anchor_generator import MapAnchorGenerator
from baseline.simulation.candidate_selector import SafetyCandidateSelector
from research_v1.execution.runtime import (
    OnlinePreferenceConditioner,
    append_runtime_trace_csv,
    append_runtime_trace_jsonl,
    build_runtime_trace_row,
)
from research_v1.stylization.runtime import ContinuousStyleRuntimeConditioner
from baseline.simulation.render import NuplanScenarioRender
from baseline.utils.config import Config

SIMULATION_PLANNER_REGISTRY: Registry[type[AbstractPlanner]] = Registry("simulation_planner")


def load_checkpoint_safely(model, ckpt_path: str, device: str):
    """加载 checkpoint，并兼容常见键名格式（EMA / DDP 前缀）。"""
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "ema_state_dict" in state_dict and state_dict["ema_state_dict"] is not None:
        state_dict = state_dict["ema_state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint state must be a dictionary, got {type(state_dict)!r}")
    ckpt_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_state_dict = model.state_dict()
    matched_numel = sum(
        value.numel()
        for key, value in model_state_dict.items()
        if key in ckpt_state_dict and tuple(ckpt_state_dict[key].shape) == tuple(value.shape)
    )
    total_numel = sum(value.numel() for value in model_state_dict.values())
    coverage = matched_numel / max(total_numel, 1)
    print(f"Checkpoint parameter coverage: {coverage:.2%}")
    minimum_coverage = float(getattr(model, "_minimum_checkpoint_coverage", 0.0))
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"Checkpoint/model compatibility check failed: {coverage:.2%} < "
            f"required {minimum_coverage:.2%}. Check args.json and checkpoint pairing."
        )
    missing, unexpected = model.load_state_dict(ckpt_state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint: {len(missing)}; first keys: {missing[:10]}")
    if unexpected:
        print(
            f"Warning: unexpected keys in checkpoint: {len(unexpected)}; "
            f"first keys: {unexpected[:10]}"
        )
    return model


class DiffusionPlanner(AbstractPlanner):
    planner_model_cls = BaseDiffusionPlannerModel
    planner_name = "diffusion_planner"
    """NuPlan 仿真环境下的 Diffusion Planner 封装。"""

    def __init__(
        self,
        config: Config,
        ckpt_path: str,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = True,
        device: str = "cpu",
    ):
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._future_horizon = future_trajectory_sampling.time_horizon
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses
        self._device = device
        self._ckpt_path = ckpt_path
        self._ema_enabled = enable_ema
        self._last_runtime_preference_debug: Optional[Dict[str, object]] = None
        self._runtime_trace_export_enabled = bool(getattr(config, "runtime_trace_export_enabled", True))
        self._runtime_preference_trace_path: Optional[str] = None
        self._runtime_preference_csv_path: Optional[str] = None
        # Cleanup-owned fields must exist even if model/data construction fails.
        self.video_writer = None
        self.renderer: Optional[NuplanScenarioRender] = None

        self._planner = self.planner_model_cls(config)
        self.data_processor = DataProcessor(config)
        self.observation_normalizer = config.observation_normalizer

        # 渲染相关
        self.render_save_dir = getattr(config, "render_save_dir", None)
        if self.render_save_dir:
            os.makedirs(self.render_save_dir, exist_ok=True)
            self.renderer = NuplanScenarioRender(future_horizon=self._future_horizon)
            print(f"Video rendering enabled. Output dir: {self.render_save_dir}")

        # 原始数据导出相关
        default_raw_dir = os.path.join(self.render_save_dir, "raw_step_data") if self.render_save_dir else None
        raw_data_cfg = getattr(config, "raw_data_save_dir", None)
        if raw_data_cfg is None or str(raw_data_cfg).strip() == "":
            self.raw_data_save_dir = default_raw_dir
        else:
            self.raw_data_save_dir = str(raw_data_cfg)
        runtime_trace_cfg = getattr(config, "runtime_trace_save_dir", None)
        if runtime_trace_cfg is None or str(runtime_trace_cfg).strip() == "":
            self.runtime_trace_save_dir = self.raw_data_save_dir
        else:
            self.runtime_trace_save_dir = str(runtime_trace_cfg)
        self._scenario_raw_dir: Optional[str] = None
        self._raw_trace_path: Optional[str] = None
        self._scenario_index = 0
        self._step_export_index = 0
        self._runtime_trace_step_index = 0
        if self.raw_data_save_dir:
            os.makedirs(self.raw_data_save_dir, exist_ok=True)
            print(f"Raw step export enabled. Output dir: {self.raw_data_save_dir}")
        if self._runtime_trace_export_enabled and self.runtime_trace_save_dir:
            os.makedirs(self.runtime_trace_save_dir, exist_ok=True)
            print(
                "Runtime preference trace export enabled. Output dir: "
                f"{self.runtime_trace_save_dir}"
            )

    def name(self) -> str:
        return self.planner_name

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialization = initialization

        # 每个 scenario 初始化时重置渲染状态
        if self.renderer is not None:
            self.renderer.reset()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # 每个 scenario 初始化时重置导出状态
        self._scenario_index += 1
        self._step_export_index = 0
        self._runtime_trace_step_index = 0
        self._last_runtime_preference_debug = None
        if self.raw_data_save_dir:
            scenario_dir = os.path.join(self.raw_data_save_dir, f"scenario_{self._scenario_index:06d}")
            os.makedirs(scenario_dir, exist_ok=True)
            self._scenario_raw_dir = scenario_dir
            self._raw_trace_path = os.path.join(scenario_dir, "step_trace.jsonl")
        else:
            self._scenario_raw_dir = None
            self._raw_trace_path = None
        if self._runtime_trace_export_enabled and self.runtime_trace_save_dir:
            runtime_scenario_dir = os.path.join(
                self.runtime_trace_save_dir,
                f"scenario_{self._scenario_index:06d}",
            )
            os.makedirs(runtime_scenario_dir, exist_ok=True)
            self._runtime_preference_trace_path = os.path.join(
                runtime_scenario_dir,
                "runtime_preference_trace.jsonl",
            )
            self._runtime_preference_csv_path = os.path.join(
                runtime_scenario_dir,
                "runtime_preference_trace.csv",
            )
        else:
            self._runtime_preference_trace_path = None
            self._runtime_preference_csv_path = None

        if self._ckpt_path:
            self._planner = load_checkpoint_safely(self._planner, self._ckpt_path, self._device)
        else:
            print("Warning: no checkpoint provided, using random weights.")

        self._planner.eval()
        self._planner = self._planner.to(self._device)

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        return self.data_processor.observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device,
        )

    def outputs_to_trajectory(
        self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]
    ) -> List[InterpolatableState]:
        predictions = outputs["prediction"][0, 0].detach().cpu().numpy().astype(np.float64)  # [T, 4]
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions_xyh = np.concatenate([predictions[..., :2], heading], axis=-1)
        return transform_predictions_to_states(
            predictions_xyh, ego_state_history, self._future_horizon, self._step_interval
        )

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        with torch.no_grad():
            self._last_runtime_preference_debug = None
            raw_inputs = self.planner_input_to_model_inputs(current_input)
            normalized_inputs = self.observation_normalizer(raw_inputs)
            model_inputs = self._augment_model_inputs(raw_inputs, normalized_inputs)
            _, outputs = self._planner(model_inputs)
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

    def _augment_model_inputs(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        normalized_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return normalized_inputs

    def _update_runtime_preference_debug(self, outputs: Dict[str, torch.Tensor]) -> None:
        return

    def _export_runtime_preference_trace(
        self,
        current_input: PlannerInput,
    ) -> None:
        if (
            self._last_runtime_preference_debug is None
            or self._runtime_preference_trace_path is None
            or self._runtime_preference_csv_path is None
        ):
            return
        step_index = self._runtime_trace_step_index
        self._runtime_trace_step_index += 1
        runtime_trace_row = build_runtime_trace_row(
            step_index=step_index,
            iteration_index=int(current_input.iteration.index),
            time_us=int(current_input.history.ego_states[-1].time_point.time_us),
            debug=self._last_runtime_preference_debug,
        )
        append_runtime_trace_jsonl(
            self._runtime_preference_trace_path,
            runtime_trace_row,
        )
        append_runtime_trace_csv(
            self._runtime_preference_csv_path,
            runtime_trace_row,
        )

    def _render_and_save(self, current_input: PlannerInput, trajectory: InterpolatedTrajectory) -> None:
        try:
            if self.video_writer is None:
                timestamp = int(current_input.history.ego_states[-1].time_point.time_us)
                video_path = os.path.join(self.render_save_dir, f"sim_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (1000, 1000))
                print(f"Started recording: {video_path}")

            planning_trajectory = trajectory.get_sampled_trajectory()
            img_rgb = self.renderer.render_from_simulation(
                current_input=current_input,
                initialization=self._initialization,
                planning_trajectory=planning_trajectory,
                predictions=getattr(self, "_last_candidate_render", None),
            )
            if img_rgb is not None and self.video_writer is not None:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                if img_bgr.shape[:2] != (1000, 1000):
                    img_bgr = cv2.resize(img_bgr, (1000, 1000))
                self.video_writer.write(img_bgr)
        except Exception as exc:
            print(f"Render failed: {exc}")
            import traceback

            traceback.print_exc()

    def _export_step_bundle(
        self,
        current_input: PlannerInput,
        raw_inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        if "prediction" not in outputs:
            return

        prediction = outputs["prediction"]
        if prediction.ndim != 4:
            return

        step_index = self._step_export_index
        self._step_export_index += 1
        time_us = int(current_input.history.ego_states[-1].time_point.time_us)
        iteration_index = int(current_input.iteration.index)
        file_name = f"step_{step_index:06d}_iter_{iteration_index:04d}_{time_us}.npz"
        file_path = os.path.join(self._scenario_raw_dir, file_name)

        prediction_np = prediction[0].detach().cpu().numpy().astype(np.float32)  # [P, T, 4]

        def _np32(key: str):
            value = raw_inputs.get(key)
            if value is None:
                return np.zeros((0,), dtype=np.float32)
            return value[0].detach().cpu().numpy().astype(np.float32)

        candidate_prediction = outputs.get("candidate_prediction")
        candidate_prediction_np = (
            candidate_prediction.detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(candidate_prediction)
            else np.zeros((0,), dtype=np.float32)
        )
        candidate_scores = outputs.get("candidate_scores")
        candidate_scores_np = (
            candidate_scores.detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(candidate_scores)
            else np.zeros((0,), dtype=np.float32)
        )
        candidate_valid = outputs.get("candidate_valid_mask")
        candidate_valid_np = (
            candidate_valid.detach().cpu().numpy().astype(np.bool_)
            if torch.is_tensor(candidate_valid)
            else np.zeros((0,), dtype=np.bool_)
        )
        candidate_anchor = outputs.get("candidate_anchor")
        candidate_anchor_np = (
            candidate_anchor.detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(candidate_anchor)
            else np.zeros((0,), dtype=np.float32)
        )

        np.savez_compressed(
            file_path,
            step_index=np.array(step_index, dtype=np.int64),
            iteration_index=np.array(iteration_index, dtype=np.int64),
            time_us=np.array(time_us, dtype=np.int64),
            map_name=np.array(str(getattr(self._map_api, "map_name", ""))),
            route_roadblock_ids=np.asarray([str(x) for x in self._route_roadblock_ids], dtype=object),
            ego_current_state=_np32("ego_current_state"),
            ego_agent_past=_np32("ego_agent_past"),
            neighbor_agents_past=_np32("neighbor_agents_past"),
            static_objects=_np32("static_objects"),
            lanes=_np32("lanes"),
            route_lanes=_np32("route_lanes"),
            generated_prediction=prediction_np,
            generated_ego_future=prediction_np[0] if prediction_np.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32),
            generated_neighbor_future=prediction_np[1:] if prediction_np.shape[0] > 1 else np.zeros((0, 0, 4), dtype=np.float32),
            candidate_prediction=candidate_prediction_np,
            candidate_anchor=candidate_anchor_np,
            candidate_scores=candidate_scores_np,
            candidate_valid_mask=candidate_valid_np,
            candidate_intents=np.asarray(outputs.get("candidate_intents", ()), dtype=str),
            selected_candidate_index=np.asarray(
                int(outputs.get("selected_candidate_index", -1)), dtype=np.int64
            ),
            safety_fallback_used=np.asarray(
                bool(outputs.get("safety_fallback_used", False)), dtype=np.bool_
            ),
        )

        if self._raw_trace_path is not None:
            row = {
                "step_index": step_index,
                "iteration_index": iteration_index,
                "time_us": time_us,
                "file_name": file_name,
            }
            if self._last_runtime_preference_debug is not None:
                row["runtime_preference"] = self._last_runtime_preference_debug
            evaluations = outputs.get("candidate_evaluations")
            if evaluations is not None:
                def _finite_or_none(value: object):
                    number = float(value)
                    return number if np.isfinite(number) else None

                row["anchor_warm_start"] = {
                    "intents": [str(item) for item in outputs.get("candidate_intents", ())],
                    "selected_candidate_index": int(
                        outputs.get("selected_candidate_index", -1)
                    ),
                    "safety_fallback_used": bool(
                        outputs.get("safety_fallback_used", False)
                    ),
                    "unavailable_reasons": dict(
                        outputs.get("anchor_unavailable_reasons", {})
                    ),
                    "evaluations": [
                        {
                            "intent": str(item.intent),
                            "hard_valid": bool(item.hard_valid),
                            "score": _finite_or_none(item.score),
                            "failure_reasons": list(item.failure_reasons),
                            "metrics": {
                                str(key): _finite_or_none(value)
                                for key, value in item.metrics.items()
                            },
                        }
                        for item in evaluations
                    ],
                }
            with open(self._raw_trace_path, "a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __del__(self):
        video_writer = getattr(self, "video_writer", None)
        if video_writer is not None:
            video_writer.release()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["video_writer"] = None
        state["renderer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.video_writer = None
        self.renderer = None


class Wayformer(AbstractPlanner):
    """
    Wayformer Planner 实现 (对齐 DiffusionPlanner 接口，增加视频保存功能)
    """

    def __init__(
            self,
            config: Config,
            ckpt_path: str,
            past_trajectory_sampling: TrajectorySampling,
            future_trajectory_sampling: TrajectorySampling,
            enable_ema: bool = True,
            device: str = "cpu",
    ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._future_horizon = future_trajectory_sampling.time_horizon
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses

        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        # ★ 差异点1: 实例化 planner
        self._planner = WayFormer(config)

        # 假设两个模型共用相同的数据处理器和 Normalizer
        self.data_processor = DataProcessor(config)
        self.observation_normalizer = config.observation_normalizer

        # 视频保存功能配置
        self.render_save_dir = getattr(config, 'render_save_dir', None)
        self.renderer = None
        self.video_writer = None
        self.current_scenario_token = None

        if self.render_save_dir:
            print(f"🎥 Video rendering enabled. Output dir: {self.render_save_dir}")
            os.makedirs(self.render_save_dir, exist_ok=True)
            # 初始化渲染器
            self.renderer = NuplanScenarioRender(future_horizon=self._future_horizon)
        else:
            print("⚠️ Render save dir not found in config. Video saving disabled.")

        # 原始数据导出配置（与 DiffusionPlanner 对齐）
        default_raw_dir = os.path.join(self.render_save_dir, "raw_step_data") if self.render_save_dir else None
        raw_data_cfg = getattr(config, "raw_data_save_dir", None)
        if raw_data_cfg is None or str(raw_data_cfg).strip() == "":
            self.raw_data_save_dir = default_raw_dir
        else:
            self.raw_data_save_dir = str(raw_data_cfg)

        self._scenario_raw_dir: Optional[str] = None
        self._raw_trace_path: Optional[str] = None
        self._scenario_index = 0
        self._step_export_index = 0
        if self.raw_data_save_dir:
            os.makedirs(self.raw_data_save_dir, exist_ok=True)
            print(f"Raw step export enabled. Output dir: {self.raw_data_save_dir}")

    def name(self) -> str:
        return "wayformer"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        # ================= 视频流管理 =================
        # 每次 Initialize 意味着一个新的 Scenario 开始
        if self.render_save_dir is not None:
            # 1. 释放上一个视频流 (如果有)
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            # 2. 重置渲染器历史
            self.renderer.reset()

            # 3. 准备新视频流的文件名 (暂时还不知道 token，会在第一次 compute_trajectory 时创建 writer)
            # 我们在这里重置状态即可
            self.current_scenario_token = None
        # =======================================================

        # 导出目录状态重置
        self._scenario_index += 1
        self._step_export_index = 0
        if self.raw_data_save_dir:
            scenario_dir = os.path.join(self.raw_data_save_dir, f"scenario_{self._scenario_index:06d}")
            os.makedirs(scenario_dir, exist_ok=True)
            self._scenario_raw_dir = scenario_dir
            self._raw_trace_path = os.path.join(scenario_dir, "step_trace.jsonl")
        else:
            self._scenario_raw_dir = None
            self._raw_trace_path = None

        if self._ckpt_path is not None:
            self._planner = load_checkpoint_safely(self._planner, self._ckpt_path, self._device)
        else:
            print("⚠️ Warning: Using random weights!")

        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        # 与 DiffusionPlanner 保持一致
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api,
                                                               self._route_roadblock_ids, self._device)
        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[
        InterpolatableState]:
        # ★ 差异点2: 输出处理
        # Wayformer 输出已经在模型内部处理成了 [B, P, T, 4] -> (x, y, heading, vel)
        # 我们取 batch=0, agent=0 (ego)
        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64)  # Shape: [T, 4]

        # transform_predictions_to_states 需要 (x, y, heading)
        # 既然 Wayformer 已经在 index 2 输出了 heading，直接切片取前 3 列即可
        # 不需要再做 arctan2
        predictions_xyh = predictions[..., :3]

        states = transform_predictions_to_states(predictions_xyh, ego_state_history, self._future_horizon,
                                                 self._step_interval)
        return states

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        with torch.no_grad():
            raw_inputs = self.planner_input_to_model_inputs(current_input)
            normalized_inputs = dict(raw_inputs)

            if 'neighbor_agents_past' in normalized_inputs:
                normalized_inputs['safety_shield_neighbors'] = normalized_inputs['neighbor_agents_past'].clone()
            if 'ego_current_state' in normalized_inputs:
                normalized_inputs['safety_shield_ego_state'] = normalized_inputs['ego_current_state'].clone()

            normalized_inputs = self.observation_normalizer(normalized_inputs)
            outputs = self._planner(normalized_inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        # ================= [新增] 渲染与视频保存 =================
        if self.render_save_dir is not None:
            self._render_and_save(current_input, trajectory)
        if self._scenario_raw_dir is not None:
            self._export_step_bundle(current_input, raw_inputs, outputs)
        # =======================================================
        return trajectory

    def _render_and_save(self, current_input, trajectory):
        """
        内部辅助函数：执行渲染并将帧写入视频
        """
        try:
            # 如果是该 Scenario 的第一帧，初始化 VideoWriter
            if self.video_writer is None:
                # 使用 map_name + 时间戳作为文件名，避免覆盖
                timestamp = current_input.history.ego_states[-1].time_point.time_us
                video_name = f"sim_{timestamp}.avi"
                video_path = os.path.join(self.render_save_dir, video_name)

                # 假设渲染 1000x1000 图像, 10 FPS (NuPlan 是 10Hz)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (1000, 1000))
                print(f"🎬 Started recording: {video_path}")

            # 2. 调用渲染器
            # 我们需要把 AbstractTrajectory 转换回 list[state] 方便渲染
            planning_trajectory_list = trajectory.get_sampled_trajectory()

            img_rgb = self.renderer.render_from_simulation(
                current_input=current_input,
                initialization=self._initialization,
                planning_trajectory=planning_trajectory_list,
                # predictions=... (如果需要画多模态预测，可以从 outputs 传进来)
            )

            # 3. 写入视频
            if img_rgb is not None and self.video_writer is not None:
                # OpenCV 使用 BGR 格式
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # 确保尺寸匹配 (resize 是一种保险措施)
                if img_bgr.shape[:2] != (1000, 1000):
                    img_bgr = cv2.resize(img_bgr, (1000, 1000))

                self.video_writer.write(img_bgr)

        except Exception as e:
            print(f"⚠️ Render failed: {e}")
            import traceback
            traceback.print_exc()

    def __del__(self):
        # 析构时确保资源释放
        if self.video_writer is not None:
            self.video_writer.release()

    # [新增] 告诉 Pickle 忽略掉不可序列化的对象
    def __getstate__(self):
        # 1. 复制当前的属性字典
        state = self.__dict__.copy()

        # 2. 移除不可被 pickle 的 OpenCV 对象和渲染器
        # 我们不需要在日志里保存视频写入器，视频文件已经独立存盘了
        if 'video_writer' in state:
            state['video_writer'] = None

        if 'renderer' in state:
            state['renderer'] = None

        return state

    # 反序列化时的恢复逻辑 (可选，防止读取日志时报错)
    def __setstate__(self, state):
        self.__dict__.update(state)
        # 恢复为 None，反正读取日志通常不需要再继续录制视频
        self.video_writer = None
        self.renderer = None

    def _export_step_bundle(
        self,
        current_input: PlannerInput,
        raw_inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        if "prediction" not in outputs:
            return

        prediction = outputs["prediction"]
        if prediction.ndim != 4:
            return

        step_index = self._step_export_index
        self._step_export_index += 1
        time_us = int(current_input.history.ego_states[-1].time_point.time_us)
        iteration_index = int(current_input.iteration.index)
        file_name = f"step_{step_index:06d}_iter_{iteration_index:04d}_{time_us}.npz"
        file_path = os.path.join(self._scenario_raw_dir, file_name)

        prediction_np = prediction[0].detach().cpu().numpy().astype(np.float32)  # [P, T, 4]

        def _np32(key: str):
            value = raw_inputs.get(key)
            if value is None:
                return np.zeros((0,), dtype=np.float32)
            return value[0].detach().cpu().numpy().astype(np.float32)

        np.savez_compressed(
            file_path,
            step_index=np.array(step_index, dtype=np.int64),
            iteration_index=np.array(iteration_index, dtype=np.int64),
            time_us=np.array(time_us, dtype=np.int64),
            map_name=np.array(str(getattr(self._map_api, "map_name", ""))),
            route_roadblock_ids=np.asarray([str(x) for x in self._route_roadblock_ids], dtype=object),
            ego_current_state=_np32("ego_current_state"),
            ego_agent_past=_np32("ego_agent_past"),
            neighbor_agents_past=_np32("neighbor_agents_past"),
            static_objects=_np32("static_objects"),
            lanes=_np32("lanes"),
            route_lanes=_np32("route_lanes"),
            generated_prediction=prediction_np,
            generated_ego_future=prediction_np[0] if prediction_np.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32),
            generated_neighbor_future=prediction_np[1:] if prediction_np.shape[0] > 1 else np.zeros((0, 0, 4), dtype=np.float32),
        )

        if self._raw_trace_path is not None:
            row = {
                "step_index": step_index,
                "iteration_index": iteration_index,
                "time_us": time_us,
                "file_name": file_name,
            }
            with open(self._raw_trace_path, "a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class StylePlanner(DiffusionPlanner):
    """Isolated closed-loop wrapper for the style-planner research branch."""

    planner_model_cls = StyleDiffusionPlannerModel
    planner_name = "style_planner"

    def __init__(
        self,
        config: Config,
        ckpt_path: str,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            config=config,
            ckpt_path=ckpt_path,
            past_trajectory_sampling=past_trajectory_sampling,
            future_trajectory_sampling=future_trajectory_sampling,
            enable_ema=enable_ema,
            device=device,
        )
        runtime_preference_enabled = bool(getattr(config, "runtime_preference_enabled", True))
        self._runtime_style_mode = str(getattr(config, "runtime_style_mode", "legacy_preference_execution"))
        style_condition_encoder = str(getattr(config, "style_condition_encoder", "mlp"))
        v6_style_condition_encoders = {"axis_router_v1", "axis_router_v2_signed"}
        if (
            runtime_preference_enabled
            and style_condition_encoder in v6_style_condition_encoders
            and self._runtime_style_mode != "continuous_v6"
        ):
            raise ValueError(
                f"A V6 {style_condition_encoder} checkpoint requires "
                "runtime_style_mode='continuous_v6'. The legacy conditioner "
                "does not implement the 12D V6 condition contract."
            )
        if not runtime_preference_enabled:
            self._online_preference_conditioner = None
        elif self._runtime_style_mode == "continuous_v6":
            self._online_preference_conditioner = ContinuousStyleRuntimeConditioner(config)
        elif self._runtime_style_mode == "legacy_preference_execution":
            self._online_preference_conditioner = OnlinePreferenceConditioner(config)
        else:
            raise ValueError(
                "runtime_style_mode must be 'legacy_preference_execution' or 'continuous_v6', got "
                f"{self._runtime_style_mode!r}"
            )

    def set_runtime_preference(
        self,
        style_label: str | None = None,
        intensity: float | None = None,
        rho: float | None = None,
    ) -> None:
        if self._online_preference_conditioner is None:
            raise RuntimeError("runtime preference control is disabled for this StylePlanner instance.")
        if self._runtime_style_mode == "continuous_v6":
            if style_label is not None or intensity is not None:
                raise ValueError("continuous_v6 accepts only rho; style_label/intensity belong to legacy_preference_execution.")
            self._online_preference_conditioner.set_command(rho=rho)
        else:
            if rho is not None:
                raise ValueError("rho is supported only when runtime_style_mode='continuous_v6'.")
            self._online_preference_conditioner.set_command(style_label=style_label, intensity=intensity)

    def set_runtime_lane_change_context(
        self,
        *,
        lane_change_intent: bool | None = None,
        intent_available: bool | None = None,
        target_lane_available: bool | None = None,
        target_lane_interaction_observable: bool | None = None,
    ) -> None:
        """Supply explicit causal route/target-lane signals to V6 at runtime."""

        if self._runtime_style_mode != "continuous_v6" or self._online_preference_conditioner is None:
            raise RuntimeError("lane-change causal context is available only for enabled continuous_v6 runtime control.")
        self._online_preference_conditioner.set_route_context(
            lane_change_intent=lane_change_intent,
            intent_available=intent_available,
            target_lane_available=target_lane_available,
            target_lane_interaction_observable=target_lane_interaction_observable,
        )

    def current_runtime_preference(self) -> Dict[str, object]:
        if self._online_preference_conditioner is None:
            return {"enabled": False}
        if self._runtime_style_mode == "continuous_v6":
            return {
                "enabled": True,
                "runtime_style_mode": "continuous_v6",
                "rho": self._online_preference_conditioner.rho,
                "normal_anchor_cfg_enabled": (
                    self._online_preference_conditioner.normal_anchor_cfg_enabled
                ),
                "cfg_guidance_scale": (
                    self._online_preference_conditioner.cfg_guidance_scale
                ),
            }
        return {
            "enabled": True,
            "runtime_style_mode": "legacy_preference_execution",
            "style_label": self._online_preference_conditioner.style_label,
            "style_intensity": self._online_preference_conditioner.style_intensity,
            "stats_path": self._online_preference_conditioner.stats_path,
        }

    def _augment_model_inputs(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        normalized_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self._online_preference_conditioner is None:
            return normalized_inputs
        model_inputs, debug = self._online_preference_conditioner.apply(
            raw_inputs,
            normalized_inputs,
            device=self._device,
        )
        self._last_runtime_preference_debug = debug
        return model_inputs

    def _update_runtime_preference_debug(self, outputs: Dict[str, torch.Tensor]) -> None:
        if self._last_runtime_preference_debug is None:
            return
        temporal_keys = (
            "temporal_near_gate",
            "temporal_far_gate",
            "temporal_near_condition",
            "temporal_far_condition",
            "axis_router_gate",
            "axis_router_learned_gate",
            "axis_router_availability",
            "normal_anchor_cfg_used",
            "empty_cfg_reference_used",
            "preference_command_strength",
            "preference_generated_axis_percentile",
            "preference_generated_raw_axis",
            "preference_generated_axis_valid_mask",
            "preference_target_axis_percentile",
            "preference_axis_reference_support_reason_code",
            "preference_axis_reference_shared_condition_count",
            "preference_axis_reference_speed_limit_source_code",
            "preference_axis_reference_speed_limit_valid",
            "preference_axis_reference_route_curvature_valid",
            "preference_axis_reference_free_drive_clear",
            "preference_axis_reference_active_traffic_control",
            "preference_axis_reference_valid_axis_mask",
        )
        for key in temporal_keys:
            value = outputs.get(key)
            if value is None or not torch.is_tensor(value):
                continue
            detached = value.detach().cpu()
            caster = bool if detached.dtype == torch.bool else float
            if detached.ndim == 0:
                self._last_runtime_preference_debug[key] = caster(detached)
            else:
                self._last_runtime_preference_debug[key] = [
                    caster(item) for item in detached[0].reshape(-1).tolist()
                ]


class AnchorWarmStartStylePlanner(StylePlanner):
    """Closed-loop style planner with map anchors and safety-first candidate selection.

    This planner intentionally shares the exact ``StyleDiffusionPlannerModel``
    parameterization with :class:`StylePlanner`, so a checkpoint trained by the
    original ``diff_planner`` can be loaded without adding or reshaping weights.
    """

    planner_name = "anchor_warm_start_style_planner"

    def __init__(
        self,
        config: Config,
        ckpt_path: str,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            config=config,
            ckpt_path=ckpt_path,
            past_trajectory_sampling=past_trajectory_sampling,
            future_trajectory_sampling=future_trajectory_sampling,
            enable_ema=enable_ema,
            device=device,
        )
        # A legacy diff_planner checkpoint should cover essentially every
        # parameter because warm-start adds no trainable modules.
        self._planner._minimum_checkpoint_coverage = 0.98
        self._anchor_generator = MapAnchorGenerator(
            config,
            horizon_s=self._future_horizon,
            step_interval=self._step_interval,
        )
        model_future_len = int(getattr(config, "future_len"))
        if self._anchor_generator.future_len != model_future_len:
            raise ValueError(
                "Simulation sampling and checkpoint future_len disagree: "
                f"{self._anchor_generator.future_len} vs {model_future_len}"
            )
        self._candidate_selector = SafetyCandidateSelector(
            config,
            step_interval=self._step_interval,
        )
        self._warm_start_t = float(getattr(config, "warm_start_t", 0.30))
        self._warm_start_shared_noise = bool(getattr(config, "warm_start_shared_noise", True))
        if not 1e-3 < self._warm_start_t <= 1.0:
            raise ValueError(f"warm_start_t must be in (1e-3, 1], got {self._warm_start_t}")
        self._last_candidate_render = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._last_candidate_render = None
        super().initialize(initialization)

    @staticmethod
    def _repeat_batch_inputs(inputs: Dict[str, object], repeats: int) -> Dict[str, object]:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        batch_size = None
        for value in inputs.values():
            if torch.is_tensor(value) and value.ndim > 0:
                batch_size = int(value.shape[0])
                break
        if batch_size is None:
            raise ValueError("No batched tensor found in planner inputs")
        repeated: Dict[str, object] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
                repeated[key] = value.repeat_interleave(repeats, dim=0)
            else:
                repeated[key] = value
        return repeated

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        with torch.no_grad():
            self._last_runtime_preference_debug = None
            ego_state = current_input.history.ego_states[-1]
            raw_inputs = self.planner_input_to_model_inputs(current_input)
            bundle = self._anchor_generator.build(
                ego_state,
                self._map_api,
                self._route_roadblock_ids,
            )
            anchors = list(bundle.candidates)
            if not anchors:
                raise RuntimeError("MapAnchorGenerator must return at least the keep/fallback anchor")

            normalized_inputs = self.observation_normalizer(raw_inputs)
            conditioned_inputs = self._augment_model_inputs(raw_inputs, normalized_inputs)
            model_inputs = self._repeat_batch_inputs(conditioned_inputs, len(anchors))
            model_inputs["warm_start_enabled"] = True
            model_inputs["warm_start_t"] = self._warm_start_t
            model_inputs["warm_start_dt"] = self._step_interval
            model_inputs["warm_start_shared_noise"] = self._warm_start_shared_noise
            model_inputs["warm_start_neighbor_past_raw"] = raw_inputs[
                "neighbor_agents_past"
            ].repeat_interleave(len(anchors), dim=0)
            model_inputs["warm_start_ego_anchor"] = torch.as_tensor(
                np.stack([anchor.ego_future_local for anchor in anchors], axis=0),
                dtype=raw_inputs["ego_current_state"].dtype,
                device=self._device,
            )

            _, model_outputs = self._planner(model_inputs)
            candidate_prediction = model_outputs["prediction"]  # [K, P, T, 4]
            selection = self._candidate_selector.select(
                candidate_prediction,
                anchors,
                bundle,
                raw_inputs,
                ego_state,
                self._map_api,
            )
            evaluations = selection.evaluations
            candidate_scores = torch.as_tensor(
                [evaluation.score for evaluation in evaluations],
                dtype=candidate_prediction.dtype,
                device=candidate_prediction.device,
            )
            candidate_valid = torch.as_tensor(
                [evaluation.hard_valid for evaluation in evaluations],
                dtype=torch.bool,
                device=candidate_prediction.device,
            )
            outputs = dict(model_outputs)
            outputs.update(
                {
                    "prediction": selection.selected_prediction.unsqueeze(0),
                    "candidate_prediction": candidate_prediction,
                    "candidate_anchor": torch.as_tensor(
                        np.stack([anchor.ego_future_local for anchor in anchors], axis=0),
                        dtype=candidate_prediction.dtype,
                        device=candidate_prediction.device,
                    ),
                    "candidate_scores": candidate_scores,
                    "candidate_valid_mask": candidate_valid,
                    "candidate_intents": tuple(anchor.intent for anchor in anchors),
                    "candidate_evaluations": evaluations,
                    "anchor_unavailable_reasons": dict(bundle.unavailable_reasons),
                    "selected_candidate_index": selection.selected_candidate_index,
                    "safety_fallback_used": selection.used_fallback,
                }
            )
            self._update_runtime_preference_debug(outputs)
            self._last_candidate_render = {
                "trajectories": candidate_prediction[:, 0, :, :2].detach().cpu().numpy(),
                "anchors": np.stack([anchor.ego_future_local[:, :2] for anchor in anchors], axis=0),
                "labels": [anchor.intent for anchor in anchors],
                "valid": [evaluation.hard_valid for evaluation in evaluations],
                "scores": [evaluation.score for evaluation in evaluations],
                "selected_index": selection.selected_candidate_index,
                "fallback_used": selection.used_fallback,
            }

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )
        if self.renderer is not None:
            self._render_and_save(current_input, trajectory)
        if self._scenario_raw_dir is not None:
            self._export_step_bundle(current_input, raw_inputs, outputs)
        self._export_runtime_preference_trace(current_input)
        return trajectory


def register_simulation_planners() -> None:
    """注册可用于 NuPlan 仿真的 planner。"""
    if len(SIMULATION_PLANNER_REGISTRY) > 0:
        return
    SIMULATION_PLANNER_REGISTRY.register("diffusion_planner", DiffusionPlanner)
    SIMULATION_PLANNER_REGISTRY.register("style_planner", StylePlanner)
    SIMULATION_PLANNER_REGISTRY.register(
        "anchor_warm_start_style_planner", AnchorWarmStartStylePlanner
    )
    SIMULATION_PLANNER_REGISTRY.register("wayformer", Wayformer)


def get_registered_simulation_planners() -> List[str]:
    """返回已注册 planner 名称列表。"""
    register_simulation_planners()
    return sorted(list(SIMULATION_PLANNER_REGISTRY.keys()))
