"""
Diffusion Planner 的仿真推理入口（精简版）。

本文件只保留 baseline 需要的 `DiffusionPlanner`：
1. 从 NuPlan 仿真输入构建模型输入；
2. 调用 diffusion 模型生成轨迹；
3. 可选渲染视频；
4. 可选按 step 导出原始输入与预测结果，便于后处理分析。
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
from baseline.model.diff_planner.diffusion_planner import Diffusion_Planner
from baseline.simulation.render import NuplanScenarioRender
from baseline.utils.config import Config


def load_checkpoint_safely(model, ckpt_path: str, device: str):
    """加载 checkpoint，并兼容常见键名格式（EMA / DDP 前缀）。"""
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "ema_state_dict" in state_dict and state_dict["ema_state_dict"] is not None:
        state_dict = state_dict["ema_state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    ckpt_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(ckpt_state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {len(unexpected)}")
    return model


class DiffusionPlanner(AbstractPlanner):
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

        self._planner = Diffusion_Planner(config)
        self.data_processor = DataProcessor(config)
        self.observation_normalizer = config.observation_normalizer

        # 渲染相关
        self.render_save_dir = getattr(config, "render_save_dir", None)
        self.renderer: Optional[NuplanScenarioRender] = None
        self.video_writer = None
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
        self._scenario_raw_dir: Optional[str] = None
        self._raw_trace_path: Optional[str] = None
        self._scenario_index = 0
        self._step_export_index = 0
        if self.raw_data_save_dir:
            os.makedirs(self.raw_data_save_dir, exist_ok=True)
            print(f"Raw step export enabled. Output dir: {self.raw_data_save_dir}")

    def name(self) -> str:
        return "diffusion_planner"

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
        if self.raw_data_save_dir:
            scenario_dir = os.path.join(self.raw_data_save_dir, f"scenario_{self._scenario_index:06d}")
            os.makedirs(scenario_dir, exist_ok=True)
            self._scenario_raw_dir = scenario_dir
            self._raw_trace_path = os.path.join(scenario_dir, "step_trace.jsonl")
        else:
            self._scenario_raw_dir = None
            self._raw_trace_path = None

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
            raw_inputs = self.planner_input_to_model_inputs(current_input)
            normalized_inputs = self.observation_normalizer(raw_inputs)
            _, outputs = self._planner(normalized_inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        if self.renderer is not None:
            self._render_and_save(current_input, trajectory)
        if self._scenario_raw_dir is not None:
            self._export_step_bundle(current_input, raw_inputs, outputs)

        return trajectory

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

    def __del__(self):
        if self.video_writer is not None:
            self.video_writer.release()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["video_writer"] = None
        state["renderer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.video_writer = None
        self.renderer = None
