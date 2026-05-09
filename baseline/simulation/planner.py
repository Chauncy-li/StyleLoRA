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
from baseline.model.diff_planner.diffusion_planner import Diffusion_Planner
from baseline.model.wayformer.wayf_planner import WayFormer
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


def register_simulation_planners() -> None:
    """注册可用于 NuPlan 仿真的 planner。"""
    if len(SIMULATION_PLANNER_REGISTRY) > 0:
        return
    SIMULATION_PLANNER_REGISTRY.register("diffusion_planner", DiffusionPlanner)
    SIMULATION_PLANNER_REGISTRY.register("wayformer", Wayformer)


def get_registered_simulation_planners() -> List[str]:
    """返回已注册 planner 名称列表。"""
    register_simulation_planners()
    return sorted(list(SIMULATION_PLANNER_REGISTRY.keys()))
