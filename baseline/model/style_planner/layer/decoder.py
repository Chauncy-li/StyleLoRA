# decoder.py
# 轨迹扩散解码器模块
#
# Author: Shangwen Li
# Date: 2026-03-09
# Description: 实现基于扩散模型的轨迹解码器，包括 DiT (Diffusion Transformer) 架构
#              使用 DPM-Solver 进行采样，支持 Classifier-Free Guidance
#              包含 Decoder、RouteEncoder 和 DiT 三个核心组件
# License: MIT

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from baseline.model.style_planner.library.sampling import dpm_sampler
from baseline.model.style_planner.library.sde import SDE, VPSDE_linear
from baseline.utils.normalizer import ObservationNormalizer, StateNormalizer
from baseline.model.style_planner.layer.mixer import MixerBlock
from baseline.model.style_planner.layer.dit import TimestepEmbedder, DiTBlock, FinalLayer
from baseline.model.style_planner.layer.preference_axis_router import (
    AxisTemporalKinematicEgoSignedOutputAdapter,
    EgoSignedOutputAdapter,
    KinematicEgoSignedOutputAdapter,
    PreferenceAxisRouter,
    SignedPreferenceAxisRouter,
)
from baseline.model.style_planner.guidance.preference_energy import ConditionalPreferenceEnergy


SIGNED_ROUTER_DIFFUSION_GATE_MODES = (
    "all_steps",
    "free_drive_terminal_only",
)


def _signed_router_diffusion_gate(
    diffusion_time: torch.Tensor,
    free_drive_mask: torch.Tensor,
    *,
    mode: str,
    terminal_t_max: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-sample A3.8 residual gate and terminal-active mask.

    ``free_drive_terminal_only`` suppresses the free-drive residual at every
    denoiser call except the final DPM denoise-to-zero call.  Non-free-drive
    samples deliberately retain the historical all-step residual path.
    """

    gate_mode = str(mode)
    if gate_mode not in SIGNED_ROUTER_DIFFUSION_GATE_MODES:
        raise ValueError(
            "signed_router_diffusion_gate_mode must be 'all_steps' or "
            f"'free_drive_terminal_only', got {gate_mode!r}"
        )
    free_drive = torch.as_tensor(
        free_drive_mask,
        device=diffusion_time.device,
        dtype=torch.bool,
    ).reshape(-1)
    time = torch.as_tensor(
        diffusion_time,
        device=free_drive.device,
    ).reshape(-1)
    if time.numel() == 1 and free_drive.numel() != 1:
        time = time.expand(free_drive.numel())
    if time.numel() != free_drive.numel():
        raise ValueError(
            "diffusion time and free-drive mask must have the same batch size"
        )

    if gate_mode == "all_steps":
        return torch.ones_like(time, dtype=dtype), torch.zeros_like(
            free_drive,
            dtype=torch.bool,
        )
    threshold = float(terminal_t_max)
    if threshold <= 0.0:
        raise ValueError("signed_router_terminal_t_max must be positive")
    terminal_active = time <= threshold
    sample_active = (~free_drive) | terminal_active
    return sample_active.to(dtype=dtype), terminal_active


def selftest_signed_router_diffusion_gate() -> dict[str, bool]:
    """Check A3.8 terminal execution and the legacy rollback path."""

    diffusion_time = torch.tensor([0.8, 0.001, 0.8, 0.0011])
    free_drive = torch.tensor([True, True, False, True])
    all_steps, _ = _signed_router_diffusion_gate(
        diffusion_time,
        free_drive,
        mode="all_steps",
        terminal_t_max=0.0011,
        dtype=torch.float32,
    )
    terminal_only, terminal_active = _signed_router_diffusion_gate(
        diffusion_time,
        free_drive,
        mode="free_drive_terminal_only",
        terminal_t_max=0.0011,
        dtype=torch.float32,
    )
    return {
        "all_steps_is_exact_legacy_path": bool(
            torch.equal(all_steps, torch.ones_like(all_steps))
        ),
        "free_drive_uses_only_terminal_call": bool(
            torch.equal(
                terminal_only,
                torch.tensor([0.0, 1.0, 1.0, 1.0]),
            )
        ),
        "car_follow_retains_all_steps": bool(terminal_only[2].item() == 1.0),
        "terminal_boundary_is_inclusive": bool(
            terminal_active.tolist() == [False, True, False, True]
        ),
    }


class Decoder(nn.Module):
    """
    扩散模型解码器：基于条件扩散模型的轨迹生成器

    该模块使用扩散概率模型（Diffusion Probabilistic Model）生成自车和周围智能体的未来轨迹：
    1. 训练阶段：通过去噪过程学习轨迹分布，预测噪声或原始数据
    2. 推理阶段：使用 DPM-Solver 从噪声中生成高质量轨迹样本
    3. 条件机制：融合场景编码、路线信息和当前状态作为生成条件
    4. 引导增强：可选的 classifier guidance 提升生成质量

    核心组件：
    - DiT (Diffusion Transformer): 基于 Transformer 的去噪网络
    - RouteEncoder: 路线特征编码器
    - VPSDE_linear: 线性方差保持随机微分方程
    - DPM-Solver: 高效的扩散模型采样器

    Attributes:
        _predicted_neighbor_num (int): 需要预测的邻居智能体数量
        _future_len (int): 预测的未来时间步长度
        _sde (VPSDE_linear): 前向扩散过程的 SDE 定义
        dit (DiT): 扩散 Transformer 主网络
        _state_normalizer (StateNormalizer): 状态归一化器
        _observation_normalizer (ObservationNormalizer): 观测归一化器
        _guidance_fn: 引导函数（用于 classifier guidance）
    """
    def __init__(self, config):
        """
        初始化解码器

        Args:
            config (Config): 配置对象，包含以下关键参数：
                - predicted_neighbor_num: 预测的邻居数量
                - future_len: 未来轨迹长度
                - decoder_drop_path_rate: Decoder 的 DropPath 比率
                - route_num: 路线数量
                - lane_len: 车道长度
                - encoder_drop_path_rate: Encoder 的 DropPath 比率
                - hidden_dim: 隐藏层维度
                - decoder_depth: Decoder 层数
                - num_heads: 注意力头数
                - diffusion_model_type: 扩散模型类型 ("score" 或 "x_start")
                - state_normalizer: 状态归一化器
                - observation_normalizer: 观测归一化器
                - guidance_fn: 引导函数
        """
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()
        self._diffusion_steps = int(getattr(config, "diffusion_steps", 10))
        self._warm_start_diffusion_steps = int(
            getattr(config, "warm_start_diffusion_steps", self._diffusion_steps)
        )
        if self._warm_start_diffusion_steps < 2:
            raise ValueError("warm_start_diffusion_steps must be at least 2 for second-order DPM-Solver")
        self._style_value_dim = int(getattr(config, "style_value_dim", 0))
        self._global_style_condition_dim = int(
            getattr(config, "global_style_condition_dim", self._style_value_dim)
        )
        self._phase_style_condition_dim = int(getattr(config, "phase_style_condition_dim", 0))
        self._phase_style_num_phases = int(getattr(config, "phase_style_num_phases", 0))
        self._phase_style_flat_dim = int(getattr(config, "phase_style_flat_dim", 0))
        self._cfg_guidance_scale = float(getattr(config, "cfg_guidance_scale", 1.0))
        self._normal_anchor_cfg_enabled = bool(
            getattr(config, "normal_anchor_cfg_enabled", False)
        )
        self._style_condition_encoder = str(
            getattr(config, "style_condition_encoder", "mlp")
        )
        self._axis_router_token_dim = int(
            getattr(config, "axis_router_token_dim", 64)
        )
        self._signed_router_injection_mode = str(
            getattr(config, "signed_router_injection_mode", "global_adaln")
        )
        self._signed_router_diffusion_gate_mode = str(
            getattr(config, "signed_router_diffusion_gate_mode", "all_steps")
        )
        self._signed_router_terminal_t_max = float(
            getattr(config, "signed_router_terminal_t_max", 0.0011)
        )
        self._kinematic_ego_basis_count = int(
            getattr(config, "kinematic_ego_basis_count", 6)
        )
        self._use_style_condition = bool(getattr(config, "use_style_condition", self._style_value_dim > 0))
        self._use_phase_style_condition = bool(
            getattr(config, "use_phase_style_condition", self._phase_style_flat_dim > 0)
        )
        self._use_temporal_style_gate = bool(getattr(config, "use_temporal_style_gate", False))
        self._temporal_gate_hidden_dim = int(getattr(config, "temporal_gate_hidden_dim", config.hidden_dim))

        self.dit = DiT(
            sde=self._sde, 
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= (config.future_len + 1) * 4, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type,
            style_condition_dim=self._style_value_dim,
            global_style_condition_dim=self._global_style_condition_dim,
            phase_style_condition_dim=self._phase_style_condition_dim,
            phase_style_num_phases=self._phase_style_num_phases,
            phase_style_flat_dim=self._phase_style_flat_dim,
            use_style_condition=self._use_style_condition,
            use_phase_style_condition=self._use_phase_style_condition,
            use_temporal_style_gate=self._use_temporal_style_gate,
            temporal_gate_hidden_dim=self._temporal_gate_hidden_dim,
            style_condition_encoder=self._style_condition_encoder,
            axis_router_token_dim=self._axis_router_token_dim,
            signed_router_injection_mode=self._signed_router_injection_mode,
            signed_router_diffusion_gate_mode=(
                self._signed_router_diffusion_gate_mode
            ),
            signed_router_terminal_t_max=self._signed_router_terminal_t_max,
            kinematic_ego_basis_count=self._kinematic_ego_basis_count,
        )
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._preference_energy_enabled = bool(
            getattr(config, "preference_energy_enabled", False)
        )
        self._preference_energy_guidance_scale = float(
            getattr(config, "preference_energy_guidance_scale", 0.0)
        )
        self._preference_energy_grad_clip = float(
            getattr(config, "preference_energy_grad_clip", 0.5)
        )
        self._preference_energy_t_min = float(
            getattr(config, "preference_energy_t_min", 0.01)
        )
        self._preference_energy_t_max = float(
            getattr(config, "preference_energy_t_max", 0.55)
        )
        if self._preference_energy_enabled:
            if self.dit.model_type != "x_start":
                raise ValueError("preference energy requires diffusion_model_type='x_start'")
            self.preference_energy = ConditionalPreferenceEnergy(
                normalization_path=str(
                    getattr(config, "preference_energy_normalization_path", "")
                ),
                conditional_rank_model_path=str(
                    getattr(config, "preference_energy_rank_model_path", "")
                ),
                state_normalizer=self._state_normalizer,
                neighbours=int(getattr(config, "preference_energy_neighbours", 64)),
                min_shared_condition_features=int(
                    getattr(config, "preference_energy_min_shared_features", 3)
                ),
                cdf_temperature=float(
                    getattr(config, "preference_energy_cdf_temperature", 0.04)
                ),
                preference_weight=float(
                    getattr(config, "preference_energy_preference_weight", 1.0)
                ),
                safety_weight=float(
                    getattr(config, "preference_energy_safety_weight", 4.0)
                ),
                free_drive_accel_support_mode=str(
                    getattr(
                        config,
                        "free_drive_accel_support_mode",
                        "self_generated",
                    )
                ),
                dt=float(getattr(config, "preference_loss_dt", 0.1)),
            )
        else:
            self.preference_energy = None
        
        self._guidance_fn = config.guidance_fn
        
    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process t in [0, 1],                 [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }
        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        is_diffusion_loss_pass = ("sampled_trajectories" in inputs) and ("diffusion_time" in inputs)

        if is_diffusion_loss_pass:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            diffusion_time = inputs['diffusion_time']
            style_condition = self._resolve_style_condition(inputs, B)

            denoised = self.dit(
                sampled_trajectories,
                diffusion_time,
                style_condition=style_condition,
                cross_c=ego_neighbor_encoding,
                route_lanes=route_lanes,
                neighbor_current_mask=neighbor_current_mask,
                phase_time_mask=self._resolve_phase_time_mask(inputs, B),
            ).reshape(B, P, -1, 4)
            temporal_debug = self.dit.pop_last_temporal_gate_outputs()
            router_debug = self.dit.pop_last_axis_router_outputs()
            if self.dit.model_type == "x_start":
                outputs = {
                    "x_start": denoised,
                    "score": denoised,
                }
                if temporal_debug is not None:
                    outputs.update(temporal_debug)
                if router_debug is not None:
                    outputs.update(router_debug)
                if (
                    self.preference_energy is not None
                    and style_condition is not None
                    and not bool(inputs.get("disable_preference_energy", False))
                ):
                    prepared_energy = self.preference_energy.prepare(
                        inputs,
                        style_condition,
                    )
                    outputs.update(
                        self.preference_energy.energy_terms(
                            denoised,
                            inputs,
                            prepared_energy,
                        )
                    )
                return outputs
            outputs = {
                "score": denoised
            }
            if temporal_debug is not None:
                outputs.update(temporal_debug)
            if router_debug is not None:
                outputs.update(router_debug)
            return outputs
        else:
            warm_start_enabled = bool(inputs.get("warm_start_enabled", False))
            if warm_start_enabled:
                xT, sampling_t_start = self._build_warm_start_state(
                    inputs,
                    current_states,
                    neighbor_current_mask,
                )
                diffusion_steps = self._warm_start_diffusion_steps
            else:
                # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
                xT = torch.cat([
                    current_states[:, :, None],
                    torch.randn(B, P, self._future_len, 4, device=current_states.device) * 0.5,
                ], dim=2).reshape(B, P, -1)
                sampling_t_start = None
                diffusion_steps = self._diffusion_steps

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)
            
            style_condition = self._resolve_style_condition(inputs, B)
            style_active = bool(
                style_condition is not None
                and torch.any(style_condition.abs() > 1e-6)
            )
            temporal_debug = None
            if style_active:
                temporal_debug = self.dit.inspect_temporal_style_condition(
                    style_condition,
                    B,
                    device=current_states.device,
                    dtype=current_states.dtype,
                )
            normal_anchor_condition = self._resolve_normal_anchor_condition(inputs, B)
            normal_anchor_active = bool(
                self._normal_anchor_cfg_enabled
                and normal_anchor_condition is not None
                and torch.any(normal_anchor_condition.abs() > 1e-6)
            )
            if style_active:
                cfg_reference = (
                    normal_anchor_condition
                    if normal_anchor_active
                    else torch.zeros_like(style_condition)
                )
                model_wrapper_params = {
                    "condition": style_condition,
                    "unconditional_condition": cfg_reference,
                    "guidance_scale": float(inputs.get("cfg_guidance_scale", self._cfg_guidance_scale)),
                    "guidance_type": "classifier-free",
                }
            else:
                model_wrapper_params = {
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                            "phase_time_mask": self._resolve_phase_time_mask(inputs, B),
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                }

            prepared_energy = None
            energy_guidance_applied = False
            command_strength = 0.0
            if style_active and style_condition is not None:
                axis_mask = style_condition[:, 3:6].clamp(0.0, 1.0)
                target_delta = (style_condition[:, 0:3] - 0.5).abs()
                command_strength = float(
                    ((target_delta * axis_mask).sum() / axis_mask.sum().clamp_min(1.0))
                    .detach()
                    .cpu()
                )
            if (
                self.preference_energy is not None
                and style_active
            ):
                prepared_energy = self.preference_energy.prepare(
                    inputs,
                    style_condition,
                )
                if (
                    bool(prepared_energy["enabled"].any())
                    and command_strength > 1e-5
                    and self._preference_energy_guidance_scale > 0.0
                ):
                    model_wrapper_params.update(
                        {
                            "energy_fn": self.preference_energy.energy_from_model_output,
                            "energy_scale": self._preference_energy_guidance_scale,
                            "energy_kwargs": {
                                "inputs": inputs,
                                "prepared": prepared_energy,
                            },
                            "energy_grad_clip": self._preference_energy_grad_clip,
                            "energy_t_min": self._preference_energy_t_min,
                            "energy_t_max": self._preference_energy_t_max,
                        }
                    )
                    energy_guidance_applied = True

            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                            "phase_time_mask": self._resolve_phase_time_mask(inputs, B),
                        },
                        dpm_solver_params={
                            "correcting_xt_fn":initial_state_constraint,
                        },
                        model_wrapper_params=model_wrapper_params,
                        diffusion_steps=diffusion_steps,
                        sample_params=(
                            {"t_start": sampling_t_start}
                            if sampling_t_start is not None
                            else {}
                        ),
                )
            if (
                self.preference_energy is not None
                and prepared_energy is not None
                and bool(prepared_energy["enabled"].any())
            ):
                # Always audit the final generated conditional percentiles,
                # including rho=0 where the energy gradient is intentionally
                # disabled. This gives the rho-sweep evaluator a common output
                # contract without changing the trajectory.
                self.preference_energy.energy_terms(
                    x0.reshape(B, P, -1, 4),
                    inputs,
                    prepared_energy,
                )
            if temporal_debug is None:
                temporal_debug = self.dit.pop_last_temporal_gate_outputs()
            router_debug = self.dit.pop_last_axis_router_outputs()
            energy_debug = (
                self.preference_energy.pop_last_diagnostics()
                if self.preference_energy is not None
                else None
            )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            outputs = {
                "prediction": x0,
                "normal_anchor_cfg_used": torch.full(
                    (B,),
                    normal_anchor_active and style_active,
                    dtype=torch.bool,
                    device=x0.device,
                ),
                "empty_cfg_reference_used": torch.full(
                    (B,),
                    style_active and not normal_anchor_active,
                    dtype=torch.bool,
                    device=x0.device,
                ),
                "preference_energy_guidance_used": torch.full(
                    (B,),
                    energy_guidance_applied,
                    dtype=torch.bool,
                    device=x0.device,
                ),
            }
            if temporal_debug is not None:
                outputs.update(temporal_debug)
            if router_debug is not None:
                outputs.update(router_debug)
            if energy_debug is not None:
                outputs.update(energy_debug)
            return outputs

    def _build_warm_start_state(self, inputs, current_states, neighbor_current_mask):
        """Build a training-consistent joint state at a shared intermediate time.

        Ego future comes from the map maneuver anchor.  Neighbor futures use a
        constant-velocity anchor so every jointly denoised token starts at the
        same VP-SDE time without changing the trained DiT architecture.
        """

        ego_anchor = inputs.get("warm_start_ego_anchor")
        if ego_anchor is None:
            raise KeyError("warm_start_enabled requires `warm_start_ego_anchor`")
        if ego_anchor.dim() == 2:
            ego_anchor = ego_anchor.unsqueeze(0)
        B, P, _ = current_states.shape
        expected_shape = (B, self._future_len, 4)
        if tuple(ego_anchor.shape) != expected_shape:
            raise ValueError(
                "warm_start_ego_anchor must have shape "
                f"{expected_shape}, got {tuple(ego_anchor.shape)}"
            )
        ego_anchor = ego_anchor.to(device=current_states.device, dtype=current_states.dtype)

        t_start_value = inputs.get("warm_start_t", 0.30)
        if torch.is_tensor(t_start_value):
            values = t_start_value.detach().reshape(-1)
            if values.numel() == 0:
                raise ValueError("warm_start_t tensor cannot be empty")
            if not torch.allclose(values, values[:1].expand_as(values), atol=1e-7, rtol=0.0):
                raise ValueError("All candidates in one DPM-Solver batch must share warm_start_t")
            t_start = float(values[0].cpu())
        else:
            t_start = float(t_start_value)
        if not 1e-3 < t_start <= 1.0:
            raise ValueError(f"warm_start_t must be in (1e-3, 1], got {t_start}")

        # ``neighbor_agents_past`` is observation-normalized before reaching
        # the model.  Constant-velocity extrapolation must use physical metres
        # and m/s, therefore the warm-start planner supplies a raw copy solely
        # for anchor construction.  It is not consumed by the encoder/DiT.
        neighbor_past = inputs.get("warm_start_neighbor_past_raw")
        if neighbor_past is None:
            raise KeyError(
                "warm_start_enabled requires `warm_start_neighbor_past_raw` in physical units"
            )
        neighbor_past = neighbor_past[:, : self._predicted_neighbor_num]
        if neighbor_past.shape[0] != B or neighbor_past.shape[-1] < 6:
            raise ValueError(
                "warm_start_neighbor_past_raw must have shape [B, N, history, >=6]; "
                f"got {tuple(neighbor_past.shape)}"
            )
        neighbor_past = neighbor_past.to(
            device=current_states.device,
            dtype=current_states.dtype,
        )
        neighbor_current = neighbor_past[:, :, -1, :]
        time = (
            torch.arange(1, self._future_len + 1, device=current_states.device, dtype=current_states.dtype)
            * float(inputs.get("warm_start_dt", 0.1))
        )
        neighbor_xy = neighbor_current[..., :2, None] + neighbor_current[..., 4:6, None] * time
        neighbor_xy = neighbor_xy.permute(0, 1, 3, 2)
        neighbor_heading = neighbor_current[..., 2:4, None].permute(0, 1, 3, 2)
        neighbor_heading = neighbor_heading.expand(-1, -1, self._future_len, -1)
        neighbor_anchor = torch.cat([neighbor_xy, neighbor_heading], dim=-1)

        joint_anchor = torch.cat([ego_anchor[:, None], neighbor_anchor], dim=1)
        joint_anchor = self._state_normalizer(joint_anchor)
        joint_anchor[:, 1:] = joint_anchor[:, 1:].masked_fill(
            neighbor_current_mask[:, :, None, None], 0.0
        )

        diffusion_time = torch.full(
            (B,), t_start, device=current_states.device, dtype=current_states.dtype
        )
        mean, std = self._sde.marginal_prob(joint_anchor, diffusion_time)
        noise = torch.randn_like(mean)
        if bool(inputs.get("warm_start_shared_noise", True)) and B > 1:
            # Common random numbers make keep/left/right comparable: candidate
            # differences are driven by their anchors, not unrelated noise.
            noise = noise[:1].expand_as(mean)
        x_t_future = mean + std * noise
        x_t = torch.cat([current_states[:, :, None], x_t_future], dim=2).reshape(B, P, -1)
        return x_t, t_start

    def _resolve_style_condition(self, inputs, batch_size: int):
        if (not self._use_style_condition) or self._style_value_dim <= 0:
            return None
        style_condition = inputs.get("style_value_condition")
        if style_condition is None:
            return None
        if style_condition.dim() == 1:
            style_condition = style_condition.unsqueeze(0)
        if style_condition.shape[0] == 1 and batch_size > 1:
            style_condition = style_condition.expand(batch_size, -1)
        return style_condition

    def _resolve_normal_anchor_condition(self, inputs, batch_size: int):
        if (not self._use_style_condition) or self._style_value_dim <= 0:
            return None
        condition = inputs.get("normal_anchor_style_value_condition")
        if condition is None:
            return None
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        if condition.shape[0] == 1 and batch_size > 1:
            condition = condition.expand(batch_size, -1)
        if condition.shape[-1] != self._style_value_dim:
            raise ValueError(
                "normal_anchor_style_value_condition dimension mismatch: "
                f"expected {self._style_value_dim}, got {condition.shape[-1]}"
            )
        return condition

    def _resolve_phase_time_mask(self, inputs, batch_size: int):
        if (not self._use_phase_style_condition) or self._phase_style_num_phases <= 0:
            return None
        phase_time_mask = inputs.get("phase_time_mask")
        if phase_time_mask is None:
            return None
        if phase_time_mask.dim() == 2:
            phase_time_mask = phase_time_mask.unsqueeze(0)
        if phase_time_mask.shape[0] == 1 and batch_size > 1:
            phase_time_mask = phase_time_mask.expand(batch_size, -1, -1)
        return phase_time_mask

        
class RouteEncoder(nn.Module):
    """
    路线编码器：处理规划路线的特征表示

    负责编码车辆的行驶路线信息，用于引导轨迹生成：
    1. 使用 MLP Mixer 架构处理路线点序列
    2. 提取路线的几何特征和方向信息
    3. 为轨迹生成提供全局路径指引

    Attributes:
        _channel (int): 通道维度
        channel_pre_project (Mlp): 通道维度预投影层
        token_pre_project (Mlp): Token 维度预投影层
        Mixer (MixerBlock): MLP Mixer 块
        norm (LayerNorm): 归一化层
        emb_project (Mlp): 特征投影层
    """
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        """
       初始化路线编码器

       Args:
           route_num (int): 路线数量
           lane_len (int): 每条车道的长度（点数）
           drop_path_rate (float): DropPath 比率，用于正则化
           hidden_dim (int): 隐藏层维度
           tokens_mlp_dim (int): Token MLP 的中间维度
           channels_mlp_dim (int): Channel MLP 的中间维度
       """
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        """
        前向传播：编码路线特征

        Args:
            x (torch.Tensor): 路线数据 [B, P_route, V_points, D]
                - B: 批次大小
                - P_route: 路线数量
                - V_points: 每条路线的点数
                - D: 特征维度（只使用前 4 维：x, y, cos, sin）

        Returns:
            torch.Tensor: 编码后的路线特征 [B, hidden_dim]
                - 如果存在有效路线，返回编码后的特征
                - 如果所有路线都无效，返回全零张量

        Note:
            该编码器只处理位置和方向信息（x, y, cos, sin），
            不包含边界、速度限制或交通灯信息。
        """
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        # 只保留前 4 维：x, y, cos, sin（位置和方向信息）
        x = x[..., :4]

        B, P, V, _ = x.shape
        # 生成掩码：判断哪些点是无效的（全 0）
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        # 判断哪些路线完全无效
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        # 判断批次中哪些样本完全无效
        mask_b = torch.sum(~mask_p, dim=-1) == 0

        # 展平为 (B, P*V, D) 便于处理
        x = x.view(B, P * V, -1)

        # 记录有效样本的索引
        valid_indices = ~mask_b.view(-1)
        # 只处理有效数据
        x = x[valid_indices]

        # ========== MLP Mixer 编码 ==========
        # 通道混合：在每个点上进行特征变换
        x = self.channel_pre_project(x)
        # 转置为 (B, D, P*V) 以便在空间维度上进行混合
        x = x.permute(0, 2, 1)
        # 空间混合：跨路线点交换信息
        x = self.token_pre_project(x)
        # 转置回 (B, P*V, D)
        x = x.permute(0, 2, 1)
        # 使用 MixerBlock 进行深度特征提取
        x = self.Mixer(x)

        # ========== 空间维度池化 ==========
        # 平均池化：聚合所有路线点的信息
        x = torch.mean(x, dim=1)

        # ========== 特征投影 ==========
        x = self.emb_project(self.norm(x))

        # ========== 填充有效部分 ==========
        # 创建全零结果张量
        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        # 填充有效部分
        x_result[valid_indices] = x

        return x_result.view(B, -1)


class DiT(nn.Module):
    """
    扩散 Transformer（Diffusion Transformer）：去噪网络核心

    基于 Transformer 架构的扩散模型去噪网络，负责预测噪声或原始数据：
    1. 使用自注意力机制处理多智能体轨迹
    2. 使用交叉注意力融合场景上下文信息
    3. 使用时间嵌入编码扩散过程的进度
    4. 使用路线编码作为全局条件

    架构设计：
    - 输入嵌入：将轨迹数据投影到隐藏维度 + 智能体类型嵌入
    - 时间嵌入：通过 MLP 编码扩散时间步
    - 路由编码：通过 RouteEncoder 提取路线特征
    - DiT Blocks：多层 Transformer 进行特征提取
    - 输出层：预测去噪结果（分数函数或原始数据）

    Attributes:
        _model_type (str): 模型类型 ("score" 预测噪声 / "x_start" 预测原始数据)
        route_encoder (RouteEncoder): 路线编码器
        agent_embedding (Embedding): 智能体类型嵌入（自车 vs 邻居）
        preproj (Mlp): 输入预投影层
        t_embedder (TimestepEmbedder): 时间步嵌入器
        blocks (ModuleList): DiTBlock 序列
        final_layer (FinalLayer): 输出层
        _sde (SDE): 随机微分方程对象
        marginal_prob_std: 边缘概率的标准差函数
    """
    def __init__(
        self,
        sde: SDE,
        route_encoder: nn.Module,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        model_type="x_start",
        style_condition_dim: int = 0,
        global_style_condition_dim: int = 0,
        phase_style_condition_dim: int = 0,
        phase_style_num_phases: int = 0,
        phase_style_flat_dim: int = 0,
        use_style_condition: bool = False,
        use_phase_style_condition: bool = False,
        use_temporal_style_gate: bool = False,
        temporal_gate_hidden_dim: int = 0,
        style_condition_encoder: str = "mlp",
        axis_router_token_dim: int = 64,
        signed_router_injection_mode: str = "global_adaln",
        signed_router_diffusion_gate_mode: str = "all_steps",
        signed_router_terminal_t_max: float = 0.0011,
        kinematic_ego_basis_count: int = 6,
    ):
        """
        初始化 DiT

        Args:
            sde (SDE): 随机微分方程对象，定义前向扩散过程
            route_encoder (RouteEncoder): 路线编码器实例
            depth (int): Transformer 层数
            output_dim (int): 输出维度（轨迹维度）
            hidden_dim (int): 隐藏层维度
            heads (int): 注意力头数
            dropout (float): Dropout 比率
            mlp_ratio (float): MLP 隐藏层扩展比例
            model_type (str): 模型类型
                - "score": 预测噪声 ε（分数匹配）
                - "x_start": 预测原始数据 x_0
        """
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self._style_condition_dim = int(style_condition_dim)
        self._global_style_condition_dim = int(
            global_style_condition_dim if global_style_condition_dim > 0 else style_condition_dim
        )
        self._phase_style_condition_dim = int(phase_style_condition_dim)
        self._phase_style_num_phases = int(phase_style_num_phases)
        self._phase_style_flat_dim = int(phase_style_flat_dim)
        self.style_condition_encoder = str(style_condition_encoder)
        self.signed_router_injection_mode = str(signed_router_injection_mode)
        self.signed_router_diffusion_gate_mode = str(
            signed_router_diffusion_gate_mode
        )
        self.signed_router_terminal_t_max = float(signed_router_terminal_t_max)
        if (
            self.signed_router_diffusion_gate_mode
            not in SIGNED_ROUTER_DIFFUSION_GATE_MODES
        ):
            raise ValueError(
                "signed_router_diffusion_gate_mode must be 'all_steps' or "
                "'free_drive_terminal_only', got "
                f"{self.signed_router_diffusion_gate_mode!r}"
            )
        if (
            self.signed_router_diffusion_gate_mode
            == "free_drive_terminal_only"
            and self.signed_router_injection_mode
            != "ego_axis_temporal_residual"
        ):
            raise ValueError(
                "free-drive terminal-only execution requires "
                "signed_router_injection_mode='ego_axis_temporal_residual'"
            )
        if (
            self.signed_router_diffusion_gate_mode
            == "free_drive_terminal_only"
            and not 0.001 <= self.signed_router_terminal_t_max <= 0.0011
        ):
            raise ValueError(
                "free-drive terminal-only execution requires "
                "signed_router_terminal_t_max in [0.001, 0.0011]"
            )
        if self.signed_router_injection_mode not in {
            "global_adaln",
            "ego_output_residual",
            "ego_kinematic_residual",
            "ego_axis_temporal_residual",
        }:
            raise ValueError(
                "signed_router_injection_mode must be 'global_adaln', "
                "'ego_output_residual', 'ego_kinematic_residual', or "
                "'ego_axis_temporal_residual', got "
                f"{self.signed_router_injection_mode!r}"
            )
        if (
            self.signed_router_injection_mode
            in {
                "ego_output_residual",
                "ego_kinematic_residual",
                "ego_axis_temporal_residual",
            }
            and self.style_condition_encoder != "axis_router_v2_signed"
        ):
            raise ValueError(
                "ego output residual injection requires style_condition_encoder="
                "'axis_router_v2_signed'"
            )
        if self.style_condition_encoder not in {
            "mlp",
            "axis_router_v1",
            "axis_router_v2_signed",
        }:
            raise ValueError(
                "style_condition_encoder must be 'mlp', 'axis_router_v1', "
                "or 'axis_router_v2_signed', "
                f"got {self.style_condition_encoder!r}"
            )

        # 智能体类型嵌入：区分自车（index=0）和邻居（index=1）
        self.agent_embedding = nn.Embedding(2, hidden_dim)

        # 输入预投影：将轨迹数据投影到隐藏维度
        self.preproj = Mlp(
            in_features=output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.
        )

        # 时间步嵌入器：将连续时间 t 编码为高维特征
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.use_style_condition = bool(use_style_condition and self._global_style_condition_dim > 0)
        self.use_phase_style_condition = bool(
            use_phase_style_condition
            and self._phase_style_num_phases > 0
            and self._phase_style_condition_dim > 0
            and self._phase_style_flat_dim > 0
        )
        self.use_temporal_style_gate = bool(
            use_temporal_style_gate
            and self.use_phase_style_condition
            and self._phase_style_num_phases == 2
        )
        self._last_temporal_gate_outputs = None
        self._last_axis_router_outputs = None
        if self.use_style_condition:
            if self.style_condition_encoder in {
                "axis_router_v1",
                "axis_router_v2_signed",
            }:
                if self._global_style_condition_dim != 12:
                    raise ValueError(
                        f"{self.style_condition_encoder} requires "
                        "global_style_condition_dim=12, "
                        f"got {self._global_style_condition_dim}"
                    )
                if self.style_condition_encoder == "axis_router_v2_signed":
                    self.style_condition_proj = SignedPreferenceAxisRouter(
                        hidden_dim=hidden_dim,
                        token_dim=int(axis_router_token_dim),
                    )
                else:
                    self.style_condition_proj = PreferenceAxisRouter(
                        hidden_dim=hidden_dim,
                        token_dim=int(axis_router_token_dim),
                        dropout=float(dropout),
                    )
            else:
                self.style_condition_proj = nn.Sequential(
                    nn.LayerNorm(self._global_style_condition_dim),
                    nn.Linear(self._global_style_condition_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        else:
            self.style_condition_proj = None
        if (
            self.use_style_condition
            and self.style_condition_encoder == "axis_router_v2_signed"
            and self.signed_router_injection_mode == "ego_output_residual"
        ):
            self.ego_style_output_proj = EgoSignedOutputAdapter(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
        elif (
            self.use_style_condition
            and self.style_condition_encoder == "axis_router_v2_signed"
            and self.signed_router_injection_mode == "ego_kinematic_residual"
        ):
            self.ego_style_output_proj = KinematicEgoSignedOutputAdapter(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                basis_count=int(kinematic_ego_basis_count),
            )
        elif (
            self.use_style_condition
            and self.style_condition_encoder == "axis_router_v2_signed"
            and self.signed_router_injection_mode == "ego_axis_temporal_residual"
        ):
            self.ego_style_output_proj = (
                AxisTemporalKinematicEgoSignedOutputAdapter(
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    basis_count=int(kinematic_ego_basis_count),
                )
            )
        else:
            self.ego_style_output_proj = None
        if self.use_phase_style_condition:
            self.phase_style_step_proj = nn.Sequential(
                nn.LayerNorm(self._phase_style_condition_dim),
                nn.Linear(self._phase_style_condition_dim, 4),
            )
        else:
            self.phase_style_step_proj = None
        if self.use_temporal_style_gate:
            gate_input_dim = self._global_style_condition_dim + self._phase_style_flat_dim
            gate_hidden_dim = int(temporal_gate_hidden_dim if temporal_gate_hidden_dim > 0 else hidden_dim)
            self.temporal_gate_stem = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(gate_hidden_dim),
            )
            self.temporal_near_gate_head = nn.Linear(gate_hidden_dim, self._phase_style_condition_dim)
            self.temporal_far_gate_head = nn.Linear(gate_hidden_dim, self._phase_style_condition_dim)
        else:
            self.temporal_gate_stem = None
            self.temporal_near_gate_head = None
            self.temporal_far_gate_head = None

        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])

        # 输出层：预测去噪结果
        self.final_layer = FinalLayer(hidden_dim, output_dim)

        # 保存 SDE 对象及其边缘概率函数
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        """获取模型类型"""
        return self._model_type

    def forward(
        self,
        x,
        t,
        style_condition=None,
        cross_c=None,
        route_lanes=None,
        neighbor_current_mask=None,
        phase_time_mask=None,
    ):
        """
        DiT 前向传播

        Args:
            x (torch.Tensor): 输入轨迹（带噪或纯净）[B, P, output_dim]
                - B: 批次大小
                - P: 智能体数量（1 个自车 + P_pred 个邻居）
                - output_dim: (1+V_future) * 4，包含当前帧和未来帧的 (x, y, cos, sin)

            t (torch.Tensor): 扩散时间步 [B]
                - t ∈ [0, 1]，0 表示无噪声，1 表示纯噪声

            cross_c (torch.Tensor): 交叉注意力上下文 [B, N, D]
                - 场景上下文特征（来自 Encoder）
                - 用于融合交通场景信息

            route_lanes (torch.Tensor): 路线车道数据 [B, M, V, D]
                - 用于条件生成的全局路径指引

            neighbor_current_mask (torch.Tensor): 邻居掩码 [B, P_pred]
                - True 表示无效邻居，False 表示有效
                - 用于自注意力掩码，避免关注无效智能体

        Returns:
            torch.Tensor: 去噪预测结果 [B, P, output_dim]
                - 如果 model_type="score"：返回 ε/σ(t)（归一化噪声）
                - 如果 model_type="x_start"：返回 x_0（原始数据）

        Note:
            对于 score 类型，输出需要除以边际标准差 σ(t)，
            这是为了匹配 SDE 的分数函数 ∇_x log p_t(x)。
        """
        B, P, _ = x.shape
        self._last_temporal_gate_outputs = None
        self._last_axis_router_outputs = None
        style_global_condition, phase_style_condition = self._split_style_condition(style_condition, B)
        phase_style_condition = self._apply_temporal_style_gate(
            style_global_condition,
            phase_style_condition,
            device=x.device,
            dtype=x.dtype,
        )
        phase_bias = self._build_phase_trajectory_bias(
            phase_style_condition,
            phase_time_mask,
            device=x.device,
            dtype=x.dtype,
        )
        if phase_bias is not None:
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + phase_bias

        # ========== 输入投影 ==========
        # 将轨迹数据投影到隐藏维度
        x = self.preproj(x)

        # ========== 智能体类型嵌入 ==========
        # 自车（index=0）和邻居（index=1）使用不同的嵌入
        # x_embedding: [P, D] = [1 个自车 + (P-1) 个邻居]
        x_embedding = torch.cat([
            self.agent_embedding.weight[0][None, :],  # [1, D] - 自车
            self.agent_embedding.weight[1][None, :].expand(P - 1, -1)  # [P-1, D] - 邻居
        ], dim=0)
        # 扩展到 batch 维度：[B, P, D]
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)
        # 添加到输入特征
        x = x + x_embedding

        # ========== 路线编码 + 时间嵌入 ==========
        # 编码路线特征：[B, D]
        route_encoding = self.route_encoder(route_lanes)
        # 作为条件特征 y
        y = route_encoding
        # 添加时间嵌入：[B, D]
        y = y + self.t_embedder(t)
        style_residual = None
        axis_style_residual = None
        router_outputs = None
        if self.style_condition_proj is not None and style_global_condition is not None:
            style_global_condition = style_global_condition.to(
                device=y.device,
                dtype=y.dtype,
            )
            if self.style_condition_encoder in {
                "axis_router_v1",
                "axis_router_v2_signed",
            }:
                scene_context = route_encoding
                if cross_c is not None:
                    scene_context = scene_context + cross_c.mean(dim=1)
                style_residual, router_outputs = self.style_condition_proj(
                    style_global_condition,
                    scene_context,
                )
                axis_style_residual = router_outputs.pop(
                    "_axis_router_axis_residual",
                    None,
                )
                self._last_axis_router_outputs = router_outputs
                if self.signed_router_injection_mode == "global_adaln":
                    y = y + style_residual
            else:
                y = y + self.style_condition_proj(style_global_condition)

        # ========== 构建注意力掩码 ==========
        # attn_mask: [B, P] - True 表示需要 mask 的位置
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        # 对邻居应用掩码（自车 index=0 永远不 mask），注意这里的是平均注意力
        attn_mask[:, 1:] = neighbor_current_mask

        # ========== DiT Blocks ==========
        # 通过多层 Transformer 进行特征提取
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        # ========== 输出层 ==========
        # 预测去噪结果
        x = self.final_layer(x, y)
        if self.ego_style_output_proj is not None and style_residual is not None:
            if self.signed_router_injection_mode == "ego_kinematic_residual":
                ego_residual = self.ego_style_output_proj(
                    style_residual,
                    x[:, 0, :],
                ).to(dtype=x.dtype)
            elif self.signed_router_injection_mode == "ego_axis_temporal_residual":
                if axis_style_residual is None:
                    raise RuntimeError(
                        "A3.7 axis-temporal adapter requires per-axis signed "
                        "Router residuals"
                    )
                free_drive_mask = style_global_condition[:, 6] > 0.5
                ego_residual, axis_temporal_debug = self.ego_style_output_proj(
                    style_residual,
                    axis_style_residual,
                    x[:, 0, :],
                    free_drive_mask,
                )
                ego_residual = ego_residual.to(dtype=x.dtype)
                diffusion_gate, terminal_active = _signed_router_diffusion_gate(
                    t,
                    free_drive_mask,
                    mode=self.signed_router_diffusion_gate_mode,
                    terminal_t_max=self.signed_router_terminal_t_max,
                    dtype=x.dtype,
                )
                ego_residual = ego_residual * diffusion_gate[:, None]
                axis_temporal_debug.update(
                    {
                        "axis_temporal_diffusion_gate": diffusion_gate,
                        "axis_temporal_terminal_only_used": (
                            free_drive_mask
                            if self.signed_router_diffusion_gate_mode
                            == "free_drive_terminal_only"
                            else torch.zeros_like(free_drive_mask)
                        ),
                        "axis_temporal_terminal_active": (
                            free_drive_mask & terminal_active
                        ),
                        "axis_temporal_diffusion_time": t.reshape(-1),
                    }
                )
                if router_outputs is not None:
                    router_outputs.update(axis_temporal_debug)
            else:
                ego_residual = self.ego_style_output_proj(style_residual).to(
                    dtype=x.dtype
                )
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + ego_residual
            if router_outputs is not None:
                router_outputs["axis_router_ego_output_residual_l2"] = (
                    torch.linalg.norm(ego_residual, dim=-1)
                )

        # ========== 模型类型处理 ==========
        if self._model_type == "score":
            # 分数匹配：返回归一化的噪声预测
            # ε/σ(t) 用于匹配分数函数 ∇_x log p_t(x)
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            # 直接预测原始数据
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def _split_style_condition(self, style_condition, batch_size: int):
        if style_condition is None:
            return None, None
        if style_condition.dim() == 1:
            style_condition = style_condition.unsqueeze(0)
        if style_condition.shape[0] == 1 and batch_size > 1:
            style_condition = style_condition.expand(batch_size, -1)
        style_condition = style_condition.to(dtype=torch.float32)

        global_condition = None
        if self.use_style_condition:
            global_condition = style_condition[..., : self._global_style_condition_dim]

        phase_condition = None
        if self.use_phase_style_condition:
            phase_flat = style_condition[
                ...,
                self._global_style_condition_dim : self._global_style_condition_dim + self._phase_style_flat_dim,
            ]
            phase_condition = phase_flat.reshape(
                style_condition.shape[0],
                self._phase_style_num_phases,
                self._phase_style_condition_dim,
            )
        return global_condition, phase_condition

    def _apply_temporal_style_gate(self, style_global_condition, phase_style_condition, *, device, dtype):
        if (
            not self.use_temporal_style_gate
            or phase_style_condition is None
            or self.temporal_gate_stem is None
            or self.temporal_near_gate_head is None
            or self.temporal_far_gate_head is None
        ):
            return phase_style_condition

        if style_global_condition is None:
            style_global_condition = torch.zeros(
                (phase_style_condition.shape[0], self._global_style_condition_dim),
                device=device,
                dtype=dtype,
            )
        gate_input = torch.cat(
            [
                style_global_condition.to(device=device, dtype=dtype),
                phase_style_condition.reshape(phase_style_condition.shape[0], -1).to(device=device, dtype=dtype),
            ],
            dim=-1,
        )
        gate_hidden = self.temporal_gate_stem(gate_input)
        near_gate = torch.sigmoid(self.temporal_near_gate_head(gate_hidden))
        far_gate = torch.sigmoid(self.temporal_far_gate_head(gate_hidden))
        stage_gates = torch.stack([near_gate, far_gate], dim=1)
        gated_phase_condition = phase_style_condition.to(device=device, dtype=dtype) * stage_gates
        self._last_temporal_gate_outputs = {
            "temporal_near_gate": near_gate,
            "temporal_far_gate": far_gate,
            "temporal_near_condition": gated_phase_condition[:, 0, :],
            "temporal_far_condition": gated_phase_condition[:, 1, :],
        }
        return gated_phase_condition

    def pop_last_temporal_gate_outputs(self):
        payload = self._last_temporal_gate_outputs
        self._last_temporal_gate_outputs = None
        return payload

    def pop_last_axis_router_outputs(self):
        payload = self._last_axis_router_outputs
        self._last_axis_router_outputs = None
        return payload

    def inspect_temporal_style_condition(self, style_condition, batch_size: int, *, device, dtype):
        self._last_temporal_gate_outputs = None
        style_global_condition, phase_style_condition = self._split_style_condition(style_condition, batch_size)
        _ = self._apply_temporal_style_gate(
            style_global_condition,
            phase_style_condition,
            device=device,
            dtype=dtype,
        )
        return self.pop_last_temporal_gate_outputs()

    def _build_phase_trajectory_bias(self, phase_style_condition, phase_time_mask, *, device, dtype):
        if (
            phase_style_condition is None
            or phase_time_mask is None
            or self.phase_style_step_proj is None
        ):
            return None
        if phase_time_mask.dim() == 2:
            phase_time_mask = phase_time_mask.unsqueeze(0)
        phase_time_mask = phase_time_mask.to(device=device, dtype=dtype)
        if phase_time_mask.shape[1] != self._phase_style_num_phases:
            raise ValueError(
                f"Expected phase_time_mask with {self._phase_style_num_phases} phases, "
                f"got shape {tuple(phase_time_mask.shape)}"
            )
        phase_steps = self.phase_style_step_proj(
            phase_style_condition.to(device=device, dtype=dtype)
        )
        phase_bias = (
            phase_steps[:, :, None, :] * phase_time_mask[:, :, :, None]
        ).sum(dim=1)
        return phase_bias.reshape(phase_bias.shape[0], -1)
