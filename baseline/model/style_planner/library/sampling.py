"""
扩散采样入口函数模块。

该文件封装 DPM-Solver 调用细节，提供统一 `dpm_sampler` 接口：
- 输入模型、初始噪声和条件；
- 输出去噪后的轨迹样本。
"""

from dataclasses import replace
from typing import Dict, Iterable, Optional
import torch
import baseline.model.style_planner.library.dpm_solver_pytorch as dpm
from baseline.model.style_planner.preference_flow import (
    CleanPredictionEditContext,
    CleanPredictionEditor,
    CleanPredictionObserver,
    CleanPredictionTraceRecorder,
    DualStreamSampleResult,
    NeutralReferenceCache,
    apply_clean_prediction_editor,
)


class TransportInjectionProbe:
    """Diagnostic-only gate for selecting DPM denoiser evaluations.

    The probe is deliberately stateful only for the lifetime of one sampler
    invocation.  ``active_call_indices=None`` means that the style residual is
    active at every model evaluation; otherwise it is active only at the
    specified zero-based evaluations.  The DPM wrapper calls
    :meth:`begin_model_evaluation` once per *logical* denoiser evaluation, so
    classifier-free conditional/unconditional branches share one gate.

    This object is not a module, has no checkpoint state, and is used only by
    the Phase-0 transport evaluator.
    """

    def __init__(
        self,
        *,
        active_call_indices: Optional[Iterable[int]] = None,
        label: str = "",
        override_style_residual_gate: bool = True,
    ) -> None:
        if active_call_indices is None:
            self._all_calls = True
            self._active_call_indices: frozenset[int] = frozenset()
        else:
            parsed = frozenset(int(index) for index in active_call_indices)
            if any(index < 0 for index in parsed):
                raise ValueError("transport probe call indices must be non-negative")
            self._all_calls = False
            self._active_call_indices = parsed
        self.label = str(label)
        self.override_style_residual_gate = bool(override_style_residual_gate)
        self.reset()

    def reset(self) -> None:
        """Clear the trace before a fresh DPM-Solver rollout."""

        self._current_call_index = -1
        self._call_times: list[float] = []
        self._active_history: list[bool] = []

    @property
    def current_call_index(self) -> int:
        return int(self._current_call_index)

    @property
    def current_gate(self) -> float:
        if self._current_call_index < 0:
            raise RuntimeError(
                "transport probe gate was read before a denoiser evaluation"
            )
        return float(self._active_history[-1])

    def begin_model_evaluation(self, diffusion_time: torch.Tensor) -> float:
        """Advance the trace and return the gate for the current evaluation."""

        time = torch.as_tensor(diffusion_time).detach().reshape(-1)
        if time.numel() == 0:
            raise ValueError("transport probe received an empty diffusion-time tensor")
        if time.numel() > 1 and not torch.allclose(
            time,
            time[:1].expand_as(time),
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(
                "all samples in one transport-probe DPM evaluation must share time"
            )
        self._current_call_index += 1
        active = self._all_calls or (
            self._current_call_index in self._active_call_indices
        )
        self._call_times.append(float(time[0].cpu()))
        self._active_history.append(bool(active))
        return float(active)

    def diagnostics(self) -> Dict[str, object]:
        """Return JSON-safe trace metadata after sampling."""

        return {
            "label": self.label,
            "override_style_residual_gate": self.override_style_residual_gate,
            "active_call_indices": (
                "all" if self._all_calls else sorted(self._active_call_indices)
            ),
            "model_evaluation_count": len(self._call_times),
            "model_evaluation_times": list(self._call_times),
            "model_evaluation_active_mask": list(self._active_history),
        }


def selftest_transport_injection_probe() -> Dict[str, bool]:
    """Verify deterministic phase selection without loading a planner."""

    selected = TransportInjectionProbe(active_call_indices=(0, 2, 4), label="toy")
    gates = [
        selected.begin_model_evaluation(torch.tensor([time]))
        for time in (1.0, 0.7, 0.4, 0.1, 0.001)
    ]
    all_calls = TransportInjectionProbe(label="all")
    all_gates = [
        all_calls.begin_model_evaluation(torch.tensor([time]))
        for time in (1.0, 0.5, 0.001)
    ]
    trace = selected.diagnostics()
    return {
        "selected_gate_pattern": gates == [1.0, 0.0, 1.0, 0.0, 1.0],
        "all_gate_pattern": all_gates == [1.0, 1.0, 1.0],
        "call_count_recorded": trace["model_evaluation_count"] == 5,
        "time_trace_recorded": len(trace["model_evaluation_times"]) == 5,
        "terminal_index_selectable": trace["model_evaluation_active_mask"][-1]
        is True,
    }


def _batched_trace_time(
        diffusion_time,
        *,
        reference: torch.Tensor,
        label: str,
) -> torch.Tensor:
    """Normalize a DPM time tensor to exactly one value per batch element."""

    time = torch.as_tensor(
        diffusion_time,
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(-1)
    batch_size = int(reference.shape[0])
    if time.numel() == 1 and batch_size != 1:
        time = time.expand(batch_size)
    if time.numel() != batch_size:
        raise ValueError(
            f"{label} received a diffusion-time batch of {time.numel()} for "
            f"state batch size {batch_size}"
        )
    if time.numel() > 1 and not torch.allclose(
        time,
        time[:1].expand_as(time),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(
            f"all samples in one {label} evaluation must share diffusion time"
        )
    return time


def _is_terminal_denoise_time(
        diffusion_time: torch.Tensor,
        *,
        terminal_time: float,
) -> bool:
    return bool(
        torch.all(
            torch.isclose(
                diffusion_time,
                torch.full_like(diffusion_time, terminal_time),
                atol=1e-7,
                rtol=0.0,
            )
        ).item()
    )


def dpm_sampler(
        model: torch.nn.Module,
        x_T,
        other_model_params: Dict = {},
        diffusion_steps=10,

        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {},
        clean_prediction_editor: Optional[CleanPredictionEditor] = None,
        clean_prediction_observer: Optional[CleanPredictionObserver] = None,
        neutral_reference_cache: Optional[NeutralReferenceCache] = None,
        stream_name: str = "single",
):
    """Run one DPM stream with an optional clean-prediction callback.

    When a callback is active, the sampler snapshots the actual DPM state
    passed to each denoiser evaluation.  The x0 corrector then consumes that
    matching snapshot immediately after the model has predicted x0.  The
    default path deliberately installs no callback and preserves the original
    StylePlanner sampling behavior.
    """

    if not isinstance(stream_name, str) or not stream_name.strip():
        raise ValueError("stream_name must be a non-empty string")
    stream_name = stream_name.strip()
    if neutral_reference_cache is not None and stream_name != "preference":
        raise ValueError(
            "neutral_reference_cache is valid only for the preference DPM stream"
        )
    with torch.no_grad():
        # Keep the caller's mapping immutable.  Phase-0 injects a mutable
        # diagnostic gate into this local copy immediately before each logical
        # denoiser evaluation; the model wrapper closes over the same mapping.
        local_model_params = dict(other_model_params)
        transport_probe = local_model_params.pop("transport_injection_probe", None)
        if transport_probe is not None:
            if not isinstance(transport_probe, TransportInjectionProbe):
                raise TypeError(
                    "transport_injection_probe must be a TransportInjectionProbe"
                )
            transport_probe.reset()
        noise_schedule = dpm.NoiseScheduleVP(
            schedule='linear',
            **noise_schedule_params
        )

        model_fn = dpm.model_wrapper(
            model,  # use your noise prediction model here
            noise_schedule,
            model_type=model.model_type,  # or "x_start" or "v" or "score"
            model_kwargs=local_model_params,
            **model_wrapper_params
        )
        if transport_probe is not None:
            base_model_fn = model_fn

            def model_fn(x, diffusion_time):
                gate = transport_probe.begin_model_evaluation(diffusion_time)
                # These keys are intentionally absent from production calls.
                # The DiT reads them only as an explicit Phase-0 override of
                # the legacy all-step/terminal gate.
                if transport_probe.override_style_residual_gate:
                    local_model_params["diagnostic_style_residual_gate"] = gate
                    local_model_params["diagnostic_style_residual_eval_index"] = (
                        transport_probe.current_call_index
                    )
                return base_model_fn(x, diffusion_time)

        # DPM-Solver invokes ``correcting_x0_fn`` after converting the wrapped
        # model output to its DPM-facing clean prediction and immediately before
        # its update.  Leaving this key absent is intentional: disabled mode
        # follows the pre-Step-1 sampler path without an extra callback.
        local_dpm_solver_params = dict(dpm_solver_params)
        hook_active = (
            clean_prediction_editor is not None
            or clean_prediction_observer is not None
            or neutral_reference_cache is not None
        )
        if hook_active:
            existing_corrector = local_dpm_solver_params.get("correcting_x0_fn")
            if existing_corrector is not None:
                raise ValueError(
                    "clean_prediction_editor cannot be combined with an existing "
                    "DPM correcting_x0_fn in Step 1"
                )

            for clean_prediction_hook in (
                clean_prediction_editor,
                clean_prediction_observer,
            ):
                if clean_prediction_hook is None:
                    continue
                reset = getattr(clean_prediction_hook, "reset", None)
                if callable(reset):
                    reset()
            if (
                clean_prediction_observer is not None
                and not callable(clean_prediction_observer)
            ):
                raise TypeError("clean_prediction_observer must be callable")

            # The solver calls model_fn(x_q, t) immediately before invoking the
            # x0 corrector.  Capture that exact x_q here, rather than passing a
            # fixed decoder input through every callback.
            base_model_fn = model_fn
            latest_model_state: Optional[torch.Tensor] = None
            latest_model_time: Optional[torch.Tensor] = None
            model_evaluation_index = -1

            def model_fn_with_current_state(x, diffusion_time):
                nonlocal latest_model_state
                nonlocal latest_model_time
                nonlocal model_evaluation_index
                if not torch.is_tensor(x):
                    raise TypeError("DPM model evaluation state must be a torch.Tensor")
                model_evaluation_index += 1
                latest_model_state = x.detach().clone()
                latest_model_time = _batched_trace_time(
                    diffusion_time,
                    reference=x,
                    label="DPM model",
                ).detach().clone()
                return base_model_fn(x, diffusion_time)

            model_fn = model_fn_with_current_state
            terminal_time = 1.0 / float(noise_schedule.total_N)

            def clean_prediction_corrector(clean_prediction, diffusion_time):
                if latest_model_state is None or latest_model_time is None:
                    raise RuntimeError(
                        "DPM x0 corrector ran without a matching model evaluation state"
                    )
                time = _batched_trace_time(
                    diffusion_time,
                    reference=clean_prediction,
                    label="DPM clean-prediction editor",
                )
                if tuple(latest_model_state.shape) != tuple(clean_prediction.shape):
                    raise RuntimeError(
                        "captured DPM state and clean prediction have different shapes: "
                        f"{tuple(latest_model_state.shape)} versus "
                        f"{tuple(clean_prediction.shape)}"
                    )
                latest_time = latest_model_time.to(
                    device=clean_prediction.device,
                    dtype=clean_prediction.dtype,
                )
                if not torch.allclose(time, latest_time, atol=1e-7, rtol=0.0):
                    raise RuntimeError(
                        "DPM x0 corrector time does not match its denoiser evaluation"
                    )
                is_terminal_denoise = _is_terminal_denoise_time(
                    time,
                    terminal_time=terminal_time,
                )
                context = CleanPredictionEditContext(
                    diffusion_time=time,
                    log_snr=noise_schedule.marginal_lambda(time),
                    model_evaluation_index=model_evaluation_index,
                    # The vendored DPM callback has no solver-step parameter;
                    # do not invent one from NFE order.
                    solver_step_index=None,
                    is_terminal_denoise=is_terminal_denoise,
                    current_state=latest_model_state,
                    stream_name=stream_name,
                )
                if neutral_reference_cache is not None:
                    context = replace(
                        context,
                        neutral_record=neutral_reference_cache.lookup(
                            evaluation_index=model_evaluation_index,
                            diffusion_time=context.diffusion_time,
                            log_snr=context.log_snr,
                            is_terminal_denoise=context.is_terminal_denoise,
                        ),
                    )
                if clean_prediction_editor is None:
                    edited_clean_prediction = clean_prediction
                else:
                    edited_clean_prediction = apply_clean_prediction_editor(
                        clean_prediction_editor,
                        clean_prediction,
                        context,
                    )
                if clean_prediction_observer is not None:
                    clean_prediction_observer(edited_clean_prediction, context)
                return edited_clean_prediction

            local_dpm_solver_params["correcting_x0_fn"] = clean_prediction_corrector

        dpm_solver = dpm.DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++", **local_dpm_solver_params)  # w.o. dynamic thresholding

        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            **sample_params
        )

    return sample_dpm


def dual_stream_dpm_sampler(
        model: torch.nn.Module,
        x_T: torch.Tensor,
        other_model_params: Dict = {},
        diffusion_steps=10,
        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {},
        neutral_editor: Optional[CleanPredictionEditor] = None,
        preference_editor: Optional[CleanPredictionEditor] = None,
) -> DualStreamSampleResult:
    """Sample independent neutral and preference DPM streams from one noise draw.

    The streams run sequentially only to make Step 2 easy to audit.  Each call
    constructs a fresh DPM-Solver and receives its own clone of ``x_T``.  The
    neutral trace is frozen into a clone-on-read cache before preference
    sampling, so a preference editor can inspect aligned neutral values without
    sharing a mutable solver tensor.
    """

    if not torch.is_tensor(x_T):
        raise TypeError("x_T must be a torch.Tensor for dual-stream sampling")
    if bool(sample_params.get("return_intermediate", False)):
        raise ValueError(
            "dual_stream_dpm_sampler exposes per-evaluation traces and does not "
            "support sample_params['return_intermediate']"
        )
    if neutral_editor is not None and neutral_editor is preference_editor:
        raise ValueError(
            "neutral and preference streams require distinct editor instances; "
            "pass None for both identity-free streams or construct two editors"
        )

    neutral_initial_state = x_T.detach().clone()
    preference_initial_state = x_T.detach().clone()
    neutral_x_T = x_T.clone()
    preference_x_T = x_T.clone()

    neutral_recorder = CleanPredictionTraceRecorder()
    neutral_sample = dpm_sampler(
        model,
        neutral_x_T,
        other_model_params=dict(other_model_params),
        diffusion_steps=diffusion_steps,
        noise_schedule_params=dict(noise_schedule_params),
        model_wrapper_params=dict(model_wrapper_params),
        dpm_solver_params=dict(dpm_solver_params),
        sample_params=dict(sample_params),
        clean_prediction_editor=neutral_editor,
        clean_prediction_observer=neutral_recorder,
        stream_name="neutral",
    )
    neutral_trace = neutral_recorder.records()
    neutral_cache = NeutralReferenceCache(neutral_trace)

    preference_recorder = CleanPredictionTraceRecorder()
    preference_sample = dpm_sampler(
        model,
        preference_x_T,
        other_model_params=dict(other_model_params),
        diffusion_steps=diffusion_steps,
        noise_schedule_params=dict(noise_schedule_params),
        model_wrapper_params=dict(model_wrapper_params),
        dpm_solver_params=dict(dpm_solver_params),
        sample_params=dict(sample_params),
        clean_prediction_editor=preference_editor,
        clean_prediction_observer=preference_recorder,
        neutral_reference_cache=neutral_cache,
        stream_name="preference",
    )
    preference_trace = preference_recorder.records()

    if len(preference_trace) != len(neutral_trace):
        raise RuntimeError(
            "neutral and preference streams produced different logical DPM "
            f"evaluation counts: {len(neutral_trace)} versus {len(preference_trace)}"
        )
    return DualStreamSampleResult(
        neutral_sample=neutral_sample,
        preference_sample=preference_sample,
        neutral_initial_state=neutral_initial_state,
        preference_initial_state=preference_initial_state,
        neutral_trace=neutral_trace,
        preference_trace=preference_trace,
    )
