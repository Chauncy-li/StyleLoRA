"""
扩散采样入口函数模块。

该文件封装 DPM-Solver 调用细节，提供统一 `dpm_sampler` 接口：
- 输入模型、初始噪声和条件；
- 输出去噪后的轨迹样本。
"""

from typing import Dict, Iterable, Optional
import torch
import baseline.model.style_planner.library.dpm_solver_pytorch as dpm


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


def dpm_sampler(
        model: torch.nn.Module,
        x_T,
        other_model_params: Dict = {},
        diffusion_steps=10,

        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {}
):
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


        dpm_solver = dpm.DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params)  # w.o. dynamic thresholding

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
