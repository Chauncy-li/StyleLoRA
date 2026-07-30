"""Clean-prediction editor contracts and read-only DPM trace recorders.

The editor is invoked after the DPM wrapper has produced a clean prediction
and immediately before the solver consumes it.  This Step-2 interface remains
agnostic to how a later preference edit will be computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch

from baseline.model.style_planner.preference_flow.contracts import (
    DPMEvaluationRecord,
    PreferenceFlowContractError,
    clone_dpm_evaluation_record,
)


@dataclass(frozen=True)
class CleanPredictionEditContext:
    """Read-only information available immediately before one DPM update.

    ``current_state`` is a detached snapshot of the actual DPM state ``x_q``
    presented to the denoiser at this evaluation.  ``neutral_record`` is set
    only for the preference stream and is independently cloned from the
    matching neutral evaluation.
    """

    diffusion_time: torch.Tensor
    log_snr: torch.Tensor
    model_evaluation_index: int
    solver_step_index: Optional[int]
    is_terminal_denoise: bool
    current_state: torch.Tensor
    stream_name: str = "single"
    neutral_record: Optional[DPMEvaluationRecord] = None


class CleanPredictionEditor(Protocol):
    """Callable contract for a future clean-prediction editor."""

    def __call__(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
    ) -> torch.Tensor:
        """Return a clean prediction with the exact same tensor contract."""


class CleanPredictionObserver(Protocol):
    """Read the solver-facing clean prediction without changing the DPM result."""

    def __call__(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
    ) -> None:
        """Record a detached observation and return ``None``."""


def _scalar_trace_value(value: torch.Tensor) -> float:
    detached = value.detach().reshape(-1)
    if detached.numel() == 0:
        raise PreferenceFlowContractError("editor diagnostics received an empty tensor")
    return float(detached[0].cpu().item())


class IdentityCleanPredictionEditor:
    """Return the original ``x0`` object and retain detached audit metadata."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._records.clear()

    def __call__(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
    ) -> torch.Tensor:
        detached_prediction = clean_prediction.detach()
        self._records.append(
            {
                "model_evaluation_index": int(context.model_evaluation_index),
                "solver_step_index": context.solver_step_index,
                "diffusion_time": _scalar_trace_value(context.diffusion_time),
                "log_snr": _scalar_trace_value(context.log_snr),
                "is_terminal_denoise": bool(context.is_terminal_denoise),
                "stream_name": str(context.stream_name),
                "has_neutral_reference": context.neutral_record is not None,
                "clean_prediction_shape": [
                    int(dimension) for dimension in detached_prediction.shape
                ],
                "clean_prediction_abs_max": float(
                    detached_prediction.abs().max().cpu().item()
                ),
                "current_state_shape": [
                    int(dimension) for dimension in context.current_state.shape
                ],
            }
        )
        return clean_prediction

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "editor": "identity",
            "model_evaluation_count": len(self._records),
            "calls": [dict(record) for record in self._records],
        }


class CleanPredictionTraceRecorder:
    """Detached snapshots of solver-facing x0 and its matching real DPM state.

    The sampler invokes this observer after any editor has returned its x0, so
    a neutral cache represents the clean prediction actually consumed by the
    DPM update.  The public snapshot helpers return CPU clones for the Step-1
    regression JSON workflow and cannot expose mutable solver tensors.
    """

    def __init__(self) -> None:
        self._records: List[DPMEvaluationRecord] = []

    def reset(self) -> None:
        self._records.clear()

    def __call__(
        self,
        clean_prediction: torch.Tensor,
        context: CleanPredictionEditContext,
    ) -> None:
        self._records.append(
            DPMEvaluationRecord(
                current_state=context.current_state.detach().clone(),
                clean_prediction=clean_prediction.detach().clone(),
                diffusion_time=context.diffusion_time.detach().clone(),
                log_snr=context.log_snr.detach().clone(),
                model_evaluation_index=int(context.model_evaluation_index),
                solver_step_index=context.solver_step_index,
                is_terminal_denoise=bool(context.is_terminal_denoise),
                stream_name=str(context.stream_name),
            )
        )

    def records(self) -> Tuple[DPMEvaluationRecord, ...]:
        """Return independent records suitable for a neutral-reference cache."""

        return tuple(clone_dpm_evaluation_record(record) for record in self._records)

    def snapshots(self) -> Tuple[torch.Tensor, ...]:
        """Return independent CPU x0 copies for numeric regression checks."""

        return tuple(
            record.clean_prediction.detach().clone().cpu()
            for record in self._records
        )

    def current_state_snapshots(self) -> Tuple[torch.Tensor, ...]:
        """Return independent CPU copies of the real per-evaluation ``x_q``."""

        return tuple(
            record.current_state.detach().clone().cpu() for record in self._records
        )

    def diagnostics(self) -> Dict[str, Any]:
        calls = []
        for record in self._records:
            calls.append(
                {
                    "model_evaluation_index": int(record.model_evaluation_index),
                    "diffusion_time": _scalar_trace_value(record.diffusion_time),
                    "log_snr": _scalar_trace_value(record.log_snr),
                    "is_terminal_denoise": bool(record.is_terminal_denoise),
                    "stream_name": str(record.stream_name),
                    "clean_prediction_shape": [
                        int(dimension) for dimension in record.clean_prediction.shape
                    ],
                    "current_state_shape": [
                        int(dimension) for dimension in record.current_state.shape
                    ],
                }
            )
        return {
            "observer": "clean_prediction_trace_recorder",
            "model_evaluation_count": len(calls),
            "calls": calls,
        }


def apply_clean_prediction_editor(
    editor: CleanPredictionEditor,
    clean_prediction: torch.Tensor,
    context: CleanPredictionEditContext,
) -> torch.Tensor:
    """Apply one edit and reject shape/device/dtype/non-finite contract breaks."""

    if not torch.is_tensor(clean_prediction):
        raise PreferenceFlowContractError(
            "DPM clean prediction must be a torch.Tensor before editor application"
        )
    if not callable(editor):
        raise PreferenceFlowContractError("clean_prediction_editor must be callable")

    edited = editor(clean_prediction, context)
    if not torch.is_tensor(edited):
        raise PreferenceFlowContractError(
            "clean_prediction_editor must return a torch.Tensor, got "
            f"{type(edited)!r}"
        )
    if tuple(edited.shape) != tuple(clean_prediction.shape):
        raise PreferenceFlowContractError(
            "clean_prediction_editor changed x0 shape: "
            f"expected {tuple(clean_prediction.shape)}, got {tuple(edited.shape)}"
        )
    if edited.device != clean_prediction.device:
        raise PreferenceFlowContractError(
            "clean_prediction_editor changed x0 device: "
            f"expected {clean_prediction.device}, got {edited.device}"
        )
    if edited.dtype != clean_prediction.dtype:
        raise PreferenceFlowContractError(
            "clean_prediction_editor changed x0 dtype: "
            f"expected {clean_prediction.dtype}, got {edited.dtype}"
        )
    if not bool(torch.isfinite(edited).all().item()):
        raise PreferenceFlowContractError("clean_prediction_editor returned non-finite x0")
    return edited


__all__ = [
    "CleanPredictionEditContext",
    "CleanPredictionEditor",
    "CleanPredictionObserver",
    "CleanPredictionTraceRecorder",
    "IdentityCleanPredictionEditor",
    "apply_clean_prediction_editor",
]
