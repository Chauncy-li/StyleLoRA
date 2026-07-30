"""Shared production contracts for the Preference Flow implementation.

This module owns common error types plus the Step-1/2 sampler records.  The
standalone Step-3 latent dynamics live in adjacent ``config``, ``vector_field``
and ``integrator`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

import torch


CLEAN_PREDICTION_EDITOR_DISABLED = "disabled"
CLEAN_PREDICTION_EDITOR_IDENTITY = "identity"
CLEAN_PREDICTION_EDITOR_MODES: Tuple[str, str] = (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    CLEAN_PREDICTION_EDITOR_IDENTITY,
)


class PreferenceFlowContractError(ValueError):
    """Raised when the narrowly scoped sampler interface is violated."""


# Retain the old public name while the research scripts migrate to this
# production module.  The alias intentionally has identical semantics.
Step1ContractError = PreferenceFlowContractError


def normalize_clean_prediction_editor_mode(value: Any) -> str:
    """Validate and canonicalize the two currently supported editor modes."""

    if not isinstance(value, str):
        raise PreferenceFlowContractError(
            "clean_prediction_editor_mode must be a string; "
            f"expected one of {CLEAN_PREDICTION_EDITOR_MODES}, got {value!r}"
        )
    mode = value.strip().lower()
    if mode not in CLEAN_PREDICTION_EDITOR_MODES:
        raise PreferenceFlowContractError(
            "Unsupported clean_prediction_editor_mode "
            f"{value!r}; permitted modes are {CLEAN_PREDICTION_EDITOR_MODES}."
        )
    return mode


def resolve_clean_prediction_editor_mode(config: Any) -> str:
    """Read the opt-in mode without changing old serialized configurations."""

    return normalize_clean_prediction_editor_mode(
        getattr(config, "clean_prediction_editor_mode", CLEAN_PREDICTION_EDITOR_DISABLED)
    )


@dataclass(frozen=True)
class DPMEvaluationRecord:
    """A detached, read-only snapshot from one logical DPM evaluation.

    ``current_state`` is the actual state passed to the denoiser for this
    evaluation, rather than the rollout's original ``x_T``.  Tensors are kept
    in their native device/dtype inside the record so a preference editor can
    use a corresponding neutral record without a CPU round trip.
    """

    current_state: torch.Tensor
    clean_prediction: torch.Tensor
    diffusion_time: torch.Tensor
    log_snr: torch.Tensor
    model_evaluation_index: int
    solver_step_index: Optional[int]
    is_terminal_denoise: bool
    stream_name: str


def clone_dpm_evaluation_record(record: DPMEvaluationRecord) -> DPMEvaluationRecord:
    """Return an independent detached tensor copy of one trace record."""

    if not isinstance(record, DPMEvaluationRecord):
        raise TypeError(
            "DPM evaluation records must be DPMEvaluationRecord instances, got "
            f"{type(record)!r}"
        )
    return replace(
        record,
        current_state=record.current_state.detach().clone(),
        clean_prediction=record.clean_prediction.detach().clone(),
        diffusion_time=record.diffusion_time.detach().clone(),
        log_snr=record.log_snr.detach().clone(),
    )


def _same_trace_value(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare scalar-or-batched trace values without tying device storage."""

    if tuple(left.shape) != tuple(right.shape):
        return False
    return bool(
        torch.allclose(
            left.detach().to(device="cpu", dtype=torch.float64),
            right.detach().to(device="cpu", dtype=torch.float64),
            atol=1e-7,
            rtol=0.0,
        )
    )


class NeutralReferenceCache:
    """Immutable neutral records indexed by logical denoiser evaluation.

    The cache owns cloned tensors and returns fresh clones for every lookup.
    Consequently a preference editor cannot mutate a neutral trace or a
    neutral solver state through the context object it receives.
    """

    def __init__(self, records: Sequence[DPMEvaluationRecord]) -> None:
        if not records:
            raise PreferenceFlowContractError(
                "neutral reference cache requires at least one DPM evaluation record"
            )
        copied = tuple(clone_dpm_evaluation_record(record) for record in records)
        expected_indices = tuple(range(len(copied)))
        actual_indices = tuple(
            int(record.model_evaluation_index) for record in copied
        )
        if actual_indices != expected_indices:
            raise PreferenceFlowContractError(
                "neutral reference records must be contiguous and zero-based; "
                f"got {actual_indices}"
            )
        if any(record.stream_name != "neutral" for record in copied):
            raise PreferenceFlowContractError(
                "neutral reference cache can only be constructed from neutral records"
            )
        self._records = copied

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> Tuple[DPMEvaluationRecord, ...]:
        """Return independent snapshots for diagnostics only."""

        return tuple(clone_dpm_evaluation_record(record) for record in self._records)

    def lookup(
        self,
        *,
        evaluation_index: int,
        diffusion_time: torch.Tensor,
        log_snr: torch.Tensor,
        is_terminal_denoise: bool,
    ) -> DPMEvaluationRecord:
        """Return the neutral record aligned to one preference evaluation."""

        index = int(evaluation_index)
        if index < 0 or index >= len(self._records):
            raise PreferenceFlowContractError(
                "preference evaluation has no aligned neutral record: "
                f"index={index}, neutral_count={len(self._records)}"
            )
        record = self._records[index]
        if not _same_trace_value(record.diffusion_time, diffusion_time):
            raise PreferenceFlowContractError(
                "neutral/preference diffusion times are not aligned at "
                f"evaluation {index}"
            )
        if not _same_trace_value(record.log_snr, log_snr):
            raise PreferenceFlowContractError(
                "neutral/preference log-SNR values are not aligned at "
                f"evaluation {index}"
            )
        if bool(record.is_terminal_denoise) != bool(is_terminal_denoise):
            raise PreferenceFlowContractError(
                "neutral/preference terminal flags are not aligned at "
                f"evaluation {index}"
            )
        return clone_dpm_evaluation_record(record)


@dataclass(frozen=True)
class DualStreamSampleResult:
    """Structured output of two sequential but state-isolated DPM rollouts."""

    neutral_sample: torch.Tensor
    preference_sample: torch.Tensor
    neutral_initial_state: torch.Tensor
    preference_initial_state: torch.Tensor
    neutral_trace: Tuple[DPMEvaluationRecord, ...]
    preference_trace: Tuple[DPMEvaluationRecord, ...]


__all__ = [
    "CLEAN_PREDICTION_EDITOR_DISABLED",
    "CLEAN_PREDICTION_EDITOR_IDENTITY",
    "CLEAN_PREDICTION_EDITOR_MODES",
    "DPMEvaluationRecord",
    "DualStreamSampleResult",
    "NeutralReferenceCache",
    "PreferenceFlowContractError",
    "Step1ContractError",
    "clone_dpm_evaluation_record",
    "normalize_clean_prediction_editor_mode",
    "resolve_clean_prediction_editor_mode",
]
