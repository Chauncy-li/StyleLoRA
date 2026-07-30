"""Backward-compatible research imports for StylePlanner production contracts.

Production sampler interfaces live under
``baseline.model.style_planner.preference_flow``.  Research scripts import
them from there (or through this thin compatibility module); baseline never
imports this package.
"""

from baseline.model.style_planner.preference_flow.contracts import (
    CLEAN_PREDICTION_EDITOR_DISABLED,
    CLEAN_PREDICTION_EDITOR_IDENTITY,
    CLEAN_PREDICTION_EDITOR_MODES,
    DPMEvaluationRecord,
    DualStreamSampleResult,
    NeutralReferenceCache,
    PreferenceFlowContractError,
    Step1ContractError,
    clone_dpm_evaluation_record,
    normalize_clean_prediction_editor_mode,
    resolve_clean_prediction_editor_mode,
)

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
