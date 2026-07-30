"""Production interfaces for clean-prediction and latent Preference Flow work."""

from baseline.model.style_planner.preference_flow.clean_prediction_editor import (
    CleanPredictionEditContext,
    CleanPredictionEditor,
    CleanPredictionObserver,
    CleanPredictionTraceRecorder,
    IdentityCleanPredictionEditor,
    apply_clean_prediction_editor,
)
from baseline.model.style_planner.preference_flow.adapter import (
    EgoTrajectoryResidualDecoder,
    LongitudinalTrajectoryEdit,
    PreferenceFlowAdapterRecord,
    PreferenceFlowCleanPredictionEditor,
    PreferenceFlowConditionEncoder,
    PreferenceFlowEditOutput,
    PreferenceFlowTrainingAdapter,
    SmoothLongitudinalTrajectoryResidualDecoder,
)
from baseline.model.style_planner.preference_flow.interaction_attention import (
    NeutralAnchoredInteractionAttention,
    NeutralInteractionAttentionOutput,
)
from baseline.model.style_planner.preference_flow.trajectory_geometry import (
    neutral_path_tangents,
    project_relative_xy,
)
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
from baseline.model.style_planner.preference_flow.config import (
    DEFAULT_INTEGRATION_METHOD,
    DEFAULT_INTEGRATION_STEPS,
    INTEGRATION_METHODS,
    PreferenceFlowConfig,
)
from baseline.model.style_planner.preference_flow.integrator import (
    integrate,
    integrate_from_neutral,
)
from baseline.model.style_planner.preference_flow.vector_field import (
    PreferenceVectorField,
)

__all__ = [
    "CLEAN_PREDICTION_EDITOR_DISABLED",
    "CLEAN_PREDICTION_EDITOR_IDENTITY",
    "CLEAN_PREDICTION_EDITOR_MODES",
    "DEFAULT_INTEGRATION_METHOD",
    "DEFAULT_INTEGRATION_STEPS",
    "CleanPredictionEditContext",
    "CleanPredictionEditor",
    "CleanPredictionObserver",
    "CleanPredictionTraceRecorder",
    "DPMEvaluationRecord",
    "DualStreamSampleResult",
    "EgoTrajectoryResidualDecoder",
    "LongitudinalTrajectoryEdit",
    "IdentityCleanPredictionEditor",
    "INTEGRATION_METHODS",
    "NeutralReferenceCache",
    "NeutralAnchoredInteractionAttention",
    "NeutralInteractionAttentionOutput",
    "PreferenceFlowContractError",
    "PreferenceFlowConfig",
    "PreferenceFlowAdapterRecord",
    "PreferenceFlowCleanPredictionEditor",
    "PreferenceFlowConditionEncoder",
    "PreferenceFlowEditOutput",
    "PreferenceFlowTrainingAdapter",
    "PreferenceVectorField",
    "SmoothLongitudinalTrajectoryResidualDecoder",
    "Step1ContractError",
    "apply_clean_prediction_editor",
    "clone_dpm_evaluation_record",
    "integrate",
    "integrate_from_neutral",
    "normalize_clean_prediction_editor_mode",
    "neutral_path_tangents",
    "project_relative_xy",
    "resolve_clean_prediction_editor_mode",
]
