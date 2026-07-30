"""Unit and structural checks for the Step-1 identity editor hook.

This module runs source/configuration checks on a lightweight workstation.  It
adds tensor-level sampler tests automatically when the active environment has
PyTorch installed (the remote ``mdsn_py39`` environment does).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch

    from baseline.model.style_planner.library.sampling import dpm_sampler
    from baseline.model.style_planner.preference_flow import (
        CLEAN_PREDICTION_EDITOR_DISABLED,
        CLEAN_PREDICTION_EDITOR_IDENTITY,
        CleanPredictionEditContext,
        CleanPredictionTraceRecorder,
        IdentityCleanPredictionEditor,
        Step1ContractError,
        apply_clean_prediction_editor,
        resolve_clean_prediction_editor_mode,
    )
    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # pragma: no cover - environment-specific
    torch = None
    _TORCH_IMPORT_ERROR = error


_REPO_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step1StaticTests(unittest.TestCase):
    def test_missing_mode_keeps_legacy_disabled_path(self) -> None:
        self.assertEqual(
            resolve_clean_prediction_editor_mode(SimpleNamespace()),
            CLEAN_PREDICTION_EDITOR_DISABLED,
        )

    def test_identity_mode_is_explicit_and_invalid_mode_fails(self) -> None:
        self.assertEqual(
            resolve_clean_prediction_editor_mode(
                SimpleNamespace(clean_prediction_editor_mode=" identity ")
            ),
            CLEAN_PREDICTION_EDITOR_IDENTITY,
        )
        with self.assertRaises(Step1ContractError):
            resolve_clean_prediction_editor_mode(
                SimpleNamespace(clean_prediction_editor_mode="preference_flow")
            )

    def test_hook_is_on_dpm_x0_path_not_final_trajectory_path(self) -> None:
        sampling_source = (
            _REPO_ROOT / "baseline/model/style_planner/library/sampling.py"
        ).read_text(encoding="utf-8")
        decoder_source = (
            _REPO_ROOT / "baseline/model/style_planner/layer/decoder.py"
        ).read_text(encoding="utf-8")
        self.assertIn('local_dpm_solver_params["correcting_x0_fn"]', sampling_source)
        self.assertIn("clean_prediction_editor=self._clean_prediction_editor", decoder_source)
        self.assertIn("dual_stream_dpm_sampler", decoder_source)
        self.assertNotIn("research_v1.execution.preference_flow", sampling_source)
        self.assertNotIn("research_v1.execution.preference_flow", decoder_source)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step1TensorTests(unittest.TestCase):
    def _context(self):
        return CleanPredictionEditContext(
            diffusion_time=torch.tensor([1.0]),
            log_snr=torch.tensor([0.0]),
            model_evaluation_index=0,
            solver_step_index=None,
            is_terminal_denoise=False,
            current_state=torch.zeros((1, 1, 4)),
        )

    def test_identity_returns_the_same_tensor_without_a_clone(self) -> None:
        clean_prediction = torch.randn((1, 2, 4))
        editor = IdentityCleanPredictionEditor()
        edited = apply_clean_prediction_editor(editor, clean_prediction, self._context())
        self.assertIs(edited, clean_prediction)
        self.assertEqual(edited.data_ptr(), clean_prediction.data_ptr())
        self.assertEqual(editor.diagnostics()["model_evaluation_count"], 1)

    def test_invalid_editor_outputs_are_rejected(self) -> None:
        clean_prediction = torch.ones((1, 2, 4))
        context = self._context()

        with self.assertRaises(Step1ContractError):
            apply_clean_prediction_editor(
                lambda value, _: value[..., :3], clean_prediction, context
            )
        with self.assertRaises(Step1ContractError):
            apply_clean_prediction_editor(
                lambda value, _: value.to(torch.float64), clean_prediction, context
            )
        with self.assertRaises(Step1ContractError):
            apply_clean_prediction_editor(
                lambda value, _: value * float("nan"), clean_prediction, context
            )

    def test_identity_sampler_is_exact_and_records_terminal_denoise(self) -> None:
        class ToyXStart(torch.nn.Module):
            model_type = "x_start"

            def forward(self, x, time, **unused_kwargs):
                del time, unused_kwargs
                return 0.125 * x

        torch.manual_seed(3407)
        x_t = torch.randn((2, 6))
        model = ToyXStart()

        disabled = dpm_sampler(
            model,
            x_t.clone(),
            diffusion_steps=2,
        )
        disabled_trace = CleanPredictionTraceRecorder()
        observed_disabled = dpm_sampler(
            model,
            x_t.clone(),
            diffusion_steps=2,
            clean_prediction_observer=disabled_trace,
        )
        editor = IdentityCleanPredictionEditor()
        identity_trace = CleanPredictionTraceRecorder()
        identity = dpm_sampler(
            model,
            x_t.clone(),
            diffusion_steps=2,
            clean_prediction_editor=editor,
            clean_prediction_observer=identity_trace,
        )

        self.assertTrue(torch.equal(disabled, observed_disabled))
        self.assertTrue(torch.equal(disabled, identity))
        self.assertEqual(len(disabled_trace.snapshots()), 3)
        self.assertEqual(len(identity_trace.snapshots()), 3)
        for raw_disabled, raw_identity in zip(
            disabled_trace.snapshots(),
            identity_trace.snapshots(),
        ):
            self.assertTrue(torch.equal(raw_disabled, raw_identity))
        diagnostics = editor.diagnostics()
        self.assertEqual(diagnostics["model_evaluation_count"], 3)
        self.assertEqual(
            [call["model_evaluation_index"] for call in diagnostics["calls"]],
            [0, 1, 2],
        )
        self.assertTrue(diagnostics["calls"][-1]["is_terminal_denoise"])
        current_states = disabled_trace.current_state_snapshots()
        self.assertTrue(torch.equal(current_states[0], x_t.cpu()))
        self.assertFalse(torch.equal(current_states[0], current_states[1]))


if __name__ == "__main__":
    if _TORCH_IMPORT_ERROR is not None:
        print(
            "PyTorch-dependent Step-1 sampler tests will be skipped: "
            f"{_TORCH_IMPORT_ERROR}"
        )
    unittest.main(verbosity=2)
