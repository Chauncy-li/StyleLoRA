"""Self-test entry point for the standalone Step-3 Preference Flow numerics."""

from __future__ import annotations

from pathlib import Path
import unittest


try:
    import torch

    from baseline.model.style_planner.preference_flow import PreferenceVectorField
    from research_v1.execution.preference_flow.run_step3_numerical_validation import (
        run_numerical_validation,
    )

    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # pragma: no cover - environment-specific
    torch = None
    _TORCH_IMPORT_ERROR = error


_REPO_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step3StaticTests(unittest.TestCase):
    def test_step3_is_not_connected_to_dpm_or_decoder(self) -> None:
        sampling_source = (
            _REPO_ROOT / "baseline/model/style_planner/library/sampling.py"
        ).read_text(encoding="utf-8")
        decoder_source = (
            _REPO_ROOT / "baseline/model/style_planner/layer/decoder.py"
        ).read_text(encoding="utf-8")
        vector_source = (
            _REPO_ROOT / "baseline/model/style_planner/preference_flow/vector_field.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("integrate_from_neutral", sampling_source)
        self.assertNotIn("integrate_from_neutral", decoder_source)
        self.assertIn("preference_coordinate", vector_source)
        self.assertNotIn("rho: torch.Tensor", vector_source)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step3TensorTests(unittest.TestCase):
    def test_complete_step3_numerical_contract(self) -> None:
        report = run_numerical_validation(seed=3407, device=torch.device("cpu"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["rho_zero_exact"])
        self.assertEqual(report["constant_field_positive_error"], 0.0)
        self.assertEqual(report["constant_field_negative_error"], 0.0)
        self.assertTrue(report["gradient_passed"])
        self.assertTrue(report["zero_initialization_passed"])


if __name__ == "__main__":
    if _TORCH_IMPORT_ERROR is not None:
        print(
            "PyTorch-dependent Step-3 tests will be skipped: "
            f"{_TORCH_IMPORT_ERROR}"
        )
    unittest.main(verbosity=2)
