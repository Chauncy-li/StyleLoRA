"""Focused Step-4 tests for the Preference Flow clean-prediction adapter.

These are toy-DPM tests only.  The companion runner executes the same adapter
against the frozen StylePlanner checkpoint and real cached scenes on CUDA.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest


try:
    import torch
    import torch.nn as nn

    from baseline.model.style_planner.library.sampling import dual_stream_dpm_sampler
    from baseline.model.style_planner.preference_flow import (
        EgoTrajectoryResidualDecoder,
        PreferenceFlowCleanPredictionEditor,
        PreferenceFlowConfig,
        PreferenceVectorField,
    )

    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # pragma: no cover - environment-specific
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = error


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ModuleBase = nn.Module if nn is not None else object


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step4StaticTests(unittest.TestCase):
    def test_adapter_is_a_preference_editor_not_a_final_trajectory_hook(self) -> None:
        source = (
            _REPO_ROOT
            / "baseline/model/style_planner/preference_flow/adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("integrate_from_neutral", source)
        self.assertIn('context.stream_name != "preference"', source)
        self.assertIn("joint_trajectory[:, 0, 1:, :]", source)
        self.assertNotIn("inverse(", source)
        self.assertNotIn("rho: torch.Tensor", (
            _REPO_ROOT
            / "baseline/model/style_planner/preference_flow/vector_field.py"
        ).read_text(encoding="utf-8"))
        signature = inspect.signature(PreferenceVectorField.forward)
        self.assertNotIn("rho", signature.parameters)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step4TensorTests(unittest.TestCase):
    class _ToyXStart:
        model_type = "x_start"

        def __call__(self, x, time, **unused_kwargs):
            del time, unused_kwargs
            return 0.125 * x

    class _ConstantProbeField(_ModuleBase):
        def forward(self, state, condition, diffusion_time, preference_coordinate):
            del condition, diffusion_time, preference_coordinate
            return torch.ones_like(state)

    @staticmethod
    def _config() -> PreferenceFlowConfig:
        return PreferenceFlowConfig(
            latent_dim=8,
            condition_dim=16,
            hidden_dim=16,
            num_layers=2,
            zero_initialize_output=True,
        )

    def _sample(self, editor):
        torch.manual_seed(3407)
        # [B, P, (current + two future poses) * 4]
        x_t = torch.randn((2, 3, 12), dtype=torch.float32)
        result = dual_stream_dpm_sampler(
            self._ToyXStart(),
            x_t,
            diffusion_steps=2,
            neutral_editor=None,
            preference_editor=editor,
        )
        return result

    def test_rho_zero_and_zero_initialized_flow_are_exact_identity(self) -> None:
        zero_rho = PreferenceFlowCleanPredictionEditor(
            rho=0.0,
            config=self._config(),
        )
        zero_rho_result = self._sample(zero_rho)
        self.assertTrue(
            torch.equal(
                zero_rho_result.neutral_sample,
                zero_rho_result.preference_sample,
            )
        )
        self.assertEqual(len(zero_rho.records()), 3)
        self.assertTrue(
            all(
                torch.equal(record.latent_start, record.latent_end)
                and float(record.ego_current_residual_abs_max) == 0.0
                and float(record.non_ego_residual_abs_max) == 0.0
                and float(record.ego_future_residual.abs().max().item()) == 0.0
                for record in zero_rho.records()
            )
        )

        zero_initialized = PreferenceFlowCleanPredictionEditor(
            rho=0.75,
            config=self._config(),
        )
        zero_initialized_result = self._sample(zero_initialized)
        self.assertTrue(
            torch.equal(
                zero_initialized_result.neutral_sample,
                zero_initialized_result.preference_sample,
            )
        )
        self.assertTrue(
            all(
                torch.equal(record.latent_start, record.latent_end)
                for record in zero_initialized.records()
            )
        )

    def test_nonzero_probe_changes_preference_ego_only_and_propagates(self) -> None:
        rho = 0.5
        decoder = EgoTrajectoryResidualDecoder(
            latent_dim=8,
            zero_initialize_output=False,
        )
        with torch.no_grad():
            decoder.projection.weight.zero_()
            decoder.projection.bias.zero_()
            # Constant field gives latent displacement rho * 1.0.  This maps
            # to a small +1e-3 x residual at the final future pose.
            decoder.projection.weight[0, 0] = 2e-3
        probe = PreferenceFlowCleanPredictionEditor(
            rho=rho,
            config=self._config(),
            vector_field=self._ConstantProbeField(),
            trajectory_decoder=decoder,
        )
        result = self._sample(probe)

        self.assertGreater(
            float(
                (
                    result.preference_trace[0].clean_prediction
                    - result.neutral_trace[0].clean_prediction
                )
                .abs()
                .max()
                .item()
            ),
            0.0,
        )
        self.assertGreater(
            max(
                float(
                    (
                        result.preference_trace[index].current_state
                        - result.neutral_trace[index].current_state
                    )
                    .abs()
                    .max()
                    .item()
                )
                for index in range(1, len(result.preference_trace))
            ),
            0.0,
        )
        self.assertGreater(
            float((result.preference_sample - result.neutral_sample).abs().max().item()),
            0.0,
        )
        self.assertTrue(
            all(
                record.ego_current_residual_abs_max == 0.0
                and record.non_ego_residual_abs_max == 0.0
                and float(record.ego_future_residual.abs().max().item()) > 0.0
                for record in probe.records()
            )
        )


if __name__ == "__main__":
    if _TORCH_IMPORT_ERROR is not None:
        print(
            "PyTorch-dependent Step-4 adapter tests will be skipped: "
            f"{_TORCH_IMPORT_ERROR}"
        )
    unittest.main(verbosity=2)
