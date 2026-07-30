"""Focused tensor and structural tests for the Step-2 dual DPM streams.

The test-only probe below edits one clean prediction by a tiny constant.  It
does not implement a preference representation or a learned flow; its sole
purpose is to prove branch isolation before such a component exists.
"""

from __future__ import annotations

from pathlib import Path
import unittest


try:
    import torch

    from baseline.model.style_planner.library.sampling import dual_stream_dpm_sampler
    from baseline.model.style_planner.preference_flow import (
        CleanPredictionEditContext,
    )

    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # pragma: no cover - environment-specific
    torch = None
    _TORCH_IMPORT_ERROR = error


_REPO_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step2StaticTests(unittest.TestCase):
    def test_production_interface_does_not_depend_on_research_scripts(self) -> None:
        sampling_source = (
            _REPO_ROOT / "baseline/model/style_planner/library/sampling.py"
        ).read_text(encoding="utf-8")
        decoder_source = (
            _REPO_ROOT / "baseline/model/style_planner/layer/decoder.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("research_v1.execution.preference_flow", sampling_source)
        self.assertNotIn("research_v1.execution.preference_flow", decoder_source)
        self.assertIn("dual_stream_dpm_sampler", sampling_source)
        self.assertIn("latest_model_state = x.detach().clone()", sampling_source)
        self.assertIn("neutral_reference_cache", sampling_source)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step2TensorTests(unittest.TestCase):
    class _ToyXStart:
        model_type = "x_start"

        def __call__(self, x, time, **unused_kwargs):
            del time, unused_kwargs
            return 0.125 * x

    class _ContextIdentity:
        def __init__(self) -> None:
            self.contexts = []

        def reset(self) -> None:
            self.contexts.clear()

        def __call__(self, clean_prediction, context: CleanPredictionEditContext):
            self.contexts.append(context)
            return clean_prediction

    class _TinyPreferenceProbe:
        def __init__(self, *, evaluation_index: int, magnitude: float) -> None:
            self.evaluation_index = int(evaluation_index)
            self.magnitude = float(magnitude)
            self.contexts = []
            self.applied_delta = 0.0

        def reset(self) -> None:
            self.contexts.clear()
            self.applied_delta = 0.0

        def __call__(self, clean_prediction, context: CleanPredictionEditContext):
            self.contexts.append(context)
            if context.model_evaluation_index != self.evaluation_index:
                return clean_prediction
            edited = clean_prediction.clone()
            # The first four values are the constrained current state in this
            # planner representation.  Edit a future coordinate instead.
            component = 4 if int(edited.shape[-1]) > 4 else 0
            edited[..., component] += self.magnitude
            self.applied_delta = float(
                (edited - clean_prediction).abs().max().item()
            )
            return edited

    def _sample(self, *, neutral_editor, preference_editor):
        torch.manual_seed(3407)
        x_t = torch.randn((2, 8))
        result = dual_stream_dpm_sampler(
            self._ToyXStart(),
            x_t,
            diffusion_steps=2,
            neutral_editor=neutral_editor,
            preference_editor=preference_editor,
        )
        return x_t, result

    def test_exact_dual_identity_and_dynamic_current_state(self) -> None:
        neutral_editor = self._ContextIdentity()
        preference_editor = self._ContextIdentity()
        x_t, result = self._sample(
            neutral_editor=neutral_editor,
            preference_editor=preference_editor,
        )

        self.assertTrue(
            torch.equal(result.neutral_initial_state, result.preference_initial_state)
        )
        self.assertTrue(torch.equal(result.neutral_initial_state, x_t))
        self.assertTrue(torch.equal(result.neutral_sample, result.preference_sample))
        self.assertEqual(len(result.neutral_trace), 3)
        self.assertEqual(len(result.preference_trace), 3)
        self.assertEqual(len(neutral_editor.contexts), 3)
        self.assertEqual(len(preference_editor.contexts), 3)

        for index, (neutral, preference) in enumerate(
            zip(result.neutral_trace, result.preference_trace)
        ):
            self.assertEqual(neutral.model_evaluation_index, index)
            self.assertEqual(preference.model_evaluation_index, index)
            self.assertEqual(neutral.stream_name, "neutral")
            self.assertEqual(preference.stream_name, "preference")
            self.assertTrue(torch.equal(neutral.diffusion_time, preference.diffusion_time))
            self.assertTrue(torch.equal(neutral.log_snr, preference.log_snr))
            self.assertTrue(torch.equal(neutral.current_state, preference.current_state))
            self.assertTrue(
                torch.equal(neutral.clean_prediction, preference.clean_prediction)
            )
            preference_context = preference_editor.contexts[index]
            self.assertIsNotNone(preference_context.neutral_record)
            self.assertTrue(
                torch.equal(
                    preference_context.neutral_record.current_state,
                    neutral.current_state,
                )
            )
            self.assertTrue(
                torch.equal(
                    preference_context.neutral_record.clean_prediction,
                    neutral.clean_prediction,
                )
            )

        self.assertTrue(torch.equal(result.neutral_trace[0].current_state, x_t))
        self.assertTrue(
            torch.equal(result.preference_trace[0].current_state, x_t)
        )
        self.assertTrue(
            any(
                not torch.equal(result.neutral_trace[0].current_state, record.current_state)
                for record in result.neutral_trace[1:]
            )
        )
        self.assertTrue(
            any(
                not torch.equal(
                    result.preference_trace[0].current_state,
                    record.current_state,
                )
                for record in result.preference_trace[1:]
            )
        )

    def test_preference_probe_isolation_and_propagation(self) -> None:
        reference_neutral_editor = self._ContextIdentity()
        reference_preference_editor = self._ContextIdentity()
        _, reference = self._sample(
            neutral_editor=reference_neutral_editor,
            preference_editor=reference_preference_editor,
        )

        probe_neutral_editor = self._ContextIdentity()
        probe = self._TinyPreferenceProbe(evaluation_index=0, magnitude=1e-3)
        _, probed = self._sample(
            neutral_editor=probe_neutral_editor,
            preference_editor=probe,
        )

        self.assertGreater(probe.applied_delta, 0.0)
        self.assertTrue(torch.equal(reference.neutral_sample, probed.neutral_sample))
        for reference_record, probed_record in zip(
            reference.neutral_trace,
            probed.neutral_trace,
        ):
            self.assertTrue(
                torch.equal(reference_record.current_state, probed_record.current_state)
            )
            self.assertTrue(
                torch.equal(
                    reference_record.clean_prediction,
                    probed_record.clean_prediction,
                )
            )

        self.assertGreater(
            float(
                (
                    probed.preference_trace[0].clean_prediction
                    - probed.neutral_trace[0].clean_prediction
                )
                .abs()
                .max()
                .item()
            ),
            0.0,
        )
        # The trace records solver-facing x0.  Its injected difference must
        # then propagate into the next DPM state.
        propagated_state_error = max(
            float(
                (
                    probed.preference_trace[index].current_state
                    - probed.neutral_trace[index].current_state
                )
                .abs()
                .max()
                .item()
            )
            for index in range(1, len(probed.preference_trace))
        )
        self.assertGreater(propagated_state_error, 0.0)
        self.assertGreater(
            float((probed.preference_sample - probed.neutral_sample).abs().max().item()),
            0.0,
        )

        # The cache supplies clone-on-read records.  Even an accidental write
        # to a test editor's context cannot mutate neutral output/trace data.
        context_record = probe.contexts[0].neutral_record
        self.assertIsNotNone(context_record)
        context_record.clean_prediction.add_(123.0)
        self.assertTrue(
            torch.equal(
                reference.neutral_trace[0].clean_prediction,
                probed.neutral_trace[0].clean_prediction,
            )
        )


if __name__ == "__main__":
    if _TORCH_IMPORT_ERROR is not None:
        print(
            "PyTorch-dependent Step-2 sampler tests will be skipped: "
            f"{_TORCH_IMPORT_ERROR}"
        )
    unittest.main(verbosity=2)
