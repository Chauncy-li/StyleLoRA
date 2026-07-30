"""CPU contracts for the Step-5 adapter, S1 bridge, and S2 attention."""

from __future__ import annotations

import unittest


try:
    import torch
    import torch.nn as nn

    from baseline.model.style_planner.preference_flow import (
        NeutralAnchoredInteractionAttention,
        PreferenceFlowConfig,
        PreferenceFlowConditionEncoder,
        PreferenceFlowTrainingAdapter,
        SmoothLongitudinalTrajectoryResidualDecoder,
    )
    from research_v1.execution.preference_flow.differentiable_behavior_alignment import (
        DifferentiableBehaviorBridge,
        FrozenAxisCalibration,
        FrozenBehaviorCalibration,
        NeutralAnchoredLeadReference,
        feasible_pathwise_loss,
        pathwise_order_loss,
    )

    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # pragma: no cover - environment-specific
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = error


_ModuleBase = nn.Module if nn is not None else object


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class Step5AdapterTests(unittest.TestCase):
    class _ConstantField(_ModuleBase):
        def forward(self, state, condition, diffusion_time, preference_coordinate):
            del condition, diffusion_time, preference_coordinate
            return torch.ones_like(state)

    @staticmethod
    def _inputs():
        # Current is physical; future uses ego mean=[10,0,0,0], std=[20,20,1,1].
        current = torch.tensor([0.0, 0.0, 1.0, 0.0])
        future_physical = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 1.0, 0.0]]
        )
        mean = torch.tensor([10.0, 0.0, 0.0, 0.0])
        std = torch.tensor([20.0, 20.0, 1.0, 1.0])
        future = (future_physical - mean) / std
        ego = torch.cat((current[None, :], future), dim=0).reshape(-1)
        neutral = torch.zeros((2, 3, ego.numel()), dtype=torch.float32)
        neutral[:, 0, :] = ego
        current_state = neutral.clone()
        task_features = torch.tensor(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 2
        )
        return neutral, current_state, task_features

    @staticmethod
    def _adapter(vector_field=None):
        config = PreferenceFlowConfig(
            latent_dim=8,
            condition_dim=24,
            hidden_dim=16,
            num_layers=2,
            zero_initialize_output=True,
        )
        decoder = SmoothLongitudinalTrajectoryResidualDecoder(
            8,
            ego_mean=torch.tensor([10.0, 0.0, 0.0, 0.0]),
            ego_std=torch.tensor([20.0, 20.0, 1.0, 1.0]),
        )
        return PreferenceFlowTrainingAdapter(
            config=config,
            trajectory_decoder=decoder,
            vector_field=vector_field,
        )

    def test_zero_initialized_field_has_exact_identity_and_live_gradients(self) -> None:
        neutral, current_state, task_features = self._inputs()
        adapter = self._adapter()
        output = adapter(
            neutral,
            current_state,
            torch.tensor([0.3, 0.7]),
            torch.tensor([0.2, -0.1]),
            task_features,
            torch.tensor([0.0, 0.75]),
        )
        self.assertTrue(torch.equal(output.clean_prediction, neutral))
        self.assertTrue(torch.equal(output.latent_start, output.latent_end))
        target = neutral[:, 0, 4:].reshape(2, 3, 4).clone()
        target[..., 0] += 0.1
        prediction = output.clean_prediction[:, 0, 4:].reshape(2, 3, 4)
        torch.mean((prediction - target).square()).backward()
        gradient = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in adapter.vector_field.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        self.assertEqual(sum(parameter.numel() for parameter in adapter.trajectory_decoder.parameters()), 0)

    def test_nonzero_probe_is_longitudinal_and_ego_future_only(self) -> None:
        neutral, current_state, task_features = self._inputs()
        output = self._adapter(self._ConstantField())(
            neutral,
            current_state,
            torch.tensor([0.4, 0.4]),
            torch.tensor([0.0, 0.0]),
            task_features,
            torch.tensor([0.0, 0.5]),
        )
        direct = output.clean_prediction - neutral
        self.assertTrue(torch.equal(direct[:, 0, :4], torch.zeros_like(direct[:, 0, :4])))
        self.assertTrue(torch.equal(direct[:, 1:], torch.zeros_like(direct[:, 1:])))
        self.assertTrue(torch.equal(output.clean_prediction[0], neutral[0]))
        self.assertGreater(float(output.ego_future_residual[1].abs().max().item()), 0.0)
        # The neutral path is horizontal, so the decoder cannot create a y edit.
        self.assertEqual(float(output.ego_future_residual[..., 1].abs().max().item()), 0.0)

    def test_decoder_reports_the_exact_tangent_used_for_physical_edits(self) -> None:
        neutral, current_state, task_features = self._inputs()
        physical_current = torch.tensor([[0.0, 0.0, 1.0, 0.0]] * 2)
        output = self._adapter(self._ConstantField())(
            neutral, current_state, torch.tensor([0.4, 0.4]), torch.tensor([0.0, 0.0]),
            task_features, torch.tensor([0.5, -0.5]),
            physical_ego_current_state=physical_current,
        )
        lateral = (
            output.physical_xy_residual[..., 0] * output.base_tangent[..., 1]
            - output.physical_xy_residual[..., 1] * output.base_tangent[..., 0]
        )
        self.assertLessEqual(float(lateral.abs().max().item()), 1e-7)

    def test_neutral_anchored_lead_and_pathwise_semantics(self) -> None:
        calibration = FrozenBehaviorCalibration(
            {
                ("straight_free_drive", 0): FrozenAxisCalibration(0.0, 1.3, "rising"),
                ("straight_car_follow", 0): FrozenAxisCalibration(0.0, 5.0, "falling"),
                ("straight_car_follow", 1): FrozenAxisCalibration(0.0, 10.0, "falling"),
            },
            "selftest",
        )
        current = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        neutral = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 1.0, 0.0]]])
        history = torch.zeros((1, 2, 2, 8))
        history[0, 0, :, :2] = torch.tensor([[8.0, 0.0], [9.0, 0.0]])
        history[0, 0, :, 2] = 1.0
        history[0, 0, :, 4] = 10.0
        history[0, 0, :, 7] = 4.5
        history[0, 1, :, :2] = torch.tensor([[2.0, 4.0], [3.0, 4.0]])
        history[0, 1, :, 2] = 1.0
        history[0, 1, :, 4] = 10.0
        history[0, 1, :, 7] = 4.5
        reference = NeutralAnchoredLeadReference().build(
            neutral_future=neutral, ego_current_state=current, neighbor_history=history
        )
        self.assertEqual(int(reference.lead_indices.item()), 0)
        self.assertFalse(reference.lead_indices.requires_grad)
        self.assertTrue(bool(reference.valid_time_mask.all().item()))
        neighbors = torch.tensor([[[[10.0, 0.0, 1.0, 0.0], [11.0, 0.0, 1.0, 0.0], [12.0, 0.0, 1.0, 0.0]],
                                   [[3.0, 4.0, 1.0, 0.0], [4.0, 4.0, 1.0, 0.0], [5.0, 4.0, 1.0, 0.0]]]])
        bridge = DifferentiableBehaviorBridge(calibration)
        kwargs = {
            "neutral_future": neutral,
            "ego_current_state": current,
            "neighbor_future": neighbors,
            "neighbor_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
            "lead_reference": reference,
            "speed_limit_mps": torch.tensor([15.0]),
            "speed_limit_valid": torch.tensor([True]),
            "free_mask": torch.tensor([False]),
            "car_mask": torch.tensor([True]),
        }
        neutral_measurement = bridge.measure(edited_future=neutral, **kwargs)
        aggressive = neutral.clone()
        aggressive[..., 0] += torch.tensor([[0.1, 0.2, 0.3]])
        aggressive_measurement = bridge.measure(edited_future=aggressive, **kwargs)
        self.assertGreater(float(aggressive_measurement.headway.item()), float(neutral_measurement.headway.item()))
        self.assertFalse(bool(neutral_measurement.ttc_valid.item()))
        increasing = torch.tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])
        decreasing = torch.flip(increasing, dims=(0,))
        grid = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.assertEqual(float(pathwise_order_loss(increasing, torch.tensor([True]), grid, margin_per_rho=0.1)[0].item()), 0.0)
        self.assertGreater(float(pathwise_order_loss(decreasing, torch.tensor([True]), grid, margin_per_rho=0.1)[0].item()), 0.0)

    def test_feasible_pathwise_loss_allows_clipped_plateaus_but_not_reversal(self) -> None:
        target = torch.tensor([[0.0], [0.1], [0.1], [0.2], [0.2]])
        valid = torch.tensor([True])
        monotonic = torch.tensor([[0.0], [0.06], [0.06], [0.12], [0.12]])
        result = feasible_pathwise_loss(monotonic, target, valid, active_fraction=0.5)
        self.assertEqual(float(result.loss.item()), 0.0)
        self.assertEqual(int(result.active_mask.sum().item()), 2)
        self.assertEqual(int(result.saturated_mask.sum().item()), 2)
        reversal = monotonic.clone()
        reversal[2] = 0.03
        self.assertGreater(float(feasible_pathwise_loss(reversal, target, valid).loss.item()), 0.0)

    def test_neutral_attention_is_masked_permutation_equivariant_and_rho_shared(self) -> None:
        torch.manual_seed(7)
        attention = NeutralAnchoredInteractionAttention(context_dim=16)
        ego_current = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        ego = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 1.0, 0.0]]])
        neighbors = torch.tensor([[[[7.0, 0.0, 1.0, 0.0], [8.0, 0.0, 1.0, 0.0], [9.0, 0.0, 1.0, 0.0]],
                                   [[2.0, 4.0, 1.0, 0.0], [3.0, 4.0, 1.0, 0.0], [4.0, 4.0, 1.0, 0.0]],
                                   [[4.0, -3.0, 1.0, 0.0], [5.0, -3.0, 1.0, 0.0], [6.0, -3.0, 1.0, 0.0]]]])
        current = torch.tensor([[[6.0, 0.0, 1.0, 0.0, 10.0, 0.0, 2.0, 4.5],
                                 [1.0, 4.0, 1.0, 0.0, 10.0, 0.0, 2.0, 4.5],
                                 [3.0, -3.0, 1.0, 0.0, 10.0, 0.0, 2.0, 4.5]]])
        mask = torch.tensor([[True, False, True]])
        kwargs = {
            "ego_current_state": ego_current, "neighbor_current_state": current,
            "agent_valid_mask": mask, "diffusion_time": torch.tensor([0.4]), "log_snr": torch.tensor([0.1]),
        }
        original = attention(ego, neighbors, **kwargs)
        permutation = torch.tensor([2, 0, 1])
        swapped = attention(
            ego, neighbors[:, permutation], ego_current_state=ego_current,
            neighbor_current_state=current[:, permutation], agent_valid_mask=mask[:, permutation],
            diffusion_time=torch.tensor([0.4]), log_snr=torch.tensor([0.1]),
        )
        inverse = torch.argsort(permutation)
        self.assertTrue(torch.allclose(original.interaction_context, swapped.interaction_context, atol=1e-6))
        self.assertTrue(torch.allclose(original.neighbor_time_attention_weights, swapped.neighbor_time_attention_weights[:, inverse], atol=1e-6))
        self.assertEqual(float(original.neighbor_time_attention_weights[:, 1].abs().max().item()), 0.0)
        self.assertTrue(torch.allclose(
            original.neighbor_time_attention_weights.sum(dim=(1, 2)) + original.null_interaction_weight,
            torch.ones(1), atol=1e-6,
        ))
        config = PreferenceFlowConfig(condition_dim=40, hidden_dim=16, num_layers=2)
        decoder = SmoothLongitudinalTrajectoryResidualDecoder(8, ego_mean=torch.zeros(4), ego_std=torch.ones(4))
        adapter = PreferenceFlowTrainingAdapter(
            config=config, trajectory_decoder=decoder,
            condition_encoder=PreferenceFlowConditionEncoder(40, task_feature_dim=9, include_diffusion_features=True, interaction_feature_dim=16),
        )
        joint = torch.cat((ego_current[:, None, None, :], ego[:, None]), dim=2).reshape(1, 1, -1)
        joint = torch.cat((joint, torch.zeros_like(joint)), dim=1)
        shared_context = original.interaction_context
        low = adapter(joint, joint, torch.tensor([0.4]), torch.tensor([0.1]), torch.ones((1, 9)), torch.tensor([-1.0]), shared_context, ego_current)
        high = adapter(joint, joint, torch.tensor([0.4]), torch.tensor([0.1]), torch.ones((1, 9)), torch.tensor([1.0]), shared_context, ego_current)
        self.assertTrue(torch.equal(low.condition, high.condition))


if __name__ == "__main__":
    if _TORCH_IMPORT_ERROR is not None:
        print(f"PyTorch-dependent Step-5 tests will be skipped: {_TORCH_IMPORT_ERROR}")
    unittest.main(verbosity=2)
