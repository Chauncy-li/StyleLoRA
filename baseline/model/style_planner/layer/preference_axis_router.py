"""Lightweight scene-aware router for the V6 three-axis preference condition."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


V6_STYLE_CONDITION_DIM = 12
V6_AXIS_COUNT = 3


class PreferenceAxisRouter(nn.Module):
    """Map the fixed V6 condition into one DiT conditioning vector.

    The router keeps the public 12-dimensional data contract unchanged:

    ``[axis_target(3), axis_mask(3), scene_one_hot(3), scene_gate(3)]``.

    Each behavior axis becomes an independent token. Existing scene and route
    encodings form the query, while hard data/runtime masks remain structural
    gates. Independent sigmoid gates are used because multiple axes may be
    simultaneously applicable; a softmax would incorrectly make them compete.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or token_dim <= 0:
            raise ValueError("hidden_dim and token_dim must be positive")

        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        # centered target, magnitude, hard mask, scene one-hot, scene gates,
        # and axis identity.
        self.axis_token_input_dim = 1 + 1 + 1 + 3 + 3 + 3

        self.token_encoder = nn.Sequential(
            nn.LayerNorm(self.axis_token_input_dim),
            nn.Linear(self.axis_token_input_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.token_dim),
        )
        self.key_encoder = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_encoder = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.axis_logit_bias = nn.Parameter(torch.zeros(V6_AXIS_COUNT))
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            # No bias is intentional: an all-zero/empty style condition must
            # remain an exact zero residual after training, not merely at
            # initialization. This preserves the original planner branch.
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Zero-init the residual so a pretrained planner starts unchanged."""

        output = self.output_projection[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("PreferenceAxisRouter output projection must end with nn.Linear")
        nn.init.zeros_(output.weight)
        if output.bias is not None:
            nn.init.zeros_(output.bias)

    @staticmethod
    def _validate_condition(condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[-1] != V6_STYLE_CONDITION_DIM:
            raise ValueError(
                "axis_router_v1 requires V6 style condition shape [B, 12], "
                f"got {tuple(condition.shape)}"
            )
        return condition

    def _axis_tokens(
        self,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = condition[:, 0:3].clamp(0.0, 1.0)
        hard_mask = condition[:, 3:6].clamp(0.0, 1.0)
        scene_one_hot = condition[:, 6:9].clamp(0.0, 1.0)
        scene_gate = condition[:, 9:12].clamp(0.0, 1.0)

        centered = target - 0.5
        magnitude = centered.abs()
        batch_size = condition.shape[0]
        axis_identity = torch.eye(
            V6_AXIS_COUNT,
            device=condition.device,
            dtype=condition.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        scene_one_hot_tokens = scene_one_hot.unsqueeze(1).expand(-1, V6_AXIS_COUNT, -1)
        scene_gate_tokens = scene_gate.unsqueeze(1).expand(-1, V6_AXIS_COUNT, -1)
        token_input = torch.cat(
            [
                centered.unsqueeze(-1),
                magnitude.unsqueeze(-1),
                hard_mask.unsqueeze(-1),
                scene_one_hot_tokens,
                scene_gate_tokens,
                axis_identity,
            ],
            dim=-1,
        )
        return token_input, hard_mask, scene_one_hot, scene_gate

    def forward(
        self,
        condition: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        condition = self._validate_condition(condition)
        if scene_context.ndim != 2 or scene_context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"scene_context must have shape [B, {self.hidden_dim}], "
                f"got {tuple(scene_context.shape)}"
            )
        if scene_context.shape[0] != condition.shape[0]:
            raise ValueError("condition and scene_context batch sizes must match")

        token_input, hard_mask, scene_one_hot, scene_gate = self._axis_tokens(condition)
        token = self.token_encoder(token_input)
        query = self.query_encoder(scene_context).unsqueeze(1)
        key = self.key_encoder(token)
        logits = (query * key).sum(dim=-1) / math.sqrt(float(self.token_dim))
        logits = logits + self.axis_logit_bias.unsqueeze(0)

        # The selected scene gate is a confidence/applicability scalar. Lane
        # style is disabled by the V6 data/runtime contract, so its selected
        # gate is zero.
        selected_scene_gate = (scene_one_hot * scene_gate).sum(dim=-1, keepdim=True)
        availability = hard_mask * selected_scene_gate
        learned_gate = torch.sigmoid(logits)
        routed_gate = availability * learned_gate

        value = self.value_encoder(token)
        denominator = routed_gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
        routed = (routed_gate.unsqueeze(-1) * value).sum(dim=1) / denominator
        residual = self.output_projection(routed)

        diagnostics = {
            "axis_router_logits": logits,
            "axis_router_learned_gate": learned_gate,
            "axis_router_gate": routed_gate,
            "axis_router_availability": availability,
            "axis_router_selected_scene_gate": selected_scene_gate.squeeze(-1),
            "axis_router_active_mask": hard_mask,
        }
        return residual, diagnostics


class SignedPreferenceAxisRouter(nn.Module):
    """Preference-factorized V6 router with structural sign and anchor guarantees.

    Applicability and axis values depend only on the current observation and
    causal condition metadata.  Preference enters the residual through the
    explicit signed scalar ``d = 2 * (p - 0.5)``.  Consequently, empty masks
    and semantic-normal targets produce an exact zero residual, while flipping
    all active preference signs flips the routed condition before it reaches
    the frozen planner.

    No affine normalization is applied after the signed weighted sum.  This is
    intentional: post-routing LayerNorm would remove most of the command
    magnitude and its affine bias would break the exact-zero anchor.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or token_dim <= 0:
            raise ValueError("hidden_dim and token_dim must be positive")

        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.axis_embedding = nn.Parameter(
            torch.empty(V6_AXIS_COUNT, self.token_dim)
        )
        nn.init.normal_(self.axis_embedding, mean=0.0, std=0.02)

        self.query_encoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.token_dim),
        )
        self.key_encoder = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.context_value_encoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.axis_value_encoder = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim, bias=False),
        )
        self.axis_logit_bias = nn.Parameter(torch.zeros(V6_AXIS_COUNT))

        # A single bias-free linear map is the only operation after the signed
        # sum.  W(0)=0 therefore remains true after arbitrary router training.
        self.output_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Start from the exact pretrained-planner function."""

        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def _condition_parts(
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = PreferenceAxisRouter._validate_condition(condition)
        target = condition[:, 0:3].clamp(0.0, 1.0)
        hard_mask = condition[:, 3:6].clamp(0.0, 1.0)
        scene_one_hot = condition[:, 6:9].clamp(0.0, 1.0)
        scene_gate = condition[:, 9:12].clamp(0.0, 1.0)
        signed_delta = 2.0 * (target - 0.5)
        return signed_delta, hard_mask, scene_one_hot, scene_gate

    def forward(
        self,
        condition: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        signed_delta, hard_mask, scene_one_hot, scene_gate = self._condition_parts(
            condition
        )
        if scene_context.ndim != 2 or scene_context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"scene_context must have shape [B, {self.hidden_dim}], "
                f"got {tuple(scene_context.shape)}"
            )
        if scene_context.shape[0] != signed_delta.shape[0]:
            raise ValueError("condition and scene_context batch sizes must match")

        query = self.query_encoder(scene_context)
        key = self.key_encoder(self.axis_embedding)
        logits = torch.einsum("bd,jd->bj", query, key) / math.sqrt(
            float(self.token_dim)
        )
        logits = logits + self.axis_logit_bias.unsqueeze(0)
        learned_gate = torch.sigmoid(logits)

        selected_scene_gate = (scene_one_hot * scene_gate).sum(
            dim=-1,
            keepdim=True,
        )
        availability = hard_mask * selected_scene_gate
        routed_gate = availability * learned_gate

        context_value = self.context_value_encoder(scene_context).unsqueeze(1)
        axis_value = self.axis_value_encoder(self.axis_embedding).unsqueeze(0)
        # Normalize the preference-independent basis, not the signed mixture.
        value = F.normalize(context_value + axis_value, dim=-1, eps=1e-6)
        signed_coefficient = routed_gate * signed_delta
        active_count = hard_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        routed = (
            signed_coefficient.unsqueeze(-1) * value
        ).sum(dim=1) / torch.sqrt(active_count)
        residual = self.output_projection(routed)
        # Keep the historical summed residual above bit-for-bit on its original
        # path, but also expose its linear per-axis decomposition for the A3.7
        # ego adapter.  The private tensor is consumed inside DiT and removed
        # before decoder diagnostics are returned to callers.
        axis_routed = (
            signed_coefficient.unsqueeze(-1) * value
        ) / torch.sqrt(active_count).unsqueeze(-1)
        axis_residual = self.output_projection(axis_routed)

        diagnostics = {
            "axis_router_logits": logits,
            "axis_router_learned_gate": learned_gate,
            "axis_router_gate": routed_gate,
            "axis_router_availability": availability,
            "axis_router_signed_delta": signed_delta,
            "axis_router_signed_coefficient": signed_coefficient,
            "axis_router_selected_scene_gate": selected_scene_gate.squeeze(-1),
            "axis_router_active_mask": hard_mask,
            "axis_router_residual_l2": torch.linalg.norm(residual, dim=-1),
            "axis_router_axis_residual_l2": torch.linalg.norm(
                axis_residual,
                dim=-1,
            ),
            "_axis_router_axis_residual": axis_residual,
        }
        return residual, diagnostics


class EgoSignedOutputAdapter(nn.Module):
    """Map a signed routed condition to an ego-only denoising residual.

    The adapter is deliberately bias-free and linear.  Combined with the
    signed router this gives two structural properties that do not depend on a
    learned loss: semantic normal/empty commands produce exactly zero, and a
    global sign flip of the active preference axes flips the output residual.
    The current-state slot is never modified; only future ego states receive
    the lightweight residual.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        current_state_dim: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= current_state_dim:
            raise ValueError(
                "ego signed output adapter needs positive hidden_dim and "
                "output_dim > current_state_dim"
            )
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.current_state_dim = int(current_state_dim)
        self.future_projection = nn.Linear(
            self.hidden_dim,
            self.output_dim - self.current_state_dim,
            bias=False,
        )
        nn.init.normal_(self.future_projection.weight, mean=0.0, std=0.02)

    def forward(self, signed_residual: torch.Tensor) -> torch.Tensor:
        if signed_residual.ndim != 2 or signed_residual.shape[-1] != self.hidden_dim:
            raise ValueError(
                "signed_residual must have shape "
                f"[B, {self.hidden_dim}], got {tuple(signed_residual.shape)}"
            )
        future = self.future_projection(signed_residual)
        current = future.new_zeros((future.shape[0], self.current_state_dim))
        return torch.cat([current, future], dim=-1)


class KinematicEgoSignedOutputAdapter(nn.Module):
    """Convert a signed style vector into a smooth ego longitudinal residual.

    Unlike :class:`EgoSignedOutputAdapter`, this adapter cannot independently
    perturb every future state.  It predicts a small set of coefficients for a
    fixed smooth acceleration basis, integrates the resulting profile twice,
    and projects the displacement along the detached base-trajectory heading.

    The coefficient projection is bias-free and every later operation is
    linear in those coefficients.  For a fixed base prediction, zero input
    therefore gives exact zero and a global preference sign flip gives the
    exact negative residual.  Only future ego ``x/y`` entries are modified;
    the current-state and heading entries remain structurally zero.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        state_dim: int = 4,
        basis_count: int = 6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= state_dim:
            raise ValueError(
                "kinematic ego signed adapter needs positive hidden_dim and "
                "output_dim > state_dim"
            )
        if state_dim != 4:
            raise ValueError(
                "kinematic ego signed adapter expects [x, y, cos, sin] states"
            )
        if output_dim % state_dim != 0:
            raise ValueError("output_dim must contain a whole number of ego states")
        if basis_count < 2:
            raise ValueError("basis_count must be at least 2")

        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.state_dim = int(state_dim)
        self.state_count = self.output_dim // self.state_dim
        self.future_steps = self.state_count - 1
        self.basis_count = int(basis_count)
        if self.future_steps < 2:
            raise ValueError("kinematic ego signed adapter needs at least two future steps")

        self.control_projection = nn.Linear(
            self.hidden_dim,
            self.basis_count,
            bias=False,
        )
        nn.init.normal_(self.control_projection.weight, mean=0.0, std=0.02)
        self.register_buffer(
            "acceleration_basis",
            self._build_acceleration_basis(
                future_steps=self.future_steps,
                basis_count=self.basis_count,
            ),
            persistent=False,
        )

    @staticmethod
    def _build_acceleration_basis(
        *,
        future_steps: int,
        basis_count: int,
    ) -> torch.Tensor:
        """Create a global Bernstein basis with a zero-onset envelope."""

        u = torch.arange(1, future_steps + 1, dtype=torch.float32) / float(
            future_steps
        )
        degree = basis_count - 1
        columns = []
        for index in range(basis_count):
            coefficient = float(math.comb(degree, index))
            bernstein = (
                coefficient
                * u.pow(index)
                * (1.0 - u).pow(degree - index)
            )
            # Starting the relative acceleration from zero protects position
            # and velocity continuity at the observed/current ego state.
            columns.append(u * bernstein)
        return torch.stack(columns, dim=-1)

    def forward(
        self,
        signed_residual: torch.Tensor,
        base_ego_output: torch.Tensor,
    ) -> torch.Tensor:
        if signed_residual.ndim != 2 or signed_residual.shape[-1] != self.hidden_dim:
            raise ValueError(
                "signed_residual must have shape "
                f"[B, {self.hidden_dim}], got {tuple(signed_residual.shape)}"
            )
        coefficients = self.control_projection(signed_residual)
        basis = self.acceleration_basis.to(
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        relative_acceleration = torch.einsum(
            "bk,tk->bt",
            coefficients,
            basis,
        )
        return self._acceleration_to_output(
            relative_acceleration,
            base_ego_output,
        )

    def _acceleration_to_output(
        self,
        relative_acceleration: torch.Tensor,
        base_ego_output: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate one relative-acceleration profile into an ego residual."""

        if (
            relative_acceleration.ndim != 2
            or relative_acceleration.shape[-1] != self.future_steps
        ):
            raise ValueError(
                "relative_acceleration must have shape "
                f"[B, {self.future_steps}], got {tuple(relative_acceleration.shape)}"
            )
        if (
            base_ego_output.ndim != 2
            or base_ego_output.shape[0] != relative_acceleration.shape[0]
            or base_ego_output.shape[-1] != self.output_dim
        ):
            raise ValueError(
                "base_ego_output must have shape "
                f"[B, {self.output_dim}], got {tuple(base_ego_output.shape)}"
            )

        # Normalized time integration keeps the basis independent of the
        # dataset sampling rate; the learned coefficients determine amplitude.
        integration_step = 1.0 / float(self.future_steps)
        relative_velocity = torch.cumsum(relative_acceleration, dim=-1)
        relative_velocity = relative_velocity * integration_step
        longitudinal_displacement = torch.cumsum(relative_velocity, dim=-1)
        longitudinal_displacement = longitudinal_displacement * integration_step

        base_states = base_ego_output.detach().reshape(
            base_ego_output.shape[0],
            self.state_count,
            self.state_dim,
        )
        future_heading = base_states[:, 1:, 2:4]
        heading_norm = torch.linalg.norm(future_heading, dim=-1, keepdim=True)

        base_xy = base_states[:, :, :2]
        fallback_tangent = F.normalize(
            base_xy[:, 1:, :] - base_xy[:, :-1, :],
            dim=-1,
            eps=1e-6,
        )
        default_tangent = torch.zeros_like(fallback_tangent)
        default_tangent[..., 0] = 1.0
        fallback_norm = torch.linalg.norm(fallback_tangent, dim=-1, keepdim=True)
        fallback_tangent = torch.where(
            fallback_norm > 1e-5,
            fallback_tangent,
            default_tangent,
        )
        tangent = torch.where(
            heading_norm > 1e-5,
            future_heading / heading_norm.clamp_min(1e-6),
            fallback_tangent,
        )

        future_xy_residual = longitudinal_displacement.unsqueeze(-1) * tangent
        future_residual = torch.zeros(
            (
                relative_acceleration.shape[0],
                self.future_steps,
                self.state_dim,
            ),
            device=relative_acceleration.device,
            dtype=relative_acceleration.dtype,
        )
        future_residual[:, :, :2] = future_xy_residual
        current_residual = future_residual.new_zeros(
            (relative_acceleration.shape[0], 1, self.state_dim)
        )
        return torch.cat([current_residual, future_residual], dim=1).reshape(
            relative_acceleration.shape[0],
            self.output_dim,
        )


class AxisTemporalKinematicEgoSignedOutputAdapter(
    KinematicEgoSignedOutputAdapter
):
    """A3.7 free-drive axis-wise temporal kinematic residual.

    Car-follow and every non-free-drive row retain the historical single-head
    Bernstein path inherited from :class:`KinematicEgoSignedOutputAdapter`.
    Free-drive rows instead keep the Router's three signed axis contributions
    separate until each has produced its own acceleration profile.  The three
    profiles are summed only in trajectory space and then integrated twice.

    All learned maps are bias-free and all temporal banks are fixed. Therefore
    semantic normal/empty commands remain exact zero and a global rho sign flip
    remains exactly odd for a fixed base prediction.  The basis shapes encode
    time scale, never response sign: speed utilization uses the long-horizon
    Bernstein bank, acceleration willingness uses localized smooth pulses, and
    two-second speed response uses a medium-width smooth bank.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        state_dim: int = 4,
        basis_count: int = 6,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            state_dim=state_dim,
            basis_count=basis_count,
        )
        self.axis_control_projections = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.basis_count, bias=False)
                for _ in range(V6_AXIS_COUNT)
            ]
        )
        for projection in self.axis_control_projections:
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
        self.register_buffer(
            "axis_acceleration_basis",
            self._build_axis_acceleration_bases(
                future_steps=self.future_steps,
                basis_count=self.basis_count,
            ),
            persistent=False,
        )

    @classmethod
    def _build_axis_acceleration_bases(
        cls,
        *,
        future_steps: int,
        basis_count: int,
    ) -> torch.Tensor:
        """Build fixed long/local/medium time banks in free-axis order."""

        long_horizon = cls._build_acceleration_basis(
            future_steps=future_steps,
            basis_count=basis_count,
        )
        u = torch.arange(1, future_steps + 1, dtype=torch.float32) / float(
            future_steps
        )
        target_column_l2 = torch.linalg.norm(
            long_horizon,
            dim=0,
            keepdim=True,
        )

        def _gaussian_bank(*, start: float, end: float, width: float) -> torch.Tensor:
            centers = torch.linspace(start, end, basis_count, dtype=torch.float32)
            bank = u.unsqueeze(-1) * torch.exp(
                -0.5 * ((u.unsqueeze(-1) - centers.unsqueeze(0)) / width).pow(2)
            )
            # Match every column's L2 energy to the historical Bernstein bank.
            # The experiment changes temporal support, not implicit axis gain.
            return (
                bank
                / torch.linalg.norm(bank, dim=0, keepdim=True).clamp_min(1e-6)
                * target_column_l2
            )

        local_acceleration = _gaussian_bank(start=0.08, end=0.92, width=0.10)
        medium_response = _gaussian_bank(start=0.12, end=0.88, width=0.20)
        return torch.stack(
            [long_horizon, local_acceleration, medium_response],
            dim=0,
        )

    @staticmethod
    def _profile_cosines(axis_acceleration: torch.Tensor) -> torch.Tensor:
        pairs = ((0, 1), (0, 2), (1, 2))
        return torch.stack(
            [
                F.cosine_similarity(
                    axis_acceleration[:, left, :],
                    axis_acceleration[:, right, :],
                    dim=-1,
                    eps=1e-6,
                )
                for left, right in pairs
            ],
            dim=-1,
        )

    def forward(
        self,
        signed_residual: torch.Tensor,
        signed_axis_residual: torch.Tensor,
        base_ego_output: torch.Tensor,
        free_drive_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if signed_residual.ndim != 2 or signed_residual.shape[-1] != self.hidden_dim:
            raise ValueError(
                "signed_residual must have shape "
                f"[B, {self.hidden_dim}], got {tuple(signed_residual.shape)}"
            )
        expected_axis_shape = (
            signed_residual.shape[0],
            V6_AXIS_COUNT,
            self.hidden_dim,
        )
        if tuple(signed_axis_residual.shape) != expected_axis_shape:
            raise ValueError(
                "signed_axis_residual must have shape "
                f"{expected_axis_shape}, got {tuple(signed_axis_residual.shape)}"
            )
        if free_drive_mask.ndim != 1 or free_drive_mask.shape[0] != signed_residual.shape[0]:
            raise ValueError(
                "free_drive_mask must have shape "
                f"[{signed_residual.shape[0]}], got {tuple(free_drive_mask.shape)}"
            )

        legacy_coefficients = self.control_projection(signed_residual)
        legacy_basis = self.acceleration_basis.to(
            device=legacy_coefficients.device,
            dtype=legacy_coefficients.dtype,
        )
        legacy_acceleration = torch.einsum(
            "bk,tk->bt",
            legacy_coefficients,
            legacy_basis,
        )

        axis_coefficients = torch.stack(
            [
                projection(signed_axis_residual[:, axis_index, :])
                for axis_index, projection in enumerate(
                    self.axis_control_projections
                )
            ],
            dim=1,
        )
        axis_basis = self.axis_acceleration_basis.to(
            device=axis_coefficients.device,
            dtype=axis_coefficients.dtype,
        )
        axis_acceleration = torch.einsum(
            "bjk,jtk->bjt",
            axis_coefficients,
            axis_basis,
        )
        free_drive_mask = free_drive_mask.to(
            device=axis_acceleration.device,
            dtype=torch.bool,
        )
        free_acceleration = axis_acceleration.sum(dim=1)
        relative_acceleration = torch.where(
            free_drive_mask.unsqueeze(-1),
            free_acceleration,
            legacy_acceleration,
        )
        output = self._acceleration_to_output(
            relative_acceleration,
            base_ego_output,
        )

        # Compact, loss-free diagnostics. Non-free rows are explicitly zeroed
        # so car-follow fallback values cannot be misread as free-axis evidence.
        diagnostic_mask = free_drive_mask.to(axis_acceleration.dtype).view(-1, 1, 1)
        diagnostic_acceleration = axis_acceleration * diagnostic_mask
        diagnostic_coefficients = axis_coefficients * diagnostic_mask
        first_end = max(self.future_steps // 3, 1)
        second_end = max(2 * self.future_steps // 3, first_end + 1)
        debug = {
            "axis_temporal_free_drive_used": free_drive_mask,
            "axis_temporal_coefficient_l2": torch.linalg.norm(
                diagnostic_coefficients,
                dim=-1,
            ),
            "axis_temporal_acceleration_rms": torch.sqrt(
                diagnostic_acceleration.pow(2).mean(dim=-1)
            ),
            "axis_temporal_acceleration_early_mean": diagnostic_acceleration[
                :, :, :first_end
            ].mean(dim=-1),
            "axis_temporal_acceleration_mid_mean": diagnostic_acceleration[
                :, :, first_end:second_end
            ].mean(dim=-1),
            "axis_temporal_acceleration_late_mean": diagnostic_acceleration[
                :, :, second_end:
            ].mean(dim=-1),
            "axis_temporal_profile_cosine": self._profile_cosines(
                diagnostic_acceleration
            ),
            "axis_temporal_total_acceleration_rms": torch.sqrt(
                relative_acceleration.pow(2).mean(dim=-1)
            ),
        }
        return output, debug


def selftest_preference_axis_router() -> Dict[str, object]:
    """Small deterministic structural self-test; no planner checkpoint needed."""

    torch.manual_seed(7)
    router = PreferenceAxisRouter(hidden_dim=16, token_dim=8)
    condition = torch.tensor(
        [
            [0.7, 0.7, 0.7, 1, 1, 1, 0, 1, 0, 0, 0.8, 0],
            [0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    residual, debug = router(condition, torch.randn(2, 16))
    with torch.no_grad():
        nn.init.normal_(router.output_projection[-1].weight, std=0.05)
    trained_residual, _ = router(condition, torch.randn(2, 16))
    return {
        "residual_shape": list(residual.shape),
        "zero_initialized_residual": bool(torch.allclose(residual, torch.zeros_like(residual))),
        "empty_condition_gate_zero": bool(torch.allclose(debug["axis_router_gate"][1], torch.zeros(3))),
        "active_condition_has_gate": bool(torch.any(debug["axis_router_gate"][0] > 0)),
        "empty_residual_stays_zero_after_training": bool(
            torch.allclose(
                trained_residual[1],
                torch.zeros_like(trained_residual[1]),
                atol=1e-7,
            )
        ),
    }


def selftest_signed_preference_axis_router() -> Dict[str, object]:
    """Verify exact zero, gate invariance, oddness, and amplitude response."""

    torch.manual_seed(11)
    router = SignedPreferenceAxisRouter(hidden_dim=16, token_dim=8)
    router.eval()
    with torch.no_grad():
        nn.init.normal_(router.output_projection.weight, std=0.05)

    scene_context = torch.randn(1, 16)
    metadata = torch.tensor(
        [[1, 1, 1, 0, 1, 0, 0, 0.8, 0]],
        dtype=torch.float32,
    )

    def _condition(target: Tuple[float, float, float]) -> torch.Tensor:
        return torch.cat(
            [torch.tensor([target], dtype=torch.float32), metadata],
            dim=-1,
        )

    negative = _condition((0.3, 0.3, 0.3))
    normal = _condition((0.5, 0.5, 0.5))
    positive = _condition((0.7, 0.7, 0.7))
    half_positive = _condition((0.6, 0.6, 0.6))
    empty = torch.zeros_like(normal)

    negative_residual, negative_debug = router(negative, scene_context)
    normal_residual, normal_debug = router(normal, scene_context)
    positive_residual, positive_debug = router(positive, scene_context)
    half_residual, half_debug = router(half_positive, scene_context)
    empty_residual, _ = router(empty, scene_context)

    return {
        "empty_residual_exact_zero": bool(
            torch.equal(empty_residual, torch.zeros_like(empty_residual))
        ),
        "normal_residual_exact_zero": bool(
            torch.equal(normal_residual, torch.zeros_like(normal_residual))
        ),
        "sign_flip_is_odd": bool(
            torch.allclose(
                positive_residual,
                -negative_residual,
                atol=1e-6,
                rtol=1e-5,
            )
        ),
        "half_command_halves_residual": bool(
            torch.allclose(
                half_residual,
                0.5 * positive_residual,
                atol=1e-6,
                rtol=1e-5,
            )
        ),
        "gate_is_preference_invariant": bool(
            torch.allclose(
                negative_debug["axis_router_gate"],
                positive_debug["axis_router_gate"],
                atol=1e-7,
            )
            and torch.allclose(
                normal_debug["axis_router_gate"],
                half_debug["axis_router_gate"],
                atol=1e-7,
            )
        ),
        "normal_gate_can_remain_active": bool(
            torch.any(normal_debug["axis_router_gate"] > 0)
        ),
        "axis_residuals_sum_to_global": bool(
            torch.allclose(
                positive_debug["_axis_router_axis_residual"].sum(dim=1),
                positive_residual,
                atol=1e-6,
                rtol=1e-5,
            )
        ),
    }


def selftest_ego_signed_output_adapter() -> Dict[str, object]:
    """Verify ego-output zero anchor, oddness, and current-state isolation."""

    torch.manual_seed(19)
    adapter = EgoSignedOutputAdapter(
        hidden_dim=16,
        output_dim=36,
        current_state_dim=4,
    )
    residual = torch.randn(2, 16)
    positive = adapter(residual)
    negative = adapter(-residual)
    zero = adapter(torch.zeros_like(residual))
    return {
        "zero_input_exact_zero": bool(torch.equal(zero, torch.zeros_like(zero))),
        "sign_flip_exact_odd": bool(
            torch.allclose(positive, -negative, atol=1e-7, rtol=1e-6)
        ),
        "current_state_exact_zero": bool(
            torch.equal(positive[:, :4], torch.zeros_like(positive[:, :4]))
        ),
        "future_residual_nonzero": bool(torch.any(positive[:, 4:].abs() > 0.0)),
    }


def selftest_kinematic_ego_signed_output_adapter() -> Dict[str, object]:
    """Verify smooth ego-only displacement, exact zero, and exact oddness."""

    torch.manual_seed(23)
    adapter = KinematicEgoSignedOutputAdapter(
        hidden_dim=16,
        output_dim=36,
        state_dim=4,
        basis_count=4,
    )
    residual = torch.randn(2, 16)
    base_states = torch.zeros(2, 9, 4)
    base_states[:, :, 0] = torch.linspace(0.0, 1.0, 9)
    base_states[:, :, 2] = 1.0
    base_output = base_states.reshape(2, 36)

    positive = adapter(residual, base_output)
    negative = adapter(-residual, base_output)
    zero = adapter(torch.zeros_like(residual), base_output)
    positive_states = positive.reshape(2, 9, 4)
    future_xy = positive_states[:, 1:, :2]
    first_difference = future_xy[:, 1:, :] - future_xy[:, :-1, :]
    second_difference = first_difference[:, 1:, :] - first_difference[:, :-1, :]
    third_difference = second_difference[:, 1:, :] - second_difference[:, :-1, :]
    return {
        "zero_input_exact_zero": bool(torch.equal(zero, torch.zeros_like(zero))),
        "sign_flip_exact_odd": bool(
            torch.allclose(positive, -negative, atol=1e-7, rtol=1e-6)
        ),
        "current_state_exact_zero": bool(
            torch.equal(
                positive_states[:, 0, :],
                torch.zeros_like(positive_states[:, 0, :]),
            )
        ),
        "future_heading_exact_zero": bool(
            torch.equal(
                positive_states[:, 1:, 2:4],
                torch.zeros_like(positive_states[:, 1:, 2:4]),
            )
        ),
        "future_xy_nonzero": bool(torch.any(future_xy.abs() > 0.0)),
        "longitudinal_only": bool(
            torch.equal(future_xy[:, :, 1], torch.zeros_like(future_xy[:, :, 1]))
        ),
        "finite_smooth_third_difference": bool(
            torch.isfinite(third_difference).all()
        ),
    }


def selftest_axis_temporal_kinematic_ego_signed_output_adapter() -> Dict[str, object]:
    """Verify A3.7 isolation, temporal diversity, zero anchor, and oddness."""

    torch.manual_seed(29)
    adapter = AxisTemporalKinematicEgoSignedOutputAdapter(
        hidden_dim=16,
        output_dim=36,
        state_dim=4,
        basis_count=4,
    )
    axis_residual = torch.randn(2, V6_AXIS_COUNT, 16)
    global_residual = axis_residual.sum(dim=1)
    base_states = torch.zeros(2, 9, 4)
    base_states[:, :, 0] = torch.linspace(0.0, 1.0, 9)
    base_states[:, :, 2] = 1.0
    base_output = base_states.reshape(2, 36)
    free_mask = torch.tensor([True, False])

    positive, debug = adapter(
        global_residual,
        axis_residual,
        base_output,
        free_mask,
    )
    negative, _ = adapter(
        -global_residual,
        -axis_residual,
        base_output,
        free_mask,
    )
    zero, _ = adapter(
        torch.zeros_like(global_residual),
        torch.zeros_like(axis_residual),
        base_output,
        free_mask,
    )
    changed_global = global_residual.clone()
    changed_global[0] = changed_global[0] + torch.randn_like(changed_global[0])
    free_same, _ = adapter(
        changed_global,
        axis_residual,
        base_output,
        free_mask,
    )
    changed_axes = axis_residual.clone()
    changed_axes[1] = changed_axes[1] + torch.randn_like(changed_axes[1])
    car_same, _ = adapter(
        global_residual,
        changed_axes,
        base_output,
        free_mask,
    )
    positive_states = positive.reshape(2, 9, 4)
    basis = adapter.axis_acceleration_basis
    return {
        "zero_input_exact_zero": bool(torch.equal(zero, torch.zeros_like(zero))),
        "sign_flip_exact_odd": bool(
            torch.allclose(positive, -negative, atol=1e-7, rtol=1e-6)
        ),
        "current_state_exact_zero": bool(
            torch.equal(
                positive_states[:, 0, :],
                torch.zeros_like(positive_states[:, 0, :]),
            )
        ),
        "future_heading_exact_zero": bool(
            torch.equal(
                positive_states[:, 1:, 2:4],
                torch.zeros_like(positive_states[:, 1:, 2:4]),
            )
        ),
        "free_drive_uses_axis_path_only": bool(
            torch.allclose(positive[0], free_same[0], atol=1e-7, rtol=1e-6)
        ),
        "car_follow_uses_legacy_path_only": bool(
            torch.allclose(positive[1], car_same[1], atol=1e-7, rtol=1e-6)
        ),
        "axis_time_banks_are_distinct": bool(
            not torch.allclose(basis[0], basis[1])
            and not torch.allclose(basis[0], basis[2])
            and not torch.allclose(basis[1], basis[2])
        ),
        "free_drive_selector_is_exact": bool(
            torch.equal(debug["axis_temporal_free_drive_used"], free_mask)
        ),
        "diagnostics_are_finite": bool(
            all(
                torch.isfinite(value.float()).all()
                for value in debug.values()
                if torch.is_tensor(value)
            )
        ),
    }
