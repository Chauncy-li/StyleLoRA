"""Tests for the minimal feedback-to-rho runtime interface."""

import pytest

from stylelora.runtime.preference_state import PreferenceState


def test_feedback_direction_accumulation_and_keep() -> None:
    state = PreferenceState(rho=0.0, step=0.25)
    assert state.update("more_aggressive")["rho_after"] == pytest.approx(0.25)
    assert state.update("more_aggressive")["rho_after"] == pytest.approx(0.50)
    assert state.update("keep")["rho_after"] == pytest.approx(0.50)
    assert state.update("more_conservative")["rho_after"] == pytest.approx(0.25)


def test_feedback_clips_at_style_endpoints() -> None:
    high = PreferenceState(rho=0.9, step=0.25)
    low = PreferenceState(rho=-0.9, step=0.25)
    assert high.update("more_aggressive")["rho_after"] == pytest.approx(1.0)
    assert low.update("more_conservative")["rho_after"] == pytest.approx(-1.0)


def test_neutral_point_is_unchanged_without_directional_feedback() -> None:
    state = PreferenceState()
    update = state.update("keep")
    assert update["rho_before"] == 0.0
    assert update["rho_after"] == 0.0
    assert update["applied_delta"] == 0.0


def test_invalid_feedback_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported feedback"):
        PreferenceState().update("faster")
