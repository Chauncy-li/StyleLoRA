import pytest
import torch

from stylelora.lora.evaluation.style_metrics import open_loop_behavior_metrics


def _lane(y: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    lane = torch.zeros((1, 4, 12), dtype=torch.float32)
    lane[0, :, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
    lane[0, :, 1] = y
    lane[0, :, 4:6] = torch.tensor([0.0, 2.0])
    lane[0, :, 6:8] = torch.tensor([0.0, -2.0])
    return lane, torch.ones((1, 4), dtype=torch.bool)


def _empty_context(steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    past = torch.zeros((0, 2, 11), dtype=torch.float32)
    future = torch.zeros((0, steps, 3), dtype=torch.float32)
    mask = torch.zeros((0, steps), dtype=torch.bool)
    return past, future, mask


def test_progress_acceleration_and_map_proxies() -> None:
    ego = torch.tensor([
        [0.1, 0.0, 1.0, 0.0],
        [0.3, 0.0, 1.0, 0.0],
        [0.6, 0.0, 1.0, 0.0],
    ])
    lane, lane_mask = _lane()
    past, future, future_mask = _empty_context(ego.shape[0])
    metrics = open_loop_behavior_metrics(
        ego_future=ego,
        ego_current=torch.tensor([0.0, 0.0]),
        neighbors_past=past,
        neighbors_future=future,
        neighbor_future_valid_mask=future_mask,
        route_lanes=lane,
        route_lanes_mask=lane_mask,
        lanes=lane,
        lanes_mask=lane_mask,
        static_objects=torch.zeros((0, 10)),
    )

    assert metrics["planned_progress_m"] == pytest.approx(0.6)
    assert metrics["route_aligned_progress_m"] == pytest.approx(0.6)
    assert metrics["planned_accel_p90_mps2"] == pytest.approx(10.0)
    assert metrics["planned_decel_p90_mps2"] == pytest.approx(0.0)
    assert metrics["drivable_area_proxy_fraction"] == pytest.approx(1.0)
    assert metrics["collision_proxy"] is False


def test_lead_gap_headway_and_ttc_use_bumper_gap() -> None:
    ego = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 1.0, 0.0],
        [3.0, 0.0, 1.0, 0.0],
    ])
    past = torch.zeros((1, 2, 11), dtype=torch.float32)
    past[0, -1, :4] = torch.tensor([10.0, 0.0, 1.0, 0.0])
    past[0, -1, 6:8] = torch.tensor([2.0, 4.0])
    future = torch.tensor([[[10.5, 0.0, 0.0], [11.0, 0.0, 0.0], [11.5, 0.0, 0.0]]])
    metrics = open_loop_behavior_metrics(
        ego_future=ego,
        ego_current=torch.tensor([0.0, 0.0]),
        neighbors_past=past,
        neighbors_future=future,
        neighbor_future_valid_mask=torch.ones((1, 3), dtype=torch.bool),
        static_objects=torch.zeros((0, 10)),
    )

    expected_gap = 11.5 - 3.0 - 4.049 - 2.0
    assert metrics["min_lead_gap_m"] == pytest.approx(expected_gap)
    assert metrics["min_time_headway_s"] == pytest.approx(expected_gap / 10.0)
    assert metrics["min_ttc_s"] == pytest.approx(expected_gap / 5.0)
    assert metrics["ttc_closing_event"] is True


def test_collision_and_offroad_are_explicit_proxies() -> None:
    ego = torch.tensor([
        [1.0, 5.0, 1.0, 0.0],
        [2.0, 5.0, 1.0, 0.0],
    ])
    past = torch.zeros((1, 2, 11), dtype=torch.float32)
    past[0, -1, :4] = torch.tensor([1.0, 5.0, 1.0, 0.0])
    past[0, -1, 6:8] = torch.tensor([2.0, 4.0])
    future = torch.tensor([[[1.0, 5.0, 0.0], [2.0, 5.0, 0.0]]])
    lane, lane_mask = _lane(y=0.0)
    metrics = open_loop_behavior_metrics(
        ego_future=ego,
        ego_current=torch.tensor([0.0, 5.0]),
        neighbors_past=past,
        neighbors_future=future,
        neighbor_future_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        lanes=lane,
        lanes_mask=lane_mask,
        static_objects=torch.zeros((0, 10)),
    )

    assert metrics["offroad_proxy_fraction"] == pytest.approx(1.0)
    assert metrics["collision_proxy"] is True
    assert metrics["min_collision_clearance_m"] < 0.0
