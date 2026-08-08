import json

import torch

from research_lora.evaluation.style_metrics import differentiable_scene_style_vector
from research_lora.training.style_prototypes import SceneStylePrototypeTable


def _artifact():
    scenes = {}
    for scene, axes in {
        "straight_free_drive": ["speed_preference", "longitudinal_intensity", "smoothness"],
        "straight_car_follow": ["headway_margin", "response_decisiveness", "response_smoothness"],
    }.items():
        scenes[scene] = {
            "axis_names": axes,
            "statistics": {"mean": [0.4, 0.5, 0.6], "std": [0.2, 0.3, 0.4]},
            "prototypes": {
                "aggressive": {"standardized_mean": [1.0, 0.5, -0.2]},
                "conservative": {"standardized_mean": [-1.0, -0.5, 0.2]},
            },
        }
    return {"artifact_version": 1, "metric_space": "expert_trajectory_proxy_axes",
            "source": {"training_split_only": True, "manifest_hash": "test-train-manifest"}, "scenes": scenes}


def test_train_only_prototype_artifact_loads_and_standardizes(tmp_path):
    path = tmp_path / "prototypes.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    table = SceneStylePrototypeTable.from_json(path, device=torch.device("cpu"))
    vector = torch.tensor((0.6, 0.8, 1.0), requires_grad=True)
    standardized = table.standardize("straight_free_drive", vector)
    assert torch.allclose(standardized, torch.ones(3))
    assert table.opposite("aggressive") == "conservative"
    standardized.sum().backward()
    assert torch.all(vector.grad > 0)


def test_differentiable_free_drive_vector_preserves_prediction_gradient():
    ego_future = torch.tensor(
        ((0.70, 0.0, 0.0), (1.42, 0.0, 0.0), (2.16, 0.0, 0.0),
         (2.90, 0.0, 0.0), (3.65, 0.0, 0.0)),
        dtype=torch.float32, requires_grad=True,
    )
    vector, valid = differentiable_scene_style_vector(
        scene="straight_free_drive", ego_future=ego_future, ego_current=torch.zeros(3),
        neighbors_past=torch.zeros(1, 2, 3), neighbors_future=torch.zeros(1, 5, 3),
        route_limits=torch.tensor([10.0]), route_has_limits=torch.tensor([True]),
        lane_limits=torch.tensor([0.0]), lane_has_limits=torch.tensor([False]),
    )
    assert bool(valid.all())
    vector.sum().backward()
    assert ego_future.grad is not None
    assert torch.isfinite(ego_future.grad).all()
    assert ego_future.grad[:, :2].abs().sum() > 0


def test_differentiable_car_follow_without_lead_is_masked():
    vector, valid = differentiable_scene_style_vector(
        scene="straight_car_follow", ego_future=torch.zeros(5, 3, requires_grad=True), ego_current=torch.zeros(3),
        neighbors_past=torch.zeros(1, 2, 3), neighbors_future=torch.zeros(1, 5, 3),
        route_limits=torch.zeros(1), route_has_limits=torch.zeros(1, dtype=torch.bool),
        lane_limits=torch.zeros(1), lane_has_limits=torch.zeros(1, dtype=torch.bool),
        lead_reference_future=torch.zeros(5, 3),
    )
    assert torch.equal(vector, torch.zeros(3))
    assert not bool(valid.any())
