from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_lora.data.dataset import StyleManifestDataset, style_collate


def test_cache_aliases_and_neighbor_slice_are_applied(tmp_path: Path):
    path = tmp_path / "one.npz"
    np.savez(path, ego_current_state=np.zeros(10), ego_agent_future=np.zeros((4, 3)),
             neighbor_agents_past=np.zeros((12, 3, 11)), neighbor_agents_future=np.zeros((12, 4, 3)),
             lanes=np.zeros((2, 2, 2)), lanes_speed_limit=np.ones((2, 1)), lanes_has_speed_limit=np.ones((2, 1)),
             route_lanes=np.zeros((2, 2, 2)), route_lanes_speed_limit=np.ones((2, 1)), route_lanes_has_speed_limit=np.ones((2, 1)),
             static_objects=np.zeros((1, 3)))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"filename": "one.npz", "split": "train", "scene": "straight_free_drive", "style": "aggr"}) + "\n")
    dataset = StyleManifestDataset(manifest, root=tmp_path, predicted_neighbor_num=10)
    batch = style_collate([dataset[0]])
    assert batch["tensors"]["ego_future_gt"].shape == (1, 4, 3)
    assert batch["tensors"]["neighbors_future_gt"].shape[:2] == (1, 10)
    assert len(batch["metadata"]) == 1
