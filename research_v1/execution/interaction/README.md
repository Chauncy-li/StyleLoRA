## Interaction State

This module implements the first executable layer of the small paper:

`Causal-Interaction-State-Conditioned Controllable Preference Diffusion Planning`

### Role

It does **not** infer user preference and it does **not** replace the scene
split pipeline. Instead, it builds an explicit proxy for the current
interaction-state:

- observable
- causal-friendly
- planner-execution oriented
- separate from the `z_style` / retrieval memory line in the 3090 project

### Inputs

The current implementation reads the existing `style_scene_split_v2`
`split_index.jsonl` and extracts a structured interaction-state proxy from
already-available fields such as:

- `lead_vehicle_present`
- `following_min_gap`
- `following_min_thw`
- `merge_min_gap`
- `merge_lateral_closure`
- `ego_speed_ratio_to_limit`
- `event_speed_drop_ratio`
- `ego_lateral_disp`
- `ego_lateral_speed_peak`
- `route_lane_count`
- `nearby_agent_count`
- `condition_density_level`
- `condition_speed_regime`
- `condition_curvature_level`

This means the current version is an **offline proxy builder** for research and
validation. Later, the same feature interface can be filled by an online
history-window extractor during planner inference.

### Outputs

By default the builder writes a new subdirectory under a split root:

- `interaction_state/interaction_state_index.jsonl`
- `interaction_state/interaction_state_summary.json`

Each record stores:

- normalized interaction-state feature vector
- feature-validity mask
- soft scene gates for the three straight-scene buckets
- flattened axis gates for the nine controllable preference axes

### Main Scripts

Build one split:

```bash
python -m research_v1.execution.interaction.build_dataset \
  --split_root /mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/style_scene_split_straight_val_v2
```

Build train and val together:

```bash
python -m research_v1.execution.interaction.build_train_val
```

Validate an exported interaction-state dataset:

```bash
python -m research_v1.execution.interaction.validate \
  --output_dir /mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/style_scene_split_straight_val_v2/interaction_state
```

### Why This Does Not Collide With Chapter 2

This module models:

- `what interaction constraints are active now`

It does **not** model:

- `what this user prefers globally`

So it is aligned with Chapter 1 execution logic, while the 3090 / SAGE-Drive
line remains the Chapter 2 representation-and-retrieval logic.
