# Style Scene Split

## Role In The Small Paper

This folder is the **offline data foundation** for the small paper:

`Causal-Interaction-State-Conditioned Controllable Preference Diffusion Planning`

Its job is not to provide an online hard scene classifier. Its job is to:

- build the three straight-scene research subsets
- preserve the existing sidecar / index / subset-list formats
- define scene-specific behavior axes for later controllability analysis
- support train / val / test-simu evaluation bucketing

## What Must Stay Unchanged

The current scripts under this folder keep the existing output organization:

- `style_cache/`
- `sidecar/`
- `subset_lists/`
- `split_index.jsonl`
- `style_cache_list.json`
- `valid_style_cache_list.json`
- `reports/`

For the train/val split helper we additionally materialize:

- `scene_lists/`
- `memory_scene_lists/`
- `scene_list_summary.json`
- `partition_summary.json`

These additions do not change the original scene-split file format. They only
make the train/val subsets easier to consume downstream.

## Current Mainline

`v2` is the primary path for current experiments.

The active scene buckets are:

- `straight_free_drive`
- `straight_car_follow`
- `straight_lane_change`

The active scene-specific behavior axes are defined in `schema.py`:

- `straight_free_drive`
  `speed_preference`, `longitudinal_intensity`, `smoothness`
- `straight_car_follow`
  `headway_margin`, `response_decisiveness`, `response_smoothness`
- `straight_lane_change`
  `gap_acceptance`, `lateral_commitment`, `execution_smoothness`

These axes are important because the small paper needs a controllable,
physically-meaningful preference space rather than a generic style token.

## Recommended Entry Points

### 1. Build Straight Scene Splits From Existing Train/Val Cache

This is the current default when using the processed cache already stored under
`Nuplan-Baseline-Record`:

```bash
python -m research_v1.scene_data.build_train_val
```

Default inputs:

- `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/boston_cache_train_val`
- `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/boston_cache_train_val_list.json`
- `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/boston_cache_train_val_manifest.json`

Default outputs:

- `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/style_scene_split_straight_train_v2`
- `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/style_scene_split_straight_val_v2`

### 2. Build V2 Directly From Planner Cache Or Raw NuPlan Scenarios

```bash
python -m research_v1.scene_data.builder
```

This entry keeps both modes:

- `planner_cache`
- `raw_scenario`

### 3. Validate And Analyze V2 Outputs

```bash
python -m research_v1.scene_data.validate
python -m research_v1.scene_data.analyze
```

## Legacy Compatibility

Legacy `v1` modules are retained under `scene_data/legacy` so older outputs can
be read or reproduced without breaking prior experiments:

- `legacy.builder`
- `legacy.validate`
- `legacy.analyze`

For new experiments in this repository, use `v2` by default.

## Relation To The Small Paper

This folder supports the paper in three ways:

1. It defines the straight-scene evaluation scope, so the paper does not
   overclaim full-scene coverage.
2. It defines the scene-specific behavior axes used later for controllability
   calibration and preference-response analysis.
3. It provides offline subset construction for train/val/test-simu without
   forcing the online planner to rely on hard scene classification.
