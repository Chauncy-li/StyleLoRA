## Projection

This module implements the first non-learned version of the preference
realizability projection:

\[
p_t^{safe} = \Pi_{\Omega(z_t)}(p_{target})
\]

### Design Choice

The current version is intentionally **non-trainable**.

Instead of learning a projector, it estimates realizable preference envelopes
from the already-built `style_scene_split_v2` outputs:

- scene bucket
- density level
- speed regime
- curvature level
- scene-specific continuous preference axes

This keeps the Chapter 1 execution logic clearly separated from the 3090
Chapter 2 representation/retrieval logic.

### What Gets Built

1. `projection_stats.json`

Built from the train split. It stores:

- scene-style preference prototypes
- hierarchical quantile envelopes
  - `scene+density+speed+curvature`
  - `scene+density+speed`
  - `scene+density`
  - `scene`

2. `projection_index.jsonl`

Built for a target split using the stats above. Each record stores:

- target preference vector
- lower / upper realizable bounds
- projected preference vector
- clipping indicators and projection magnitude

3. `projection_summary.json`

Run summary for coverage and clipping statistics.

### Main Commands

Build stats from the train split:

```bash
python -m research.preference_execution.projection.build_projection_stats
```

Apply those stats to the val split:

```bash
python -m research.preference_execution.projection.apply_preference_projection
```

Validate the resulting projection dataset:

```bash
python -m research.preference_execution.projection.validate_projection_dataset
```

### Current Offline Approximation

In this first version, `b(z_t)` is approximated by the existing offline
condition labels:

- `scene_bucket`
- `condition_density_level`
- `condition_speed_regime`
- `condition_curvature_level`

Later we can swap this bucket assignment with a direct `z_t`-conditioned lookup
without changing the file interface.
