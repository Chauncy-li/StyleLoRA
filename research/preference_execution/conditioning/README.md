## Conditioning

This module exports the first research-side conditioning interface for the
small paper:

\[
p_{target} \rightarrow p_{safe} \rightarrow p_{eff}
\]

### Role

It does **not** modify the baseline planner yet.

Instead, it merges the already-built:

- `interaction_state/`
- `projection/`

into one conditioning dataset that a later planner wrapper can consume
directly.

### What Gets Built

1. `conditioning_index.jsonl`

Each record stores:

- `feature_values` / `feature_mask`
- `scene_gate_values`
- `axis_gate_values`
- `target_preference_scene_vec`
- `safe_preference_scene_vec`
- `effective_preference_scene_vec`
- global 9D preference vectors aligned with the fixed axis order

2. `conditioning_summary.json`

Run summary for scene coverage, gate usage, and attenuation statistics.

3. `validate_conditioning.json`

Validation checks for:

- vector shapes
- finite values
- gate ranges
- `p_eff = axis_gate * p_safe`
- local/global consistency

### Main Commands

Build conditioning for a split after `interaction_state` and `projection`
already exist:

```bash
python -m research.preference_execution.conditioning.build_conditioning_dataset
```

Validate the resulting export:

```bash
python -m research.preference_execution.conditioning.validate_conditioning_dataset
```

### Current Semantics

- `p_target`: scene-conditioned target preference prototype
- `p_safe`: projected realizable preference under current offline bucket
- `axis_gate`: soft local activation derived from the causal interaction-state proxy
- `p_eff`: the current effective preference after local gating
