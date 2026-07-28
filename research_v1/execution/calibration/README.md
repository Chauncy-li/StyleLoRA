## Calibration

This module builds the first offline calibration layer for the small paper.

It does not train the planner yet. Instead, it exports calibration targets and
analysis primitives from:

- `style_scene_split_v2`
- `conditioning/`
- train-split projection statistics

### What Gets Built

1. `calibration_index.jsonl`

Each record stores:

- current observed scene-style vector
- current target / safe / effective preference vectors
- direction-alignment targets
- three-style sweep targets (`conservative / normal / aggressive`)
- sweep monotonicity checks
- frontier-related offline features

2. `calibration_summary.json`

Run summary for:

- direction-alignment rates
- sweep monotonicity rates
- observed-to-target / safe / effective distances
- frontier feature averages

3. `validate_calibration.json`

Validation checks for:

- vector shapes
- style sweep structure
- finite values
- rate ranges

### Main Commands

Build calibration for a split:

```bash
python -m research_v1.execution.calibration.build
```

Validate the resulting export:

```bash
python -m research_v1.execution.calibration.validate
```

### Current Meaning

- `direction_alignment_rate_*` measures whether the projected/effective update
  still moves in the same axis direction as the original target preference.
- `*_monotonic_rate` measures whether the three-style sweep preserves the style
  ordering implied by the train-split prototypes.
- `frontier_features` are offline safety/feasibility-related primitives for the
  next-stage response calibration and frontier analysis.
