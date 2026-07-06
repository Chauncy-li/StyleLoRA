## Research Workspace

`research/` is the small-paper workspace that stays separate from `baseline/`.
The goal is to keep the baseline training and planning code stable while
placing paper-specific data preparation, controllable preference execution, and
evaluation code here.

### Current Foundation

- `research/_runtime.py`
  Shared runtime defaults for the server environment. The existing
  `/home/lisw/...` and `/mnt/mydata/...` paths are preserved here.
- `research/data/`
  Offline split and cache utilities. These scripts organize raw-db splits,
  train/val cache construction, and cache manifests without changing the
  baseline preprocessing outputs.
- `research/style_scene_split/`
  Offline straight-scene split foundation. This is the current data base for the
  small paper: it builds the three straight interaction subsets, preserves the
  existing sidecar/index/subset-list formats, and defines scene-specific
  behavior axes for later controllability analysis.

### Planned Method Modules

The small paper will extend the foundation above through
`research/preference_execution/`:

- `interaction_state/`
  Causal interaction-state feature construction from currently observable
  traffic context.
- `projection/`
  Preference realizability projection from `p_target` to `p_safe`.
- `conditioning/`
  Preference-conditioned diffusion wrappers and soft gating.
- `calibration/`
  Preference-response calibration and safety-frontier constraints.
- `eval/`
  Offline and closed-loop evaluation utilities for controllability,
  monotonicity, and safety.

### Scope Reminder

The scene split is an offline research foundation, not an online hard scene
classifier. For this small-paper version:

- training/evaluation subsets are organized offline with `style_scene_split`
- online execution should depend on causal interaction state, not hard scene IDs

