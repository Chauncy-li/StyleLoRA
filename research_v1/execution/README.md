## Preference Execution

This package is reserved for the small paper:

`Causal-Interaction-State-Conditioned Controllable Preference Diffusion Planning`

It will sit on top of the current `research_v1/data` and
`research_v1/style_scene_split` foundation without changing the baseline planner
entrypoints more than necessary.

### Planned Submodules

- `interaction_state/`
  Extract causal, observable interaction-state features for training and
  inference.
- `projection/`
  Project explicit target preference into a realizable preference under the
  current interaction state.
- `conditioning/`
  Inject projected preference into diffusion planning with soft local gating.
- `calibration/`
  Build offline targets and summaries for direction alignment, monotonicity,
  and frontier-related analysis.
- `diffusion/`
  Train the baseline diffusion planner with exported preference conditions.
- `eval/`
  Report controllability metrics on the straight-scene subsets.

Current status:

- `interaction_state/` is implemented.
- `projection/` is implemented.
- `conditioning/` is implemented as a research-side data export layer.
- `calibration/` is implemented as an offline calibration export layer.
- `diffusion/` is implemented as the first trainable integration path.
- `eval/` remains for post-training controllability reporting.
