# Continuous Style V6: direct axes with a scalar rho interface

V6 replaces the V5 training assumption that a single empirical latent rho
must explain all three axes.  Its training condition is the observed,
condition-calibrated three-axis vector:

`p_theta(tau | observation, u_label, m_train, b_causal, g_causal)`.

At inference, a human supplies one scalar `rho in [-1, 1]` and V6 uses the
transparent first-version command curve:

`p_des_j(rho) = clip(0.5 + 0.25 * rho, 0, 1)`.

The resulting command is passed through hard causal masks and explicit bounds
only:

`p_exec = F(p_des, m_causal, bounds)`.

No V6 stage reads `rho`, `rho_oof`, `rho_kappa`, or
`rho_style_train_valid` from the retired V5 latent-rho experiment. Normal-anchor
CFG uses the semantic `rho=0` condition and does not require a discrete C/N/A
anchor subset.

## Required masks

- `m_label`: an offline future metric/rank label can be measured.
- `m_causal`: a car-follow/free-drive axis is applicable online.
- `m_train = m_label & m_causal & router_training_agreement`.

The data export writes an all-zero `style_value_condition` when `m_train` is
empty, so the sample still trains the base diffusion model but does not train a
style condition.  Conversely, `normal_anchor` at rho=0 has nonzero target,
mask, scene, and gate entries whenever a causal axis is active; it is not CFG
unconditional zero.

## Two-gate and lane-change policy

V6 does not train or expose an online three-class driving-task router.  It
keeps only two continuous longitudinal applicability gates:

`[g_free(o), g_car_follow(o), 0]`.

The sum may be below one under lateral or ambiguous interaction, which softly
reduces style applicability without declaring a lane-change class.  Offline
lane-change samples remain in the diffusion dataset but receive an all-zero
style condition.  Route/map inputs and the original diffusion backbone remain
responsible for lateral planning.  Offline `m_gap`, future reached-lane fields,
and route-intent labels are not style-router inputs.

## Artifacts

Starting from `v5_conditional_rank.jsonl`, run:

1. `build_continuous_style_v6_conditions --base-index-path <split_index.jsonl>` ->
   `v6_direct_axis_conditions.jsonl`, `v6_style_command_spec.json`.
2. `validate_continuous_style_v6_conditions` -> interface/mask contract.
3. `audit_continuous_style_v6_router` -> offline-vs-two-gate agreement audit.
4. `audit_continuous_style_v6_support` -> rho endpoint support audit.

The V6 `style_value_condition` layout is fixed at 12 dimensions:

`[axis_target(3), axis_mask(3), causal_scene_one_hot(3), scene_gate(3)]`.

The base index is the authoritative diffusion-sample list.  A matching,
quality-qualified free-drive/car-follow rank row enables style supervision;
all unmatched rows stay in the dataset with an all-zero style condition.
This prevents metric filtering from silently deleting ordinary planning data.

Train the existing preference-conditioned diffusion entrypoint with
`--condition_field style_value_condition --style_condition_feature_set global_only
--base_style_condition_dim 12` and direct the train/validation index overrides
to the V6 sidecars.

For closed-loop runtime, set `runtime_style_mode=continuous_v6` and use a V6
checkpoint whose saved args also specify `style_value_dim=12`.  Calling
`StylePlanner.set_runtime_preference(rho=...)` changes the scalar command.
The legacy lane-context setter remains API-compatible but is not consumed by
the two-gate style router.

## Post-training sweep evaluation

`evaluate_continuous_style_v6_rho_sweep` consumes a model-agnostic JSONL.  A
row must contain:

```json
{
  "sample_id": "...",
  "rho_requested": 0.4,
  "seed": 3,
  "generated_axis_percentile_vec": [0.58, 0.61, 0.55],
  "generated_axis_valid_mask": [true, true, true]
}
```

The generated axis values must be measured from generated trajectories with
the frozen train-only V5 conditional-percentile reference.  The evaluator
reports per-axis target MAE, projected intensity MAE/calibration, paired rho
order accuracy, vector direction cosine relative to rho=0, and same-command
multi-seed variance.

For trained checkpoints, first generate that JSONL with
`research.preference_execution.eval.evaluate_styleplanner_v6_checkpoints`.
It uses the V6 validation sidecar and planner cache, holds diffusion noise
fixed across rho values, compares semantic rho=0 with the empty condition,
and audits offline lane-change rows under an empty style condition.  It can
also compare router-only and normal-anchor-CFG inference.

NuPlan DB closed-loop evaluation remains a separate stage.  Use
`research.preference_execution.eval.run_styleplanner_v6_closed_loop_suite` to
preflight DB/maps/train artifacts and launch a checkpoint/rho grid.  The
launcher is dry-run by default and needs `--execute` to start simulations.
