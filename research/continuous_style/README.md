# Continuous Preference Data and Runtime Interface

This package contains the active data and runtime path for continuous,
context-dependent preference control in StylePlanner.

The implementation is intentionally split into two roles:

- **V5 data calibration** measures interpretable behavior axes, fits train-only
  robust normalization, and converts each valid axis to a conditional
  percentile under similar causal context.
- **V6 conditioning** trains directly on the calibrated three-axis vector and
  exposes a scalar expert command `rho` only at inference time.

The retired prototype/C/N/A pipeline and the empirical single-latent-rho
pipeline are not part of the current method.

## Current data flow

Train:

```text
style_scene_split_straight_train_v2/split_index.jsonl
  -> v5_candidates.jsonl
  -> v5_normalized.jsonl + frozen normalization model
  -> v5_conditional_rank.jsonl + frozen train conditional-CDF references
  -> v6_direct_axis_conditions.jsonl
  -> V6 validation/router/support audits
```

Validation:

```text
style_scene_split_straight_val_v2/split_index.jsonl
  -> v5_candidates.jsonl
  -> apply frozen train normalization
  -> apply frozen train conditional-CDF references
  -> v6_direct_axis_conditions.jsonl
  -> V6 validation/router/support audits
```

Validation/test must never fit their own normalization or conditional-rank
references.

## Behavior axes

All calibrated axes use the same orientation: a larger value means a more
aggressive preference realization.

| Offline scene | Raw metrics | Canonical axes |
|---|---|---|
| `straight_car_follow` | `h`, `ttc_margin`, `r_closing` | headway tightness, TTC tightness, closing-response tolerance |
| `straight_free_drive` | `r_v`, `r_a`, `r_response` | speed utilization, acceleration willingness, speed-response intensity |
| `straight_lane_change` | `r_init`, `r_commit`, `m_gap` | retained for offline metric compatibility only |

The active V6 style policy controls car-follow and free-drive. Lane-change
samples remain in the diffusion training set with an all-zero style condition,
so route/map inputs and the original planner remain responsible for lateral
planning.

## Calibration

For raw metric `z_j`, train-only q05/q95 statistics produce an
aggressive-aligned robust value. Rising axes use

```text
clip((z_j - q05_j) / (q95_j - q05_j), 0, 1)
```

and falling axes use one minus that value.

The conditional-percentile stage then estimates

```text
u_ij = F_train,b,j(z_ij | c_i)
```

with weighted k-nearest-neighbor empirical CDFs. Training uses five-fold OOF
references. Validation/test use the frozen all-train reference model.
Unobserved metrics and insufficiently supported contexts remain masked; they
are never filled with a neutral value.

## V6 condition

The fixed 12-dimensional layout is

```text
[axis_target(3), causal_axis_mask(3), causal_scene_one_hot(3), scene_gate(3)]
```

Training uses the observed conditional-percentile target. At inference, an
expert supplies `rho in [-1, 1]`, mapped in the first implementation to

```text
p_des,j(rho) = clip(0.5 + 0.25 * rho, 0, 1).
```

The all-zero vector is reserved for classifier-free/unconditioned planning.
The semantic normal anchor at `rho=0` is a nonzero condition whenever a causal
car-follow or free-drive axis is active.

## Active entry points

V5 train stages:

```bash
python -m research.continuous_style.build_continuous_style_v5_candidates --help
python -m research.continuous_style.fit_continuous_style_v5_normalization --help
python -m research.continuous_style.fit_continuous_style_v5_conditional_rank --help
```

V5 frozen-statistics application:

```bash
python -m research.continuous_style.apply_continuous_style_v5_normalization --help
python -m research.continuous_style.apply_continuous_style_v5_conditional_rank --help
```

V6 export and audits:

```bash
python -m research.continuous_style.build_continuous_style_v6_conditions --help
python -m research.continuous_style.validate_continuous_style_v6_conditions --help
python -m research.continuous_style.audit_continuous_style_v6_router --help
python -m research.continuous_style.audit_continuous_style_v6_support --help
python -m research.continuous_style.selftest_continuous_style_v6
```

Post-training rho sweep:

```bash
python -m research.continuous_style.evaluate_continuous_style_v6_rho_sweep --help
```

## Modules that must remain aligned

- `metrics.py`: hard, interpretable behavior measurements.
- `soft_metrics.py`: differentiable surrogates for future preference-energy guidance.
- `router.py`: causal car-follow/free-drive applicability signals.
- `runtime.py`: online scalar-rho to V6 condition adapter.
- `v5.py`: retained data measurement and calibration implementation.
- `v6.py`: direct-axis condition, command, validation, and evaluation logic.

See `README_V5.md` for the retained calibration stages and `README_V6.md` for
the model-facing condition contract.
