# Continuous Style V5: retained data calibration

V5 is the active data-calibration layer used before V6 conditioning. It does
not infer a single empirical style latent and does not create discrete
conservative/normal/aggressive targets.

The retained pipeline has three train stages and two frozen-statistics
application stages.

## 1. Candidate measurement

`build_continuous_style_v5_candidates.py` reads
`style_scene_split_v2/split_index.jsonl` only as a scene and quality index.
It recomputes behavior metrics from the original planner-cache `.npz` files.
Legacy fields such as `style_label`, `subset_id`, `split_valid`, and
`memory_eligible` are not used as preference supervision.

The current controlled scenes are:

- `straight_car_follow`
- `straight_free_drive`

Lane-change records are retained later through the V6 base-index merge, but
they do not receive active style supervision.

## 2. Train-only robust normalization

`fit_continuous_style_v5_normalization.py` fits q05/q95 independently for each
scene and axis on train candidates only. Falling raw metrics are flipped so
every canonical axis is increasing in aggressiveness.

The output includes:

- `v5_normalized.jsonl`
- `v5_normalization.json`

Validation/test use `apply_continuous_style_v5_normalization.py` with that
frozen train model.

## 3. Conditional percentile calibration

`fit_continuous_style_v5_conditional_rank.py` estimates

```text
u_ij = P(z_j < z_ij | c_i) + 0.5 P(z_j = z_ij | c_i)
```

using a weighted local empirical CDF. Training labels are five-fold OOF:
the query sample is never ranked against the fold used to fit its own
reference distribution.

Default support rules:

- 64 retained nearest valid references;
- effective-neighbor threshold 32;
- at least three shared observed causal context features;
- missing conditions are masked, not zero-imputed.

The output includes:

- `v5_conditional_rank.jsonl`
- `v5_conditional_rank_model.json`
- frozen train reference arrays used by validation/test.

Validation/test use `apply_continuous_style_v5_conditional_rank.py`; they do
not fit a separate conditional distribution.

## Train commands

```bash
python -m research.continuous_style.build_continuous_style_v5_candidates \
  --index-path "${TRAIN_SPLIT}/split_index.jsonl" \
  --planner-cache-dir "${PLANNER_CACHE}" \
  --output-dir "${TRAIN_OUT}" \
  --scene-filter controlled \
  --sample-fraction 1.0 \
  --log-interval 5000

python -m research.continuous_style.fit_continuous_style_v5_normalization \
  --candidate-index-path "${TRAIN_OUT}/v5_candidates.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --scene-filter controlled

python -m research.continuous_style.fit_continuous_style_v5_conditional_rank \
  --normalized-index-path "${TRAIN_OUT}/v5_normalized.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --scene-filter controlled \
  --folds 5 \
  --neighbours 64 \
  --min-effective-neighbours 32
```

## Validation commands

First build validation candidates with the same metric code, then:

```bash
python -m research.continuous_style.apply_continuous_style_v5_normalization \
  --candidate-index-path "${VAL_OUT}/v5_candidates.jsonl" \
  --normalization-model-path "${TRAIN_OUT}/v5_normalization.json" \
  --output-dir "${VAL_OUT}" \
  --scene-filter controlled

python -m research.continuous_style.apply_continuous_style_v5_conditional_rank \
  --normalized-index-path "${VAL_OUT}/v5_normalized.jsonl" \
  --conditional-rank-model-path "${TRAIN_OUT}/v5_conditional_rank_model.json" \
  --output-dir "${VAL_OUT}" \
  --scene-filter controlled
```

The V6 builder consumes `v5_conditional_rank.jsonl` directly. It does not
consume empirical-rho, scene-constrained-target, or C/N/A-anchor artifacts.
