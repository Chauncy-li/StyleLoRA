# Preference-Conditioned Diffusion

This folder contains the first **trainable** integration path for the small
paper:

`Causal-Interaction-State-Conditioned Controllable Preference Diffusion Planning`

## What It Does

It consumes the already prepared research-side exports:

- `style_scene_split_v2`
- `interaction_state`
- `projection`
- `conditioning`

and trains the baseline diffusion planner with an explicit 9D preference
condition.

The default condition is:

- `effective_preference_global_vec`

which corresponds to:

- `p_target -> p_safe -> p_eff`

after local interaction-state gating.

## Design Choice

This first training path is intentionally lightweight:

- it reuses the existing baseline diffusion loss
- it injects only an external preference value condition
- it keeps the backbone unchanged except for the thin decoder condition path

## Main Entry

```bash
python -m research.preference_execution.diffusion.train_preference_conditioned_diffusion
```

## Monitoring

This entry reuses the baseline online logger wrapper.

By default it enables:

- `swanlab`
- local TensorBoard logs under `tb/`

You can also switch to:

- `wandb`
- `disabled`

## Important Note

For this first version, `use_data_augment` is disabled by default. The reason
is that the conditioning vectors are derived from the original split sample,
and aggressive state perturbation would otherwise create a mismatch between the
augmented state and the offline preference condition.
