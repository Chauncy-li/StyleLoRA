# Preference Flow — Steps 1 to 5

The frozen base StylePlanner remains unchanged in these stages. Production
Preference Flow interfaces live in:

```text
baseline/model/style_planner/preference_flow/
```

`research_v1` contains only reproducibility tests and server execution scripts.
The StylePlanner production path does not import this research directory.

## Step 1: safe clean-prediction edit location

```text
DiT clean x0 -> optional clean-prediction editor -> DPM-Solver update
```

`clean_prediction_editor_mode=disabled` preserves the old sampler path with no
x0 callback. `identity` returns the exact same x0 tensor. The real regression
compares the two paths on fixed cache scenes and requires all measured errors to
be exactly zero.

## Step 2: neutral/preference dual DPM streams

```text
x_T -> neutral DPM solver    -> neutral sample
x_T -> preference DPM solver -> preference sample
```

Both streams receive independent clones of the same initial noise and use fresh
solvers. The neutral trace is cached by logical evaluation index. The preference
editor receives a clone of the aligned neutral record. Its `current_state` is a
detached snapshot of the actual DPM `x_q` from that denoiser evaluation.

Dual mode is opt-in regression machinery; ordinary StylePlanner inference
remains single-stream.

## Step 3: standalone preference-coordinate dynamics

Step 3 does not touch DPM, clean predictions, or trajectories. It introduces a
separate low-dimensional mathematical module only:

```text
z(q, r) in R^8
V_phi(z, condition, q, r) -> dz/dr
integrator: start coordinate -> end coordinate
```

`r` is the current internal integration coordinate supplied to the vector
field. `rho` is only the integration endpoint passed to
`integrate_from_neutral`; it is never an argument of
`PreferenceVectorField.forward`.

The executable convention follows the required constant-field contract:

```text
V(z, r) = c  =>  z_end = z_start + (end - start) * c
```

Positive and negative endpoints are handled by the signed interval, without an
assumption of directional symmetry. The two explicit solvers are:

```text
Euler: z_next = z + h * V(z, r)
Heun:  z_predict = z + h * V(z, r)
       z_next = z + h/2 * (V(z, r) + V(z_predict, r + h))
```

The real `PreferenceVectorField` has a zero-initialized final layer. At initial
weights it therefore produces exactly zero velocity and identity integration
for every allowed endpoint.

## Step 4: install the Flow Adapter in the preference DPM stream

Step 4 uses the existing Step-2 preference editor callback; it does not change
the DPM solver or the frozen base checkpoint. At every preference-stream DPM
evaluation, the adapter performs:

```text
aligned neutral clean x0 + preference live x_q -> deterministic ego condition
z(q, 0)=0 --integrate to rho--> z(q, rho)
z(q, rho)-z(q, 0) -> ego future residual -> preference clean x0
```

The condition pools the aligned neutral ego clean prediction and the
preference stream's real-time ego DPM state. Diffusion time is passed directly
to the vector field as `q`. The user endpoint `rho` is passed only to
`integrate_from_neutral`; the field still receives only the evolving internal
coordinate `r`.

The editor writes only `ego[:, 1:, :]` of solver-facing x0. It never writes
the ego current pose or another agent slice, and it does not edit a final
denormalized trajectory. Step-4 keeps the legacy zero-output decoder for its
probe; Step-5 additionally validates the smooth longitudinal decoder, which
is still an exact fresh no-op because its zero-initialized vector field returns
zero latent displacement. The nonzero field used by the Step-4 probe is
test-only and is not a model or checkpoint.

## Files

- `baseline/model/style_planner/preference_flow/config.py`: K=8 dimensions,
  coordinate bounds, and numerical defaults.
- `vector_field.py`: the production field over `(state, condition, q, r)`.
- `integrator.py`: general-interval Euler/Heun integration and the neutral
  endpoint wrapper.
- `clean_prediction_editor.py`, `contracts.py`: Step-1/2 DPM interfaces.
- `adapter.py`, `trajectory_geometry.py`: Step-4 condition bridge plus the
  Step-5 differentiable single-phase adapter, the physical-current geometry
  contract, and the exact neutral tangent returned by the decoder.
- `interaction_attention.py`: opt-in neutral-anchored ego-to-neighbor temporal
  attention with a null-interaction token. It has no `r` or `rho` input.
- `selftest_step1.py`, `selftest_step2.py`, `selftest_step3.py`: focused tests.
- `run_step1_base_regression.py`, `run_step2_dual_regression.py`: real frozen
  planner regressions.
- `run_step3_numerical_validation.py`: synthetic, no-data Step-3 validation.
- `selftest_step4.py`: toy-DPM Step-4 identity, propagation, and ego-write
  boundary tests.
- `run_step4_flow_adapter_regression.py`: real frozen-checkpoint CUDA
  regression for the adapter.
- `build_step5_tiny_cohort.py`: creates a fixed 8--16 sample free-drive /
  car-follow cohort without exporting future labels to inference features.
- `selftest_step5.py`: CPU checks for exact zero identity and first backward
  gradient flow.
- `run_step5_tiny_preference_learning.py`: frozen-base tiny overfit,
  gradient bootstrap, reload check, and rho direction smoke test.
- `differentiable_behavior_alignment.py`: Step-5-S1 neutral-anchored lead
  reference, differentiable THW/TTC bridge, and formal-metric audit adapter.
- `run_step5_semantic_alignment.py`: fresh-flow, five-point pathwise semantic
  alignment proof. It is separate from the failed Step-5-R output.
- `run_step5_interaction_attention.py`: Step-5-S2 same-cohort comparison of
  the legacy hard single-lead reference and unified neutral-anchored soft
  interaction attention.

## Remote validation commands

Run after uploading the patch:

```bash
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

python -m py_compile \
  baseline/model/style_planner/preference_flow/config.py \
  baseline/model/style_planner/preference_flow/vector_field.py \
  baseline/model/style_planner/preference_flow/integrator.py \
  baseline/model/style_planner/preference_flow/adapter.py \
  baseline/model/style_planner/preference_flow/trajectory_geometry.py \
  baseline/model/style_planner/preference_flow/interaction_attention.py \
  baseline/model/style_planner/preference_flow/contracts.py \
  baseline/model/style_planner/preference_flow/clean_prediction_editor.py \
  baseline/model/style_planner/preference_flow/__init__.py \
  research_v1/execution/preference_flow/selftest_step1.py \
  research_v1/execution/preference_flow/selftest_step2.py \
  research_v1/execution/preference_flow/selftest_step3.py \
  research_v1/execution/preference_flow/selftest_step4.py \
  research_v1/execution/preference_flow/selftest_step5.py \
  research_v1/execution/preference_flow/run_step1_base_regression.py \
  research_v1/execution/preference_flow/run_step2_dual_regression.py \
  research_v1/execution/preference_flow/run_step3_numerical_validation.py \
  research_v1/execution/preference_flow/run_step4_flow_adapter_regression.py \
  research_v1/execution/preference_flow/build_step5_tiny_cohort.py \
  research_v1/execution/preference_flow/run_step5_tiny_preference_learning.py \
  research_v1/execution/preference_flow/differentiable_behavior_alignment.py \
  research_v1/execution/preference_flow/run_step5_semantic_alignment.py \
  research_v1/execution/preference_flow/run_step5_interaction_attention.py

python -m research_v1.execution.preference_flow.selftest_step1
python -m research_v1.execution.preference_flow.selftest_step2
python -m research_v1.execution.preference_flow.selftest_step3
python -m research_v1.execution.preference_flow.selftest_step4
python -m research_v1.execution.preference_flow.selftest_step5
```

The already-approved Step-1/2 commands remain unchanged. Step 3 has no
checkpoint or cache requirement:

```bash
python -m research_v1.execution.preference_flow.run_step3_numerical_validation \
  --seed 3407 \
  --device cpu \
  --output-dir "$PF_ROOT/step3_numerics"
```

It writes only measured synthetic numerical results:

```text
step3_preference_flow_numerics.json
```

Step 4 requires the same frozen checkpoint, cache root, five-scene JSON, and
normalization artifact already used for Steps 1--2. Run the actual DPM test on
CUDA (for example physical GPU 1) as follows:

```bash
CUDA_VISIBLE_DEVICES=1 python -m research_v1.execution.preference_flow.run_step4_flow_adapter_regression \
  --base-checkpoint "$BASE_CKPT" \
  --model-args "$MODEL_ARGS" \
  --cache-root "$PLANNER_CACHE" \
  --scene-token-file "$PF_ROOT/cohorts/step1_scenes.json" \
  --normalization-file-path "$REPO/baseline/resources/normalization_train.json" \
  --seed 3407 \
  --device cuda \
  --output-dir "$PF_ROOT/step4_adapter" \
  --overwrite
```

It writes:

```text
step4_adapter_identity_regression.json
step4_adapter_probe_regression.json
```

## Step 5: trainability retrofit and tiny preference proof

Step 5 keeps the base StylePlanner frozen and uses one fixed noisy diffusion
phase per fixed sample, not an 11-step DPM rollout. The frozen planner produces
neutral clean `x0`; the Flow receives the same live `x_q` and only Flow-side
parameters are optimized.

The formal decoder is now:

```text
8-D latent -> smooth progress basis -> neutral-path tangent -> ego future x/y
                                                     -> path-consistent cos/sin
```

It is exactly zero at zero latent but has a nonzero latent-to-residual Jacobian,
so the zero-initialized vector field gets a useful first backward gradient. It
never writes ego current, non-ego, or independent lateral offsets. Future
expert trajectories and V6 future axis targets are used only for cohort/loss
construction, never as Flow inference features. See
[the Step-5 audit](STEP5_TRAINING_AUDIT.md) for channel and normalization
semantics.

Create the fixed 12-sample cohort first, then run CUDA smoke training. The
Step-5-R rerun always uses that complete 12-scene cohort as one batch. Its
ordering loss evaluates the same noisy scene at `rho=-1, 0, +1`; its smoke
report verifies all six free-drive and all six car-follow scenes. Add
`--use-swanlab` to enable SwanLab loss/gradient monitoring.

```bash
python -m research_v1.execution.preference_flow.build_step5_tiny_cohort \
  --conditioning-index "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --cache-root "$PLANNER_CACHE" \
  --output-file "$PF_ROOT/cohorts/step5_tiny_cohort.json" \
  --per-scene 6 \
  --seed 3407

CUDA_VISIBLE_DEVICES=1 python -m research_v1.execution.preference_flow.run_step5_tiny_preference_learning \
  --base-checkpoint "$BASE_CKPT" \
  --model-args "$MODEL_ARGS" \
  --cache-root "$PLANNER_CACHE" \
  --cohort-file "$PF_ROOT/cohorts/step5_tiny_cohort.json" \
  --normalization-file-path "$REPO/baseline/resources/normalization_train.json" \
  --seed 3407 \
  --device cuda \
  --batch-size 12 \
  --steps 300 \
  --output-dir "$PF_ROOT/step5_tiny_rerun" \
  --use-swanlab \
  --swanlab-project preference-flow-step5 \
  --swanlab-run-name tiny-overfit-rerun-gpu1 \
  --overwrite
```

This writes `step5_gradient_bootstrap.json`, `step5_tiny_overfit.json`,
`step5_rho_direction_smoke.json`, and a Flow-only checkpoint. The tiny report
contains separate target-fit/order curves, physical-unit longitudinal residual
mean/P95/max, and frozen-base/reload checks. The smoke report contains every
scene's `M-`, `M0`, `M+`, both margins, and per-task summaries. A failed
acceptance check still leaves the measured JSON reports before raising.

## Step 5-S1: identity-anchored pathwise semantic alignment

Step 5-S1 replaces the old all-neighbor minimum-distance supervision; it does
not add a second model, PCGrad, a task-specific head, or a MoE fallback.
For each scene, neutral `x0` plus observable neighbor history chooses one
same-lane front-vehicle ID and a detached neutral validity mask. Every
`rho = [-1, -0.5, 0, 0.5, 1]` branch shares that reference, the same `x_q`,
the same diffusion phase, and the same neutral prediction. The Flow can alter
how the ego follows the lead, but cannot change which vehicle defines the
interaction merely by changing `rho`.

The optimization loss is the neutral-relative calibrated behavior curve plus
every adjacent pathwise ordering interval. `rho=0` remains an exact adapter
identity. The former future-expert proxy is reported only as a diagnostic, not
used in the total loss. After training, the script substitutes each edited ego
future into a short-lived cache clone and calls the unchanged formal metric
implementation for the rho-sweep direction audit. That formal path is outside
autograd and never becomes a Flow input.

```bash
CUDA_VISIBLE_DEVICES=1 python -m research_v1.execution.preference_flow.run_step5_semantic_alignment \
  --base-checkpoint "$BASE_CKPT" \
  --model-args "$MODEL_ARGS" \
  --cache-root "$PLANNER_CACHE" \
  --cohort-file "$PF_ROOT/cohorts/step5_tiny_cohort.json" \
  --behavior-normalization-path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --normalization-file-path "$REPO/baseline/resources/normalization_train.json" \
  --seed 3407 \
  --device cuda \
  --batch-size 12 \
  --steps 300 \
  --output-dir "$PF_ROOT/step5_semantic_alignment" \
  --use-swanlab \
  --swanlab-project preference-flow-step5 \
  --swanlab-run-name semantic-alignment-gpu1 \
  --overwrite
```

It writes `step5_proxy_sensitivity_audit.json`,
`step5_pathwise_alignment.json`, `step5_all_scene_direction.json`, and
`step5_gradient_alignment_history.json`, plus a fresh Flow-only checkpoint.
The run fails closed unless all 12 scenes pass every adjacent rho interval and
the unchanged formal metrics agree with the differentiable direction.

## Step 5-S2: unified neutral-anchored interaction attention

S2 keeps one frozen base planner, one Vector Field, and one ego-only decoder.
It does not create free-drive/car-follow experts. At the fixed noisy phase, the
neutral clean prediction supplies the ego query and neutral neighbor tokens;
observable current geometry and `q`/log-SNR provide relation bias. This produces
one attention map over neighbor-time tokens plus a null-interaction token:

```text
neutral ego x0 + neutral neighbor x0 + current geometry
    -> A_q^0 (shared unchanged by every rho endpoint)
    -> interaction_context -> V_phi(z, c_q, q, r)
```

`rho` remains only the integrator endpoint; neither the attention nor the
Vector Field receives it as a regular condition. The behavior loss uses the
neutral weights with gradient stopped, so a rho branch cannot evade a difficult
interaction by changing who it attends to; neutral interaction confidence is
also a detached validity gate. Neighbor ground-truth futures are loss-only
data. The path loss distinguishes active calibrated intervals from
clipped/saturated intervals: active intervals need positive movement, while a
saturated interval may plateau but cannot reverse.

The smooth decoder now accepts raw physical ego current geometry in this
training bridge and returns the exact neutral tangent it used. Training, lateral
validation, and physical residual reporting all consume that same tensor.
This stage keeps the existing speed/headway/TTC bridge as the semantic proof;
the later full-DPM stage is where the same unified condition is extended to the
complete six-axis offline audit.

```bash
CUDA_VISIBLE_DEVICES=1 python -m research_v1.execution.preference_flow.run_step5_interaction_attention \
  --base-checkpoint "$BASE_CKPT" \
  --model-args "$MODEL_ARGS" \
  --cache-root "$PLANNER_CACHE" \
  --cohort-file "$PF_ROOT/cohorts/step5_tiny_cohort.json" \
  --behavior-normalization-path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --normalization-file-path "$REPO/baseline/resources/normalization_train.json" \
  --seed 3407 \
  --device cuda \
  --batch-size 12 \
  --steps 300 \
  --output-dir "$PF_ROOT/step5_interaction_attention" \
  --use-swanlab \
  --swanlab-project preference-flow-step5 \
  --swanlab-run-name interaction-attention-gpu1 \
  --overwrite
```

It writes the hard-reference report, soft-attention report, direct comparison,
attention diagnostics, and one fresh Flow+attention checkpoint. The run fails
closed on direction reversal, missing 6/6 car-follow interaction coverage,
attention-mask/normalization violations, non-identity `rho=0`, base mutation,
reload mismatch, or lateral residual above `1e-5` m.

## Step 5-S3A-R: normalized neutral/expert longitudinal pairing audit

S3A-R is an audit-only correction to the first S3A run. Full-DPM inference
must receive `observation_normalizer(raw_inputs)` exactly once; its returned
`prediction` has already been state-inversed by the decoder and is never
inversed again by the audit. Raw cache current/future geometry remains physical
for all sanity, projection, and feasibility measurements.

The fixed 12-scene cohort is evaluated with two independent frozen neutral
sources: the existing teacher-forced fixed-q clean prediction used by Step-5
training, and a correctly normalized full-DPM rollout from observed inputs.
The future expert constructs fixed-q `xq` only; it is never a full-DPM input or
a Vector Field condition. Before projection, each source reports physical
trajectory sanity, direct ADE/FDE, input summaries, and discontinuities.

The longitudinal target remains `s0 + lambda * (s_expert_projected - s0)` for
coherent labels. `rho_star` stays continuous; active-axis-count groups make
single-axis scalar labels distinct from actual two/three-axis consistency
evidence. Projection keeps the 5 m guard and reports local (`<=0.05 m`) versus
semantic backtracking without correcting either one.

```bash
CUDA_VISIBLE_DEVICES=1 python -m research_v1.execution.preference_flow.run_step5_s3a_transport_audit \
  --base-checkpoint "$BASE_CKPT" \
  --model-args "$MODEL_ARGS" \
  --cache-root "$PLANNER_CACHE" \
  --conditioning-index "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --cohort-file "$PF_ROOT/cohorts/step5_tiny_cohort.json" \
  --normalization-file-path "$REPO/baseline/resources/normalization_train.json" \
  --output-dir "$PF_ROOT/step5_longitudinal_transport_audit_rerun" \
  --max-axis-spread 0.25 \
  --max-projection-distance-m 5.0 \
  --seed 3407 \
  --device cuda
```

It writes `step5_s3ar_axis_coherence.json`,
`step5_s3ar_neutral_sanity.json`, `step5_s3ar_fixed_q_projection.json`,
`step5_s3ar_full_dpm_projection.json`, and `step5_s3ar_comparison.json`.
An empty valid-contract set reports finite aggregation as `null`, never as a
vacuous pass. This stage intentionally does not use SwanLab because no
parameters or losses are updated.

## Still deliberately absent

Step 5 is not full-scale training. It still has no six-axis simultaneous
optimization, full DPM rollout training, feasibility/CBF/QP, content-curve or
route continuation, A3.8/B3, other-planner edits, or final-trajectory
post-processing.
