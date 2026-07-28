# V6 Preference-Conditioned StylePlanner

This package trains the StylePlanner branch with the frozen V6 direct-axis
sidecars. The original `diff_planner`, Wayformer, scene encoder, DiT blocks,
and trajectory output head are unchanged.

## Condition

The model receives the fixed 12-dimensional condition:

```text
[axis_target(3), axis_mask(3), scene_one_hot(3), scene_gate(3)]
```

Lane-change/base samples carry the all-zero condition. A semantic normal
condition is separate:

```text
[0.5, 0.5, 0.5, active_mask, scene_one_hot, scene_gate]
```

## Model-side extensions

### Signed Preference Axis Router (V2)

`axis_router_v2_signed` replaces only the StylePlanner condition adapter. For
axis `j`, applicability is computed from observation context, axis identity,
the causal hard mask, and the selected scene gate; it never reads preference
magnitude. The command enters only through the signed coordinate
`d_j = 2 * (p_j - 0.5)`:

```text
c_style = W sum_j a_j(o) d_j v_j(o) / sqrt(number_of_active_axes)
```

The final map `W` is bias-free and zero-initialized, and there is no nonlinear
normalization after the signed mixture. Consequently, semantic normal and
empty conditions give an exact zero residual, while reversing the command
reverses the residual direction. `axis_router_v1` remains available only for
reproducing the first experiment.

### Normal-referenced Conditional Quantile Transport (NCQT)

The frozen train-only conditional CDF specifies a local quantile displacement,
not an absolute destination:

```text
delta_z_target(p) = F^-1(p | context) - F^-1(0.5 | context)
delta_z_model(p)  = z(tau_p) - stopgrad(z(tau_normal))
```

Both trajectories reuse the same observation, noisy state, diffusion time, and
frozen encoder context. NCQT is therefore exactly compatible with structural
base preservation even when the pretrained normal trajectory is not located at
the empirical median. Generated coordinates remain unclamped, avoiding the
saturated sigmoid-CDF gradient of the first version.

### Same-noise signed response consistency

Synthetic `0.5-delta` and `0.5+delta` commands share the same noisy state and
context. A monotonic hinge enforces positive physical-axis ordering, while a
normal-referenced oddness term aligns the positive response with the negative
of the conservative response. This supplies counterfactual direction and
symmetry supervision without requiring paired expert trajectories.

Stage A freezes the encoder, DiT blocks, and output head, keeps them in eval
mode, and trains only the signed router. Every batch is formed from explicit
controlled and preservation streams (default 70/30). Metrics are aggregated by
active sample/pair counts, and checkpoint selection uses NCQT error plus
pairwise order error instead of total diffusion loss.

### Ego-isolated counterfactual transport (Stage A2)

Stage A2 keeps the original Stage-A implementation behind explicit config
switches and adds a stricter personalized-planning path:

```text
x0 = frozen_joint_planner(x_t, observation)
     + ego_mask * R_ego(c_style)
```

`R_ego` is bias-free, changes future ego states only, and cannot directly
rewrite predicted neighbors. For car-follow axes, every command is measured
against the detached `rho=0` neighbor forecast:

```text
delta_z(p) = z(tau_ego(p), stopgrad(tau_neighbor(0)))
             - z(tau_ego(0), stopgrad(tau_neighbor(0)))
```

The same fixed-neighbor contract is used by plus/minus pair supervision and
the StylePlanner rho-sweep evaluator. A small exogenous-neighbor invariance
term audits direct denoiser leakage, while rollout evaluation separately
reports any indirect neighbor response. Checkpoint selection uses the worse
of free-drive and car-follow signed-control scores. The evaluator also exports
unclamped canonical-axis response and percentile saturation rates.

### Normal-Anchor CFG (Stage B)

For a conservative/aggressive target condition `c_rho`, sampling uses:

```text
epsilon_NA = epsilon(c_0) + s * (epsilon(c_rho) - epsilon(c_0))
```

where `c_0` is the semantic `rho=0` condition. The all-zero condition remains
the base/unconditioned branch used by classifier-free dropout and by samples
without applicable style axes.

It is intentionally disabled in Stage A. Because the backbone is frozen and
the V2 router residual is structurally zero at `rho=0`, base capability
preservation does not depend on a student-to-student anchor loss. CFG should be
reintroduced only after signed controllability passes the smoke criteria,
starting at scale 1.0 and then testing 1.1/1.2.

Stage-B checkpoint evaluation reuses the frozen Stage-A checkpoint without
training or changing its state dict. In
`research_v1.execution.evaluation.evaluate_checkpoints`, the `router_only` variant explicitly
disables Normal-Anchor CFG and the `anchor_cfg` variant explicitly enables it
at runtime. This override is required because Stage-A `args.json` correctly
saves `normal_anchor_cfg_enabled=false`; changing only
`--cfg-guidance-scale` would otherwise use the all-zero reference and would not
be Stage B.

Run `router_only,anchor_cfg` together with the same samples and noise. Each
generated row records the requested CFG module/scale, whether the semantic
normal or empty reference was actually used, and its trajectory distance to
the paired router-only rollout. The evaluator writes
`v6_stage_b_contract.json` with these hard contracts:

```text
scale=1.0: anchor_cfg trajectory == router_only trajectory
any scale: rho=0 anchor_cfg trajectory == router_only trajectory
anchor_cfg: semantic normal reference used on every controlled rollout
anchor_cfg: empty CFG reference never used
```

Only after scale 1.0 passes exactly should scales 1.1 and 1.2 be compared for
raw-axis strength, calibration, jerk, collision, and preservation.

After the unit-scale contract has passed, both effect scales can be evaluated
in one paired invocation:

```text
--variants router_only,anchor_cfg
--cfg-guidance-scales 1.1,1.2
```

The evaluator runs router-only once, then writes separate
`anchor_cfg_s1p1` and `anchor_cfg_s1p2` branches using the same selected
samples, seed replicas, rho commands, and diffusion noise. The historical
single-value `--cfg-guidance-scale` interface remains unchanged and is
mutually exclusive with the new list argument.

Every checkpoint directory additionally receives:

```text
v6_stage_b_contracts.json
v6_stage_b_scale_diagnostics.json
v6_stage_b_scale_axis_transitions.csv
```

The scale diagnostic performs exact row matching and reports two transitions:
router-only (the validated scale-1 equivalent) to 1.1, and 1.1 to 1.2. For
each physical axis it records the signed response increment, amplified and
regressed fractions, newly wrong-direction flips, and trajectory-proxy deltas.
This separates failure of the first amplification step from saturation or
regression introduced only by the second scale.

Closed-loop DB evaluation must not treat a random NuPlan scenario as a valid
style-control sample. NuPlan scenario types are only a coarse candidate filter;
the final applicability decision is the same causal runtime Router used by the
planner. Build the closed-loop cohort in two stages:

1. Run exactly one frozen `router_only`, `rho=0` discovery job. This keeps
   selection independent of CFG and rho treatment.
2. In the initial Router audit window, require enough controlled active steps
   and a stable dominant `straight_free_drive` or `straight_car_follow` gate.
3. Export a balanced exact `scenario_tokens` JSON cohort.
4. Reuse that exact token list for every rho/variant job and enable the hard
   Router-eligibility audit. The audit records per-scenario active-step ratio,
   dominant-scene purity, Router confidence, token/log identity, and both
   controlled-scene counts.

The closed-loop launcher supports this protocol through
`--export-router-eligible-tokens-json` in discovery mode and
`--scenario-tokens-json --require-router-eligible-scenarios` in formal mode.
It also overrides a debug scenario-filter limit to the exact token count, so a
fixed cohort cannot be silently truncated by `boston.yaml`.

Each closed-loop job writes
`v6_closed_loop_metrics.json` and `v6_closed_loop_scenario_metrics.csv` so
activation, safety, progress, and style effects can be inspected separately.

## Stage-A3.3 rollout-consistent pair ablation

```text
v6_signed_router_ncqt_stage_a3_3_rollout
```

Stage A3.3 preserves the A3.2 Router, ego kinematic residual, NCQT targets,
loss weights, frozen backbone, and inference path. It changes only the existing
minus/normal/plus training branch. The three commands share recovered forward
noise at `t=0.10`, traverse two stop-gradient first-order DPM-Solver++ state
updates on the inference log-SNR schedule, and receive differentiable signed
response supervision at `t=0.01`. This targets the remaining discrepancy
between correct one-pass low-noise responses and incorrect complete-rollout
free-drive acceleration responses without backpropagating through the full
sampler.

The completed smoke did not correct acceleration willingness and slightly
weakened several other axes, so A3.3 is retained as a negative ablation rather
than the base for later stages.

Both earlier ablations remain directly selectable:

```text
v6_signed_router_ncqt_stage_a3_2_terminal
v6_signed_router_ncqt_stage_a3_kinematic
```

## Stage-A3.4 global-rho pair smoke preset

```text
v6_signed_router_ncqt_stage_a3_4_global_pair
```

Stage A3.4 uses Stage A3.2 as its complete model and training baseline. It
keeps the signed Router, kinematic ego residual, fixed-normal-neighbor NCQT,
low-noise pair interval, all loss weights, frozen backbone, data interface,
and inference path unchanged. The only change is the command manifold used by
the existing paired monotonic/symmetry branch:

```text
p_minus = 0.5 - delta * m_causal
p_normal = 0.5
p_plus = 0.5 + delta * m_causal
```

Thus every active axis receives its own response constraint while all active
targets move together exactly as in the runtime scalar-rho sweep. This tests
whether the remaining free-drive acceleration reversal comes from cross-axis
interference hidden by axis-isolated training. A3.2 and A3.3 remain selectable
unchanged, so rollback requires only choosing their previous preset/checkpoint.

## Stage-A3.5 normal-opportunity acceleration preset

```text
v6_signed_router_ncqt_stage_a3_5_normal_opportunity
```

Stage A3.5 retains the complete A3.4 model, global-rho paired command,
low-noise interval, fixed-normal-neighbor NCQT, loss weights, frozen planner,
data interface, and inference path. It changes only the differentiable
free-drive acceleration-willingness measurement. A detached semantic-normal
trajectory defines one common acceleration-opportunity support:

```text
w_normal(t) = stopgrad(sigmoid((v_limit - v_normal(t) - 2) / 0.35))
z_accel(tau_rho; tau_normal)
  = high_quantile(relu(acceleration(tau_rho)), w_normal)
```

The commanded, plus, minus, and normal branches therefore compare positive
acceleration over the same feasible time support. A command cannot make its
own willingness score easier or harder by changing speed headroom. The frozen
conditional-rank target and the definitions of speed utilization and response
intensity are unchanged. The evaluator supplies the same rho=0 ego reference
to every row in a fixed-noise rho sweep and records
`accel_opportunity_anchor_used` for contract verification.

Earlier presets retain `free_drive_accel_support_mode=self_generated`, so A3.5
can be reverted without source rollback by selecting the A3.4 preset and its
checkpoint.

## Stage-A3.6 per-sample worst-axis preset

```text
v6_signed_router_ncqt_stage_a3_6_worst_axis
```

Stage A3.6 branches directly from A3.4, not A3.5. It preserves the global-rho
command, self-generated acceleration-opportunity support, terminal interval,
NCQT and symmetry weights, frozen planner, signed Router, kinematic ego
residual, data path, CFG-off state, and inference behavior.
Only the monotonic hinge aggregation changes.

The legacy `axis_mean` mode still samples eligible `(sample, axis)` rows and is
the default for every earlier preset. A3.6 uses
`soft_worst_per_sample`: it first samples at most
`signed_monotonic_max_pairs` eligible samples, constructs exactly one shared
minus/normal/plus global-rho triplet per sample, and measures all causally valid
axes from that triplet. For sample `i`, valid-axis hinge errors are aggregated
as

```text
L_i = T * (logsumexp(log(w_ij) + e_ij / T) - logsumexp(log(w_ij)))
e_ij = relu(margin - (z_plus_ij - z_minus_ij))
```

where `w_ij` is detached semantic-normal measurement confidence multiplied by
causal/reference validity and `T=0.05`. The batch loss is the mean of `L_i`
over valid samples; no maximum is taken across samples or scenes. The symmetry
term deliberately remains the historical confidence-weighted mean.

Training logs expose per-axis order, mean response, pair count, and worst-axis
ratio. In particular, `free_speed_utilization_order`,
`free_accel_willingness_order`, and `free_speed_response_order` prevent the
aggregate free-drive order score from hiding the acceleration axis. The
acceleration definition itself remains unchanged: it is the opportunity-
weighted high quantile of positive acceleration, so a larger aggressive
command is expected to produce a larger value rather than a sign inversion.

Rollback requires only switching back to the A3.4 preset and checkpoint; no
source or data change is required.

## Stage-A3.8 terminal-executor preset

```text
v6_signed_router_ncqt_stage_a3_8_terminal_executor
```

Stage A3.8 is the trainable version of the fixed-32-sample DPM step-gating
counterfactual. Relative to A3.7, it changes only when the existing
free-drive axis-temporal ego residual is allowed to enter the denoiser output:

```text
g_i(t) = 1                                      for non-free-drive rows
g_i(t) = 1[t <= 0.0011]                         for free-drive rows
x0_ego(t) = x0_base_ego(t) + g_i(t) r_axis(t)
```

With the canonical ten-step DPM-Solver and `denoise_to_zero=True`, this selects
the final `t=0.001` model call and suppresses historical accumulation through
earlier solver calls. All three free-drive branches use the same diffusion-time
gate; their already separated trajectory-time banks remain unchanged. These
two notions of time are deliberately distinct: A3.7 controls the shape of a
40-step physical trajectory, whereas A3.8 controls at which diffusion solver
call that shaped residual is executed.

The global-rho paired monotonic objective is sampled in `[0.001, 0.0011]`, the
same active execution window. The six-axis definitions/data, signed Router and
scene gates, one global rho, bias-free heads, ego-only scope, NCQT/soft-worst
weights, frozen planner, CFG-off state, and inference behavior are unchanged.
Car-follow rows retain the historical all-step A3.7 kinematic path. Therefore
rho=0 and empty conditions remain exact zero-residual cases.

The gate adds no parameter or checkpoint buffer. Old `args.json` files default
to `signed_router_diffusion_gate_mode=all_steps`; selecting the A3.7 preset and
checkpoint exactly restores the prior execution path. An A3.7 checkpoint can
also initialize A3.8 with full state-dict coverage.

Evaluation JSONL rows expose `axis_temporal_diffusion_gate`,
`axis_temporal_terminal_only_used`, `axis_temporal_terminal_active`, and
`axis_temporal_diffusion_time` so the final-call contract can be audited without
changing the generated trajectory.

## B3 complete scene-axis executor

```text
v6_signed_router_ncqt_stage_b3_scene_axis_executor
```

B3 preserves A3.8's fixed 12-D V6 command, generated-data axis masks, frozen
router-only training scope, free-drive terminal gate, and all objective weights.
It changes only the executor for `straight_car_follow`: the three existing
car-follow axis residuals now pass through separate bias-free projections and
fixed headway/TTC/closing-tolerance time banks.  The historical global
`control_projection` / `legacy_acceleration` branch is never selected by B3.

Initialize B3 from an A3.8 checkpoint with `--pretrained_model_path`.  The
free-drive A3.8 branch retains identical parameter names and is loaded exactly;
the three new car-follow projections are the only freshly initialized adapter
parameters.  Run the structural self-test and the Phase-0 transport evaluator
after training: the self-test requires `car_follow_uses_axis_path_only=true`,
and Phase-0 should be compared with the fixed A3.8 cohort before any scheduler
or loss-stage change.

### Multi-seed candidate audit and later scene scaling

`research_v1.execution.evaluation.evaluate_checkpoints` accepts multiple checkpoint epochs in
one invocation. `--num-seeds 3` repeats every fixed `(sample, rho)` command with
three paired diffusion-noise replicas; it does not update model weights. The
evaluator now writes both the historical aggregate report and:

```text
v6_multiseed_checkpoint_comparison.json
v6_multiseed_axis_metrics.csv
epoch_<N>/router_only/v6_multiseed_robustness.json
```

The multi-seed artifact reports every seed separately and summarizes mean,
standard deviation, and worst seed for canonical response span, Spearman,
extreme/adjacent direction accuracy, acceleration, jerk, safety, and exact
rho-zero retention. `seed_index` identifies the same noise replica across
samples; the absolute rollout seed remains sample-specific.

Scene count and diffusion-seed count remain separate experimental dimensions.
Use `--max-samples-per-controlled-scene 32` for the candidate audit, then raise
it to `128`, `256`, or another declared count for the paper-scale test.
`--max-samples-per-controlled-scene 0` and
`--max-lane-change-samples 0` request every available candidate. Increasing
seeds tests sampling stochasticity; increasing samples tests scene
generalization, so one must not be reported as a substitute for the other.

An existing output can be re-summarized without loading a checkpoint:

```bash
python -m research_v1.execution.evaluation.summarize_multiseed \
  --output-root /path/to/rho_sweep \
  --checkpoint-tags epoch_2,epoch_5 \
  --variants router_only
```

## Stage-A3.7 free-drive axis-temporal preset

```text
v6_signed_router_ncqt_stage_a3_7_axis_temporal
```

Stage A3.7 is an independent structural branch from A3.6. It keeps the same
global-rho command, sample-grouped soft-worst monotonic loss, NCQT and symmetry
weights, self-generated acceleration support, terminal interval, frozen
planner, Router gates, 12-D data contract, CFG-off state, and inference behavior.
Only the ego kinematic residual changes.

The signed Router still produces its historical summed residual and now also
exposes the exactly corresponding three linear axis contributions internally.
For free-drive rows, each contribution passes through its own bias-free
six-coefficient head and one fixed smooth acceleration bank before the three
profiles are summed and integrated twice:

```text
a_rel(t) = sum_j B_j(t) beta_j(r_j),  beta_j(0) = 0
```

The fixed banks encode time scale but never response sign: speed utilization
uses the long-horizon Bernstein bank, acceleration willingness uses localized
smooth pulses, and two-second speed response uses a medium-width smooth bank.
Car-follow rows continue to use the original single six-coefficient Bernstein
path, so any car-follow change cannot be attributed to a new temporal basis.
All paths remain ego-only; current-state and heading residuals stay zero.
Bias-free heads and rho-independent bases preserve exact zero residual at
rho=0/empty and exact oddness under a global sign flip.

Training logs add passive per-axis coefficient norms, acceleration RMS values,
and pairwise profile cosines. Fixed rho-sweep JSONL rows also contain per-axis
early/mid/late acceleration contributions. These are diagnostics only and do
not add a loss. Earlier injection modes, presets, and checkpoints are unchanged;
rollback is a preset/checkpoint switch to A3.4 or A3.6.

## Stage-A3.2 terminal-pair smoke preset

```text
v6_signed_router_ncqt_stage_a3_2_terminal
```

Stage A3.2 keeps the complete A3.1 Router, kinematic adapter, NCQT objective,
and loss weights. The base diffusion and main NCQT passes still use the
original full-range random diffusion time. Only the existing minus/normal/plus
paired monotonic branch reconstructs the same-noise trajectory at
`t_pair in [0.01, 0.10]`; validation uses the deterministic midpoint `0.055`.
This directly tests whether near-terminal supervision transfers the signed
free-drive acceleration response to the final diffusion rollout.

Reverting requires only the previous preset and its checkpoint:

```text
v6_signed_router_ncqt_stage_a3_kinematic
```

## Stage-A3.1 kinematic smoke preset

```text
v6_signed_router_ncqt_stage_a3_kinematic
```

Stage A3.1 keeps the signed Router, fixed-normal-neighbor NCQT objective,
frozen planner, sampler, and all loss weights identical to Stage A2. Its only
method change is the ego residual parameterization: six smooth Bernstein
acceleration coefficients are integrated into a longitudinal displacement and
projected along the detached base ego heading. Current-state and heading
residuals stay exactly zero, as do normal/empty commands. This prevents the
adapter from independently perturbing all 80 future states to manufacture
finite-difference acceleration or braking events.

The Stage-A2 pointwise ego adapter remains available unchanged:

```text
v6_signed_router_ncqt_stage_a2
```

Thus Stage A3.1 can be reverted by selecting the A2 preset and its checkpoint;
no source or data rollback is needed.

## Stage-A2 preset

```text
v6_signed_router_ncqt_stage_a2
```

It enables:

- 12D V6 condition;
- signed Preference Axis Router V2 with ego-only output injection;
- frozen pretrained planner backbone;
- fixed-neighbor NCQT relative raw-axis transport;
- exogenous-neighbor leakage diagnostics;
- same-noise monotonic and physical odd-symmetry losses;
- controlled/preservation 70/30 batch sampling;
- CFG disabled during training and the standard diffusion sampler unchanged;
- worst-scene signed-control checkpoint selection.

### Reverting without source rollback

The previous NCQT architecture is retained unchanged:

```text
v6_signed_router_ncqt_stage_a
```

It reconstructs `global_adaln + self_generated`, does not instantiate the new
ego adapter, and remains compatible with existing Stage-A checkpoints. Thus an
unsuccessful Stage-A2 run can be abandoned by changing only the preset; no data
or source rollback is required. `v6_signed_router_stage_a` remains the earlier
absolute-target ablation.

The train V5 normalization and conditional-rank model paths are supplied only
as frozen raw-axis references and must be explicit:

```bash
python -m research_v1.execution.diffusion.train_stylized_diffusion \
  --experiment_preset v6_signed_router_ncqt_stage_a2 \
  --pretrained_model_path /path/to/verified_base_planner.pth \
  --train_conditioning_index_override /path/to/train/v6_direct_axis_conditions.jsonl \
  --val_conditioning_index_override /path/to/val/v6_direct_axis_conditions.jsonl \
  --preference_energy_normalization_path /path/to/train/v5_normalization.json \
  --preference_energy_rank_model_path /path/to/train/v5_conditional_rank_model.json
```

## Structural self-test

```bash
python -m research_v1.execution.diffusion.selftest_stylization \
  --normalization-path /path/to/train/v5_normalization.json \
  --conditional-rank-model-path /path/to/train/v5_conditional_rank_model.json
```
