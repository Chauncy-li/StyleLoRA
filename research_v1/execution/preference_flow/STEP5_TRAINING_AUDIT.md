# Step-5 training-chain audit

This audit is intentionally completed before optimizing a trajectory residual.

1. The StylePlanner joint state has four channels: `(x, y, cos(theta),
   sin(theta))`. They are not independent position/heading residual channels.
2. The cached current frame is physical scene geometry, but the solver-facing
   current state follows `ObservationNormalizer`, just as the frozen planner
   input does. Future clean-prediction frames are `StateNormalizer`
   coordinates. Step 5-S2 therefore passes raw cached ego current separately
   wherever physical path geometry is needed; it never treats normalized `x_q`
   current as metres.
3. `StateNormalizer.__call__` is used on future supervision before noising;
   `StateNormalizer.inverse` is used on final sampled futures. Step 5 performs
   the equivalent future-only physical conversion inside its longitudinal
   decoder, never on the current frame.
4. Step 4 had two zero output maps: the vector-field final layer and the
   four-channel residual projection. This blocks the first useful gradient.
   Step 5 retains the zero-initialized vector-field output but replaces the
   production decoder with a fixed nonzero smooth latent-to-progress basis.
   Thus `z=0` gives an exact zero residual, while `d residual / d z` is nonzero.
5. A Step-5 training forward draws one fixed noisy phase per sample, calls the
   frozen base planner under `torch.no_grad()` to obtain neutral clean `x0`, and
   feeds the same `x_q` to the preference branch. It does not execute an
   11-evaluation DPM rollout.
6. V6 causal metadata is sourced from `style_value_condition[3:12]` only:
   causal three-axis mask, causal task one-hot, and scene gate. The condition
   uses `[one_hot, gate, signed_mask]`; all signed masks follow the persisted
   V6 conservative-to-aggressive orientation.
7. `style_value_condition[:3]` is a future-expert axis target. It is never
   passed to the adapter or vector field. The cohort builder may use it only to
   choose a training rho endpoint; raw future ego/neighbor trajectories are
   used only by behavior losses.

The Step-5 decoder shifts only ego future positions along the neutral-path
tangent and re-derives future heading from the edited path. It directly writes
neither ego current state nor any non-ego state, creates no independent lateral
offset, and contains no feasibility, route-continuation, A3.8/B3, or full-scale
training logic.

## Step-5-R direction proof

The tiny rerun is a deterministic full-cohort overfit: all six free-drive and
all six car-follow scenes share one optimizer batch at every update. For every
scene, the target-fitting branch still uses its persisted `training_rho`, while
the order branch separately evaluates the same neutral prediction, `x_q`,
diffusion phase, and noise at exactly `rho=-1, 0, +1`. It enforces positive
free-drive speed/headway-tightness margins on both sides of neutral. The smoke
report repeats that three-way comparison for all 12 scenes and records both
margins, pass counts, physical longitudinal residual statistics, and direct
ego-current/non-ego invariants. The fresh zero-initialized Flow is never loaded
from a failed Step-5 checkpoint.

## Step-5-S1 semantic correction

Step-5-R established that the base can remain frozen while a Flow receives a
gradient, but its car-follow proxy was `-min_{agent,time} distance`. That value
can select a side vehicle and can switch agent or time when the ego is edited;
it is neither a fixed-lead THW nor a TTC measurement. Step-5-S1 therefore does
not tune a failed scene. It changes the supervision contract:

```text
neutral x0 + observed neighbor history
    -> detached lead ID j0 and neutral valid-time mask T0
    -> shared by rho = [-1, -.5, 0, .5, 1]
```

The lead selector uses positive neutral-path longitudinal distance and a
same-lane lateral bound. It reads no neighbor future. Neighbor futures are used
only after `j0,T0` are frozen, inside the training loss and in offline
evaluation. The car proxy is canonical headway tightness, optionally combined
with TTC tightness only on a closing-event mask frozen from neutral. Invalid
TTC events contribute neither a default value nor a gradient.

For free-drive, the bridge is speed utilization. For both tasks, canonical
values are oriented so larger means more aggressive using the frozen V5 ranges.
The desired curve is anchored at the neutral measurement:

```text
p_des(rho) = clip(p0 + rho * d_b)
```

Positive and negative `d_b` are configurable fractions of the existing frozen
calibration range. The total loss contains neutral-relative curve fit, all four
adjacent rho-grid order terms, and small content/smooth regularizers. The
legacy future-expert fit remains report-only.

After optimization, each edited ego future is evaluated by the unmodified hard
metric implementation using a temporary cache clone; this supplies a formal
metric rho sweep without changing `metrics.py` or exposing it to the Flow. The
audit records the legacy min-distance selection versus `j0`, four fixed
longitudinal perturbations, all twelve five-point paths, formal/proxy direction
agreement, invariant checks, physical longitudinal residuals, and
initial/mid/final free-car gradient cosine. It does not add PCGrad, task heads,
a MoE, or any filename-specific rule.

## Step-5-S2 neutral-anchored soft interaction correction

S2 replaces only S1's hard single-lead selection for the unified method. The
base planner is still frozen and there is still one Vector Field. At each fixed
diffusion phase, neutral ego clean `x0` queries neutral neighbor clean `x0`
tokens. Relative longitudinal/lateral displacement, relative speed, THW, TTC,
and current observed geometry form relation bias; a null token represents no
material interaction. The module has no `r` or `rho` parameter.

The resulting neighbor-time weights and interaction context are computed once
from neutral data and reused verbatim for all five rho branches. Weights are
detached before weighted headway/TTC measurement. Thus gradients can train the
unified Flow through the interaction context, but no branch can lower its loss
by changing the identity/weighting of the vehicle it is measured against.
Ground-truth neighbor future remains strictly loss-only.

The desired path is still neutral-relative and calibrated. Adjacent intervals
with an unclipped target change are active and require positive same-direction
movement. Intervals clipped at a calibration boundary may be flat, but negative
movement is always penalized. This is a feasible-boundary contract, not a
relaxation of directionality.

The persisted `training_rho` still selects a labeled endpoint for a separate
GT-behavior fit. It does not turn the three behavior axes into averaged scalar
personality labels: free-drive uses speed utilization, while car-follow uses
the neutral-attention-weighted headway/TTC measurement.

Finally, `SmoothLongitudinalTrajectoryResidualDecoder.decode()` receives the
raw physical ego current frame and returns `base_tangent`; behavior loss,
trajectory-residual reporting, and the lateral invariant use that exact tensor.
The old normalized `x_q` current is never treated as physical geometry.
