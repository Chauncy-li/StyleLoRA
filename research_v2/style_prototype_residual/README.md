# Same-State Residual Style Prototypes

This is the FC-first replacement for the earlier Preference Flow experiments.
It lives entirely in `research_v2`; it does not modify `baseline` or the
previous `research_v1` code.

## Method contract

For a frozen planner prediction at the same noisy state,

`Delta x0,q = expert x0 - base clean x0,q`.

The deterministic executor is centred structurally:

`A_ctrl(h,c,q,u) = A_raw(h,c,q,u) - A_raw(h,c,q,0)`.

The deployed clean prediction is therefore:

`final clean x0 = base clean x0 + A_ctrl`.

This makes `u = 0` exactly identity, without relying on an approximate loss or
dropout-free inference coincidence.  Normal defines the centre only; it is not
asked to reconstruct every Normal expert residual.

The direct baseline learns three controls (`aggr`, `norm`, `cons`) and the same
executor.  The prototype path trains a residual encoder only from expert data
at training time, estimates three class prototypes, then deploys the saved
prototypes.  The encoder and expert future never enter deployment.

Sample-code loss reconstructs individual aggressive/conservative residuals.
Prototype loss reconstructs only a `scene × style` mean residual.  All loss
terms are added with positive weights.  No six-axis average is used to create a
single personality label.

## Stages

1. `data-audit` builds fixed FC manifests and checks label/cache overlap.
2. `same-state-contract` verifies that targets are expert-minus-base at the
   same diffusion state.
3. `train-direct` and `full-dpm` establish the fair direct-embedding baseline.
4. `train-encoder` learns training-only sample codes and exports prototypes;
   repeat it with `--label-shuffle` as a control.
5. `train-executor` trains sample and prototype supervision jointly.
6. `full-dpm` validates a single DPM stream.  It computes the neutral base and
   `rho=0` rollout with identical noise, and only opens the expert future after
   all rollouts for evaluation metrics.
7. `summary` records evidence separately from scientific claims.

FCL is intentionally blocked until FC `summary.json` says
`fc_full_dpm_supported=true`.  It reuses exactly these commands with `--mode
fcl`; no free/follow/lane expert or task-specific head exists.
