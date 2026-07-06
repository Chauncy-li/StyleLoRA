## Data Foundation

This folder holds the offline data utilities that remain useful for the small
paper even though the method itself will live under
`research/preference_execution/`.

### Current Responsibilities

- raw-db train / val / test-simu log splitting
- planner-cache construction from the selected log subsets
- train-first / val-second manifest generation so existing processed caches can
  still be consumed without changing downstream formats

### Why It Still Matters

The small paper depends on a clean offline data boundary:

- `test_simu` stays held out from training
- `train` and `val` stay disjoint
- straight-scene subsets are built from already partitioned cache data

That means the later controllability experiments can focus on the right
evaluation population without touching the baseline preprocessing contract.

