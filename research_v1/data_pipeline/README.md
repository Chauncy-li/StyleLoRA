# Data Pipeline

This package contains the offline data utilities retained by the current
stylized-planning study:

- raw-database train/validation/test-simulation log splitting;
- planner-cache construction for selected log subsets;
- train-first and validation-second manifest generation for existing caches;
- joint diffusion-dataset assembly.

The split boundary is unchanged: test-simulation logs remain held out, training
and validation logs remain disjoint, and scene subsets are built only after the
cache partition is established.

Server data and output roots come from `research_v1.paths`. Existing command
options and environment-variable overrides remain available.
