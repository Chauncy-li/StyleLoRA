# Research V1

`research_v1` is the cleaned server-facing research package for the current
stylized Diffusion Planner work. The original `research` directory is preserved
unchanged as a rollback source.

## Package Layout

```text
research_v1/
├── paths.py                 # Server paths and repository bootstrap
├── data_pipeline/           # Raw DB split and planner-cache preparation
├── scene_data/              # Canonical straight-scene dataset pipeline
│   └── legacy/              # Frozen foundations still used by the canonical pipeline
├── stylization/             # Rho commands, metrics, routing, and runtime control
└── execution/               # Training, conditioning, runtime, and evaluation
```

The canonical modules no longer carry implementation-version suffixes in their
file names. Artifact names and schema identifiers such as `v2`, `v5`, and `v6`
remain unchanged because existing datasets, checkpoints, and reports depend on
those contracts.

## Server Root

The default server repository root is:

```text
/home/lisw/programs/Nuplan-Diffusion-Baseline
```

The default data and cache roots remain:

```text
/mnt/mydata/lishangwen/TrafficDataSetSource/dataset
/mnt/mydata/lishangwen/Nuplan-Baseline-Record
```

All defaults are centralized in `research_v1.paths` and can still be overridden
with the existing environment variables.

## Main Server Commands

Run commands from the server repository root:

```bash
cd /home/lisw/programs/Nuplan-Diffusion-Baseline
```

Build the canonical train/validation scene data:

```bash
python -m research_v1.scene_data.build_train_val
```

Build V5 reference-calibration artifacts required by the current V6 command
contract:

```bash
python -m research_v1.stylization reference candidates ...
python -m research_v1.stylization reference normalization ...
python -m research_v1.stylization reference rank ...
```

Build and validate continuous rho command sidecars:

```bash
python -m research_v1.stylization command build ...
python -m research_v1.stylization command validate ...
python -m research_v1.stylization command router_audit ...
python -m research_v1.stylization command support_audit ...
```

Train the current StylePlanner mainline:

```bash
python -m research_v1.execution.diffusion.train_stylized_diffusion ...
```

Evaluate checkpoints and diffusion transport:

```bash
python -m research_v1.execution.evaluation.evaluate_checkpoints ...
python -m research_v1.execution.evaluation.analyze_diffusion_transport ...
```

Run the closed-loop suite:

```bash
python -m research_v1.execution.evaluation.run_closed_loop_suite ...
```

## Compatibility Contract

- Output file names, CLI option names, schema versions, model preset names, and
  numerical implementations are unchanged.
- The baseline planner now imports runtime research components from
  `research_v1`.
- `research/` is not deleted and can be used to compare or roll back behavior.
- `MIGRATION_MAP.json` records every migrated source and destination file.
