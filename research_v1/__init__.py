"""Clean research package for the current stylized Diffusion Planner pipeline.

The package is organized by responsibility:

- ``data_pipeline``: raw split and planner-cache preparation.
- ``scene_data``: canonical straight-scene dataset construction.
- ``stylization``: continuous rho commands and behavior measurements.
- ``execution``: conditioning, training, runtime, and evaluation.

The original ``research`` package is intentionally left untouched for rollback.
"""

from research_v1.paths import ensure_repo_on_path

__all__ = ["ensure_repo_on_path"]
