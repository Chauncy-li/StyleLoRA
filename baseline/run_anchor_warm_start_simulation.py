"""Remote Boston closed-loop entrypoint for anchor warm-start validation.

Run from the repository root on the server:

    python baseline/run_anchor_warm_start_simulation.py

Every path can be overridden through the environment, while defaults match the
server layout used by this project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"

# Resolve this checkout before any globally installed/stale nuplan package.
for import_root in (str(DEVKIT_ROOT), str(REPO_ROOT)):
    if import_root in sys.path:
        sys.path.remove(import_root)
sys.path[0:0] = [str(DEVKIT_ROOT), str(REPO_ROOT)]

from baseline import run_simulation as runner  # noqa: E402


DEFAULT_CHECKPOINT_DIR = (
    "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline/train_log/"
    "diffusion-planner/2026_04_03-14_05_53"
)


def configure() -> None:
    checkpoint_dir = os.environ.get("ANCHOR_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    runner.DEVKIT_ROOT = str(DEVKIT_ROOT)
    # Do not inherit generic NUPLAN_* variables here: long-lived server shells
    # often contain paths from a different checkout/dataset.  Anchor-specific
    # variables are explicit and cannot silently redirect this experiment.
    runner.NUPLAN_DATA_ROOT = os.environ.get(
        "ANCHOR_NUPLAN_DATA_ROOT",
        "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston",
    )
    runner.NUPLAN_MAPS_ROOT = os.environ.get(
        "ANCHOR_NUPLAN_MAPS_ROOT",
        "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps",
    )
    runner.NUPLAN_EXP_ROOT = os.environ.get(
        "ANCHOR_NUPLAN_EXP_ROOT",
        "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline",
    )
    runner.SAVE_ROOT = os.environ.get("ANCHOR_SAVE_ROOT", runner.NUPLAN_EXP_ROOT)

    runner.PLANNER = "anchor_warm_start_style_planner"
    if runner.PLANNER not in runner.SUPPORTED_PLANNERS_FALLBACK:
        runner.SUPPORTED_PLANNERS_FALLBACK.append(runner.PLANNER)
    runner.CHECKPOINT_DIR = checkpoint_dir
    runner.ARGS_FILE = os.environ.get("ANCHOR_ARGS_FILE", os.path.join(checkpoint_dir, "args.json"))
    runner.CKPT_FILE = os.environ.get(
        "ANCHOR_CKPT_FILE",
        os.path.join(checkpoint_dir, "best_model-epoch_116-train_loss_0.0701.pth"),
    )
    runner.SPLIT = os.environ.get("ANCHOR_SCENARIO_FILTER", "boston")
    runner.CHALLENGE = os.environ.get("ANCHOR_CHALLENGE", "closed_loop_nonreactive_agents")
    runner.BRANCH_NAME = os.environ.get("ANCHOR_BRANCH_NAME", "anchor_warm_start_boston")
    runner.ONLINE_LOGGER = os.environ.get("ANCHOR_ONLINE_LOGGER", "disabled")


def validate_inputs() -> None:
    required = {
        "training args": runner.ARGS_FILE,
        "checkpoint": runner.CKPT_FILE,
        "Boston database path": runner.NUPLAN_DATA_ROOT,
        "NuPlan maps path": runner.NUPLAN_MAPS_ROOT,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not os.path.exists(path)]
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Anchor warm-start simulation prerequisites are missing:\n  - " + details
        )


if __name__ == "__main__":
    configure()
    validate_inputs()
    runner.main()
