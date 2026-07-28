"""Unified command-line entry point for stylization data stages."""

from __future__ import annotations

import argparse
import sys

from research_v1.stylization.command_cli import run as run_command_stage
from research_v1.stylization.reference_cli import run as run_reference_stage


REFERENCE_STAGES = (
    "candidates",
    "normalization",
    "rank",
    "apply_normalization",
    "apply_rank",
)
COMMAND_STAGES = (
    "build",
    "validate",
    "router_audit",
    "support_audit",
    "sweep_evaluation",
    "selftest",
)


def main() -> None:
    """Dispatch to the retained reference or current command pipeline."""

    parser = argparse.ArgumentParser(
        description="Run continuous-stylization data and command stages."
    )
    parser.add_argument("pipeline", choices=("reference", "command"))
    parser.add_argument("stage")
    args, remaining = parser.parse_known_args()

    if args.pipeline == "reference":
        if args.stage not in REFERENCE_STAGES:
            parser.error(
                f"reference stage must be one of {REFERENCE_STAGES}, got {args.stage!r}"
            )
        runner = run_reference_stage
    else:
        if args.stage not in COMMAND_STAGES:
            parser.error(
                f"command stage must be one of {COMMAND_STAGES}, got {args.stage!r}"
            )
        runner = run_command_stage

    sys.argv = [sys.argv[0], *remaining]
    runner(args.stage)


if __name__ == "__main__":
    main()
