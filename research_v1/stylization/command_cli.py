"""Command-line entry points for V6 direct-axis conditioning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from research_v1.paths import ensure_repo_on_path

ensure_repo_on_path()

from research_v1.stylization.schema import SCENE_BUCKET_ORDER
from research_v1.stylization.commands import (
    audit_v6_causal_router,
    audit_v6_command_support,
    build_v6_direct_axis_conditions,
    evaluate_v6_rho_sweep,
    selftest_v6_style_command,
    validate_v6_direct_axis_conditions,
)


SCENE_CHOICES = ("all", *SCENE_BUCKET_ORDER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V6: export direct three-axis diffusion conditions plus causal rho command interface"
    )
    parser.add_argument("--rank-index-path", required=True, help="V5 v5_conditional_rank.jsonl; rho fields are ignored")
    parser.add_argument(
        "--base-index-path",
        required=True,
        help=(
            "authoritative original train_v2/val_v2 split_index.jsonl; keeps every base sample and fills style only "
            "for matched rank records"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-filter", choices=SCENE_CHOICES, default="all")
    parser.add_argument("--min-router-confidence", type=float, default=0.60)
    return parser


def validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V6: validate direct-axis/causal-mask condition contract")
    parser.add_argument("--condition-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-active-train-rate", type=float, default=0.0)
    return parser


def router_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V6: audit offline buckets against causal router/gates")
    parser.add_argument("--condition-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def support_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V6: audit rho command endpoints against direct-axis support")
    parser.add_argument("--condition-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rho-values", default="-0.8,-0.4,0,0.4,0.8")
    parser.add_argument("--min-references", type=int, default=40)
    parser.add_argument("--support-radius", type=float, default=0.30)
    parser.add_argument("--amplitude", type=float, default=0.25)
    return parser


def sweep_evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V6: evaluate generated trajectory axes under a rho sweep")
    parser.add_argument("--condition-index-path", required=True)
    parser.add_argument("--generated-axis-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--amplitude", type=float, default=0.25)
    return parser


def _parse_rho_values(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise ValueError(f"Invalid --rho-values item {item!r}") from exc
    if not values:
        raise ValueError("--rho-values must contain at least one numeric rho")
    return values


def run(stage: str) -> None:
    if stage == "build":
        args = build_parser().parse_args()
        result = build_v6_direct_axis_conditions(
            rank_index_path=args.rank_index_path,
            output_dir=args.output_dir,
            base_index_path=args.base_index_path,
            scene_filter=args.scene_filter,
            min_router_confidence=args.min_router_confidence,
        )
    elif stage == "validate":
        args = validation_parser().parse_args()
        result = validate_v6_direct_axis_conditions(
            condition_index_path=args.condition_index_path,
            output_dir=args.output_dir,
            min_active_train_rate=args.min_active_train_rate,
        )
    elif stage == "router_audit":
        args = router_audit_parser().parse_args()
        result = audit_v6_causal_router(
            condition_index_path=args.condition_index_path,
            output_dir=args.output_dir,
        )
    elif stage == "support_audit":
        args = support_audit_parser().parse_args()
        result = audit_v6_command_support(
            condition_index_path=args.condition_index_path,
            output_dir=args.output_dir,
            rho_values=_parse_rho_values(args.rho_values),
            min_references=args.min_references,
            support_radius=args.support_radius,
            amplitude=args.amplitude,
        )
    elif stage == "sweep_evaluation":
        args = sweep_evaluation_parser().parse_args()
        result = evaluate_v6_rho_sweep(
            condition_index_path=args.condition_index_path,
            generated_axis_path=args.generated_axis_path,
            output_dir=args.output_dir,
            amplitude=args.amplitude,
        )
    elif stage == "selftest":
        result = selftest_v6_style_command()
    else:
        raise ValueError(f"Unknown V6 stage: {stage}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
