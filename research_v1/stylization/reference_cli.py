"""Command-line entry points for the retained V5 data-calibration stages."""

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

from research_v1.stylization.reference_calibration import (
    apply_v5_conditional_ranks,
    apply_v5_normalization,
    build_v5_candidates,
    fit_v5_conditional_ranks,
    fit_v5_normalization,
)
from research_v1.stylization.schema import SCENE_BUCKET_ORDER


SCENE_CHOICES = ["all", "controlled", "interaction", *SCENE_BUCKET_ORDER]


def _common_scene(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-filter", choices=SCENE_CHOICES, default="all")


def candidate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V5: measure continuous-style candidates from raw planner cache")
    parser.add_argument("--index-path", required=True, help="style_scene_split_v2 split_index.jsonl; scene/quality source only")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-cache-dir", default="", help="remote boston_cache_train_val root; overrides stale paths in index")
    _common_scene(parser)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=20260714)
    parser.add_argument("--min-scene-confidence", type=float, default=0.60)
    parser.add_argument("--no-require-sample-quality", action="store_true")
    parser.add_argument("--min-ego-path-length-m", type=float, default=12.0)
    parser.add_argument("--min-ego-forward-progress-m", type=float, default=8.0)
    parser.add_argument("--max-selected-records", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser


def normalization_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V5: fit train-only robust axis normalization")
    parser.add_argument("--candidate-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    _common_scene(parser)
    parser.add_argument("--low-quantile", type=float, default=0.05)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--min-axis-samples", type=int, default=40)
    return parser


def rank_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V5: create out-of-fold conditional percentile labels")
    parser.add_argument("--normalized-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    _common_scene(parser)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--neighbours", type=int, default=64)
    parser.add_argument("--min-effective-neighbours", type=float, default=32.0)
    parser.add_argument(
        "--opportunity-candidate-pool-multiplier",
        type=int,
        default=8,
        help=(
            "retrieve this many times k nearest causal contexts only for opportunity-masked axes, "
            "then retain the nearest k valid labels; ordinary axes always use 3k"
        ),
    )
    parser.add_argument(
        "--min-shared-condition-features",
        type=int,
        default=3,
        help="minimum causal condition dimensions shared by a query and each kNN reference; missing values are never imputed",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def normalization_application_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V5: apply frozen train normalization to val/test")
    parser.add_argument("--candidate-index-path", required=True)
    parser.add_argument("--normalization-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    _common_scene(parser)
    return parser


def rank_application_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V5: apply frozen train conditional-CDF references to val/test")
    parser.add_argument("--normalized-index-path", required=True)
    parser.add_argument("--conditional-rank-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    _common_scene(parser)
    return parser


def run(stage: str) -> None:
    if stage == "candidates":
        args = candidate_parser().parse_args()
        result = build_v5_candidates(
            index_path=args.index_path, output_dir=args.output_dir, planner_cache_dir=args.planner_cache_dir,
            scene_filter=args.scene_filter, sample_fraction=args.sample_fraction, sample_seed=args.sample_seed,
            min_scene_confidence=args.min_scene_confidence, require_sample_quality=not args.no_require_sample_quality,
            min_ego_path_length_m=args.min_ego_path_length_m,
            min_ego_forward_progress_m=args.min_ego_forward_progress_m,
            max_selected_records=args.max_selected_records, log_interval=args.log_interval,
        )
    elif stage == "normalization":
        args = normalization_parser().parse_args()
        result = fit_v5_normalization(
            candidate_index_path=args.candidate_index_path, output_dir=args.output_dir, scene_filter=args.scene_filter,
            low_quantile=args.low_quantile, high_quantile=args.high_quantile, min_axis_samples=args.min_axis_samples,
        )
    elif stage == "rank":
        args = rank_parser().parse_args()
        result = fit_v5_conditional_ranks(
            normalized_index_path=args.normalized_index_path, output_dir=args.output_dir, scene_filter=args.scene_filter,
            folds=args.folds, neighbours=args.neighbours, min_effective_neighbours=args.min_effective_neighbours,
            min_shared_condition_features=args.min_shared_condition_features,
            opportunity_candidate_pool_multiplier=args.opportunity_candidate_pool_multiplier, seed=args.seed,
        )
    elif stage == "apply_normalization":
        args = normalization_application_parser().parse_args()
        result = apply_v5_normalization(
            candidate_index_path=args.candidate_index_path,
            normalization_model_path=args.normalization_model_path,
            output_dir=args.output_dir,
            scene_filter=args.scene_filter,
        )
    elif stage == "apply_rank":
        args = rank_application_parser().parse_args()
        result = apply_v5_conditional_ranks(
            normalized_index_path=args.normalized_index_path,
            conditional_rank_model_path=args.conditional_rank_model_path,
            output_dir=args.output_dir,
            scene_filter=args.scene_filter,
        )
    else:
        raise ValueError(f"Unknown V5 stage: {stage}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
