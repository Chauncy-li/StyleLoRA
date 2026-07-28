"""Run the full per-scene controllability evaluation suite for one experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCENE_BUCKET_TO_PREFIX = {
    "straight_free_drive": "free_drive",
    "straight_car_follow": "car_follow",
    "straight_lane_change": "lane_change",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all scene-bucket evaluations for a preference-conditioned diffusion experiment."
    )
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition_source", default="effective", choices=("effective", "safe", "target"))
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_rollouts_per_condition", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--traj_smooth_window", type=int, default=5)
    parser.add_argument("--kinematics_stride", type=int, default=3)
    parser.add_argument("--cfg_guidance_scale", type=float, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)
    for scene_bucket, prefix in SCENE_BUCKET_TO_PREFIX.items():
        cmd = [
            sys.executable,
            "-m",
            "research_v1.execution.evaluation.evaluate_conditioned_diffusion",
            "--experiment_dir",
            str(experiment_dir),
            "--device",
            str(args.device),
            "--condition_source",
            str(args.condition_source),
            "--scene_bucket",
            scene_bucket,
            "--max_samples",
            str(args.max_samples),
            "--num_rollouts_per_condition",
            str(args.num_rollouts_per_condition),
            "--sample_stride",
            str(args.sample_stride),
            "--traj_smooth_window",
            str(args.traj_smooth_window),
            "--kinematics_stride",
            str(args.kinematics_stride),
            "--summary_json_path",
            str(experiment_dir / f"eval_{prefix}_{args.condition_source}_summary_proxyv2.json"),
            "--detail_jsonl_path",
            str(experiment_dir / f"eval_{prefix}_{args.condition_source}_details_proxyv2.jsonl"),
        ]
        if args.cfg_guidance_scale is not None:
            cmd.extend(["--cfg_guidance_scale", str(args.cfg_guidance_scale)])
        print(f"[PrefCondEvalSuite] running scene_bucket={scene_bucket}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
