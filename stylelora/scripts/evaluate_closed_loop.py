"""Run actual NuPlan closed-loop jobs with post-initialize LoRA attachment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stylelora.lora.evaluation.reports import write_json


def _metric_summary(result_dir: Path, minimums: dict[str, float]) -> dict:
    from baseline.simulation.simulation_metrics import load_metrics_dataframe

    frame = load_metrics_dataframe(str(result_dir))
    if frame is None or frame.empty:
        return {"closed_loop_pass": False, "failure": "NuPlan produced no readable per-scenario metrics"}
    numeric = {column: float(frame[column].mean()) for column in frame.select_dtypes(include="number").columns}
    missing = [name for name in minimums if name not in numeric]
    failed = {name: numeric[name] for name, threshold in minimums.items() if name in numeric and numeric[name] < threshold}
    return {"closed_loop_pass": not missing and not failed, "metric_means": numeric,
            "missing_required_metrics": missing, "below_threshold": failed}


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Execute NuPlan closed-loop rho runs; no external factory is required.")
    parser.add_argument("--args-file", required=True); parser.add_argument("--baseline-checkpoint", required=True)
    # high/low 是当前偏好空间语义；保留旧参数名作为兼容别名。
    parser.add_argument("--high-adapter", "--aggressive-adapter", dest="high_adapter", required=True)
    parser.add_argument("--low-adapter", "--conservative-adapter", dest="low_adapter", required=True)
    parser.add_argument("--normalization-file", required=True); parser.add_argument("--output-root", required=True)
    parser.add_argument("--rhos", default="-1,-0.5,0,0.5,1"); parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--challenge", default="closed_loop_nonreactive_agents")
    parser.add_argument("--config-root", default=str(repository / "baseline" / "config"))
    parser.add_argument("--nuplan-overrides-json", required=True, help="JSON list of remaining NuPlan overrides, e.g. scenario_builder and scenario_filter.")
    parser.add_argument("--required-minimums-json", default='{"no_ego_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0, "time_to_collision_within_bound": 1.0, "ego_is_comfortable": 1.0}')
    args = parser.parse_args()
    extra = json.loads(args.nuplan_overrides_json); minimums = {k: float(v) for k, v in json.loads(args.required_minimums_json).items()}
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("--nuplan-overrides-json must be a JSON list of Hydra override strings")
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    search = f"[pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments,file://{args.config_root},file://{repository / 'stylelora' / 'config'}]"
    records = []
    for rho in (float(item) for item in args.rhos.split(",")):
        result_dir = root / f"rho_{rho:+.2f}".replace("+", "plus").replace("-", "minus")
        command = [sys.executable, "-m", "nuplan.planning.script.run_simulation", f"+simulation={args.challenge}",
                   "planner=lora_diffusion_planner", f"planner.lora_diffusion_planner.config.args_file={args.args_file}",
                   f"planner.lora_diffusion_planner.ckpt_path={args.baseline_checkpoint}",
                   f"planner.lora_diffusion_planner.config.lora_aggressive_adapter={args.high_adapter}",
                   f"planner.lora_diffusion_planner.config.lora_conservative_adapter={args.low_adapter}",
                   f"planner.lora_diffusion_planner.config.lora_normalization_file={args.normalization_file}",
                   f"planner.lora_diffusion_planner.config.lora_rank={args.rank}",
                   f"planner.lora_diffusion_planner.config.lora_rho={rho}", f"output_dir={result_dir}",
                   f"hydra.run.dir={result_dir}", f"hydra.searchpath={search}", *extra]
        subprocess.run(command, check=True)
        records.append({"rho": rho, "result_dir": str(result_dir), **_metric_summary(result_dir, minimums)})
    write_json(root / "closed_loop_lora_report.json", {"records": records, "closed_loop_pass": all(r["closed_loop_pass"] for r in records)})


if __name__ == "__main__": main()
