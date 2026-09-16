"""生成实验小节 A 的数据、模型与运行配置表。"""

from __future__ import annotations

import argparse
from pathlib import Path

from stylelora.eval.common import count_jsonl, load_json, section_dir, write_csv


def _size_mb(path: str | None) -> float | None:
    if not path:
        return None
    file = Path(path)
    return round(file.stat().st_size / (1024 ** 2), 4) if file.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文表1：实验配置与数据统计。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--closed-loop-report", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--adapter-high", required=True)
    parser.add_argument("--adapter-low", required=True)
    parser.add_argument("--scene-gate-checkpoint", required=True)
    parser.add_argument("--rho-grid", default="-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1")
    parser.add_argument("--training-seeds", default="17")
    parser.add_argument("--hardware", default="not_recorded")
    args = parser.parse_args()

    print("[stage 1/2] 读取数据清单、闭环报告和模型文件信息。", flush=True)
    closed = load_json(args.closed_loop_report)
    records = closed.get("records", [])
    scenario_counts = [int(row.get("scenario_count", 0)) for row in records if isinstance(row, dict)]
    rows = [
        {"category": "dataset", "item": "training_samples", "value": count_jsonl(args.train_manifest), "unit": "samples"},
        {"category": "dataset", "item": "validation_samples", "value": count_jsonl(args.val_manifest), "unit": "samples"},
        {"category": "dataset", "item": "closed_loop_scenarios", "value": min(scenario_counts) if scenario_counts else 0, "unit": "scenarios"},
        {"category": "evaluation", "item": "rho_grid", "value": args.rho_grid, "unit": ""},
        {"category": "evaluation", "item": "training_seeds", "value": args.training_seeds, "unit": ""},
        {"category": "hardware", "item": "platform", "value": args.hardware, "unit": ""},
    ]
    for item, path in (
        ("baseline_checkpoint", args.baseline_checkpoint),
        ("preference_encoder", args.encoder_checkpoint),
        ("high_style_adapter", args.adapter_high),
        ("low_style_adapter", args.adapter_low),
        ("adaptive_gate", args.scene_gate_checkpoint),
    ):
        rows.append({"category": "model", "item": item, "value": _size_mb(path), "unit": "MiB"})

    output = section_dir(args.output_root, "A") / "table_1_experimental_setup.csv"
    write_csv(output, rows, ("category", "item", "value", "unit"))
    print("[stage 2/2] 实验配置表写入完成。", flush=True)
    print(f"表1 -> {output}")


if __name__ == "__main__":
    main()
