"""生成实验小节 F 的模块消融与工程效率表。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from stylelora.eval.common import load_json, parse_named_paths, section_dir, write_csv


def _read_efficiency(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["method"]): dict(row) for row in csv.DictReader(handle)}


def _checkpoint_groups(values: list[str]) -> dict[str, list[Path]]:
    """同一方法可重复传入多个 checkpoint，例如 High、Low 与门控文件。"""
    groups: dict[str, list[Path]] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"checkpoint 参数必须采用 名称=路径：{raw}")
        name, path = (part.strip() for part in raw.split("=", 1))
        if not name or not path:
            raise ValueError(f"checkpoint 名称或路径为空：{raw}")
        groups.setdefault(name, []).append(Path(path))
    return groups


def _checkpoint_statistics(paths: list[Path]) -> tuple[float | None, int | None]:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None, None
    size_mib = sum(path.stat().st_size for path in existing) / (1024 ** 2)
    try:
        import torch
    except ImportError:
        return size_mib, None

    parameter_count = 0
    for path in existing:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            continue
        state = next((payload[key] for key in ("adapter_state", "model_state", "state_dict")
                      if isinstance(payload.get(key), dict)), payload)
        parameter_count += sum(int(value.numel()) for value in state.values()
                               if torch.is_tensor(value))
    return size_mib, parameter_count


def _component_flags(values: list[str]) -> dict[str, dict[str, int]]:
    """解析 NAME=STRUCTURED,LATERAL,GATE，生成论文消融表的三个组件开关。"""
    output: dict[str, dict[str, int]] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"components 参数必须采用 名称=0,0,0：{raw}")
        name, encoded = (part.strip() for part in raw.split("=", 1))
        parts = [part.strip() for part in encoded.split(",")]
        if not name or len(parts) != 3 or any(part not in {"0", "1"} for part in parts):
            raise ValueError(f"components 参数必须采用 名称=0,0,0：{raw}")
        if name in output:
            raise ValueError(f"components 名称重复：{name}")
        output[name] = {
            "structured_alignment": int(parts[0]),
            "lateral_constraint": int(parts[1]),
            "adaptive_gate": int(parts[2]),
        }
    return output


def _open_metrics(path: Path) -> dict:
    report = load_json(path)
    per_rho = [entry for entry in report.get("per_rho", {}).values() if isinstance(entry, dict)]
    ade = [entry.get("ade", {}).get("mean") for entry in per_rho]
    fde = [entry.get("fde", {}).get("mean") for entry in per_rho]
    lateral = [float(row["lateral_shift_vs_baseline"]) for row in report.get("records", [])
               if isinstance(row, dict) and row.get("lateral_shift_vs_baseline") is not None]
    return {
        "style_direction_rate": report.get("direction_check", {}).get("sign_correct_vs_zero_rate"),
        "identity_max_error": report.get("identity_max_mismatch"),
        "mean_ade_over_rho": float(np.mean([v for v in ade if v is not None])) if any(v is not None for v in ade) else None,
        "mean_fde_over_rho": float(np.mean([v for v in fde if v is not None])) if any(v is not None for v in fde) else None,
        "mean_lateral_shift": float(np.mean(lateral)) if lateral else None,
    }


def _closed_metrics(path: Path) -> dict:
    report = load_json(path)
    scores, collision, ttc = [], [], []
    for row in report.get("records", []):
        if abs(float(row.get("rho", 0.0))) < 1e-9:
            continue
        official = row.get("official_aggregator", {})
        if official.get("challenge_score") is not None:
            scores.append(float(official["challenge_score"]))
        safety = official.get("safety_metric_means", {})
        if safety.get("no_ego_at_fault_collisions") is not None:
            collision.append(float(safety["no_ego_at_fault_collisions"]))
        if safety.get("time_to_collision_within_bound") is not None:
            ttc.append(float(safety["time_to_collision_within_bound"]))
    return {
        "closed_challenge_mean_nonzero": float(np.mean(scores)) if scores else None,
        "closed_no_collision_mean_nonzero": float(np.mean(collision)) if collision else None,
        "closed_ttc_mean_nonzero": float(np.mean(ttc)) if ttc else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAST 论文表6：模块消融与工程效率。")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--open-report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--closed-report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--components",
        action="append",
        default=[],
        metavar="NAME=STRUCTURED,LATERAL,GATE",
        help="为每个变体记录结构化对齐、横向约束和自适应门控三个0/1开关",
    )
    parser.add_argument("--efficiency-csv", default=None,
                        help="可选列：method,trainable_parameters,training_hours,inference_ms,peak_memory_mb。")
    args = parser.parse_args()

    open_reports = parse_named_paths(args.open_report)
    closed_reports = parse_named_paths(args.closed_report)
    checkpoints = _checkpoint_groups(args.checkpoint)
    components = _component_flags(args.components)
    efficiency = _read_efficiency(args.efficiency_csv)
    names = list(dict.fromkeys([*open_reports, *closed_reports, *checkpoints, *components, *efficiency]))
    if not names:
        raise ValueError("至少需要提供一个消融变体")
    if components:
        missing_components = [name for name in names if name not in components]
        if missing_components:
            raise ValueError("以下变体缺少 --components 定义：" + ", ".join(missing_components))
    print(f"[stage 1/2] 开始汇总 {len(names)} 个消融/效率条目。", flush=True)
    rows = []
    for index, name in enumerate(names, start=1):
        print(f"[progress] 消融条目: {index}/{len(names)} ({name})", flush=True)
        row = {"method": name}
        row.update(components.get(name, {}))
        if name in open_reports:
            row.update(_open_metrics(open_reports[name]))
        if name in closed_reports:
            row.update(_closed_metrics(closed_reports[name]))
        if name in checkpoints:
            size_mib, parameter_count = _checkpoint_statistics(checkpoints[name])
            row["checkpoint_mib"] = size_mib
            row["trainable_parameters"] = parameter_count
        row.update({key: value for key, value in efficiency.get(name, {}).items() if key != "method"})
        rows.append(row)

    fields = ["method", "structured_alignment", "lateral_constraint", "adaptive_gate",
              "style_direction_rate", "identity_max_error", "mean_ade_over_rho",
              "mean_fde_over_rho", "mean_lateral_shift", "closed_challenge_mean_nonzero",
              "closed_no_collision_mean_nonzero", "closed_ttc_mean_nonzero", "trainable_parameters",
              "checkpoint_mib", "training_hours", "inference_ms", "peak_memory_mb"]
    output = section_dir(args.output_root, "F") / "table_6_ablation_and_efficiency.csv"
    write_csv(output, rows, fields)
    print("[stage 2/2] 表6写入完成。", flush=True)
    print(f"表6 -> {output}")


if __name__ == "__main__":
    main()
