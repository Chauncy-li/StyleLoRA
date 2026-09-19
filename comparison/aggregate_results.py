"""汇总三个模型在统一 test split 上的闭环评分 -> 对比表 + summary.json。

用法：
  python comparison/aggregate_results.py \
      --style <stylelora 结果目录> \
      --diff  <diffusion_planner 结果目录> \
      --admlp <ego_status_planner 结果目录> \
      --stage <stage_vector_planner 结果目录> \
      --out comparison/summary.json

每个结果目录来自 baseline/run_simulation.py 的 <SAVE_ROOT>/simulation/<challenge>/<planner>/<timestamp>。
读取其中的 aggregator_metric/*.parquet（官方聚合总分）与逐场景 metrics parquet，
汇总 final_score 与关键子指标，输出对比表与 summary.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (REPO_ROOT / "nuplan-devkit", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline.simulation.simulation_metrics import (  # noqa: E402
    load_aggregator_metric,
    load_metrics_dataframe,
)

SUB_METRICS = [
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "ego_is_comfortable",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "speed_limit_compliance",
    "ego_progress_along_expert_route",
]


def _extract_final_score(result_dir):
    agg = load_aggregator_metric(result_dir)
    if agg is None or agg.empty:
        return None
    # 官方聚合 parquet：metric_score 列存综合分，其余列为各子指标均值
    score = None
    if "metric_score" in agg.columns:
        score = float(pd.to_numeric(agg["metric_score"], errors="coerce").mean())
    sub = {}
    for col in SUB_METRICS:
        if col in agg.columns:
            v = pd.to_numeric(agg[col], errors="coerce").mean()
            if np.isfinite(v):
                sub[col] = float(v)
    return {"final_score": score, "sub_metrics": sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default=None)
    ap.add_argument("--diff", default=None)
    ap.add_argument("--admlp", default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--out", default=str(SCRIPT_DIR / "summary.json"))
    args = ap.parse_args()

    models = [("StyleLoRA", args.style), ("DiffPlanner", args.diff), ("AD-MLP", args.admlp), ("StageVec", args.stage)]
    models = [(n, d) for n, d in models if d]

    rows = {}
    for name, d in models:
        if not os.path.isdir(d):
            print(f"[aggregate] ⚠️ 目录不存在，跳过 {name}: {d}")
            continue
        res = _extract_final_score(d)
        if res is None or res["final_score"] is None:
            print(f"[aggregate] ⚠️ 未找到聚合分数，跳过 {name}: {d}")
            continue
        rows[name] = res
        print(f"[aggregate] {name}: final_score={res['final_score']:.4f}")

    # 对比表
    print("\n===== 对比表 =====")
    header = f"{'模型':<14} | {'final_score':>11}"
    for m in SUB_METRICS:
        header += f" | {m[:22]:>22}"
    print(header)
    print("-" * len(header))
    for name, res in rows.items():
        line = f"{name:<14} | {res['final_score']:>11.4f}"
        for m in SUB_METRICS:
            v = res["sub_metrics"].get(m)
            line += f" | {('%.4f' % v) if v is not None else '—':>22}"
        print(line)

    out = {name: res for name, res in rows.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[aggregate] saved -> {args.out}")


if __name__ == "__main__":
    main()
