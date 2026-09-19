"""DiffPlanner 训练入口（薄封装，不重写训练逻辑）。

DiffPlanner 已有完整的 Hydra 训练管线（baseline/train.py + baseline/config/train.yaml），
为了论文严谨性我们**不改变其训练逻辑**，本脚本只负责把“对齐后的训练量”翻译成 Hydra
override 并原样调用 baseline/train.py。

数据口径：DiffPlanner 训练数据来自 train.yaml 的 `data.train_set`（学长现有缓存），
该缓存由 DataProcessor.work() 产出，与统一抽取器里复用同一套 process_scenario 口径，
故三个模型数据源一致（只是 DiffPlanner 走既有缓存、AD-MLP/StageVec 走 unified .npz）。

用法：
  # 全量（不传 sample 数，沿用 train.yaml 默认；四个模型需一致）
  python comparison/train_diff_planner.py
  # 显式指定训练量（与 AD-MLP / StageVec 的 --train-samples/--val-samples 一致）
  python comparison/train_diff_planner.py --train-samples 60000 --val-samples 14000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=None,
                    help="训练样本数 -> training.train_num_samples（四个模型需一致）")
    ap.add_argument("--val-samples", type=int, default=None,
                    help="验证样本数 -> training.val_num_samples")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="追加的 hydra override，如 training.batch_size=200")
    args = ap.parse_args()

    overrides = ["method=diffusion_planner"]
    if args.train_samples is not None:
        overrides.append(f"training.train_num_samples={args.train_samples}")
    if args.val_samples is not None:
        overrides.append(f"training.val_num_samples={args.val_samples}")
    overrides += args.extra

    cmd = [sys.executable, str(REPO_ROOT / "baseline" / "train.py"), *overrides]
    print("[diff_planner] 调用:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
