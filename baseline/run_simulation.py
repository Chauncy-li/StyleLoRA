"""
NuPlan 闭环仿真启动脚本（注册制入口）。

功能说明：
1. 通过 `PLANNER` 选择仿真模型（diffusion_planner / wayformer）；
2. 自动生成对应的 Hydra override（args / ckpt / render / raw export）；
3. 启动 NuPlan 官方 run_simulation，并在结束后可选执行本地评估。
"""

from __future__ import annotations

import datetime
import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ==============================================================================
# 基础路径配置
# ==============================================================================
DEVKIT_ROOT = str(REPO_ROOT / "nuplan-devkit")

NUPLAN_DATA_ROOT = "/media/lsw/Work/ubuntu_system/DATASET/nuplan-v1.1/splits/mini"
NUPLAN_MAPS_ROOT = "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps"
NUPLAN_EXP_ROOT = "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline"
SAVE_ROOT = NUPLAN_EXP_ROOT

# ==============================================================================
# 仿真模型选择（注册制）
# ==============================================================================
# 可选：
# - diffusion_planner
# - wayformer
PLANNER = "diffusion_planner"
SUPPORTED_PLANNERS_FALLBACK = ["diffusion_planner", "wayformer"]

# ==============================================================================
# 模型 checkpoint 配置，注意要和模型对应上
# ==============================================================================
CHECKPOINT_DIR = "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline/train_log/diffusion-planner/2026_02_02-13_05_43"
ARGS_FILE = os.path.join(CHECKPOINT_DIR, "args.json")
CKPT_FILE = os.path.join(CHECKPOINT_DIR, "best_model-epoch_60-train_loss_0.0947.pth")

# ==============================================================================
# 仿真参数
# ==============================================================================
SPLIT = "mini"  # 可选: val14 / test14-random / test14-hard
CHALLENGE = "closed_loop_nonreactive_agents"
BRANCH_NAME = "diffusion_debug"
SCENARIO_BUILDER = "nuplan"


def _resolve_local_config_path() -> Path:
    """优先使用 baseline/config，兼容回落到 nuplan_baseline/config。"""
    candidates = [REPO_ROOT / "baseline" / "config", REPO_ROOT / "nuplan_baseline" / "config"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _planner_override(planner_name: str, key: str, value: str) -> str:
    """构造 planner 动态 override，例如 planner.wayformer.ckpt_path=..."""
    return f"planner.{planner_name}.{key}={value}"


def _build_sys_argv(timestamp: str) -> Tuple[List[str], str, str, str]:
    """构建 NuPlan 主脚本所需的 sys.argv，并返回输出目录信息。"""
    filename_wo_ext = Path(CKPT_FILE).stem if os.path.exists(CKPT_FILE) else "unknown_ckpt"
    experiment_uid = f"{PLANNER}/{SPLIT}/{BRANCH_NAME}/{filename_wo_ext}_{timestamp}"

    full_output_dir = os.path.join(SAVE_ROOT, "simulation", CHALLENGE, PLANNER, timestamp)
    video_output_dir = os.path.join(full_output_dir, "simulation_video")
    raw_output_dir = os.path.join(full_output_dir, "raw_step_data")

    local_config_path = _resolve_local_config_path()
    search_path = (
        "[pkg://nuplan.planning.script.config.common, "
        "pkg://nuplan.planning.script.experiments, "
        f"file://{local_config_path}]"
    )

    planner_overrides = [
        _planner_override(PLANNER, "config.args_file", ARGS_FILE),
        _planner_override(PLANNER, "ckpt_path", CKPT_FILE),
        _planner_override(PLANNER, "device", "cuda"),
        _planner_override(PLANNER, "config.render_save_dir", video_output_dir),
        _planner_override(PLANNER, "config.raw_data_save_dir", raw_output_dir),
    ]

    argv = [
        "run_simulation.py",
        f"+simulation={CHALLENGE}",
        f"planner={PLANNER}",
        *planner_overrides,
        f"scenario_builder={SCENARIO_BUILDER}",
        f"scenario_filter={SPLIT}",
        f"scenario_builder.db_files={NUPLAN_DATA_ROOT}",
        f"experiment_uid={experiment_uid}",
        f"output_dir={full_output_dir}",
        f"hydra.run.dir={full_output_dir}",
        "verbose=true",
        "worker=sequential",
        "enable_simulation_progress_bar=true",
        "number_of_gpus_allocated_per_simulation=1.0",
        f"hydra.searchpath={search_path}",
    ]
    return argv, full_output_dir, video_output_dir, raw_output_dir


def _load_registered_planners() -> List[str]:
    """
    读取 simulation 注册表里的 planner 名单。
    如果导入失败，回退到本地兜底名单，避免入口直接崩溃。
    """
    try:
        from baseline.simulation.planner import get_registered_simulation_planners

        return get_registered_simulation_planners()
    except ModuleNotFoundError as exc:
        print(f"⚠️ 无法读取 simulation 注册表，使用兜底名单: {exc}")
        return SUPPORTED_PLANNERS_FALLBACK


def _load_evaluator():
    """可选导入仿真评估函数，不可用时返回 None。"""
    try:
        from baseline.simulation.simulation_metrics import evaluate_simulation_results

        return evaluate_simulation_results
    except Exception:
        try:
            from nuplan_baseline.simulation.simulation_metrics import evaluate_simulation_results

            return evaluate_simulation_results
        except Exception:
            return None


def main() -> None:
    """脚本主入口。"""
    if DEVKIT_ROOT not in sys.path:
        sys.path.append(DEVKIT_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    os.environ["NUPLAN_DEVKIT_ROOT"] = DEVKIT_ROOT
    os.environ["NUPLAN_DATA_ROOT"] = NUPLAN_DATA_ROOT
    os.environ["NUPLAN_MAPS_ROOT"] = NUPLAN_MAPS_ROOT
    os.environ["NUPLAN_EXP_ROOT"] = NUPLAN_EXP_ROOT
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    registered_planners = _load_registered_planners()
    if PLANNER not in registered_planners:
        print(f"❌ 未注册 planner: {PLANNER}")
        print(f"可选: {registered_planners}")
        raise SystemExit(1)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    argv, full_output_dir, video_output_dir, raw_output_dir = _build_sys_argv(timestamp)
    sys.argv = argv

    print(f"🚀 [{PLANNER}] 启动 NuPlan 闭环仿真...")
    print(f"🕒 任务 ID: {timestamp}")
    print(f"📂 结果输出: {full_output_dir}")
    print(f"🎞️ 视频输出: {video_output_dir}")
    print(f"🧾 原始 step 输出: {raw_output_dir}")
    print(f"🔧 模型路径: {CKPT_FILE}")

    if not os.path.exists(CKPT_FILE):
        print(f"\n❌ [Error] 找不到模型文件: {CKPT_FILE}")
        print("请先修改脚本顶部 CHECKPOINT_DIR / CKPT_FILE。")
        raise SystemExit(1)

    from nuplan.planning.script.run_simulation import main as nuplan_main

    evaluate_simulation_results = _load_evaluator()

    try:
        nuplan_main()
        print("\n" + "=" * 50)
        print("✅ 仿真执行完毕")
        print("=" * 50 + "\n")

        if os.path.exists(full_output_dir):
            parquet_files = glob.glob(os.path.join(full_output_dir, "**", "*.parquet"), recursive=True)
            if parquet_files:
                print(f"📊 发现 {len(parquet_files)} 个统计文件 (Parquet)")
            else:
                print("⚠️  警告: 未找到统计文件")

            if evaluate_simulation_results is not None:
                print("Running custom evaluation metrics...")
                evaluate_simulation_results(full_output_dir)
    except Exception as exc:
        print(f"\n❌ [Error] 仿真异常: {exc}")
        raise


if __name__ == "__main__":
    main()
