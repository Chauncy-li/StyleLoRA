"""
NuPlan 闭环仿真启动脚本（Diffusion Planner）。

功能说明：
1. 组装 NuPlan `run_simulation.py` 所需 Hydra 参数；
2. 固定使用 `planner=diffusion_planner`；
3. 注入模型 `args.json` / checkpoint / 输出目录；
4. 仿真结束后可选调用本地统计评估函数。

使用建议：
- 先修改本文件顶部的路径常量；
- 再运行：`python baseline/run_simulation.py`。
"""

from __future__ import annotations

import datetime
import glob
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ==============================================================================
# 基础路径配置
# ==============================================================================
# NuPlan Devkit 路径（需要与本地安装一致）
DEVKIT_ROOT = "/home/lisw/programs/Nuplan-Baseline-3090/nuplan-devkit"

# 数据与地图路径
NUPLAN_DATA_ROOT = "/media/lsw/Work/ubuntu_system/DATASET/nuplan-v1.1/splits/mini"
NUPLAN_MAPS_ROOT = "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps"
NUPLAN_EXP_ROOT = "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline"

SAVE_ROOT = NUPLAN_EXP_ROOT

# ==============================================================================
# 模型 checkpoint 配置
# ==============================================================================
CHECKPOINT_DIR = "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline/train_log/diffusion-planner/2026_02_02-13_05_43"
ARGS_FILE = os.path.join(CHECKPOINT_DIR, "args.json")
CKPT_FILE = os.path.join(CHECKPOINT_DIR, "best_model-epoch_60-train_loss_0.0947.pth")

# ==============================================================================
# 仿真参数
# ==============================================================================
PLANNER = "diffusion_planner"
SPLIT = "mini"  # 可选: val14 / test14-random / test14-hard
CHALLENGE = "closed_loop_nonreactive_agents"
BRANCH_NAME = "diffusion_debug"
SCENARIO_BUILDER = "nuplan"


def _resolve_local_config_path() -> Path:
    """优先使用 baseline/config，兼容回落到 nuplan_baseline/config。"""
    candidates = [
        REPO_ROOT / "baseline" / "config",
        REPO_ROOT / "nuplan_baseline" / "config",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _build_sys_argv(timestamp: str) -> tuple[list[str], str, str]:
    """构建 NuPlan 主脚本所需的 sys.argv，并返回输出目录信息。"""
    filename_wo_ext = Path(CKPT_FILE).stem if os.path.exists(CKPT_FILE) else "unknown_ckpt"
    experiment_uid = f"{PLANNER}/{SPLIT}/{BRANCH_NAME}/{filename_wo_ext}_{timestamp}"

    full_output_dir = os.path.join(SAVE_ROOT, "simulation", CHALLENGE, PLANNER, timestamp)
    video_output_dir = os.path.join(full_output_dir, "simulation_video")

    local_config_path = _resolve_local_config_path()
    search_path = (
        "[pkg://nuplan.planning.script.config.common, "
        "pkg://nuplan.planning.script.experiments, "
        f"file://{local_config_path}]"
    )

    argv = [
        "run_simulation.py",
        f"+simulation={CHALLENGE}",
        f"planner={PLANNER}",
        f"planner.diffusion_planner.config.args_file={ARGS_FILE}",
        f"planner.diffusion_planner.ckpt_path={CKPT_FILE}",
        "planner.diffusion_planner.device=cuda",
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
        f"planner.diffusion_planner.config.render_save_dir={video_output_dir}",
    ]
    return argv, full_output_dir, video_output_dir


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
    sys.path.append(DEVKIT_ROOT)
    sys.path.append(str(REPO_ROOT))

    os.environ["NUPLAN_DEVKIT_ROOT"] = DEVKIT_ROOT
    os.environ["NUPLAN_DATA_ROOT"] = NUPLAN_DATA_ROOT
    os.environ["NUPLAN_MAPS_ROOT"] = NUPLAN_MAPS_ROOT
    os.environ["NUPLAN_EXP_ROOT"] = NUPLAN_EXP_ROOT
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    argv, full_output_dir, _ = _build_sys_argv(timestamp)
    sys.argv = argv

    print(f"🚀 [Diffusion] 启动 NuPlan 闭环仿真...")
    print(f"🕒 任务 ID: {timestamp}")
    print(f"📂 结果输出: {full_output_dir}")
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
