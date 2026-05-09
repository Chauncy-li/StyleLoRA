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
import json
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

# 在线日志后端（默认 swanlab）
# - swanlab: 优先 swanlab，抑制 wandb 自动初始化
# - wandb:   使用 wandb
# - disabled: 关闭在线日志
ONLINE_LOGGER = "swanlab"

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
MINI_TEST_LOG_JSON = REPO_ROOT / "baseline" / "resources" / "mini" / "splits" / "mini_test_logs.json"


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


def _normalize_online_logger(name: str) -> str:
    value = str(name or "swanlab").strip().lower()
    alias = {
        "none": "disabled",
        "off": "disabled",
        "no": "disabled",
    }
    return alias.get(value, value)


def _load_log_name_list(path: Path) -> List[str]:
    """读取 JSON 日志列表；不存在或格式异常时返回空列表。"""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, list):
            return []
        return [str(item) for item in payload]
    except Exception:
        return []


def _apply_online_logger_env(name: str) -> str:
    """
    给仿真进程设置在线日志相关环境变量。
    说明：run_simulation 本身不直接写 wandb/swanlab，这里用于避免依赖侧自动初始化冲突。
    """
    backend = _normalize_online_logger(name)
    os.environ["ONLINE_LOGGER"] = backend

    if backend == "wandb":
        os.environ["WANDB_MODE"] = "online"
        os.environ.pop("WANDB_DISABLED", None)
    elif backend in {"swanlab", "disabled"}:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

    if backend == "disabled":
        os.environ["SWANLAB_MODE"] = "disabled"
    else:
        os.environ.pop("SWANLAB_MODE", None)

    return backend


def _build_sys_argv(timestamp: str) -> Tuple[List[str], str, str, str, int]:
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
    scenario_filter_overrides: List[str] = []
    mini_test_count = 0
    if SPLIT == "mini":
        mini_test_logs = _load_log_name_list(MINI_TEST_LOG_JSON)
        if mini_test_logs:
            # 仅覆盖 log_names；其余参数（如 limit_total_scenarios）继续使用 mini.yaml。
            scenario_filter_overrides.extend(
                [
                    f"scenario_filter.log_names={json.dumps(mini_test_logs, ensure_ascii=False, separators=(',', ':'))}",
                ]
            )
            mini_test_count = len(mini_test_logs)

    argv = [
        "run_simulation.py",
        f"+simulation={CHALLENGE}",
        f"planner={PLANNER}",
        *planner_overrides,
        f"scenario_builder={SCENARIO_BUILDER}",
        f"scenario_filter={SPLIT}",
        *scenario_filter_overrides,
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
    return argv, full_output_dir, video_output_dir, raw_output_dir, mini_test_count


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
    active_online_logger = _apply_online_logger_env(ONLINE_LOGGER)

    registered_planners = _load_registered_planners()
    if PLANNER not in registered_planners:
        print(f"❌ 未注册 planner: {PLANNER}")
        print(f"可选: {registered_planners}")
        raise SystemExit(1)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    argv, full_output_dir, video_output_dir, raw_output_dir, mini_test_count = _build_sys_argv(timestamp)
    sys.argv = argv

    print(f"🚀 [{PLANNER}] 启动 NuPlan 闭环仿真...")
    print(f"🕒 任务 ID: {timestamp}")
    print(f"📂 结果输出: {full_output_dir}")
    print(f"🎞️ 视频输出: {video_output_dir}")
    print(f"🧾 原始 step 输出: {raw_output_dir}")
    print(f"📝 在线日志后端: {active_online_logger}")
    print(f"🔧 模型路径: {CKPT_FILE}")
    if SPLIT == "mini":
        if mini_test_count > 0:
            print(f"🧪 mini 测试日志: {mini_test_count} 条（来自 {MINI_TEST_LOG_JSON}）")
        else:
            print(f"⚠️ 未读取到 mini_test_logs，回退为 scenario_filter=mini 默认配置。")

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
