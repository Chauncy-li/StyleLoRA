#!/usr/bin/env bash
# 对单个模型在学长的固定闭环 token 名单上跑 NuPlan 闭环仿真。
#
# 与学长 stylelora/scripts/evaluate_closed_loop.py 同口径：
#   - 场景集合用学长 select_closed_loop_scenarios 产出的固定 token 名单（tokens_file）；
#   - 通过 scenario_filter.scenario_tokens + limit_total_scenarios 注入。
#
# 用法：
#   bash comparison/run_closed_loop.sh \
#       --model diff_planner|ego_status_planner|stage_vector_planner \
#       --ckpt <checkpoint.pth> \
#       [--stats <stage_vec_stats.npz>] \
#       [--args-file <diff_planner 的 args.json>] \
#       [--style aggressive|normal|conservative] \
#       [--style-control <float>] \
#       [--tokens-file <tokens.json>] \
#       [--data-root <dir>] [--maps-root <dir>]
#
# --style 语义按模型分派：
#   - ego_status_planner：开启 with_style，拼接 3 维风格 one-hot（激进/正常/保守）。
#   - stage_vector_planner：激进=+σ、保守=-σ（σ 来自 stats 的 style_value_std）。
# --style-control：stage_vector_planner 的原始 style_control 覆盖（未传 --style 时生效）。
#
# 所有路径都可调：未显式传参时从 stylelora/config/paths.local.json 读取，
# 也可用环境变量 NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT / COMPARISON_TOKENS_FILE 覆盖。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RESOLVER="$REPO_ROOT/stylelora/config/runtime_paths.py"
config_path() { python "$RESOLVER" --get "$1"; }
optional_config_path() { python "$RESOLVER" --get-optional "$1"; }

MODEL=""
CKPT=""
STATS=""
ARGS_FILE=""
TOKENS_FILE=""
STYLE=""
STYLE_CONTROL=""
DATA_ROOT="${NUPLAN_DATA_ROOT:-}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --stats) STATS="$2"; shift 2 ;;
    --args-file) ARGS_FILE="$2"; shift 2 ;;
    --tokens-file) TOKENS_FILE="$2"; shift 2 ;;
    --style) STYLE="$2"; shift 2 ;;
    --style-control) STYLE_CONTROL="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --maps-root) MAPS_ROOT="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

case "$MODEL" in
  diff_planner)         PLANNER="diffusion_planner" ;;
  ego_status_planner)   PLANNER="ego_status_planner" ;;
  stage_vector_planner) PLANNER="stage_vector_planner" ;;
  *) echo "未知 --model: $MODEL（可选 diff_planner/ego_status_planner/stage_vector_planner）" >&2; exit 1 ;;
esac

# 路径解析：未传参/未设环境变量时回退到 paths.local.json
TOKENS_FILE="${COMPARISON_TOKENS_FILE:-${TOKENS_FILE:-$(config_path tokens_file)}}"
DATA_ROOT="${DATA_ROOT:-$(config_path data_root)}"
MAPS_ROOT="${MAPS_ROOT:-$(config_path maps_root)}"

[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "请提供有效的 --ckpt 路径" >&2; exit 1; }
[[ -n "$TOKENS_FILE" && -f "$TOKENS_FILE" ]] || { echo "找不到 tokens_file: $TOKENS_FILE" >&2; exit 1; }
if [[ "$MODEL" == "stage_vector_planner" ]]; then
  [[ -n "$STATS" && -f "$STATS" ]] || { echo "stage_vector_planner 需要 --stats stage_vec_stats.npz" >&2; exit 1; }
fi
# DiffPlanner 需要训练时的 args.json：默认从 checkpoint 同目录 args.json 推导，可显式 --args-file 覆盖。
if [[ "$MODEL" == "diff_planner" ]]; then
  ARGS_FILE="${ARGS_FILE:-$(dirname "$CKPT")/args.json}"
  [[ -f "$ARGS_FILE" ]] || { echo "找不到 DiffPlanner args_file: $ARGS_FILE（可用 --args-file 指定）" >&2; exit 1; }
fi

export COMPARISON_PLANNER="$PLANNER"
export COMPARISON_CKPT="$CKPT"
export COMPARISON_STATS="${STATS:-}"
export COMPARISON_ARGS_FILE="${ARGS_FILE:-}"
export COMPARISON_SPLIT="boston"
export COMPARISON_TOKENS_FILE="$TOKENS_FILE"
export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$MAPS_ROOT"

# 风格覆盖：--style 对 ego_status 开启 with_style（拼接 one-hot），对 stage_vector 走 ±σ。
if [[ -n "$STYLE" ]]; then
  case "$STYLE" in
    aggressive|normal|conservative) ;;
    *) echo "--style 只接受 aggressive|normal|conservative，收到: $STYLE" >&2; exit 1 ;;
  esac
fi
if [[ "$MODEL" == "ego_status_planner" && -n "$STYLE" ]]; then
  export COMPARISON_EGO_WITH_STYLE="true"
else
  export COMPARISON_EGO_WITH_STYLE="false"
fi
export COMPARISON_EGO_STYLE="${STYLE:-normal}"
export COMPARISON_STAGE_STYLE="$STYLE"
export COMPARISON_STAGE_STYLE_CONTROL="${STYLE_CONTROL:-0.0}"

echo "[run_closed_loop] PLANNER=$PLANNER"
echo "[run_closed_loop] CKPT=$CKPT"
[[ -n "$STATS" ]] && echo "[run_closed_loop] STATS=$STATS"
[[ -n "$ARGS_FILE" ]] && echo "[run_closed_loop] ARGS_FILE=$ARGS_FILE"
echo "[run_closed_loop] TOKENS=$TOKENS_FILE"
echo "[run_closed_loop] DATA=$DATA_ROOT"

cd "$REPO_ROOT"
python baseline/run_simulation.py
