#!/usr/bin/env bash
# Bounded V2 条件 LoRA + 纯场景门控。
# LoRA 与条件路由保持冻结；门控仅裁剪一次 rho，不启用候选验证或 baseline 回退。
set -Eeuo pipefail

REPO_ROOT="${CAST_REPO_ROOT:-/home/lisw/programs/Nuplan-Diffusion-Baseline}"
RECORD_ROOT="${CAST_RECORD_ROOT:-/mnt/mydata/lishangwen/Nuplan-Baseline-Record}"
SOURCE_ROOT="${CAST_SOURCE_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS}"
CAST_ROOT="${CAST_OUTPUT_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS2}"
CAST_GPU="${CAST_GPU:-2}"
CAST_START_STEP="${CAST_START_STEP:-1}"
CAST_RUN_CLOSED_LOOP="${CAST_RUN_CLOSED_LOOP:-0}"

INPUT_ROOT="$SOURCE_ROOT/INPUTS"
SOURCE_MODELS="$SOURCE_ROOT/MODELS"
CACHE_ROOT="$RECORD_ROOT/CACHE/boston_cache_train_val"

ARGS_FILE="$INPUT_ROOT/args.json"
BASELINE_CKPT="$SOURCE_MODELS/baseline_diffplanner.pth"
ENCODER_CKPT="$SOURCE_MODELS/preference_encoder.pt"
TRAIN_MANIFEST="$INPUT_ROOT/weak_preference_train.jsonl"
VAL_MANIFEST="$INPUT_ROOT/weak_preference_val.jsonl"

V2_ROOT="$CAST_ROOT/MODELS/BOUNDED_CONDITIONAL_V2"
V2_HIGH="$V2_ROOT/conditional_high_bounded_v2.pt"
V2_LOW="$V2_ROOT/conditional_low_bounded_v2.pt"
V2_ROUTER="$V2_ROOT/conditional_router_bounded_v2.pt"

TARGET_ROOT="$CAST_ROOT/GATE_TARGETS/V2_PURE_GATE"
MODEL_ROOT="$CAST_ROOT/MODELS/V2_PURE_GATE"
OPEN_ROOT="$CAST_ROOT/OPEN_LOOP/V2_PURE_GATE"
CLOSED_ROOT="$CAST_ROOT/CLOSED_LOOP/V2_PURE_GATE"
LOG_ROOT="$CAST_ROOT/LOGS/V2_PURE_GATE"
STATUS_FILE="$CAST_ROOT/STATUS/V2_PURE_GATE.status"

TRAIN_TARGETS="$TARGET_ROOT/gate_targets_train.jsonl"
VAL_TARGETS="$TARGET_ROOT/gate_targets_val.jsonl"
GATE_CKPT="$MODEL_ROOT/scene_gate_v2_pure_3k_ow2.pt"
OPEN_NO_GATE="$OPEN_ROOT/open_v2_no_gate_full.json"
OPEN_GATE="$OPEN_ROOT/open_v2_pure_gate_full.json"

NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps}"
TOKENS_FILE="${CAST_TOKENS_FILE:-$SOURCE_ROOT/MAIN_RESULTS/CLOSED_LOOP/05_COMMON_SUBSET/common_valid_tokens.json}"

CURRENT_STEP="初始化"

write_status() {
  local state="$1"
  printf '%s\nstep=%s\ngpu=%s\nstart_step=%s\n' \
    "$state" "$CURRENT_STEP" "$CAST_GPU" "$CAST_START_STEP" > "$STATUS_FILE"
}

pipeline_failed() {
  local exit_code="$1"
  local line_number="$2"
  set +e
  printf 'FAILED\nstep=%s\nline=%s\nexit_code=%s\ngpu=%s\n' \
    "$CURRENT_STEP" "$line_number" "$exit_code" "$CAST_GPU" > "$STATUS_FILE"
  echo "[FAILED] 步骤：$CURRENT_STEP"
  echo "[FAILED] 行号：$line_number，退出码：$exit_code"
  echo "[FAILED] 修复后设置 CAST_START_STEP 从当前步骤继续。"
  exit "$exit_code"
}

run_step() {
  local number="$1"
  local description="$2"
  local log_file="$3"
  shift 3
  if (( number < CAST_START_STEP )); then
    echo "[skip] 步骤 $number：$description"
    return 0
  fi
  CURRENT_STEP="$number $description"
  write_status RUNNING
  echo "[start] 步骤 $number：$description"
  "$@" 2>&1 | tee "$log_file"
  echo "[done] 步骤 $number：$description"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "缺少文件：$1" >&2
    return 1
  fi
}

trap 'pipeline_failed "$?" "$LINENO"' ERR

mkdir -p "$TARGET_ROOT" "$MODEL_ROOT" "$OPEN_ROOT" "$CLOSED_ROOT" "$LOG_ROOT" "$(dirname "$STATUS_FILE")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/nuplan-devkit:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CAST_GPU"

CURRENT_STEP="0 检查 V2 与公共输入"
write_status RUNNING
for required_path in \
  "$ARGS_FILE" \
  "$BASELINE_CKPT" \
  "$ENCODER_CKPT" \
  "$TRAIN_MANIFEST" \
  "$VAL_MANIFEST" \
  "$INPUT_ROOT/latent_bank_train.npy" \
  "$INPUT_ROOT/latent_bank_train_index.jsonl" \
  "$INPUT_ROOT/latent_bank_val.npy" \
  "$INPUT_ROOT/latent_bank_val_index.jsonl" \
  "$INPUT_ROOT/scene_features_train.npy" \
  "$INPUT_ROOT/scene_features_train_index.jsonl" \
  "$INPUT_ROOT/scene_features_val.npy" \
  "$INPUT_ROOT/scene_features_val_index.jsonl" \
  "$V2_HIGH" \
  "$V2_LOW" \
  "$V2_ROUTER"
do
  require_file "$required_path"
done
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "缺少目录：$CACHE_ROOT" >&2
  false
fi
if [[ "$CAST_RUN_CLOSED_LOOP" == "1" ]]; then
  require_file "$TOKENS_FILE"
  if [[ ! -d "$NUPLAN_DATA_ROOT" ]]; then
    echo "缺少闭环 DB 目录：$NUPLAN_DATA_ROOT" >&2
    false
  fi
  if [[ ! -d "$NUPLAN_MAPS_ROOT" ]]; then
    echo "缺少地图目录：$NUPLAN_MAPS_ROOT" >&2
    false
  fi
fi

run_step 1 "使用冻结 V2 生成训练门控标签" "$LOG_ROOT/01_build_train_targets.log" \
  python -u -m stylelora.scripts.build_scene_gate_targets \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --adapter-high "$V2_HIGH" \
    --adapter-low "$V2_LOW" \
    --conditional-router-checkpoint "$V2_ROUTER" \
    --cspq-checkpoint "$ENCODER_CKPT" \
    --manifest "$TRAIN_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --latent-bank "$INPUT_ROOT/latent_bank_train.npy" \
    --latent-bank-index "$INPUT_ROOT/latent_bank_train_index.jsonl" \
    --feature-npy "$INPUT_ROOT/scene_features_train.npy" \
    --feature-index "$INPUT_ROOT/scene_features_train_index.jsonl" \
    --output "$TRAIN_TARGETS" \
    --rho-magnitudes "0.25,0.5,0.75,1.0" \
    --batch-size 16 --workers 4 --rank 4 --seed 17 --device cuda:0

run_step 2 "使用冻结 V2 生成验证门控标签" "$LOG_ROOT/02_build_val_targets.log" \
  python -u -m stylelora.scripts.build_scene_gate_targets \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --adapter-high "$V2_HIGH" \
    --adapter-low "$V2_LOW" \
    --conditional-router-checkpoint "$V2_ROUTER" \
    --cspq-checkpoint "$ENCODER_CKPT" \
    --manifest "$VAL_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --latent-bank "$INPUT_ROOT/latent_bank_val.npy" \
    --latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl" \
    --feature-npy "$INPUT_ROOT/scene_features_val.npy" \
    --feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
    --output "$VAL_TARGETS" \
    --rho-magnitudes "0.25,0.5,0.75,1.0" \
    --batch-size 16 --workers 4 --rank 4 --seed 17 --device cuda:0

run_step 3 "训练独立纯门控，V2 参数保持冻结" "$LOG_ROOT/03_train_gate.log" \
  python -u -m stylelora.scripts.train_scene_gate \
    --train-targets "$TRAIN_TARGETS" \
    --train-feature-npy "$INPUT_ROOT/scene_features_train.npy" \
    --train-feature-index "$INPUT_ROOT/scene_features_train_index.jsonl" \
    --val-targets "$VAL_TARGETS" \
    --val-feature-npy "$INPUT_ROOT/scene_features_val.npy" \
    --val-feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
    --output "$GATE_CKPT" \
    --hidden-dim 64 --steps 3000 --batch-size 64 --workers 4 \
    --lr 1e-3 --weight-decay 1e-4 --val-every 50 \
    --minimum-target-std 0.02 --overestimate-weight 2 \
    --seed 17 --device cuda:0

OPEN_COMMON=(
  --args-file "$ARGS_FILE"
  --baseline-checkpoint "$BASELINE_CKPT"
  --adapter-high "$V2_HIGH"
  --adapter-low "$V2_LOW"
  --conditional-router-checkpoint "$V2_ROUTER"
  --cspq-checkpoint "$ENCODER_CKPT"
  --manifest "$VAL_MANIFEST"
  --cache-root "$CACHE_ROOT"
  --feature-npy "$INPUT_ROOT/scene_features_val.npy"
  --feature-index "$INPUT_ROOT/scene_features_val_index.jsonl"
  --latent-bank "$INPUT_ROOT/latent_bank_val.npy"
  --latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl"
  --sampling-mode full --batch-size 16 --n-eval-batches 2000
  --rho-min -1 --rho-max 1 --rho-steps 9
  --rank 4 --seed 17 --device cuda:0
)

run_step 4 "全量开环：V2 无门控参照" "$LOG_ROOT/04_open_no_gate.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${OPEN_COMMON[@]}" \
    --output-report "$OPEN_NO_GATE"

run_step 5 "全量开环：V2 纯门控" "$LOG_ROOT/05_open_pure_gate.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${OPEN_COMMON[@]}" \
    --enable-scene-gate --scene-gate-checkpoint "$GATE_CKPT" \
    --output-report "$OPEN_GATE"

if [[ "$CAST_RUN_CLOSED_LOOP" == "1" ]]; then
  CLOSED_COMMON=(
    --args-file "$ARGS_FILE"
    --baseline-checkpoint "$BASELINE_CKPT"
    --high-adapter "$V2_HIGH"
    --low-adapter "$V2_LOW"
    --conditional-router-checkpoint "$V2_ROUTER"
    --data-root "$NUPLAN_DATA_ROOT"
    --maps-root "$NUPLAN_MAPS_ROOT"
    --scenario-filter boston
    --scenario-tokens-file "$TOKENS_FILE"
    --rhos=-1,-0.5,0,0.5,1
    --rank 4 --challenge closed_loop_nonreactive_agents
    --device cuda --worker sequential --allow-partial-scenarios --skip-existing
  )
  run_step 6 "闭环：V2 无门控参照" "$LOG_ROOT/06_closed_no_gate.log" \
    python -u -m stylelora.scripts.evaluate_closed_loop \
      "${CLOSED_COMMON[@]}" \
      --output-root "$CLOSED_ROOT/NO_GATE"

  # 不传 --enable-bounded-style：仅由门控裁剪 rho，每个规划周期只生成一个风格轨迹。
  run_step 7 "闭环：V2 纯门控" "$LOG_ROOT/07_closed_pure_gate.log" \
    python -u -m stylelora.scripts.evaluate_closed_loop \
      "${CLOSED_COMMON[@]}" \
      --enable-scene-gate --scene-gate-checkpoint "$GATE_CKPT" \
      --output-root "$CLOSED_ROOT/PURE_GATE"
else
  echo "[skip] CAST_RUN_CLOSED_LOOP=$CAST_RUN_CLOSED_LOOP，暂不运行闭环。"
fi

CURRENT_STEP="全部完成"
write_status DONE
echo "[DONE] V2 + 纯场景门控流程完成。"
echo "[DONE] 门控模型：$GATE_CKPT"
echo "[DONE] 无门控开环：$OPEN_NO_GATE"
echo "[DONE] 纯门控开环：$OPEN_GATE"
echo "[DONE] 状态文件：$STATUS_FILE"
