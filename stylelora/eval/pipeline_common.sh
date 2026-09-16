#!/usr/bin/env bash
# CAST EAAI 实验流水线的唯一公共路径与错误处理定义。

set -Eeuo pipefail

EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CAST_REPO_ROOT:-$(cd "$EVAL_SCRIPT_DIR/../.." && pwd)}"
RECORD_ROOT="${CAST_RECORD_ROOT:-/mnt/mydata/lishangwen/Nuplan-Baseline-Record}"
CAST_ROOT="${CAST_OUTPUT_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS}"

# 只读来源：公共准备脚本会把本轮需要的模型和中间输入复制到 CAST_ROOT。
SOURCE_ARGS_FILE="$RECORD_ROOT/args.json"
SOURCE_BASELINE_CKPT="$RECORD_ROOT/best_model-epoch_116-train_loss_0.0701.pth"
SOURCE_PREF_ROOT="$RECORD_ROOT/STYLE_LORA_PREFERENCE"
SOURCE_ENCODER_ROOT="$RECORD_ROOT/STYLE_LORA_ENCODER"
SOURCE_LORA_ROOT="$RECORD_ROOT/STYLE_LORA_CONTINUOUS_LORA_DYN_Q_LAT_TOPK"
SOURCE_GATE_TARGET_ROOT="$RECORD_ROOT/STYLE_LORA_SCENE_GATE"
SOURCE_GATE_ROOT="$RECORD_ROOT/STYLE_LORA_SCENE_GATE_3K_OW2"

TEST_CACHE_ROOT="$RECORD_ROOT/CACHE/boston_cache_test_simu"
TEST_CACHE_LIST="$RECORD_ROOT/CACHE/boston_cache_test_simu_list.json"
TEST_CACHE_MANIFEST="$RECORD_ROOT/CACHE/boston_cache_test_simu_manifest.json"
CACHE_ROOT="$RECORD_ROOT/CACHE/boston_cache_train_val"
NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps}"

# 本轮统一目录。所有新模型、日志、开环、闭环和论文图表都位于 CAST_ROOT。
MODELS_ROOT="$CAST_ROOT/MODELS"
INPUTS_ROOT="$CAST_ROOT/INPUTS"
COMMON_ROOT="$CAST_ROOT/COMMON_RESULTS"
MAIN_ROOT="$CAST_ROOT/MAIN_RESULTS"
FULL_ROOT="$CAST_ROOT/FULL_RESULTS"
LOG_ROOT="$CAST_ROOT/LOGS"
STATUS_ROOT="$CAST_ROOT/STATUS"

ARGS_FILE="$INPUTS_ROOT/args.json"
TRAIN_MANIFEST="$INPUTS_ROOT/weak_preference_train.jsonl"
VAL_MANIFEST="$INPUTS_ROOT/weak_preference_val.jsonl"
TRAIN_FEATURE_NPY="$INPUTS_ROOT/scene_features_train.npy"
TRAIN_FEATURE_INDEX="$INPUTS_ROOT/scene_features_train_index.jsonl"
VAL_FEATURE_NPY="$INPUTS_ROOT/scene_features_val.npy"
VAL_FEATURE_INDEX="$INPUTS_ROOT/scene_features_val_index.jsonl"
TRAIN_LATENT_BANK="$INPUTS_ROOT/latent_bank_train.npy"
TRAIN_LATENT_INDEX="$INPUTS_ROOT/latent_bank_train_index.jsonl"
VAL_LATENT_BANK="$INPUTS_ROOT/latent_bank_val.npy"
VAL_LATENT_INDEX="$INPUTS_ROOT/latent_bank_val_index.jsonl"
GATE_TARGET_VAL="$INPUTS_ROOT/gate_targets_val.jsonl"
ENCODER_REPORT="$INPUTS_ROOT/evaluate_encoder.json"

BASELINE_CKPT="$MODELS_ROOT/baseline_diffplanner.pth"
ENCODER_CKPT="$MODELS_ROOT/preference_encoder.pt"
HIGH_ADAPTER="$MODELS_ROOT/preference_adapter_high.pt"
LOW_ADAPTER="$MODELS_ROOT/preference_adapter_low.pt"
GATE_CKPT="$MODELS_ROOT/scene_gate.pt"
GATE_REPORT="$MODELS_ROOT/scene_gate.report.json"
SCALAR_HIGH_ADAPTER="$MODELS_ROOT/ABLATIONS/preference_adapter_high_scalar_only.pt"
SCALAR_LOW_ADAPTER="$MODELS_ROOT/ABLATIONS/preference_adapter_low_scalar_only.pt"
NO_LAT_HIGH_ADAPTER="$MODELS_ROOT/ABLATIONS/preference_adapter_high_no_lateral.pt"
NO_LAT_LOW_ADAPTER="$MODELS_ROOT/ABLATIONS/preference_adapter_low_no_lateral.pt"

PREDICTIONS="$COMMON_ROOT/STYLE_REPRESENTATION/val_predictions.jsonl"
TEST_SCENE_SPLIT="$COMMON_ROOT/CLOSED_LOOP_SCENES/test_scene_split"
TOKENS_300="$COMMON_ROOT/CLOSED_LOOP_SCENES/closed_loop_candidates_boston_300.json"

MAIN_OPEN_ROOT="$MAIN_ROOT/OPEN_LOOP"
OPEN_NOGATE="$MAIN_OPEN_ROOT/open_no_gate_balanced.json"
OPEN_GATE="$MAIN_OPEN_ROOT/open_scene_gate_balanced.json"
OPEN_SCALAR="$MAIN_OPEN_ROOT/open_scalar_only_balanced.json"
OPEN_NO_LAT="$MAIN_OPEN_ROOT/open_no_lateral_balanced.json"

FULL_OPEN_ROOT="$FULL_ROOT/OPEN_LOOP"
OPEN_NOGATE_FULL="$FULL_OPEN_ROOT/open_no_gate_full.json"
OPEN_GATE_FULL="$FULL_OPEN_ROOT/open_scene_gate_full.json"

CLOSED_ROOT="$MAIN_ROOT/CLOSED_LOOP"
CLOSED_SCALAR="$CLOSED_ROOT/01_SCALAR_ONLY"
CLOSED_NO_LAT="$CLOSED_ROOT/02_NO_LATERAL"
CLOSED_NOGATE="$CLOSED_ROOT/03_NO_GATE"
CLOSED_GATE="$CLOSED_ROOT/04_FULL_CAST"
COMMON_CLOSED_ROOT="$CLOSED_ROOT/05_COMMON_SUBSET"
COMMON_VALID_TOKENS="$COMMON_CLOSED_ROOT/common_valid_tokens.json"

CLOSED_SCALAR_RAW_REPORT="$CLOSED_SCALAR/closed_loop_lora_report.json"
CLOSED_NO_LAT_RAW_REPORT="$CLOSED_NO_LAT/closed_loop_lora_report.json"
CLOSED_NOGATE_RAW_REPORT="$CLOSED_NOGATE/closed_loop_lora_report.json"
CLOSED_GATE_RAW_REPORT="$CLOSED_GATE/closed_loop_lora_report.json"
CLOSED_SCALAR_REPORT="$COMMON_CLOSED_ROOT/scalar_only_alignment.filtered.json"
CLOSED_NO_LAT_REPORT="$COMMON_CLOSED_ROOT/no_lateral_constraint.filtered.json"
CLOSED_NOGATE_REPORT="$COMMON_CLOSED_ROOT/without_adaptive_gate.filtered.json"
CLOSED_GATE_REPORT="$COMMON_CLOSED_ROOT/full_cast.filtered.json"

CAST_GPU_ID="${CAST_GPU:-${CUDA_VISIBLE_DEVICES:-2}}"
CAST_START_STEP="${CAST_START_STEP:-1}"

PIPELINE_NAME=""
PIPELINE_SCRIPT=""
PIPELINE_STATUS_FILE=""
CURRENT_STEP="初始化"

make_pipeline_directories() {
  mkdir -p \
    "$MODELS_ROOT/ABLATIONS" "$INPUTS_ROOT" \
    "$COMMON_ROOT/STYLE_REPRESENTATION" "$COMMON_ROOT/CLOSED_LOOP_SCENES" \
    "$MAIN_OPEN_ROOT" "$FULL_OPEN_ROOT" "$CLOSED_ROOT" \
    "$MAIN_ROOT" "$FULL_ROOT" "$LOG_ROOT" "$STATUS_ROOT"
}

begin_pipeline() {
  PIPELINE_NAME="$1"
  PIPELINE_SCRIPT="$2"
  make_pipeline_directories
  PIPELINE_STATUS_FILE="$STATUS_ROOT/${PIPELINE_NAME}.status"
  printf 'RUNNING\nscript=%s\nstart_step=%s\ngpu=%s\n' \
    "$PIPELINE_SCRIPT" "$CAST_START_STEP" "$CAST_GPU_ID" > "$PIPELINE_STATUS_FILE"
  trap 'pipeline_failed "$?" "$LINENO"' ERR
  export CUDA_VISIBLE_DEVICES="$CAST_GPU_ID"
  cd "$REPO_ROOT"
  echo "[$PIPELINE_NAME] 使用 GPU: $CUDA_VISIBLE_DEVICES"
  echo "[$PIPELINE_NAME] 输出根目录: $CAST_ROOT"
}

pipeline_failed() {
  local exit_code="$1"
  local line_number="$2"
  set +e
  printf 'FAILED\nstep=%s\nline=%s\nexit_code=%s\nscript=%s\n' \
    "$CURRENT_STEP" "$line_number" "$exit_code" "$PIPELINE_SCRIPT" > "$PIPELINE_STATUS_FILE"
  echo
  echo "[FAILED] 流水线：$PIPELINE_NAME"
  echo "[FAILED] 步骤：$CURRENT_STEP"
  echo "[FAILED] 脚本行号：$line_number，退出码：$exit_code"
  echo "[FAILED] 修复后按 README 对应步骤继续；也可设置 CAST_START_STEP 后重跑："
  echo "CAST_GPU=$CAST_GPU_ID CAST_START_STEP=${CURRENT_STEP%% *} bash $PIPELINE_SCRIPT"
  exit "$exit_code"
}

finish_pipeline() {
  printf 'DONE\nscript=%s\ngpu=%s\n' "$PIPELINE_SCRIPT" "$CAST_GPU_ID" > "$PIPELINE_STATUS_FILE"
  echo "[$PIPELINE_NAME] 全部步骤完成。"
}

run_stage() {
  local stage_number="$1"
  local description="$2"
  local log_file="$3"
  shift 3
  if (( stage_number < CAST_START_STEP )); then
    echo "[$PIPELINE_NAME] 跳过步骤 $stage_number：$description"
    return 0
  fi
  CURRENT_STEP="$stage_number $description"
  mkdir -p "$(dirname "$log_file")"
  printf 'RUNNING\nstep=%s\nscript=%s\ngpu=%s\n' \
    "$CURRENT_STEP" "$PIPELINE_SCRIPT" "$CAST_GPU_ID" > "$PIPELINE_STATUS_FILE"
  echo
  echo "[$PIPELINE_NAME] 开始步骤 $stage_number：$description"
  "$@" 2>&1 | tee "$log_file"
  echo "[$PIPELINE_NAME] 完成步骤 $stage_number：$description"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "缺少文件：$path" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "缺少目录：$path" >&2
    return 1
  fi
}

copy_required() {
  local source="$1"
  local destination="$2"
  require_file "$source"
  mkdir -p "$(dirname "$destination")"
  cp -f -- "$source" "$destination"
  cmp -s -- "$source" "$destination"
  echo "已归档：$destination"
}

require_pipeline_done() {
  local name="$1"
  local status_file="$STATUS_ROOT/${name}.status"
  require_file "$status_file"
  grep -q '^DONE$' "$status_file"
}
