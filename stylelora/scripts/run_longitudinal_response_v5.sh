#!/usr/bin/env bash
# Optional V5 short fine-tuning from V4 with a physical longitudinal-response margin.
set -Eeuo pipefail

REPO_ROOT="${CAST_REPO_ROOT:-/home/lisw/programs/Nuplan-Diffusion-Baseline}"
RECORD_ROOT="${CAST_RECORD_ROOT:-/mnt/mydata/lishangwen/Nuplan-Baseline-Record}"
SOURCE_ROOT="${CAST_SOURCE_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS}"
CAST_ROOT="${CAST_OUTPUT_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS3}"
CAST_GPU="${CAST_GPU:-2}"
CAST_START_STEP="${CAST_START_STEP:-1}"

INPUT_ROOT="$SOURCE_ROOT/INPUTS"
SOURCE_MODELS="$SOURCE_ROOT/MODELS"
CACHE_ROOT="$RECORD_ROOT/CACHE/boston_cache_train_val"

ARGS_FILE="$INPUT_ROOT/args.json"
BASELINE_CKPT="$SOURCE_MODELS/baseline_diffplanner.pth"
ENCODER_CKPT="$SOURCE_MODELS/preference_encoder.pt"
TRAIN_MANIFEST="$INPUT_ROOT/weak_preference_train.jsonl"
VAL_MANIFEST="$INPUT_ROOT/weak_preference_val.jsonl"

V4_ROOT="$CAST_ROOT/MODELS/ORDERED_FEASIBLE_V4"
V4_HIGH="$V4_ROOT/conditional_high_ordered_feasible_v4.pt"
V4_LOW="$V4_ROOT/conditional_low_ordered_feasible_v4.pt"
V4_ROUTER="$V4_ROOT/conditional_router_ordered_feasible_v4.pt"

MODEL_ROOT="$CAST_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5"
OPEN_ROOT="$CAST_ROOT/OPEN_LOOP/LONGITUDINAL_RESPONSE_V5"
FIGURE_ROOT="$OPEN_ROOT/GRID_BALANCED_10"
LOG_ROOT="$CAST_ROOT/LOGS/LONGITUDINAL_RESPONSE_V5"
STATUS_FILE="$CAST_ROOT/STATUS/LONGITUDINAL_RESPONSE_V5.status"

V5_HIGH="$MODEL_ROOT/conditional_high_longitudinal_response_v5.pt"
V5_LOW="$MODEL_ROOT/conditional_low_longitudinal_response_v5.pt"
V5_ROUTER="$MODEL_ROOT/conditional_router_longitudinal_response_v5.pt"
OPEN_REPORT="$OPEN_ROOT/open_longitudinal_response_v5_full.json"

CURRENT_STEP="initialization"

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
  echo "[FAILED] step: $CURRENT_STEP"
  echo "[FAILED] line: $line_number; exit code: $exit_code"
  echo "[FAILED] resume with CAST_START_STEP set to the failed step."
  exit "$exit_code"
}

run_step() {
  local number="$1"
  local description="$2"
  local log_file="$3"
  shift 3
  if (( number < CAST_START_STEP )); then
    echo "[skip] step $number: $description"
    return 0
  fi
  CURRENT_STEP="$number $description"
  write_status RUNNING
  echo "[start] step $number: $description"
  "$@" 2>&1 | tee "$log_file"
  echo "[done] step $number: $description"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing file: $1" >&2
    return 1
  fi
}

trap 'pipeline_failed "$?" "$LINENO"' ERR

mkdir -p "$MODEL_ROOT" "$OPEN_ROOT" "$FIGURE_ROOT" "$LOG_ROOT" "$(dirname "$STATUS_FILE")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/nuplan-devkit:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CAST_GPU"

CURRENT_STEP="0 input validation"
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
  "$V4_HIGH" \
  "$V4_LOW" \
  "$V4_ROUTER"
do
  require_file "$required_path"
done
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "missing cache directory: $CACHE_ROOT" >&2
  false
fi

run_step 1 "focused response-loss tests" "$LOG_ROOT/01_tests.log" \
  python -m pytest stylelora/tests/test_conditional_preference_lora.py -q

run_step 2 "short V5 longitudinal-response fine-tuning from V4" "$LOG_ROOT/02_train.log" \
  python -u -m stylelora.scripts.train_conditional_preference_lora \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --init-adapter-high "$V4_HIGH" \
    --init-adapter-low "$V4_LOW" \
    --init-router "$V4_ROUTER" \
    --cspq-checkpoint "$ENCODER_CKPT" \
    --manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --latent-bank "$INPUT_ROOT/latent_bank_train.npy" \
    --latent-bank-index "$INPUT_ROOT/latent_bank_train_index.jsonl" \
    --feature-npy "$INPUT_ROOT/scene_features_train.npy" \
    --feature-index "$INPUT_ROOT/scene_features_train_index.jsonl" \
    --val-latent-bank "$INPUT_ROOT/latent_bank_val.npy" \
    --val-latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl" \
    --val-feature-npy "$INPUT_ROOT/scene_features_val.npy" \
    --val-feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
    --output-high "$V5_HIGH" \
    --output-low "$V5_LOW" \
    --output-router "$V5_ROUTER" \
    --steps 1200 --batch-size 32 --workers 4 --lr 5e-5 --rank 4 \
    --lambda-n 1.0 --lambda-z 1.0 --lambda-s 1.0 --lambda-q 1.0 \
    --lambda-dyn 0.1 --lambda-lat 1.0 \
    --lambda-order 0.25 --order-margin-scale 0.25 --min-rho-gap 0.25 \
    --order-pair-mode mixed_adjacent \
    --order-rho-grid=-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1 \
    --local-order-ratio 0.75 \
    --lambda-feasibility 0.1 \
    --lambda-response 0.5 \
    --response-margin-m-per-rho 1.0 \
    --response-min-baseline-progress-m 3.0 \
    --lateral-tolerance 0.3 --lateral-topk-ratio 0.2 --lateral-smooth-weight 0.1 \
    --feasible-max-mean-accel-degradation 0.5 \
    --feasible-max-mean-jerk-degradation 2.0 \
    --feasible-max-progress-loss-m 2.0 \
    --feasible-max-mean-lateral-deviation-m 1.5 \
    --val-every 100 --val-batches 50 --seed 17 --device cuda:0

require_file "$V5_HIGH"
require_file "$V5_LOW"
require_file "$V5_ROUTER"

run_step 3 "full nine-rho open-loop evaluation" "$LOG_ROOT/03_open_full.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --adapter-high "$V5_HIGH" \
    --adapter-low "$V5_LOW" \
    --conditional-router-checkpoint "$V5_ROUTER" \
    --cspq-checkpoint "$ENCODER_CKPT" \
    --manifest "$VAL_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --feature-npy "$INPUT_ROOT/scene_features_val.npy" \
    --feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
    --latent-bank "$INPUT_ROOT/latent_bank_val.npy" \
    --latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl" \
    --output-report "$OPEN_REPORT" \
    --sampling-mode full --batch-size 16 --n-eval-batches 2000 \
    --rho-min -1 --rho-max 1 --rho-steps 9 \
    --style-response-epsilon 0.01 \
    --max-ade-cost 0.5 --max-fde-cost 1.0 --max-jerk-cost 2.0 \
    --rank 4 --seed 17 --device cuda:0

require_file "$OPEN_REPORT"

run_step 4 "ten balanced 3x3 trajectory figures" "$LOG_ROOT/04_grid.log" \
  python -u -m stylelora.scripts.plot_open_loop_grid \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --adapter-high "$V5_HIGH" \
    --adapter-low "$V5_LOW" \
    --conditional-router-checkpoint "$V5_ROUTER" \
    --manifest "$VAL_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --feature-npy "$INPUT_ROOT/scene_features_val.npy" \
    --feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
    --latent-bank "$INPUT_ROOT/latent_bank_val.npy" \
    --latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl" \
    --output-dir "$FIGURE_ROOT" \
    --num-scenes 10 --scene-type balanced \
    --rank 4 --seed 17 --dpi 180 --device cuda:0

CURRENT_STEP="complete"
write_status DONE
echo "[DONE] model directory: $MODEL_ROOT"
echo "[DONE] open-loop directory: $OPEN_ROOT"
