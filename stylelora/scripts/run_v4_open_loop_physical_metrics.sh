#!/usr/bin/env bash
# Re-evaluate the existing Ordered Feasible V4 checkpoints and store open-loop
# physical behavior/safety proxy data. This script performs no training.
set -Eeuo pipefail

REPO_ROOT="${CAST_REPO_ROOT:-/home/lisw/programs/Nuplan-Diffusion-Baseline}"
RECORD_ROOT="${CAST_RECORD_ROOT:-/mnt/mydata/lishangwen/Nuplan-Baseline-Record}"
SOURCE_ROOT="${CAST_SOURCE_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS}"
CAST_ROOT="${CAST_OUTPUT_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS3}"
CAST_GPU="${CAST_GPU:-2}"
BATCH_SIZE="${CAST_BATCH_SIZE:-16}"
EVAL_BATCHES="${CAST_EVAL_BATCHES:-2000}"

INPUT_ROOT="$SOURCE_ROOT/INPUTS"
SOURCE_MODELS="$SOURCE_ROOT/MODELS"
CACHE_ROOT="$RECORD_ROOT/CACHE/boston_cache_train_val"
MODEL_ROOT="$CAST_ROOT/MODELS/ORDERED_FEASIBLE_V4"
OPEN_ROOT="$CAST_ROOT/OPEN_LOOP/ORDERED_FEASIBLE_V4"
LOG_ROOT="$CAST_ROOT/LOGS/ORDERED_FEASIBLE_V4"
STATUS_FILE="$CAST_ROOT/STATUS/ORDERED_FEASIBLE_V4_PHYSICAL_OPEN.status"

ARGS_FILE="$INPUT_ROOT/args.json"
BASELINE_CKPT="$SOURCE_MODELS/baseline_diffplanner.pth"
ENCODER_CKPT="$SOURCE_MODELS/preference_encoder.pt"
VAL_MANIFEST="$INPUT_ROOT/weak_preference_val.jsonl"
V4_HIGH="$MODEL_ROOT/conditional_high_ordered_feasible_v4.pt"
V4_LOW="$MODEL_ROOT/conditional_low_ordered_feasible_v4.pt"
V4_ROUTER="$MODEL_ROOT/conditional_router_ordered_feasible_v4.pt"
OPEN_REPORT="$OPEN_ROOT/open_ordered_feasible_v4_physical_full.json"
LOG_FILE="$LOG_ROOT/04_open_physical_full.log"

mkdir -p "$OPEN_ROOT" "$LOG_ROOT" "$(dirname "$STATUS_FILE")"
for required_path in \
  "$ARGS_FILE" \
  "$BASELINE_CKPT" \
  "$ENCODER_CKPT" \
  "$VAL_MANIFEST" \
  "$INPUT_ROOT/latent_bank_val.npy" \
  "$INPUT_ROOT/latent_bank_val_index.jsonl" \
  "$INPUT_ROOT/scene_features_val.npy" \
  "$INPUT_ROOT/scene_features_val_index.jsonl" \
  "$V4_HIGH" \
  "$V4_LOW" \
  "$V4_ROUTER"
do
  if [[ ! -f "$required_path" ]]; then
    echo "missing file: $required_path" >&2
    exit 1
  fi
done
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "missing cache directory: $CACHE_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/nuplan-devkit:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CAST_GPU"
printf 'RUNNING\ngpu=%s\nreport=%s\n' "$CAST_GPU" "$OPEN_REPORT" > "$STATUS_FILE"

set +e
python -u -m stylelora.scripts.evaluate_preference_lora \
  --args-file "$ARGS_FILE" \
  --baseline-checkpoint "$BASELINE_CKPT" \
  --adapter-high "$V4_HIGH" \
  --adapter-low "$V4_LOW" \
  --conditional-router-checkpoint "$V4_ROUTER" \
  --cspq-checkpoint "$ENCODER_CKPT" \
  --manifest "$VAL_MANIFEST" \
  --cache-root "$CACHE_ROOT" \
  --feature-npy "$INPUT_ROOT/scene_features_val.npy" \
  --feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
  --latent-bank "$INPUT_ROOT/latent_bank_val.npy" \
  --latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl" \
  --output-report "$OPEN_REPORT" \
  --sampling-mode full --batch-size "$BATCH_SIZE" --n-eval-batches "$EVAL_BATCHES" \
  --rho-min -1 --rho-max 1 --rho-steps 9 \
  --style-response-epsilon 0.01 \
  --max-ade-cost 0.5 --max-fde-cost 1.0 --max-jerk-cost 2.0 \
  --rank 4 --seed 17 --device cuda:0 \
  2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e

if (( exit_code != 0 )); then
  printf 'FAILED\nexit_code=%s\nlog=%s\n' "$exit_code" "$LOG_FILE" > "$STATUS_FILE"
  exit "$exit_code"
fi
printf 'DONE\nreport=%s\nlog=%s\n' "$OPEN_REPORT" "$LOG_FILE" > "$STATUS_FILE"
echo "[DONE] open-loop physical report: $OPEN_REPORT"
