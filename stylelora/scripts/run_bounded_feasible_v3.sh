#!/usr/bin/env bash
# Bounded V2 + 可行同场景风格配对：标签构建、训练和全量开环验证。
set -euo pipefail

REPO_ROOT="${CAST_REPO_ROOT:-/home/lisw/programs/Nuplan-Diffusion-Baseline}"
RECORD_ROOT="${CAST_RECORD_ROOT:-/mnt/mydata/lishangwen/Nuplan-Baseline-Record}"
SOURCE_ROOT="${CAST_SOURCE_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS}"
CAST_ROOT="${CAST_OUTPUT_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS2}"
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

V2_ROOT="$CAST_ROOT/MODELS/BOUNDED_CONDITIONAL_V2"
V2_HIGH="$V2_ROOT/conditional_high_bounded_v2.pt"
V2_LOW="$V2_ROOT/conditional_low_bounded_v2.pt"
V2_ROUTER="$V2_ROOT/conditional_router_bounded_v2.pt"

PAIR_ROOT="$CAST_ROOT/FEASIBLE_STYLE_PAIRS_V3"
V3_MODEL_ROOT="$CAST_ROOT/MODELS/BOUNDED_FEASIBLE_V3"
V3_OPEN_ROOT="$CAST_ROOT/OPEN_LOOP/BOUNDED_FEASIBLE_V3"
V3_LOG_ROOT="$CAST_ROOT/LOGS/BOUNDED_FEASIBLE_V3"
STATUS_FILE="$CAST_ROOT/STATUS/BOUNDED_FEASIBLE_V3.status"

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

trap 'pipeline_failed "$?" "$LINENO"' ERR

mkdir -p "$PAIR_ROOT" "$V3_MODEL_ROOT" "$V3_OPEN_ROOT" "$V3_LOG_ROOT" "$(dirname "$STATUS_FILE")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/nuplan-devkit:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CAST_GPU"

CURRENT_STEP="0 检查输入"
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
  if [[ ! -f "$required_path" ]]; then
    echo "缺少文件：$required_path" >&2
    false
  fi
done

if (( CAST_START_STEP <= 1 )); then
  CURRENT_STEP="1 生成训练集可行风格配对"
  write_status RUNNING
  python -u -m stylelora.scripts.build_counterfactual_preferences \
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
    --output-bank "$PAIR_ROOT/train_pairs.f32" \
    --output-index "$PAIR_ROOT/train_pairs.jsonl" \
    --rho-magnitudes "0.25,0.5,0.75,1.0" \
    --batch-size 16 --workers 4 --rank 4 --seed 17 --device cuda:0 \
    --min-style-delta 0.001 \
    --max-mean-accel-degradation 0.5 \
    --max-mean-jerk-degradation 2.0 \
    --max-progress-loss-m 2.0 \
    --max-mean-lateral-deviation-m 1.5 \
    2>&1 | tee "$V3_LOG_ROOT/01_build_train_pairs.log"
fi

if (( CAST_START_STEP <= 2 )); then
  CURRENT_STEP="2 生成验证集可行风格配对"
  write_status RUNNING
  python -u -m stylelora.scripts.build_counterfactual_preferences \
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
    --output-bank "$PAIR_ROOT/val_pairs.f32" \
    --output-index "$PAIR_ROOT/val_pairs.jsonl" \
    --rho-magnitudes "0.25,0.5,0.75,1.0" \
    --batch-size 16 --workers 4 --rank 4 --seed 17 --device cuda:0 \
    --min-style-delta 0.001 \
    --max-mean-accel-degradation 0.5 \
    --max-mean-jerk-degradation 2.0 \
    --max-progress-loss-m 2.0 \
    --max-mean-lateral-deviation-m 1.5 \
    2>&1 | tee "$V3_LOG_ROOT/02_build_val_pairs.log"
fi

if (( CAST_START_STEP <= 3 )); then
  CURRENT_STEP="3 从 Bounded V2 继续联合训练"
  write_status RUNNING
  python -u -m stylelora.scripts.train_conditional_preference_lora \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --init-adapter-high "$V2_HIGH" \
    --init-adapter-low "$V2_LOW" \
    --init-router "$V2_ROUTER" \
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
    --cf-pair-bank "$PAIR_ROOT/train_pairs.f32" \
    --cf-pair-index "$PAIR_ROOT/train_pairs.jsonl" \
    --val-cf-pair-bank "$PAIR_ROOT/val_pairs.f32" \
    --val-cf-pair-index "$PAIR_ROOT/val_pairs.jsonl" \
    --output-high "$V3_MODEL_ROOT/conditional_high_feasible_v3.pt" \
    --output-low "$V3_MODEL_ROOT/conditional_low_feasible_v3.pt" \
    --output-router "$V3_MODEL_ROOT/conditional_router_feasible_v3.pt" \
    --steps 3000 --batch-size 32 --workers 4 --lr 1e-4 --rank 4 \
    --lambda-n 1.0 --lambda-z 1.0 --lambda-s 1.0 --lambda-q 1.0 \
    --lambda-dyn 0.1 --lambda-lat 1.0 --lambda-order 1.0 \
    --order-margin-scale 0.25 --min-rho-gap 0.25 \
    --lambda-cf 0.1 --cf-margin 0.001 \
    --lateral-tolerance 0.3 --lateral-topk-ratio 0.2 --lateral-smooth-weight 0.1 \
    --feasible-max-mean-accel-degradation 0.5 \
    --feasible-max-mean-jerk-degradation 2.0 \
    --feasible-max-progress-loss-m 2.0 \
    --feasible-max-mean-lateral-deviation-m 1.5 \
    --val-every 100 --val-batches 50 --seed 17 --device cuda:0 \
    2>&1 | tee "$V3_LOG_ROOT/03_train.log"
fi

if (( CAST_START_STEP <= 4 )); then
  CURRENT_STEP="4 全量开环验证"
  write_status RUNNING
  python -u -m stylelora.scripts.evaluate_preference_lora \
    --args-file "$ARGS_FILE" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --adapter-high "$V3_MODEL_ROOT/conditional_high_feasible_v3.pt" \
    --adapter-low "$V3_MODEL_ROOT/conditional_low_feasible_v3.pt" \
    --conditional-router-checkpoint "$V3_MODEL_ROOT/conditional_router_feasible_v3.pt" \
    --cspq-checkpoint "$ENCODER_CKPT" \
    --manifest "$VAL_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --feature-npy "$INPUT_ROOT/scene_features_val.npy" \
    --feature-index "$INPUT_ROOT/scene_features_val_index.jsonl" \
    --latent-bank "$INPUT_ROOT/latent_bank_val.npy" \
    --latent-bank-index "$INPUT_ROOT/latent_bank_val_index.jsonl" \
    --output-report "$V3_OPEN_ROOT/open_feasible_v3_full.json" \
    --sampling-mode full --batch-size 16 --n-eval-batches 2000 \
    --rho-min -1 --rho-max 1 --rho-steps 9 \
    --rank 4 --seed 17 --device cuda:0 \
    2>&1 | tee "$V3_LOG_ROOT/04_open_full.log"
fi

CURRENT_STEP="全部完成"
write_status DONE
echo "[DONE] Bounded Feasible V3 全流程完成"
echo "[DONE] 状态：$STATUS_FILE"
echo "[DONE] 开环报告：$V3_OPEN_ROOT/open_feasible_v3_full.json"
