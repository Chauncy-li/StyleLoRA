#!/usr/bin/env bash
# FULL 全量线：遍历完整验证 manifest，并生成对应的 C、D 结果。

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"
begin_pipeline "FULL" "$SCRIPT_DIR/run_01_full.sh"

check_prepare() {
  require_pipeline_done "PREPARE"
  for path in \
    "$ARGS_FILE" "$BASELINE_CKPT" "$ENCODER_CKPT" \
    "$HIGH_ADAPTER" "$LOW_ADAPTER" "$GATE_CKPT" "$GATE_REPORT" \
    "$VAL_MANIFEST" "$VAL_FEATURE_NPY" "$VAL_FEATURE_INDEX" \
    "$VAL_LATENT_BANK" "$VAL_LATENT_INDEX" "$GATE_TARGET_VAL"
  do
    require_file "$path"
  done
  require_dir "$CACHE_ROOT"
}

verify_full_outputs() {
  for path in \
    "$OPEN_NOGATE_FULL" "$OPEN_GATE_FULL" \
    "$FULL_ROOT/C_continuous_tuning/table_3_continuous_tuning.csv" \
    "$FULL_ROOT/C_continuous_tuning/figure_2_continuous_style_tuning.pdf" \
    "$FULL_ROOT/D_adaptive_tuning/table_4_adaptive_gate_comparison.csv" \
    "$FULL_ROOT/D_adaptive_tuning/figure_3_adaptive_intensity.pdf"
  do
    require_file "$path"
  done
}

run_stage 1 "检查公共准备产物" "$LOG_ROOT/FULL/01_check_prepare.log" check_prepare

FULL_OPEN_COMMON=(
  --args-file "$ARGS_FILE"
  --baseline-checkpoint "$BASELINE_CKPT"
  --adapter-high "$HIGH_ADAPTER"
  --adapter-low "$LOW_ADAPTER"
  --cspq-checkpoint "$ENCODER_CKPT"
  --manifest "$VAL_MANIFEST"
  --cache-root "$CACHE_ROOT"
  --feature-npy "$VAL_FEATURE_NPY"
  --feature-index "$VAL_FEATURE_INDEX"
  --latent-bank "$VAL_LATENT_BANK"
  --latent-bank-index "$VAL_LATENT_INDEX"
  --sampling-mode full
  --batch-size 16 --n-eval-batches 2000
  --rho-min -1 --rho-max 1 --rho-steps 9
  --rank 4 --seed 17 --device cuda:0
)

run_stage 2 "全量开环：关闭门控" "$LOG_ROOT/FULL/02_open_no_gate_full.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${FULL_OPEN_COMMON[@]}" --output-report "$OPEN_NOGATE_FULL"

run_stage 3 "全量开环：启用场景门控" "$LOG_ROOT/FULL/03_open_gate_full.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${FULL_OPEN_COMMON[@]}" \
    --enable-scene-gate --scene-gate-checkpoint "$GATE_CKPT" \
    --output-report "$OPEN_GATE_FULL"

run_stage 4 "生成全量 C 连续风格调节表图" "$LOG_ROOT/FULL/04_section_c.log" \
  python -u -m stylelora.eval.section_c_continuous_tuning \
    --output-root "$FULL_ROOT" --cast-report "$OPEN_NOGATE_FULL"

run_stage 5 "生成全量 D 自适应门控表图" "$LOG_ROOT/FULL/05_section_d.log" \
  python -u -m stylelora.eval.section_d_adaptive_tuning \
    --output-root "$FULL_ROOT" --gate-checkpoint "$GATE_CKPT" \
    --gate-report "$GATE_REPORT" --val-targets "$GATE_TARGET_VAL" \
    --val-feature-npy "$VAL_FEATURE_NPY" --val-feature-index "$VAL_FEATURE_INDEX" \
    --adaptive-open-report "$OPEN_GATE_FULL" \
    --comparison-report "Without adaptive gate=$OPEN_NOGATE_FULL" \
    --batch-size 512 --device cuda:0

run_stage 6 "核对 FULL 开环产物" "$LOG_ROOT/FULL/06_verify.log" verify_full_outputs

finish_pipeline
