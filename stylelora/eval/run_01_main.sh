#!/usr/bin/env bash
# MAIN 主线：平衡开环、四组消融闭环、共同场景过滤以及论文主图表。

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"
begin_pipeline "MAIN" "$SCRIPT_DIR/run_01_main.sh"

check_prepare() {
  require_pipeline_done "PREPARE"
  for path in \
    "$ARGS_FILE" "$BASELINE_CKPT" "$ENCODER_CKPT" \
    "$HIGH_ADAPTER" "$LOW_ADAPTER" "$GATE_CKPT" "$GATE_REPORT" \
    "$SCALAR_HIGH_ADAPTER" "$SCALAR_LOW_ADAPTER" \
    "$NO_LAT_HIGH_ADAPTER" "$NO_LAT_LOW_ADAPTER" \
    "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$VAL_FEATURE_NPY" "$VAL_FEATURE_INDEX" \
    "$VAL_LATENT_BANK" "$VAL_LATENT_INDEX" "$GATE_TARGET_VAL" \
    "$PREDICTIONS" "$TOKENS_300"
  do
    require_file "$path"
  done
  require_dir "$CACHE_ROOT"
  require_dir "$NUPLAN_DATA_ROOT"
  require_dir "$NUPLAN_MAPS_ROOT"
}

verify_main_outputs() {
  for path in \
    "$MAIN_ROOT/A_experimental_setup/table_1_experimental_setup.csv" \
    "$MAIN_ROOT/B_style_representation/table_2_style_representation.csv" \
    "$MAIN_ROOT/B_style_representation/figure_1_style_representation.pdf" \
    "$MAIN_ROOT/C_continuous_tuning/table_3_continuous_tuning.csv" \
    "$MAIN_ROOT/C_continuous_tuning/figure_2_continuous_style_tuning.pdf" \
    "$MAIN_ROOT/D_adaptive_tuning/table_4_adaptive_gate_comparison.csv" \
    "$MAIN_ROOT/D_adaptive_tuning/figure_3_adaptive_intensity.pdf" \
    "$MAIN_ROOT/E_closed_loop/table_5_closed_loop_performance.csv" \
    "$MAIN_ROOT/E_closed_loop/figure_4_closed_loop_performance.pdf" \
    "$MAIN_ROOT/F_ablation_efficiency/table_6_ablation_and_efficiency.csv" \
    "$MAIN_ROOT/G_qualitative/figure_5_qualitative_cases.pdf" \
    "$COMMON_VALID_TOKENS"
  do
    require_file "$path"
  done
  python - "$COMMON_VALID_TOKENS" <<'PY'
import json
import sys
from pathlib import Path

tokens = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not tokens or len(tokens) != len(set(tokens)):
    raise RuntimeError("最终共同闭环 token 为空或包含重复项")
print(f"MAIN 最终使用完全配对闭环场景：{len(tokens)} 个")
PY
}

run_stage 1 "检查公共准备产物" "$LOG_ROOT/MAIN/01_check_prepare.log" check_prepare

OPEN_COMMON=(
  --args-file "$ARGS_FILE"
  --baseline-checkpoint "$BASELINE_CKPT"
  --cspq-checkpoint "$ENCODER_CKPT"
  --manifest "$VAL_MANIFEST"
  --cache-root "$CACHE_ROOT"
  --feature-npy "$VAL_FEATURE_NPY"
  --feature-index "$VAL_FEATURE_INDEX"
  --latent-bank "$VAL_LATENT_BANK"
  --latent-bank-index "$VAL_LATENT_INDEX"
  --batch-size 16 --n-eval-batches 2000
  --rho-min -1 --rho-max 1 --rho-steps 9
  --rank 4 --seed 17 --device cuda:0
)

run_stage 2 "平衡开环：无门控完整适配器" "$LOG_ROOT/MAIN/02_open_no_gate.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${OPEN_COMMON[@]}" \
    --adapter-high "$HIGH_ADAPTER" --adapter-low "$LOW_ADAPTER" \
    --output-report "$OPEN_NOGATE"

run_stage 3 "平衡开环：启用场景门控" "$LOG_ROOT/MAIN/03_open_gate.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${OPEN_COMMON[@]}" \
    --adapter-high "$HIGH_ADAPTER" --adapter-low "$LOW_ADAPTER" \
    --enable-scene-gate --scene-gate-checkpoint "$GATE_CKPT" \
    --output-report "$OPEN_GATE"

run_stage 4 "平衡开环：标量监督消融" "$LOG_ROOT/MAIN/04_open_scalar.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${OPEN_COMMON[@]}" \
    --adapter-high "$SCALAR_HIGH_ADAPTER" --adapter-low "$SCALAR_LOW_ADAPTER" \
    --output-report "$OPEN_SCALAR"

run_stage 5 "平衡开环：无横向约束消融" "$LOG_ROOT/MAIN/05_open_no_lateral.log" \
  python -u -m stylelora.scripts.evaluate_preference_lora \
    "${OPEN_COMMON[@]}" \
    --adapter-high "$NO_LAT_HIGH_ADAPTER" --adapter-low "$NO_LAT_LOW_ADAPTER" \
    --output-report "$OPEN_NO_LAT"

CLOSED_COMMON=(
  --args-file "$ARGS_FILE"
  --baseline-checkpoint "$BASELINE_CKPT"
  --data-root "$NUPLAN_DATA_ROOT"
  --maps-root "$NUPLAN_MAPS_ROOT"
  --scenario-filter boston
  --scenario-tokens-file "$TOKENS_300"
  --expected-scenario-count 300
  --allow-partial-scenarios
  --rhos=-1,-0.5,0,0.5,1
  --rank 4
  --challenge closed_loop_nonreactive_agents
  --device cuda
  --worker sequential
)

run_stage 6 "300候选闭环：标量监督消融" "$LOG_ROOT/MAIN/06_closed_scalar.log" \
  python -u -m stylelora.scripts.evaluate_closed_loop \
    "${CLOSED_COMMON[@]}" \
    --high-adapter "$SCALAR_HIGH_ADAPTER" --low-adapter "$SCALAR_LOW_ADAPTER" \
    --output-root "$CLOSED_SCALAR"

run_stage 7 "300候选闭环：无横向约束消融" "$LOG_ROOT/MAIN/07_closed_no_lateral.log" \
  python -u -m stylelora.scripts.evaluate_closed_loop \
    "${CLOSED_COMMON[@]}" \
    --high-adapter "$NO_LAT_HIGH_ADAPTER" --low-adapter "$NO_LAT_LOW_ADAPTER" \
    --output-root "$CLOSED_NO_LAT"

run_stage 8 "300候选闭环：关闭门控" "$LOG_ROOT/MAIN/08_closed_no_gate.log" \
  python -u -m stylelora.scripts.evaluate_closed_loop \
    "${CLOSED_COMMON[@]}" \
    --high-adapter "$HIGH_ADAPTER" --low-adapter "$LOW_ADAPTER" \
    --output-root "$CLOSED_NOGATE"

run_stage 9 "300候选闭环：完整 CAST" "$LOG_ROOT/MAIN/09_closed_full_cast.log" \
  python -u -m stylelora.scripts.evaluate_closed_loop \
    "${CLOSED_COMMON[@]}" \
    --high-adapter "$HIGH_ADAPTER" --low-adapter "$LOW_ADAPTER" \
    --enable-scene-gate --scene-gate-checkpoint "$GATE_CKPT" \
    --output-root "$CLOSED_GATE"

run_stage 10 "生成所有配置和 rho 的共同可行场景" "$LOG_ROOT/MAIN/10_common_closed_subset.log" \
  python -u -m stylelora.eval.build_common_closed_loop_subset \
    --report "Scalar-only alignment=$CLOSED_SCALAR_RAW_REPORT" \
    --report "No lateral constraint=$CLOSED_NO_LAT_RAW_REPORT" \
    --report "Without adaptive gate=$CLOSED_NOGATE_RAW_REPORT" \
    --report "Full CAST=$CLOSED_GATE_RAW_REPORT" \
    --output-root "$COMMON_CLOSED_ROOT"

run_stage 11 "生成 A 实验设置表" "$LOG_ROOT/MAIN/11_section_a.log" \
  python -u -m stylelora.eval.section_a_setup \
    --output-root "$MAIN_ROOT" \
    --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
    --closed-loop-report "$CLOSED_GATE_REPORT" \
    --baseline-checkpoint "$BASELINE_CKPT" --encoder-checkpoint "$ENCODER_CKPT" \
    --adapter-high "$HIGH_ADAPTER" --adapter-low "$LOW_ADAPTER" \
    --scene-gate-checkpoint "$GATE_CKPT" \
    --rho-grid=-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1 \
    --training-seeds 17 --hardware "NVIDIA RTX 3090"

run_stage 12 "生成 B 风格表征表图" "$LOG_ROOT/MAIN/12_section_b.log" \
  python -u -m stylelora.eval.section_b_representation \
    --output-root "$MAIN_ROOT" --encoder-report "$ENCODER_REPORT" \
    --predictions "$PREDICTIONS" --max-points 6000 --seed 17

run_stage 13 "生成 C 连续风格调节表图" "$LOG_ROOT/MAIN/13_section_c.log" \
  python -u -m stylelora.eval.section_c_continuous_tuning \
    --output-root "$MAIN_ROOT" --cast-report "$OPEN_NOGATE"

run_stage 14 "生成 D 自适应门控表图" "$LOG_ROOT/MAIN/14_section_d.log" \
  python -u -m stylelora.eval.section_d_adaptive_tuning \
    --output-root "$MAIN_ROOT" --gate-checkpoint "$GATE_CKPT" \
    --gate-report "$GATE_REPORT" --val-targets "$GATE_TARGET_VAL" \
    --val-feature-npy "$VAL_FEATURE_NPY" --val-feature-index "$VAL_FEATURE_INDEX" \
    --adaptive-open-report "$OPEN_GATE" \
    --comparison-report "Without adaptive gate=$OPEN_NOGATE" \
    --batch-size 512 --device cuda:0

run_stage 15 "生成 E 正式闭环表图" "$LOG_ROOT/MAIN/15_section_e.log" \
  python -u -m stylelora.eval.section_e_closed_loop \
    --output-root "$MAIN_ROOT" --cast-report "$CLOSED_GATE_REPORT" \
    --comparison-report "Without adaptive gate=$CLOSED_NOGATE_REPORT" --seed 17

run_stage 16 "生成 F 完整消融表" "$LOG_ROOT/MAIN/16_section_f.log" \
  python -u -m stylelora.eval.section_f_ablation_efficiency \
    --output-root "$MAIN_ROOT" \
    --open-report "Scalar-only alignment=$OPEN_SCALAR" \
    --open-report "No lateral constraint=$OPEN_NO_LAT" \
    --open-report "Without adaptive gate=$OPEN_NOGATE" \
    --open-report "Full CAST=$OPEN_GATE" \
    --closed-report "Scalar-only alignment=$CLOSED_SCALAR_REPORT" \
    --closed-report "No lateral constraint=$CLOSED_NO_LAT_REPORT" \
    --closed-report "Without adaptive gate=$CLOSED_NOGATE_REPORT" \
    --closed-report "Full CAST=$CLOSED_GATE_REPORT" \
    --checkpoint "Scalar-only alignment=$SCALAR_HIGH_ADAPTER" \
    --checkpoint "Scalar-only alignment=$SCALAR_LOW_ADAPTER" \
    --checkpoint "No lateral constraint=$NO_LAT_HIGH_ADAPTER" \
    --checkpoint "No lateral constraint=$NO_LAT_LOW_ADAPTER" \
    --checkpoint "Without adaptive gate=$HIGH_ADAPTER" \
    --checkpoint "Without adaptive gate=$LOW_ADAPTER" \
    --checkpoint "Full CAST=$HIGH_ADAPTER" \
    --checkpoint "Full CAST=$LOW_ADAPTER" \
    --checkpoint "Full CAST=$GATE_CKPT" \
    --components "Scalar-only alignment=0,0,0" \
    --components "No lateral constraint=1,0,0" \
    --components "Without adaptive gate=1,1,0" \
    --components "Full CAST=1,1,1"

run_stage 17 "生成 G 定性案例图" "$LOG_ROOT/MAIN/17_section_g.log" \
  python -u -m stylelora.eval.section_g_qualitative \
    --output-root "$MAIN_ROOT" \
    --raw-root=-1="$CLOSED_GATE/rho_minus1.00/raw_step_data" \
    --raw-root=0="$CLOSED_GATE/rho_plus0.00/raw_step_data" \
    --raw-root=1="$CLOSED_GATE/rho_plus1.00/raw_step_data" \
    --allowed-tokens-file "$COMMON_VALID_TOKENS" --max-cases 3 --dt 0.1

run_stage 18 "核对 MAIN 论文产物" "$LOG_ROOT/MAIN/18_verify.log" verify_main_outputs

finish_pipeline
