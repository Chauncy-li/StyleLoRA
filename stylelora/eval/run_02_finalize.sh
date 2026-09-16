#!/usr/bin/env bash
# 最终汇总：等待 MAIN/FULL 均成功后生成全量消融表并核对全部产物。

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"
begin_pipeline "FINALIZE" "$SCRIPT_DIR/run_02_finalize.sh"

check_parallel_lines() {
  require_pipeline_done "MAIN"
  require_pipeline_done "FULL"
  for path in \
    "$OPEN_NOGATE_FULL" "$OPEN_GATE_FULL" \
    "$CLOSED_NOGATE_REPORT" "$CLOSED_GATE_REPORT" \
    "$HIGH_ADAPTER" "$LOW_ADAPTER" "$GATE_CKPT"
  do
    require_file "$path"
  done
}

verify_all_outputs() {
  require_file "$FULL_ROOT/F_ablation_efficiency/table_6_ablation_and_efficiency.csv"
  require_file "$COMMON_CLOSED_ROOT/common_valid_scenarios_report.json"
  require_file "$COMMON_VALID_TOKENS"
  find "$MODELS_ROOT" "$COMMON_ROOT" "$MAIN_ROOT" "$FULL_ROOT" \
    -type f -print | sort > "$CAST_ROOT/ARTIFACT_MANIFEST.txt"
  local artifact_count
  artifact_count="$(wc -l < "$CAST_ROOT/ARTIFACT_MANIFEST.txt")"
  echo "最终产物清单：$CAST_ROOT/ARTIFACT_MANIFEST.txt"
  echo "已登记文件数：$artifact_count"
}

run_stage 1 "确认 MAIN 与 FULL 均已完成" "$LOG_ROOT/FINALIZE/01_check_lines.log" \
  check_parallel_lines

run_stage 2 "生成基于全量开环结果的 F 消融表" "$LOG_ROOT/FINALIZE/02_full_section_f.log" \
  python -u -m stylelora.eval.section_f_ablation_efficiency \
    --output-root "$FULL_ROOT" \
    --open-report "Without adaptive gate=$OPEN_NOGATE_FULL" \
    --open-report "Full CAST=$OPEN_GATE_FULL" \
    --closed-report "Without adaptive gate=$CLOSED_NOGATE_REPORT" \
    --closed-report "Full CAST=$CLOSED_GATE_REPORT" \
    --checkpoint "Without adaptive gate=$HIGH_ADAPTER" \
    --checkpoint "Without adaptive gate=$LOW_ADAPTER" \
    --checkpoint "Full CAST=$HIGH_ADAPTER" \
    --checkpoint "Full CAST=$LOW_ADAPTER" \
    --checkpoint "Full CAST=$GATE_CKPT" \
    --components "Without adaptive gate=1,1,0" \
    --components "Full CAST=1,1,1"

run_stage 3 "生成最终产物清单并核对" "$LOG_ROOT/FINALIZE/03_verify_all.log" \
  verify_all_outputs

finish_pipeline
