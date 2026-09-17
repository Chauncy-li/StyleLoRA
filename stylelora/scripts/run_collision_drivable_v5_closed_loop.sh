#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <negative|center|positive> <comma-separated-rhos>" >&2
  exit 2
fi

SHARD_NAME="$1"
RHOS="$2"
case "$SHARD_NAME" in
  negative|center|positive) ;;
  *) echo "invalid shard: $SHARD_NAME" >&2; exit 2 ;;
esac

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY"

PATHS_RESOLVER="$REPOSITORY/stylelora/config/runtime_paths.py"
config_path() {
  python "$PATHS_RESOLVER" --get "$1"
}
optional_config_path() {
  python "$PATHS_RESOLVER" --get-optional "$1"
}

RECORD_ROOT="${CAST_RECORD_ROOT:-$(config_path record_root)}"
if [[ -n "${CAST_SOURCE_ROOT:-}" ]]; then
  SOURCE_ROOT="$CAST_SOURCE_ROOT"
elif [[ -n "${CAST_RECORD_ROOT:-}" ]]; then
  SOURCE_ROOT="$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS"
else
  SOURCE_ROOT="$(config_path source_root)"
fi
if [[ -n "${CAST_OUTPUT_ROOT:-}" ]]; then
  CAST_ROOT="$CAST_OUTPUT_ROOT"
elif [[ -n "${CAST_RECORD_ROOT:-}" ]]; then
  CAST_ROOT="$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS3"
else
  CAST_ROOT="$(config_path output_root)"
fi
DATA_ROOT="${NUPLAN_DATA_ROOT:-$(config_path data_root)}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$(config_path maps_root)}"
if [[ -n "${CAST_TOKENS_FILE:-}" ]]; then
  TOKENS="$CAST_TOKENS_FILE"
elif [[ -n "${CAST_RECORD_ROOT:-}" || -n "${CAST_OUTPUT_ROOT:-}" ]]; then
  TOKENS="$CAST_ROOT/INPUTS/closed_loop_tokens_balanced_50.json"
else
  TOKENS="$(config_path tokens_file)"
fi
V5_ROOT="$CAST_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5"
ARGS_FILE="$(optional_config_path args_file)"
NORMALIZATION_FILE="$(optional_config_path normalization_file)"
BASELINE_CKPT="$(optional_config_path baseline_checkpoint)"
HIGH_ADAPTER="$(optional_config_path v5_high_adapter)"
LOW_ADAPTER="$(optional_config_path v5_low_adapter)"
ROUTER_CKPT="$(optional_config_path v5_router_checkpoint)"
ARGS_FILE="${ARGS_FILE:-$SOURCE_ROOT/INPUTS/args.json}"
NORMALIZATION_FILE="${NORMALIZATION_FILE:-$SOURCE_ROOT/INPUTS/normalization.json}"
BASELINE_CKPT="${BASELINE_CKPT:-$SOURCE_ROOT/MODELS/baseline_diffplanner.pth}"
HIGH_ADAPTER="${HIGH_ADAPTER:-$V5_ROOT/conditional_high_longitudinal_response_v5.pt}"
LOW_ADAPTER="${LOW_ADAPTER:-$V5_ROOT/conditional_low_longitudinal_response_v5.pt}"
ROUTER_CKPT="${ROUTER_CKPT:-$V5_ROOT/conditional_router_longitudinal_response_v5.pt}"
OUTPUT="$CAST_ROOT/CLOSED_LOOP/LONGITUDINAL_RESPONSE_V5_COLLISION_DRIVABLE/SHARD_${SHARD_NAME^^}"
LOG_ROOT="$CAST_ROOT/LOGS/LONGITUDINAL_RESPONSE_V5_COLLISION_DRIVABLE"
LOG="$LOG_ROOT/${SHARD_NAME}.log"

mkdir -p "$OUTPUT" "$LOG_ROOT"

python -u -m stylelora.scripts.evaluate_closed_loop \
  --args-file "$ARGS_FILE" \
  --normalization-file "$NORMALIZATION_FILE" \
  --baseline-checkpoint "$BASELINE_CKPT" \
  --high-adapter "$HIGH_ADAPTER" \
  --low-adapter "$LOW_ADAPTER" \
  --conditional-router-checkpoint "$ROUTER_CKPT" \
  --output-root "$OUTPUT" \
  --data-root "$DATA_ROOT" \
  --maps-root "$MAPS_ROOT" \
  --scenario-filter boston \
  --scenario-tokens-file "$TOKENS" \
  --expected-scenario-count 50 \
  "--rhos=$RHOS" \
  --allow-rho-shard \
  --rank 4 \
  --challenge closed_loop_nonreactive_agents \
  --device cuda \
  --worker sequential \
  --allow-partial-scenarios \
  --enable-trajectory-repair \
  --trajectory-repair-mode collision_parallel \
  --collision-repair-scales 1,0.875,0.75,0.625,0.5,0.375,0.25,0.125,0 \
  --collision-repair-horizon-s 2.0 \
  --collision-repair-min-clearance-m 0.5 \
  --repair-check-drivable-area \
  --repair-drivable-horizon-s 2.0 \
  --skip-existing 2>&1 | tee "$LOG"
