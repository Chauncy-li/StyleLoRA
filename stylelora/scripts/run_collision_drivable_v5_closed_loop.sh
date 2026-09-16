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

RECORD_ROOT="${CAST_RECORD_ROOT:-/mnt/mydata/lishangwen/Nuplan-Baseline-Record}"
SOURCE_ROOT="${CAST_SOURCE_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS}"
CAST_ROOT="${CAST_OUTPUT_ROOT:-$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS3}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps}"
TOKENS="${CAST_TOKENS_FILE:-$CAST_ROOT/INPUTS/closed_loop_tokens_balanced_50.json}"
V5_ROOT="$CAST_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5"
OUTPUT="$CAST_ROOT/CLOSED_LOOP/LONGITUDINAL_RESPONSE_V5_COLLISION_DRIVABLE/SHARD_${SHARD_NAME^^}"
LOG_ROOT="$CAST_ROOT/LOGS/LONGITUDINAL_RESPONSE_V5_COLLISION_DRIVABLE"
LOG="$LOG_ROOT/${SHARD_NAME}.log"

mkdir -p "$OUTPUT" "$LOG_ROOT"

python -u -m stylelora.scripts.evaluate_closed_loop \
  --args-file "$SOURCE_ROOT/INPUTS/args.json" \
  --normalization-file "$SOURCE_ROOT/INPUTS/normalization.json" \
  --baseline-checkpoint "$SOURCE_ROOT/MODELS/baseline_diffplanner.pth" \
  --high-adapter "$V5_ROOT/conditional_high_longitudinal_response_v5.pt" \
  --low-adapter "$V5_ROOT/conditional_low_longitudinal_response_v5.pt" \
  --conditional-router-checkpoint "$V5_ROOT/conditional_router_longitudinal_response_v5.pt" \
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
