#!/usr/bin/env bash
# 公共准备：归档依赖、导出表征、训练两个消融模型并固定300个候选场景。

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"
begin_pipeline "PREPARE" "$SCRIPT_DIR/run_00_prepare.sh"

archive_required_files() {
  python - "$SOURCE_ARGS_FILE" "$ARGS_FILE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source_args = Path(sys.argv[1]).resolve()
target_args = Path(sys.argv[2]).resolve()
payload = json.loads(source_args.read_text(encoding="utf-8"))
key = "normalization_file_path" if "normalization_file_path" in payload else "normalization_file"
if key not in payload:
    raise RuntimeError("args.json 缺少 normalization_file_path/normalization_file")
source_normalization = Path(str(payload[key])).expanduser()
if not source_normalization.is_absolute():
    source_normalization = source_args.parent / source_normalization
source_normalization = source_normalization.resolve()
if not source_normalization.is_file():
    raise FileNotFoundError(f"缺少归一化文件：{source_normalization}")
target_args.parent.mkdir(parents=True, exist_ok=True)
target_normalization = target_args.parent / f"normalization{source_normalization.suffix}"
shutil.copy2(source_normalization, target_normalization)
payload[key] = str(target_normalization)
target_args.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"已归档：{target_args}")
print(f"已归档：{target_normalization}")
PY
  copy_required "$SOURCE_BASELINE_CKPT" "$BASELINE_CKPT"
  copy_required "$SOURCE_ENCODER_ROOT/preference_encoder.pt" "$ENCODER_CKPT"
  copy_required "$SOURCE_LORA_ROOT/preference_lora_high_dyn_q_lat_topk.pt" "$HIGH_ADAPTER"
  copy_required "$SOURCE_LORA_ROOT/preference_lora_low_dyn_q_lat_topk.pt" "$LOW_ADAPTER"
  copy_required "$SOURCE_GATE_ROOT/scene_gate_3k_ow2.pt" "$GATE_CKPT"
  copy_required "$SOURCE_GATE_ROOT/scene_gate_3k_ow2.report.json" "$GATE_REPORT"

  copy_required "$SOURCE_PREF_ROOT/weak_preference_train.jsonl" "$TRAIN_MANIFEST"
  copy_required "$SOURCE_PREF_ROOT/weak_preference_val.jsonl" "$VAL_MANIFEST"
  copy_required "$SOURCE_PREF_ROOT/scene_features_train.npy" "$TRAIN_FEATURE_NPY"
  copy_required "$SOURCE_PREF_ROOT/scene_features_train_index.jsonl" "$TRAIN_FEATURE_INDEX"
  copy_required "$SOURCE_PREF_ROOT/scene_features_val.npy" "$VAL_FEATURE_NPY"
  copy_required "$SOURCE_PREF_ROOT/scene_features_val_index.jsonl" "$VAL_FEATURE_INDEX"
  copy_required "$SOURCE_ENCODER_ROOT/latent_bank_train.npy" "$TRAIN_LATENT_BANK"
  copy_required "$SOURCE_ENCODER_ROOT/latent_bank_train_index.jsonl" "$TRAIN_LATENT_INDEX"
  copy_required "$SOURCE_ENCODER_ROOT/latent_bank_val.npy" "$VAL_LATENT_BANK"
  copy_required "$SOURCE_ENCODER_ROOT/latent_bank_val_index.jsonl" "$VAL_LATENT_INDEX"
  copy_required "$SOURCE_ENCODER_ROOT/evaluate_encoder.json" "$ENCODER_REPORT"
  copy_required "$SOURCE_GATE_TARGET_ROOT/gate_targets_val.jsonl" "$GATE_TARGET_VAL"
}

verify_candidate_tokens() {
  python - "$TOKENS_300" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tokens = json.loads(path.read_text(encoding="utf-8"))
if len(tokens) != 300 or len(set(tokens)) != 300:
    raise RuntimeError(f"候选 token 应为300个且无重复，实际为 {len(tokens)} 个")
print(f"候选闭环场景检查通过：{len(tokens)} 个")
PY
}

run_stage 1 "归档模型与本轮必要输入" "$LOG_ROOT/PREPARE/01_archive_inputs.log" \
  archive_required_files

run_stage 2 "运行方法与评测相关单测" "$LOG_ROOT/PREPARE/02_tests.log" \
  python -m pytest \
    stylelora/tests/test_preference_lora_stage.py \
    stylelora/tests/test_scene_gate.py \
    stylelora/tests/test_closed_loop_style.py \
    stylelora/tests/test_continuous_rho_evaluation.py \
    stylelora/tests/test_eval_ablation.py -q

run_stage 3 "导出验证集偏好表征预测" "$LOG_ROOT/PREPARE/03_export_predictions.log" \
  python -u -m stylelora.scripts.export_preference_predictions \
    --checkpoint "$ENCODER_CKPT" \
    --manifest "$VAL_MANIFEST" \
    --feature-npy "$VAL_FEATURE_NPY" \
    --feature-index "$VAL_FEATURE_INDEX" \
    --cache-root "$CACHE_ROOT" \
    --output "$PREDICTIONS" \
    --summary "$COMMON_ROOT/STYLE_REPRESENTATION/val_predictions_summary.json" \
    --batch-size 128 --workers 4 --device cuda:0 --seed 17

ABLATION_TRAIN_COMMON=(
  --args-file "$ARGS_FILE"
  --baseline-checkpoint "$BASELINE_CKPT"
  --manifest "$TRAIN_MANIFEST"
  --val-manifest "$VAL_MANIFEST"
  --cache-root "$CACHE_ROOT"
  --latent-bank "$TRAIN_LATENT_BANK"
  --latent-bank-index "$TRAIN_LATENT_INDEX"
  --cspq-checkpoint "$ENCODER_CKPT"
  --feature-npy "$TRAIN_FEATURE_NPY"
  --feature-index "$TRAIN_FEATURE_INDEX"
  --val-latent-bank "$VAL_LATENT_BANK"
  --val-latent-bank-index "$VAL_LATENT_INDEX"
  --val-feature-npy "$VAL_FEATURE_NPY"
  --val-feature-index "$VAL_FEATURE_INDEX"
  --steps 500 --batch-size 32 --workers 16 --lr 1e-4
  --lambda-n 1.0 --lambda-s 1.0 --lambda-dyn 0.1
  --rank 4 --seed 17 --device cuda:0
  --swanlab-project continuous-preference-lora-ablation
  --swanlab-mode online
  --swanlab-logdir "$LOG_ROOT/swanlab"
)

run_stage 4 "训练标量监督 High 消融" "$LOG_ROOT/PREPARE/04_scalar_high.log" \
  python -u -m stylelora.scripts.train_preference_lora \
    "${ABLATION_TRAIN_COMMON[@]}" \
    --direction high --lambda-z 0 --lambda-q 0 --lambda-lat 0 \
    --output "$SCALAR_HIGH_ADAPTER" \
    --swanlab-experiment-name scalar-only-high-seed17

run_stage 5 "训练标量监督 Low 消融" "$LOG_ROOT/PREPARE/05_scalar_low.log" \
  python -u -m stylelora.scripts.train_preference_lora \
    "${ABLATION_TRAIN_COMMON[@]}" \
    --direction low --lambda-z 0 --lambda-q 0 --lambda-lat 0 \
    --output "$SCALAR_LOW_ADAPTER" \
    --swanlab-experiment-name scalar-only-low-seed17

run_stage 6 "训练无横向约束 High 消融" "$LOG_ROOT/PREPARE/06_no_lateral_high.log" \
  python -u -m stylelora.scripts.train_preference_lora \
    "${ABLATION_TRAIN_COMMON[@]}" \
    --direction high --lambda-z 1 --lambda-q 1 --lambda-lat 0 \
    --output "$NO_LAT_HIGH_ADAPTER" \
    --swanlab-experiment-name no-lateral-high-seed17

run_stage 7 "训练无横向约束 Low 消融" "$LOG_ROOT/PREPARE/07_no_lateral_low.log" \
  python -u -m stylelora.scripts.train_preference_lora \
    "${ABLATION_TRAIN_COMMON[@]}" \
    --direction low --lambda-z 1 --lambda-q 1 --lambda-lat 0 \
    --output "$NO_LAT_LOW_ADAPTER" \
    --swanlab-experiment-name no-lateral-low-seed17

run_stage 8 "构建 held-out test 场景划分" "$LOG_ROOT/PREPARE/08_build_test_scene_split.log" \
  python -u -m stylelora.scripts.build_scene_split \
    --planner_cache_dir "$TEST_CACHE_ROOT" \
    --data_list_path "$TEST_CACHE_LIST" \
    --manifest_path "$TEST_CACHE_MANIFEST" \
    --split_name test \
    --test_output_dir "$TEST_SCENE_SPLIT" \
    --num_workers 16

run_stage 9 "固定300个闭环候选 token" "$LOG_ROOT/PREPARE/09_select_tokens.log" \
  python -u -m stylelora.scripts.select_closed_loop_scenarios \
    --split-index "$TEST_SCENE_SPLIT/split_index.jsonl" \
    --output "$TOKENS_300" \
    --total 300 --min-scene-confidence 0.8 \
    --exclude-known-route-failures --seed 17

run_stage 10 "核对公共准备产物" "$LOG_ROOT/PREPARE/10_verify.log" \
  verify_candidate_tokens

finish_pipeline
