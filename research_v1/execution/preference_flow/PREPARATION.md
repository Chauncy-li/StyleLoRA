# Phase 0 前置产物重建顺序（服务器）

这份说明只重建当前 StylePlanner 基线所需的输入与 checkpoint，不实现
preference flow、双 DPM stream 或轨迹后处理。

旧的 A3.8 / B3 checkpoint 若确实已经删除，重新训练得到的是**新的重建
谱系**，不能在论文中称为旧运行目录的逐位复现。后续 Phase 0 manifest 会
锁定这条新谱系。

## 不需要重建的已有输入

从当前服务器目录截图看，下列内容应先作为只读输入复用：

- `CACHE/boston_cache_train_val/` 与同级 `boston_cache_train_val_list.json`；
- `CACHE/style_scene_split_straight_train_v2/split_index.jsonl`；
- `CACHE/style_scene_split_straight_val_v2/split_index.jsonl`；
- `DATA_SPLITS_CONFIG/boston_raw_seed3407/`。

不要先运行原始 cache 构建器或 straight-scene split builder。只有上述文件
缺失或损坏时，才单独恢复这些原始输入。

## 统一的输出根

```bash
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

REPO=/home/lisw/programs/Nuplan-Diffusion-Baseline
RECORD=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
CACHE="$RECORD/CACHE"
PF_ROOT="$RECORD/preference_flow_v1"
PLANNER_CACHE="$CACHE/boston_cache_train_val"
TRAIN_SPLIT="$CACHE/style_scene_split_straight_train_v2"
VAL_SPLIT="$CACHE/style_scene_split_straight_val_v2"

mkdir -p "$PF_ROOT"/{calibration/train,calibration/val,conditioning/train,conditioning/val,checkpoints/base,checkpoints/a3_8,checkpoints/b3,cohorts,manifests}

test -f "$TRAIN_SPLIT/split_index.jsonl"
test -f "$VAL_SPLIT/split_index.jsonl"
test -f "$CACHE/boston_cache_train_val_list.json"
test -f "$REPO/baseline/resources/normalization_train.json"
```

All newly generated artifacts below write beneath `$PF_ROOT`. Existing
`CACHE/` contents are never overwritten.

## 1. 重建 V5 训练参考

These three commands create the frozen normalization and conditional-rank
reference used by the signed V6 StylePlanner training objective.

```bash
python -m research_v1.stylization reference candidates \
  --index-path "$TRAIN_SPLIT/split_index.jsonl" \
  --planner-cache-dir "$PLANNER_CACHE" \
  --output-dir "$PF_ROOT/calibration/train" \
  --scene-filter controlled \
  --sample-fraction 1.0 \
  --log-interval 5000

python -m research_v1.stylization reference normalization \
  --candidate-index-path "$PF_ROOT/calibration/train/v5_candidates.jsonl" \
  --output-dir "$PF_ROOT/calibration/train" \
  --scene-filter controlled

python -m research_v1.stylization reference rank \
  --normalized-index-path "$PF_ROOT/calibration/train/v5_normalized.jsonl" \
  --output-dir "$PF_ROOT/calibration/train" \
  --scene-filter controlled \
  --folds 5 \
  --neighbours 64 \
  --min-effective-neighbours 32 \
  --seed 3407
```

Required outputs:

```text
$PF_ROOT/calibration/train/v5_normalization.json
$PF_ROOT/calibration/train/v5_conditional_rank_model.json
$PF_ROOT/calibration/train/v5_conditional_rank.jsonl
```

## 2. 建立验证集的冻结参考标签

Validation data measures candidates afresh, but applies the train-only V5
normalization/reference. It must not fit a new validation reference.

```bash
python -m research_v1.stylization reference candidates \
  --index-path "$VAL_SPLIT/split_index.jsonl" \
  --planner-cache-dir "$PLANNER_CACHE" \
  --output-dir "$PF_ROOT/calibration/val" \
  --scene-filter controlled \
  --sample-fraction 1.0 \
  --log-interval 5000

python -m research_v1.stylization reference apply_normalization \
  --candidate-index-path "$PF_ROOT/calibration/val/v5_candidates.jsonl" \
  --normalization-model-path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --output-dir "$PF_ROOT/calibration/val" \
  --scene-filter controlled

python -m research_v1.stylization reference apply_rank \
  --normalized-index-path "$PF_ROOT/calibration/val/v5_normalized.jsonl" \
  --conditional-rank-model-path "$PF_ROOT/calibration/train/v5_conditional_rank_model.json" \
  --output-dir "$PF_ROOT/calibration/val" \
  --scene-filter controlled
```

## 3. 建立并审计 V6 条件 sidecar

The base split remains authoritative: unmatched samples stay in the diffusion
dataset with an all-zero style condition. Build train and validation sidecars,
then validate their masks and support.

```bash
python -m research_v1.stylization command selftest

python -m research_v1.stylization command build \
  --rank-index-path "$PF_ROOT/calibration/train/v5_conditional_rank.jsonl" \
  --base-index-path "$TRAIN_SPLIT/split_index.jsonl" \
  --output-dir "$PF_ROOT/conditioning/train" \
  --scene-filter all

python -m research_v1.stylization command validate \
  --condition-index-path "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --output-dir "$PF_ROOT/conditioning/train"

python -m research_v1.stylization command router_audit \
  --condition-index-path "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --output-dir "$PF_ROOT/conditioning/train"

python -m research_v1.stylization command support_audit \
  --condition-index-path "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --output-dir "$PF_ROOT/conditioning/train"

python -m research_v1.stylization command build \
  --rank-index-path "$PF_ROOT/calibration/val/v5_conditional_rank.jsonl" \
  --base-index-path "$VAL_SPLIT/split_index.jsonl" \
  --output-dir "$PF_ROOT/conditioning/val" \
  --scene-filter all

python -m research_v1.stylization command validate \
  --condition-index-path "$PF_ROOT/conditioning/val/v6_direct_axis_conditions.jsonl" \
  --output-dir "$PF_ROOT/conditioning/val"

python -m research_v1.stylization command router_audit \
  --condition-index-path "$PF_ROOT/conditioning/val/v6_direct_axis_conditions.jsonl" \
  --output-dir "$PF_ROOT/conditioning/val"

python -m research_v1.stylization command support_audit \
  --condition-index-path "$PF_ROOT/conditioning/val/v6_direct_axis_conditions.jsonl" \
  --output-dir "$PF_ROOT/conditioning/val"

python -m research_v1.execution.diffusion.selftest_stylization \
  --normalization-path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --conditional-rank-model-path "$PF_ROOT/calibration/train/v5_conditional_rank_model.json"
```

## 4. 恢复 checkpoint 谱系（仅在旧 checkpoint 不可恢复时，本轮忽略）

First obtain a verified plain Diffusion Planner checkpoint. If none can be
recovered, the existing baseline entrypoint can train one with the current
planner cache. The legacy YAML contains stale paths, so pass all relevant
paths explicitly instead of editing `baseline/`.

```bash
python baseline/train.py \
  save_dir="$PF_ROOT/checkpoints/base" \
  data.train_set="$PLANNER_CACHE" \
  data.train_set_list="$CACHE/boston_cache_train_val_list.json" \
  data.normalization_file_path="$REPO/baseline/resources/normalization_train.json" \
  use_online_logger=false
```

Choose one verified base `.pth` from the timestamped output and set
`BASE_CKPT` manually. Do not silently select a different checkpoint.

```bash
BASE_CKPT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record/best_model-epoch_116-train_loss_0.0701.pth
```

Train A3.8 from that base checkpoint, then train B3 from the selected A3.8
checkpoint. The presets own their frozen-backbone and signed-router settings;
do not replace them with manually assembled flags.

```bash
CUDA_VISIBLE_DEVICES=1 python -m research_v1.execution.diffusion.train_stylized_diffusion \
  --experiment_preset v6_signed_router_ncqt_stage_a3_8_terminal_executor \
  --pretrained_model_path "$BASE_CKPT" \
  --train_split_root "$TRAIN_SPLIT" \
  --val_split_root "$VAL_SPLIT" \
  --train_cache_dir "$PLANNER_CACHE" \
  --val_cache_dir "$PLANNER_CACHE" \
  --normalization_file_path "$REPO/baseline/resources/normalization_train.json" \
  --train_conditioning_index_override "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --val_conditioning_index_override "$PF_ROOT/conditioning/val/v6_direct_axis_conditions.jsonl" \
  --preference_energy_normalization_path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --preference_energy_rank_model_path "$PF_ROOT/calibration/train/v5_conditional_rank_model.json" \
  --save_dir "$PF_ROOT/checkpoints/a3_8" \
  --experiment_name a3_8_rebuild \
  --seed 3407 \
  --disable_online_logger

A38_CKPT=/absolute/path/to/the/selected/a3_8_checkpoint.pth

python -m research_v1.execution.diffusion.train_stylized_diffusion \
  --experiment_preset v6_signed_router_ncqt_stage_b3_scene_axis_executor \
  --pretrained_model_path "$A38_CKPT" \
  --train_split_root "$TRAIN_SPLIT" \
  --val_split_root "$VAL_SPLIT" \
  --train_cache_dir "$PLANNER_CACHE" \
  --val_cache_dir "$PLANNER_CACHE" \
  --normalization_file_path "$REPO/baseline/resources/normalization_train.json" \
  --train_conditioning_index_override "$PF_ROOT/conditioning/train/v6_direct_axis_conditions.jsonl" \
  --val_conditioning_index_override "$PF_ROOT/conditioning/val/v6_direct_axis_conditions.jsonl" \
  --preference_energy_normalization_path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --preference_energy_rank_model_path "$PF_ROOT/calibration/train/v5_conditional_rank_model.json" \
  --save_dir "$PF_ROOT/checkpoints/b3" \
  --experiment_name b3_rebuild \
  --seed 3407 \
  --disable_online_logger
```

Select the B3 checkpoint explicitly (normally the declared best signed-control
checkpoint), then set:

```bash
B3_CKPT=/absolute/path/to/the/selected/b3_checkpoint.pth
```

## 5. 固定 smoke cohort

Do not create this list by arbitrarily taking the first split rows. After B3
exists, use one `router_only`, `rho=0` closed-loop discovery job to export a
balanced exact-token JSON file. It requires the actual validation DB and map
roots, so those two paths remain explicit placeholders:

```bash
python -m research_v1.execution.evaluation.run_closed_loop_suite \
  --experiment-dir /absolute/path/to/the/b3_timestamped_run \
  --checkpoint-paths "$B3_CKPT" \
  --rho-values 0 \
  --variants router_only \
  --output-root "$PF_ROOT/cohort_discovery" \
  --db-files /absolute/path/to/nuplan_validation_db_root \
  --maps-root /absolute/path/to/nuplan_maps_root \
  --log-names-json "$RECORD/DATA_SPLITS_CONFIG/boston_raw_seed3407/cache_val_log_names.json" \
  --map-names us-ma-boston \
  --worker sequential \
  --export-router-eligible-tokens-json "$PF_ROOT/cohorts/smoke_cohort_tokens.json" \
  --eligible-scenarios-per-controlled-scene 4 \
  --execute
```

The resulting JSON is the fixed smoke cohort. Reuse it unchanged after the
manifest is written.

## 6. Only then run the Phase 0 manifest

```bash
python -m research_v1.execution.preference_flow.phase0_manifest --strict \
  --record-root "$RECORD" \
  --preference-flow-root "$PF_ROOT" \
  --a3-8-checkpoint-root "$PF_ROOT/checkpoints/a3_8" \
  --base-checkpoint-path "$BASE_CKPT" \
  --style-checkpoint-path "$B3_CKPT" \
  --dataset-split-path "$TRAIN_SPLIT/split_index.jsonl" \
  --calibration-path "$PF_ROOT/calibration/train/v5_normalization.json" \
  --reference-path "$PF_ROOT/calibration/train/v5_conditional_rank_model.json" \
  --config-path "$REPO/baseline/config/planner/style_planner.yaml" \
  --smoke-cohort-path "$PF_ROOT/cohorts/smoke_cohort_tokens.json" \
  --rho-grid=-1.0,-0.5,0.0,0.5,1.0 \
  --seed 3407 \
  --output-path "$PF_ROOT/manifests/phase0_manifest.json"
```

`research_v1.execution.calibration.build` is intentionally not in this chain:
it exports offline evaluation targets, not the frozen V5 normalization and
conditional-rank reference required by the V6 training objective.
