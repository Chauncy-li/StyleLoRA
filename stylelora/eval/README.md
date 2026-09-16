# CAST EAAI 论文实验流水线

本文档是本轮论文实验的唯一运行说明。流程不增加新方法、新损失或新指标，只整理已有的偏好表征、连续调节、场景门控、消融、开环、闭环和论文图表生成步骤。

本轮所有新模型与结果统一写入：

```text
/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS
```

旧的 `STYLE_LORA_*` 目录只作为公共准备阶段的只读来源。脚本不会删除或移动旧文件，而是复制并校验本轮需要的模型和输入，避免误删历史实验。

## 目录结构

```text
CAST_EAAI_PAPER_RESULTS/
├── MODELS/
│   ├── baseline_diffplanner.pth
│   ├── preference_encoder.pt
│   ├── preference_adapter_high.pt
│   ├── preference_adapter_low.pt
│   ├── scene_gate.pt
│   ├── scene_gate.report.json
│   └── ABLATIONS/
│       ├── preference_adapter_high_scalar_only.pt
│       ├── preference_adapter_low_scalar_only.pt
│       ├── preference_adapter_high_no_lateral.pt
│       └── preference_adapter_low_no_lateral.pt
├── INPUTS/                         # 本轮实际使用的 manifest、特征和 latent bank 副本
├── COMMON_RESULTS/
│   ├── STYLE_REPRESENTATION/
│   └── CLOSED_LOOP_SCENES/
│       ├── test_scene_split/
│       └── closed_loop_candidates_boston_300.json
├── MAIN_RESULTS/
│   ├── OPEN_LOOP/
│   ├── CLOSED_LOOP/
│   │   ├── 01_SCALAR_ONLY/
│   │   ├── 02_NO_LATERAL/
│   │   ├── 03_NO_GATE/
│   │   ├── 04_FULL_CAST/
│   │   └── 05_COMMON_SUBSET/
│   └── A_experimental_setup ... G_qualitative/
├── FULL_RESULTS/
│   ├── OPEN_LOOP/
│   └── C_continuous_tuning、D_adaptive_tuning、F_ablation_efficiency/
├── LOGS/
├── STATUS/
└── ARTIFACT_MANIFEST.txt
```

闭环首先对四个配置、五个 `rho` 统一采集300个候选场景。最终论文数量不是强制200，而是所有配置、所有 `rho` 都成功产生官方指标的 token 交集。例如交集为249，后续闭环表格、配对差异、风格代理和定性图全部统一使用这249个场景。

## 自动脚本

| 脚本 | 作用 | 是否使用 GPU |
|---|---|---|
| `run_00_prepare.sh` | 归档模型与输入、单测、表征导出、训练两个消融变体、固定300个闭环候选 | 是 |
| `run_01_main.sh` | 平衡开环、四组闭环、共同场景过滤、MAIN 的 A–G 图表 | 是，长时间占用一张 GPU |
| `run_01_full.sh` | 完整验证集的无门控/有门控开环，以及 FULL 的 C、D 图表 | 是，可与 MAIN 使用不同 GPU 并行 |
| `run_02_finalize.sh` | MAIN/FULL 完成检查、FULL 的 F 表格、最终产物清单 | 基本为 CPU |

所有脚本都使用 `set -Eeuo pipefail`。任一步骤出错时会：

- 立即停止当前流水线；
- 在终端和总日志中打印流水线名称、失败步骤、行号和退出码；
- 将状态写入 `STATUS/<流水线>.status`；
- 给出带 `CAST_START_STEP` 的恢复命令。

## 1. 检查并恢复基础数据划分

截图中缺少的根目录是：

```text
DATA_SPLITS_CONFIG/boston_raw_seed3407/
```

它与 `CACHE/` 平级，保存原始 DB 的 train/val/test_simu 固定日志划分；`CACHE/` 内部则可能还有 `style_scene_split_straight_*_v2` 场景划分。先检查两类目录：

```bash
conda activate mdsn_py39
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record

find "$RECORD_ROOT/DATA_SPLITS_CONFIG" -maxdepth 3 -type f -print 2>/dev/null
find "$RECORD_ROOT/CACHE" -maxdepth 2 -type d -iname "*split*" -print
```

如果 `DATA_SPLITS_CONFIG/boston_raw_seed3407` 确实不存在，运行下面两条命令恢复。它们只重建 JSON/YAML 划分配置，不重新生成或覆盖 cache `.npz`：

```bash
SPLIT_ROOT=$RECORD_ROOT/DATA_SPLITS_CONFIG/boston_raw_seed3407
RAW_DB_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston

mkdir -p "$SPLIT_ROOT/scenario_filters"

python -u -m stylelora.scripts.build_raw_split \
  --log_names_path "" \
  --data_path "$RAW_DB_ROOT" \
  --output_dir "$SPLIT_ROOT" \
  --config_output_dir "$SPLIT_ROOT/scenario_filters" \
  --prefix boston \
  --test_ratio 0.10 \
  --seed 3407 \
  --group_by session \
  --write_filters 1 \
  2>&1 | tee "$SPLIT_ROOT/rebuild_raw_split.log"
```

```bash
python -u -m stylelora.scripts.build_cache_split \
  --split_name cache_train_val \
  --cache_log_names_path "$SPLIT_ROOT/cache_log_names.json" \
  --output_dir "$SPLIT_ROOT" \
  --config_output_dir "$SPLIT_ROOT/scenario_filters" \
  --prefix boston \
  --val_ratio 0.10 \
  --seed 3408 \
  --group_by session \
  --write_filters 1 \
  2>&1 | tee "$SPLIT_ROOT/rebuild_train_val_split.log"
```

核对应该恢复的八个核心 JSON：

```bash
python - "$SPLIT_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (
    "cache_log_names.json", "test_simu_log_names.json",
    "test_simu_group_ids.json", "split_summary.json",
    "cache_train_log_names.json", "cache_val_log_names.json",
    "cache_val_group_ids.json", "cache_train_val_summary.json",
)
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise RuntimeError("仍缺少：" + ", ".join(missing))
print("DATA_SPLITS_CONFIG 恢复完成：8个核心 JSON 均存在")
PY
```

如果缺少的是 `CACHE/style_scene_split_straight_train_v2` 或 `val_v2`，但 `boston_cache_train_val` 及其 list/manifest 仍存在，可单独恢复：

```bash
CACHE_ROOT=$RECORD_ROOT/CACHE

python -u -m stylelora.scripts.build_scene_split \
  --planner_cache_dir "$CACHE_ROOT/boston_cache_train_val" \
  --data_list_path "$CACHE_ROOT/boston_cache_train_val_list.json" \
  --manifest_path "$CACHE_ROOT/boston_cache_train_val_manifest.json" \
  --split_name all \
  --train_output_dir "$CACHE_ROOT/style_scene_split_straight_train_v2" \
  --val_output_dir "$CACHE_ROOT/style_scene_split_straight_val_v2" \
  --num_workers 16 \
  2>&1 | tee "$CACHE_ROOT/rebuild_train_val_scene_split.log"
```

当前 CAST 流水线不依赖旧的 test scene-split 目录；公共准备第8步会从已有 `boston_cache_test_simu` 自动重建到 `CAST_EAAI_PAPER_RESULTS/COMMON_RESULTS`。

## 2. 从新终端建立环境

每个新终端都先执行：

```bash
conda activate mdsn_py39
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
CAST_ROOT=$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS
mkdir -p "$CAST_ROOT/LOGS"
```

脚本内部统一定义所有路径和数组，因此不再需要在终端手动定义 `ARGS_FILE`、`ABLATION_TRAIN_COMMON`、`CLOSED_300_COMMON` 等变量。

## 3. 运行公共准备流水线

公共准备必须先完成一次：

```bash
nohup env CAST_GPU=2 \
  bash stylelora/eval/run_00_prepare.sh \
  > "$CAST_ROOT/LOGS/nohup_prepare.log" 2>&1 &

echo "PREPARE PID=$!"
```

查看进度：

```bash
tail -f "$CAST_ROOT/LOGS/nohup_prepare.log"
```

检查是否完成：

```bash
cat "$CAST_ROOT/STATUS/PREPARE.status"
```

只有第一行显示 `DONE` 才能启动 MAIN 和 FULL。

### 公共准备内部顺序

| 步骤 | 内容 | 主要产物 |
|---:|---|---|
| 1 | 复制并校验已有模型与必要输入 | `MODELS/`、`INPUTS/` |
| 2 | 运行方法和评测相关单测 | `LOGS/PREPARE/02_tests.log` |
| 3 | 导出验证集逐样本偏好表征 | `COMMON_RESULTS/STYLE_REPRESENTATION/val_predictions.jsonl` |
| 4 | 训练 Scalar-only High | `MODELS/ABLATIONS/*high_scalar_only.pt` |
| 5 | 训练 Scalar-only Low | `MODELS/ABLATIONS/*low_scalar_only.pt` |
| 6 | 训练 No-lateral High | `MODELS/ABLATIONS/*high_no_lateral.pt` |
| 7 | 训练 No-lateral Low | `MODELS/ABLATIONS/*low_no_lateral.pt` |
| 8 | 从 held-out test cache 构建场景划分 | `COMMON_RESULTS/CLOSED_LOOP_SCENES/test_scene_split/` |
| 9 | 固定300个候选 token | `closed_loop_candidates_boston_300.json` |
| 10 | 核对 token 数量与重复项 | 准备状态 `DONE` |

步骤1复制以下既有模型，不重新训练它们：

| 原路径 | 统一目标 |
|---|---|
| 根目录 baseline checkpoint | `MODELS/baseline_diffplanner.pth` |
| `STYLE_LORA_ENCODER/preference_encoder.pt` | `MODELS/preference_encoder.pt` |
| `STYLE_LORA_CONTINUOUS_LORA_DYN_Q_LAT_TOPK` 的 High/Low | `MODELS/preference_adapter_high.pt`、`preference_adapter_low.pt` |
| `STYLE_LORA_SCENE_GATE_3K_OW2` 的 checkpoint/report | `MODELS/scene_gate.pt`、`scene_gate.report.json` |

## 4. 运行 MAIN 主线

PREPARE 完成后，在终端一启动：

```bash
conda activate mdsn_py39
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
CAST_ROOT=$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS

nohup env \
  CAST_GPU=2 \
  CAST_START_STEP=1 \
  NUPLAN_DATA_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston \
  NUPLAN_MAPS_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps \
  bash stylelora/eval/run_01_main.sh \
  > "$CAST_ROOT/LOGS/nohup_main.log" 2>&1 &

echo "MAIN PID=$!"
```

实时查看运行输出：
```bash
tail -f "$CAST_ROOT/LOGS/nohup_main.log"
```

MAIN 内部顺序：

| 步骤 | 内容 |
|---:|---|
| 1 | 检查 PREPARE 和所有输入 |
| 2 | 平衡开环：无门控完整适配器 |
| 3 | 平衡开环：完整 CAST 场景门控 |
| 4 | 平衡开环：Scalar-only 消融 |
| 5 | 平衡开环：No-lateral 消融 |
| 6 | 300候选闭环：Scalar-only |
| 7 | 300候选闭环：No-lateral |
| 8 | 300候选闭环：Without adaptive gate |
| 9 | 300候选闭环：Full CAST |
| 10 | 取四种配置、五个 `rho` 的共同成功 token 交集并重算正式报告 |
| 11–17 | 依次生成 MAIN 的 A–G 表格和图 |
| 18 | 核对 MAIN 固定产物与最终闭环场景数 |

闭环步骤允许单场景失败并继续采集。失败 token、所属方法、`rho` 和异常信息保存在：

```text
MAIN_RESULTS/CLOSED_LOOP/05_COMMON_SUBSET/common_valid_scenarios_report.json
```

## 4.1 同时运行 FULL 全量线

FULL 必须使用另一张空闲 GPU。PREPARE 完成后，在终端二启动，例如 GPU 3：

```bash
conda activate mdsn_py39
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
CAST_ROOT=$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS

nohup env \
  CAST_GPU=1 \
  NUPLAN_DATA_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston \
  NUPLAN_MAPS_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps \
  bash stylelora/eval/run_01_full.sh \
  > "$CAST_ROOT/LOGS/nohup_full.log" 2>&1 &

echo "FULL PID=$!"


```

实时查看运行输出：
```bash
tail -f "$CAST_ROOT/LOGS/nohup_full.log"
```

FULL 内部顺序：

| 步骤 | 内容 |
|---:|---|
| 1 | 检查 PREPARE 和所有输入 |
| 2 | 完整验证 manifest：关闭门控开环 |
| 3 | 完整验证 manifest：启用门控开环 |
| 4 | 生成 FULL 的 C 表图 |
| 5 | 生成 FULL 的 D 表图 |
| 6 | 核对 FULL 开环产物 |

MAIN 和 FULL 可以同时运行，但必须各自独占一张物理 GPU，并保证 CPU、内存和磁盘带宽充足。不要在同一张 GPU 上同时启动两条流水线。如果只有 GPU 2 可用，就先完成 MAIN，再运行 FULL。

查看并行状态：

```bash
cat "$CAST_ROOT/STATUS/MAIN.status"
cat "$CAST_ROOT/STATUS/FULL.status"

tail -f "$CAST_ROOT/LOGS/nohup_main.log"
tail -f "$CAST_ROOT/LOGS/nohup_full.log"
```

## 5. 最终汇总

确认 MAIN 和 FULL 的状态第一行都为 `DONE` 后执行：

```bash
conda activate mdsn_py39
cd /home/lisw/programs/Nuplan-Diffusion-Baseline

RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
CAST_ROOT=$RECORD_ROOT/CAST_EAAI_PAPER_RESULTS

nohup env CAST_GPU=2 \
  bash stylelora/eval/run_02_finalize.sh \
  > "$CAST_ROOT/LOGS/nohup_finalize.log" 2>&1 &

echo "FINALIZE PID=$!"
```

最终检查：

```bash
cat "$CAST_ROOT/STATUS/FINALIZE.status"
tail -n 50 "$CAST_ROOT/LOGS/nohup_finalize.log"
wc -l "$CAST_ROOT/ARTIFACT_MANIFEST.txt"
```

## 6. 某一步失败后的恢复

先查看状态和对应日志：

```bash
cat "$CAST_ROOT/STATUS/PREPARE.status"
cat "$CAST_ROOT/STATUS/MAIN.status"
cat "$CAST_ROOT/STATUS/FULL.status"
cat "$CAST_ROOT/STATUS/FINALIZE.status"
```

状态文件会明确显示类似：

```text
FAILED
step=7 300候选闭环：无横向约束消融
line=...
exit_code=...
script=.../run_01_main.sh
```

修复问题后从失败步骤继续。例如 MAIN 第7步失败：

```bash
nohup env CAST_GPU=2 CAST_START_STEP=7 \
  bash stylelora/eval/run_01_main.sh \
  > "$CAST_ROOT/LOGS/nohup_main_resume_step7.log" 2>&1 &
```

其他流水线同理：

```bash
# PREPARE 从第6步继续
nohup env CAST_GPU=2 CAST_START_STEP=6 bash stylelora/eval/run_00_prepare.sh \
  > "$CAST_ROOT/LOGS/nohup_prepare_resume_step6.log" 2>&1 &

# FULL 从第3步继续
nohup env CAST_GPU=3 CAST_START_STEP=3 bash stylelora/eval/run_01_full.sh \
  > "$CAST_ROOT/LOGS/nohup_full_resume_step3.log" 2>&1 &
```

`CAST_START_STEP` 之前的产物会原样保留。恢复时必须从状态文件显示的失败步骤开始，不能直接跳到后续步骤。

## 7. 论文图表与证明目的

| 小节 | MAIN 固定产物 | 证明内容 |
|---|---|---|
| A. Experimental Setup | `A_experimental_setup/table_1_experimental_setup.csv` | 数据、模型、强度网格和最终闭环规模透明 |
| B. Structured Style Representation | `table_2`、`figure_1` | 高维 latent 与结构化风格锚点具有一致性 |
| C. Continuous Style Tuning | `table_3`、`figure_2` | 外部强度 `rho` 连续控制整体风格，且 `rho=0` 保持 baseline |
| D. Adaptive Style Tuning | `table_4`、`figure_3` | 同一连续门控根据环境表征限制有效强度 |
| E. Closed-loop Evaluation | `table_5`、`figure_4` | 在完全相同 token 上报告官方综合分、安全指标和逐场景配对差异 |
| F. Ablation and Efficiency | `table_6` | 验证结构化对齐、横向约束和自适应门控三部分 |
| G. Qualitative Analysis | `figure_5` | 从最终共同成功 token 中展示 `rho=-1,0,+1` 的轨迹和速度差异 |

`FULL_RESULTS` 只作为完整验证 manifest 的补充证据，不替代 MAIN 的平衡开环主结果。论文统一讨论纵向驾驶风格调节，不再把样本拆成 free-drive 和 car-follow 两套任务；车距与 THW 仅在存在有效前车时作为辅助物理指标。

## 8. 旧目录整理与最终下载

在 PREPARE 完成前，必须保留：

```text
CACHE/
STYLE_LORA_CONTINUOUS_LORA_DYN_Q_LAT_TOPK/
STYLE_LORA_ENCODER/
STYLE_LORA_PREFERENCE/
STYLE_LORA_SCENE_GATE/
STYLE_LORA_SCENE_GATE_3K_OW2/
args.json
best_model-epoch_116-train_loss_0.0701.pth
```

以下目录不被新流水线读取，可以先移动到统一归档目录，不建议立刻永久删除：

```text
STYLE_LORA_CLOSED_LOOP_FORMAL_DYN_Q/
STYLE_LORA_CLOSED_LOOP_FORMAL_DYN_Q_LAT_TOPK/
STYLE_LORA_CLOSED_LOOP_FORMAL_SCENE_GATE_3K_OW2/
STYLE_LORA_CONTINUOUS_LORA_DYN_Q/
STYLE_LORA_CONTINUOUS_LORA_DYN_Q_LAT/
STYLE_LORA_PREFERENCE_ALIGNMENT/
STYLE_LORA_RHO_CONTINUITY_COMPARISON/
```

安全归档命令：

```bash
RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
ARCHIVE_ROOT=$RECORD_ROOT/ARCHIVE_OLD_STYLE_LORA_RESULTS
mkdir -p "$ARCHIVE_ROOT"

for NAME in \
  STYLE_LORA_CLOSED_LOOP_FORMAL_DYN_Q \
  STYLE_LORA_CLOSED_LOOP_FORMAL_DYN_Q_LAT_TOPK \
  STYLE_LORA_CLOSED_LOOP_FORMAL_SCENE_GATE_3K_OW2 \
  STYLE_LORA_CONTINUOUS_LORA_DYN_Q \
  STYLE_LORA_CONTINUOUS_LORA_DYN_Q_LAT \
  STYLE_LORA_PREFERENCE_ALIGNMENT \
  STYLE_LORA_RHO_CONTINUITY_COMPARISON
do
  if test -d "$RECORD_ROOT/$NAME"; then
    mv -n -- "$RECORD_ROOT/$NAME" "$ARCHIVE_ROOT/"
    echo "已归档：$NAME"
  fi
done
```

不要移动或删除 `CACHE`、恢复后的 `DATA_SPLITS_CONFIG`、根目录 baseline/args，或者六个 PREPARE 来源目录。至少等 `STATUS/FINALIZE.status` 显示 `DONE`、完整结果已下载并核对之后，再决定是否清除归档。

最终只需下载整个目录：

```text
/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS
```

其中 `ARTIFACT_MANIFEST.txt` 是本轮 MODELS、COMMON、MAIN 和 FULL 所有文件的最终清单。
