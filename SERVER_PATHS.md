# StyleLoRA 服务器路径清单

本文记录服务器路径默认值。StyleLoRA 主流程优先从 `stylelora/config/paths.local.json` 读取机器路径；若该文件尚未创建，则使用 `paths.example.json` 中的服务器默认值。`paths.local.json` 不进入 Git，协作者可以各自修改。环境变量仍可覆盖相应路径。当前已知的运行环境为 `mdsn_py39`。

新增机器路径时，在本机 `paths.local.json` 的 `paths` 对象中增加键值，然后由 Python 代码调用 `get_path("new_key")`，或由 shell 调用 `python stylelora/config/runtime_paths.py --get new_key`。模板使用说明见 `stylelora/config/README.md`。

## 1. 当前 V5 主流程实际使用的默认路径

| 用途 | 服务器默认路径 | 来自/用途 |
|---|---|---|
| 仓库目录 | `/home/lisw/programs/Nuplan-Diffusion-Baseline` | V5 训练/开环脚本默认 `REPO_ROOT`；闭环脚本从自身位置自动定位仓库 |
| 实验记录根目录 | `/mnt/mydata/lishangwen/Nuplan-Baseline-Record` | V5 与闭环脚本的 `CAST_RECORD_ROOT` 默认值 |
| 已有输入结果根目录 | `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS` | `CAST_SOURCE_ROOT` 默认值，放置 baseline、编码器和训练/验证输入 |
| 当前结果根目录 | `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS3` | `CAST_OUTPUT_ROOT` 默认值，V5 模型、开环、闭环、图和日志写入位置 |
| 训练/验证 cache | `/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CACHE/boston_cache_train_val` | V5 训练脚本从 `paths.local.json: cache_root` 读取，也可用 `CAST_CACHE_ROOT` 覆盖 |
| NuPlan Boston 数据 | `/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston` | 闭环 `NUPLAN_DATA_ROOT` 默认值 |
| NuPlan 地图 | `/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps` | 闭环 `NUPLAN_MAPS_ROOT` 默认值 |

### V5 脚本需要存在的输入

下列文件由 `stylelora/scripts/run_longitudinal_response_v5.sh` 使用。训练前要确认它们在服务器上真实存在：

```text
SOURCE_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS
SOURCE_ROOT/INPUTS/args.json
SOURCE_ROOT/INPUTS/weak_preference_train.jsonl
SOURCE_ROOT/INPUTS/weak_preference_val.jsonl
SOURCE_ROOT/INPUTS/latent_bank_train.npy
SOURCE_ROOT/INPUTS/latent_bank_train_index.jsonl
SOURCE_ROOT/INPUTS/latent_bank_val.npy
SOURCE_ROOT/INPUTS/latent_bank_val_index.jsonl
SOURCE_ROOT/INPUTS/scene_features_train.npy
SOURCE_ROOT/INPUTS/scene_features_train_index.jsonl
SOURCE_ROOT/INPUTS/scene_features_val.npy
SOURCE_ROOT/INPUTS/scene_features_val_index.jsonl
SOURCE_ROOT/MODELS/baseline_diffplanner.pth
SOURCE_ROOT/MODELS/preference_encoder.pt

OUTPUT_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS3
OUTPUT_ROOT/MODELS/ORDERED_FEASIBLE_V4/conditional_high_ordered_feasible_v4.pt
OUTPUT_ROOT/MODELS/ORDERED_FEASIBLE_V4/conditional_low_ordered_feasible_v4.pt
OUTPUT_ROOT/MODELS/ORDERED_FEASIBLE_V4/conditional_router_ordered_feasible_v4.pt
```

V5 产生的主要文件：

```text
OUTPUT_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5/conditional_high_longitudinal_response_v5.pt
OUTPUT_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5/conditional_low_longitudinal_response_v5.pt
OUTPUT_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5/conditional_router_longitudinal_response_v5.pt
OUTPUT_ROOT/OPEN_LOOP/LONGITUDINAL_RESPONSE_V5/open_longitudinal_response_v5_full.json
OUTPUT_ROOT/OPEN_LOOP/LONGITUDINAL_RESPONSE_V5/GRID_BALANCED_10/
OUTPUT_ROOT/LOGS/LONGITUDINAL_RESPONSE_V5/
OUTPUT_ROOT/STATUS/LONGITUDINAL_RESPONSE_V5.status
```

### 闭环额外路径

V5 闭环脚本 `run_collision_drivable_v5_closed_loop.sh` 还需要：

```text
NuPlan DB:  /mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston
Maps:      /mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps
Tokens:    /mnt/mydata/lishangwen/Nuplan-Baseline-Record/CAST_EAAI_PAPER_RESULTS3/INPUTS/closed_loop_tokens_balanced_50.json
```

默认读取 `SOURCE_ROOT/INPUTS/args.json`、`SOURCE_ROOT/INPUTS/normalization.json`、`SOURCE_ROOT/MODELS/baseline_diffplanner.pth`，以及 `OUTPUT_ROOT/MODELS/LONGITUDINAL_RESPONSE_V5/` 下的 V5 adapters/router。若要使用仓库 `weights/` 目录中的文件，可在 `paths.local.json` 设置 `args_file`、`normalization_file` 及对应模型路径；设为 `null` 或不填写时保持上述默认位置。闭环输出默认位于：

```text
OUTPUT_ROOT/CLOSED_LOOP/LONGITUDINAL_RESPONSE_V5_COLLISION_DRIVABLE/SHARD_<NEGATIVE|CENTER|POSITIVE>/
OUTPUT_ROOT/LOGS/LONGITUDINAL_RESPONSE_V5_COLLISION_DRIVABLE/
```

## 2. 当前主脚本可覆盖的路径/运行变量

| 环境变量 | 默认值 | 影响范围 |
|---|---|---|
| `CAST_REPO_ROOT` | `paths.local.json: repo_root` | V5 训练/开环脚本仓库位置 |
| `CAST_RECORD_ROOT` | `paths.local.json: record_root` | 输入记录和 cache 根目录 |
| `CAST_SOURCE_ROOT` | `paths.local.json: source_root` | V5 训练/闭环所需既有输入 |
| `CAST_OUTPUT_ROOT` | `paths.local.json: output_root` | 当前模型、评测和日志输出 |
| `CAST_CACHE_ROOT` | `paths.local.json: cache_root` | V5 训练 cache，可独立覆盖 |
| `NUPLAN_DATA_ROOT` | `paths.local.json: data_root` | V5 闭环 DB 路径 |
| `NUPLAN_MAPS_ROOT` | `paths.local.json: maps_root` | V5 闭环地图路径 |
| `CAST_TOKENS_FILE` | `paths.local.json: tokens_file` | 闭环共同场景 token 文件 |
| `CAST_GPU` | `paths.local.json: gpu`（默认 `2`） | V5 训练/开环脚本使用的 GPU 编号 |
| `CAST_START_STEP` | `1` | V5 一键脚本恢复起始步骤，不是路径 |

示例：

```bash
export CAST_REPO_ROOT=/home/lisw/programs/Nuplan-Diffusion-Baseline
export CAST_RECORD_ROOT=/mnt/mydata/lishangwen/Nuplan-Baseline-Record
export CAST_SOURCE_ROOT="$CAST_RECORD_ROOT/CAST_EAAI_PAPER_RESULTS"
export CAST_OUTPUT_ROOT="$CAST_RECORD_ROOT/CAST_EAAI_PAPER_RESULTS3"
export NUPLAN_DATA_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston
export NUPLAN_MAPS_ROOT=/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps
export CAST_TOKENS_FILE="$CAST_OUTPUT_ROOT/INPUTS/closed_loop_tokens_balanced_50.json"
```

注意：闭环脚本会从脚本文件位置自动定位仓库，因此 `CAST_REPO_ROOT` 不是闭环脚本的控制变量；它只在 V5 训练/开环一键脚本中生效。V5 训练 cache 已独立配置，可通过 `cache_root` 或 `CAST_CACHE_ROOT` 设置。

## 3. Python 数据流水线的通用默认路径

`stylelora/paths.py` 与 `stylelora/pipeline/paths.py` 还为分割、缓存和偏好数据准备提供独立默认值。这些不是 V5 一键训练脚本的主要路径来源：

| 环境变量 | 当前服务器默认值/派生路径 |
|---|---|
| `NUPLAN_SERVER_PROGRAM_ROOT` | `paths.local.json: program_root` |
| `NUPLAN_SERVER_REPO_ROOT` | `paths.local.json: repo_root` |
| `NUPLAN_SERVER_DEVKIT_ROOT` | `paths.local.json: devkit_root` |
| `NUPLAN_SERVER_STYLE_RECORD_ROOT` | `paths.local.json: style_record_root` |
| `NUPLAN_SERVER_CACHE_RECORD_ROOT` | `paths.local.json: record_root` |
| `NUPLAN_SERVER_DATA_ROOT` | `paths.local.json: data_root` |
| `NUPLAN_SERVER_MAP_ROOT` | `paths.local.json: maps_root` |
| `NUPLAN_SERVER_LOG_NAMES_PATH` | `paths.local.json: log_names_path` |
| `NUPLAN_RECORD_ROOT` | `$NUPLAN_SERVER_CACHE_RECORD_ROOT` |
| `NUPLAN_CACHE_ROOT` | `$NUPLAN_RECORD_ROOT/CACHE` |
| `NUPLAN_DATA_SPLITS_ROOT` | `$NUPLAN_RECORD_ROOT/DATA_SPLITS_CONFIG/boston_raw_seed3407` |
| `NUPLAN_CACHE_TRAIN_VAL_DIR` | `$NUPLAN_CACHE_ROOT/boston_cache_train_val` |
| `NUPLAN_CACHE_TRAIN_VAL_LIST_PATH` | `$NUPLAN_CACHE_ROOT/boston_cache_train_val_list.json` |
| `NUPLAN_CACHE_TRAIN_VAL_MANIFEST_PATH` | `$NUPLAN_CACHE_ROOT/boston_cache_train_val_manifest.json`（`pipeline/paths.py` 使用） |
| `NUPLAN_PLANNER_CACHE_DIR` | 默认等于 `NUPLAN_CACHE_TRAIN_VAL_DIR` |
| `NUPLAN_PLANNER_CACHE_LIST_PATH` | 默认等于 `NUPLAN_CACHE_TRAIN_VAL_LIST_PATH` |
| `NUPLAN_STYLE_MANIFESTS_ROOT` | `$NUPLAN_RECORD_ROOT/STYLE_LORA_MANIFESTS` |
| `NUPLAN_STYLE_TRAIN_MANIFEST` | `$NUPLAN_RECORD_ROOT/STYLE_LORA_MANIFESTS/train.jsonl` |
| `NUPLAN_STYLE_VAL_MANIFEST` | `$NUPLAN_RECORD_ROOT/STYLE_LORA_MANIFESTS/val.jsonl` |
| `NUPLAN_STYLE_TEST_MANIFEST` | `$NUPLAN_RECORD_ROOT/STYLE_LORA_MANIFESTS/test.jsonl` |
| `NUPLAN_PREFERENCE_ROOT` | `$NUPLAN_RECORD_ROOT/STYLE_LORA_PREFERENCE` |
| `NUPLAN_PREFERENCE_MANIFEST` / `NUPLAN_PREFERENCE_VAL_MANIFEST` | `STYLE_LORA_PREFERENCE/weak_preference_train.jsonl` / `weak_preference_val.jsonl` |
| `NUPLAN_PREFERENCE_FEATURE_NPY` / `NUPLAN_PREFERENCE_FEATURE_INDEX` | `STYLE_LORA_PREFERENCE/scene_features_train.npy` / `scene_features_train_index.jsonl` |
| `NUPLAN_PREFERENCE_FEATURE_VAL_NPY` / `NUPLAN_PREFERENCE_FEATURE_VAL_INDEX` | `STYLE_LORA_PREFERENCE/scene_features_val.npy` / `scene_features_val_index.jsonl` |
| `NUPLAN_PREFERENCE_AUDIT` | `STYLE_LORA_PREFERENCE/audit_preference.json` |
| `NUPLAN_ENCODER_ROOT` | `$NUPLAN_RECORD_ROOT/STYLE_LORA_ENCODER` |
| `NUPLAN_ENCODER_CHECKPOINT` / `NUPLAN_ENCODER_LAST_CHECKPOINT` | `STYLE_LORA_ENCODER/preference_encoder.pt` / `preference_encoder_last.pt` |
| `NUPLAN_ENCODER_LATENT_BANK` / `NUPLAN_ENCODER_LATENT_BANK_INDEX` | `STYLE_LORA_ENCODER/latent_bank_train.npy` / `latent_bank_train_index.jsonl` |
| `NUPLAN_ENCODER_VAL_LATENT` / `NUPLAN_ENCODER_VAL_LATENT_INDEX` | `STYLE_LORA_ENCODER/latent_bank_val.npy` / `latent_bank_val_index.jsonl` |
| `NUPLAN_ENCODER_EVAL_REPORT` | `STYLE_LORA_ENCODER/evaluate_encoder.json` |

另外，`NUPLAN_DATA_PATH`、`NUPLAN_MAP_PATH`、`NUPLAN_LOG_NAMES_PATH` 分别覆盖通用流水线的数据、地图和日志名文件；缺省时使用上方对应的 `NUPLAN_SERVER_*` 值。

以上 `NUPLAN_STYLE_*`、`NUPLAN_PREFERENCE_*` 和 `NUPLAN_ENCODER_*` 是一般数据准备 CLI 的默认目录约定。当前 V5 一键脚本直接读取 `CAST_SOURCE_ROOT/INPUTS` 与 `CAST_SOURCE_ROOT/MODELS`，不会自动采用这些独立默认产物路径。

## 4. 仍留在 baseline 的旧路径

以下路径属于 baseline 自身的旧训练/仿真入口，不是 StyleLoRA V5 主线使用的路径；如果之后单独运行这些入口，需要另行检查：

| 文件 | 当前硬编码/默认路径 | 备注 |
|---|---|---|
| `baseline/config/train.yaml` | `/mnt/mydata/lishangwen/NuplanBaselinesRecord/...` 和 `/home/lsw/meta_programs/Nuplan-Diffusion-Baseline/...` | 旧 baseline 训练数据、日志和 normalization 路径 |
| `baseline/run_simulation.py` | 数据默认在 `/media/lsw/.../splits/mini`；checkpoint 位于 `/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline/...` | 旧的 baseline 闭环入口 |
| `baseline/process_data.py` | `/media/lsw/.../splits/mini`、`/media/lsw/.../maps`、`/media/lsw/.../CACHE/minicache` | 本地 Linux 机器的旧数据处理默认值 |
| `baseline/build_data_index_json.py` | `/media/lsw/Work/ubuntu_system/DATACACHE/...` | 旧数据索引工具 |
| `baseline/run_nuboard_viewer.py` | `/mnt/mydata/lishangwen/NuplanBaselinesRecord/...` 和 NuPlan 数据/地图目录 | 旧 nuBoard 入口 |
| `baseline/run_anchor_warm_start_simulation.py` | `/mnt/mydata/lishangwen/NuplanBaselinesRecord/...` 和 NuPlan 数据/地图目录 | 早期 AnchorWarmStart planner，不是当前 StyleLoRA planner |

## 5. 需要你重点确认的差异

1. **两个记录盘路径分别保留**：`Nuplan-Baseline-Record` 用作 V5 记录/cache 根目录，`NuplanBaselinesRecord` 用作通用 StyleLoRA 记录根目录；服务器模板分别保留为 `record_root` 与 `style_record_root`。
2. **仓库目录仍叫旧名**：服务器默认路径是 `Nuplan-Diffusion-Baseline`，虽然 GitHub 仓库现在叫 `StyleLoRA`。如果服务器 clone 目录不同，在本机 `paths.local.json` 中更新 `repo_root`。
3. **训练 cache 路径可独立配置**：V5 训练脚本读取 `paths.local.json` 的 `cache_root`，也可用 `CAST_CACHE_ROOT` 临时覆盖。
4. **输入与输出分属 RESULTS、RESULTS3**：V5 训练从 `CAST_EAAI_PAPER_RESULTS` 读取基础模型/数据产物、从 `CAST_EAAI_PAPER_RESULTS3` 读取 V4 初始化权重并写出 V5 结果。请确认 V4 三个 checkpoint 和 closed-loop tokens 确实都放在 RESULTS3 对应子目录。

## 6. 旧实验脚本

`run_v2_pure_gate.sh`、`run_bounded_feasible_v3.sh`、`run_ordered_feasible_v4.sh`、`run_v4_open_loop_physical_metrics.sh` 仍保留在代码中，用于历史实验或对照。它们也有服务器路径默认值，但不是当前 V5 主线；需要重跑时应分别检查各脚本顶部的 `REPO_ROOT`、`RECORD_ROOT`、`SOURCE_ROOT`、`CAST_ROOT`、cache 和输入模型路径。
