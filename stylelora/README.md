# StyleLoRA 新人运行指南

## 1. 研究故事

StyleLoRA 研究的是：在不重训基础 DiffPlanner、尽量保持原始驾驶能力的前提下，让规划器按照可解释的连续驾驶偏好改变轨迹风格。

整条技术线可以概括为：

1. **结构化偏好学习**：从专家轨迹中计算速度利用、纵向动态、跟车间距等具有物理含义的指标；在同类场景内部转成经验百分位和弱排序偏好，而不是依赖 aggressive/normal/conservative 人工分类标签。
2. **偏好空间蒸馏**：偏好编码器把场景表征 `h_c` 和专家轨迹映射到结构化偏好空间，输出可排序标量 `s` 与高维偏好表示 `z`；训练 LoRA 时再用冻结编码器约束生成轨迹，使规划器继承这个偏好空间。
3. **基线保持的参数高效控制**：冻结 DiffPlanner，只训练 ego 相关 LoRA 参数；用 high/low 两个方向和连续强度 `rho` 控制风格，同时检查 `rho=0` 与原始 baseline 的一致性。

这里的重点不是给每个场景重新定义一个孤立“风格类别”，而是：

- 场景决定哪些物理偏好轴有效；
- 偏好编码器把不同场景下的有效证据映射到统一表示；
- LoRA 把统一偏好表示蒸馏进基础规划器；
- `rho` 负责运行时连续调节风格强度。

当前正式支持的主场景是：

- `straight_free_drive`
- `straight_car_follow`

## 2. 目录说明

```text
stylelora/
├── pipeline/     原始 DB、cache 生成、场景划分
├── data/         无分类标签的数据读取、物理指标和弱排序偏好
├── model/        CSPQ 偏好编码器
├── training/     偏好空间与 Preference-LoRA 训练
├── lora/         LoRA 注入、checkpoint、rollout、风格指标、闭环桥接
├── scripts/      所有正式 main 入口
├── config/       NuPlan 闭环 planner 配置
└── tests/        偏好编码器与 LoRA 核心测试
```

`stylelora` 内部导入已经统一为 `stylelora.*`。旧的 `research_lora_2`、`research_lora`、`research_v1`、`research_v2` 不再是运行依赖。

`baseline/` 和 `nuplan-devkit/` 仍是外部基础设施：前者提供 DiffPlanner、DataProcessor 和 NuPlan simulation planner，后者提供 NuPlan 场景与仿真框架。

## 3. 运行前准备

所有命令都应在仓库根目录运行：

```bash
python -m stylelora.scripts.<脚本名> --help
```

推荐优先设置以下环境变量，避免在脚本中硬编码服务器路径：

```bash
export NUPLAN_DATA_PATH=/path/to/nuplan/db
export NUPLAN_MAP_PATH=/path/to/nuplan/maps
export NUPLAN_RECORD_ROOT=/path/to/experiment_record
export NUPLAN_CACHE_ROOT=/path/to/cache
```

下面示例中的路径只是占位符，实际参数以各脚本 `--help` 为准。

## 4. 完整运行顺序

### 步骤 1：划分原始 DB

```bash
python -m stylelora.scripts.build_raw_split \
  --log_names_path /path/to/all_log_names.json \
  --output_dir /path/to/data_splits
```

输出保留测试日志，并生成后续 cache 阶段使用的日志列表。

### 步骤 2：划分 cache 的 train/val 日志

```bash
python -m stylelora.scripts.build_cache_split --split_name cache_train_val
```

该入口复用历史数据流水线的默认目录和环境变量。第一次运行前务必先执行 `--help`，确认服务器路径。

### 步骤 3：原始 DB 转 planner cache

```bash
python -m stylelora.scripts.process_cache \
  --data_path /path/to/nuplan/db \
  --map_path /path/to/nuplan/maps \
  --train_log_names_path /path/to/train_log_names.json \
  --val_log_names_path /path/to/val_log_names.json \
  --save_path /path/to/planner_cache \
  --output_list_path /path/to/cache_list.json \
  --manifest_path /path/to/cache_manifest.json
```

这一步调用 `baseline.data_process.data_processor.DataProcessor` 生成 DiffPlanner 使用的 `.npz`。

### 步骤 4：构建直线场景划分

```bash
python -m stylelora.scripts.build_scene_split \
  --planner_cache_dir /path/to/planner_cache \
  --data_list_path /path/to/cache_list.json \
  --manifest_path /path/to/cache_manifest.json \
  --split_name all \
  --train_output_dir /path/to/scene_split/train \
  --val_output_dir /path/to/scene_split/val
```

核心输出：

```text
/path/to/scene_split/train/split_index.jsonl
/path/to/scene_split/val/split_index.jsonl
```

### 步骤 5：生成无分类标签的 source manifest

```bash
python -m stylelora.scripts.build_preference_source_manifest \
  --train-index /path/to/scene_split/train/split_index.jsonl \
  --val-index /path/to/scene_split/val/split_index.jsonl \
  --output-dir /path/to/source_manifests \
  --cache-root /path/to/planner_cache
```

这一步只保留场景、cache 路径和溯源信息，不使用旧的三分类 style label。

### 步骤 6：生成弱排序偏好标签

先在训练集拟合经验 CDF：

```bash
python -m stylelora.scripts.build_weak_preference \
  --manifest /path/to/source_manifests/train.jsonl \
  --cache-root /path/to/planner_cache \
  --output /path/to/preference/weak_preference_train.jsonl \
  --save-cdf /path/to/preference/train_cdf.json
```

验证集必须复用训练集 CDF，不能重新拟合：

```bash
python -m stylelora.scripts.build_weak_preference \
  --manifest /path/to/source_manifests/val.jsonl \
  --cache-root /path/to/planner_cache \
  --output /path/to/preference/weak_preference_val.jsonl \
  --load-cdf /path/to/preference/train_cdf.json
```

随后审计偏好覆盖率和分布：

```bash
python -m stylelora.scripts.audit_preference \
  --manifest /path/to/preference/weak_preference_train.jsonl \
  --output /path/to/preference/audit_preference.json
```

### 步骤 7：提取冻结的环境场景表征

训练集和验证集分别运行：

```bash
python -m stylelora.scripts.extract_scene_features \
  --args-file /path/to/baseline_args.json \
  --baseline-checkpoint /path/to/baseline.ckpt \
  --manifest /path/to/preference/weak_preference_train.jsonl \
  --cache-root /path/to/planner_cache \
  --output /path/to/preference/scene_features_train.npy \
  --feature-index /path/to/preference/scene_features_train_index.jsonl
```

验证集只需把 manifest 和输出路径替换成 val 对应文件。

### 步骤 8：训练结构化偏好空间

```bash
python -m stylelora.scripts.train_preference_encoder \
  --train-manifest /path/to/preference/weak_preference_train.jsonl \
  --val-manifest /path/to/preference/weak_preference_val.jsonl \
  --train-feature-npy /path/to/preference/scene_features_train.npy \
  --train-feature-index /path/to/preference/scene_features_train_index.jsonl \
  --val-feature-npy /path/to/preference/scene_features_val.npy \
  --val-feature-index /path/to/preference/scene_features_val_index.jsonl \
  --cache-root /path/to/planner_cache \
  --output /path/to/encoder/preference_encoder.pt \
  --output-last /path/to/encoder/preference_encoder_last.pt
```

### 步骤 9：评估编码器并导出 latent bank

```bash
python -m stylelora.scripts.evaluate_preference_encoder \
  --checkpoint /path/to/encoder/preference_encoder.pt \
  --train-manifest /path/to/preference/weak_preference_train.jsonl \
  --val-manifest /path/to/preference/weak_preference_val.jsonl \
  --train-feature-npy /path/to/preference/scene_features_train.npy \
  --train-feature-index /path/to/preference/scene_features_train_index.jsonl \
  --val-feature-npy /path/to/preference/scene_features_val.npy \
  --val-feature-index /path/to/preference/scene_features_val_index.jsonl \
  --cache-root /path/to/planner_cache \
  --latent-bank-train /path/to/encoder/latent_bank_train.npy \
  --latent-bank-train-index /path/to/encoder/latent_bank_train_index.jsonl \
  --latent-bank-val /path/to/encoder/latent_bank_val.npy \
  --latent-bank-val-index /path/to/encoder/latent_bank_val_index.jsonl
```

只有编码器质量、排序单调性和 latent 分布检查通过后，才进入 LoRA 训练。

### 步骤 10：分别训练 high/low LoRA

high 方向：

```bash
python -m stylelora.scripts.train_preference_lora \
  --args-file /path/to/baseline_args.json \
  --baseline-checkpoint /path/to/baseline.ckpt \
  --direction high \
  --manifest /path/to/preference/weak_preference_train.jsonl \
  --cache-root /path/to/planner_cache \
  --latent-bank /path/to/encoder/latent_bank_train.npy \
  --latent-bank-index /path/to/encoder/latent_bank_train_index.jsonl \
  --cspq-checkpoint /path/to/encoder/preference_encoder.pt \
  --feature-npy /path/to/preference/scene_features_train.npy \
  --feature-index /path/to/preference/scene_features_train_index.jsonl \
  --output /path/to/lora/preference_high.pt
```

low 方向使用相同命令，把 `--direction` 改为 `low`，输出改为 `preference_low.pt`。

正式实验建议同时传入 val manifest、val feature 和 val latent bank；先用小步数确认管线时可以省略验证集。

### 步骤 11：开环 rho 扫描

```bash
python -m stylelora.scripts.evaluate_open_loop \
  --args-file /path/to/baseline_args.json \
  --baseline-checkpoint /path/to/baseline.ckpt \
  --adapter-high /path/to/lora/preference_high.pt \
  --adapter-low /path/to/lora/preference_low.pt \
  --cspq-checkpoint /path/to/encoder/preference_encoder.pt \
  --manifest /path/to/preference/weak_preference_val.jsonl \
  --cache-root /path/to/planner_cache \
  --feature-npy /path/to/preference/scene_features_val.npy \
  --feature-index /path/to/preference/scene_features_val_index.jsonl \
  --latent-bank /path/to/encoder/latent_bank_val.npy \
  --latent-bank-index /path/to/encoder/latent_bank_val_index.jsonl \
  --output-report /path/to/reports/open_loop.json
```

必须关注：

- `rho=0` identity 是否通过；
- `s` 是否随 `rho` 单调变化；
- high/low 是否朝正确方向移动；
- ADE/FDE、邻车变化和三轴物理指标是否可接受。

### 步骤 12：NuPlan 闭环验证

```bash
python -m stylelora.scripts.evaluate_closed_loop \
  --args-file /path/to/baseline_args.json \
  --baseline-checkpoint /path/to/baseline.ckpt \
  --high-adapter /path/to/lora/preference_high.pt \
  --low-adapter /path/to/lora/preference_low.pt \
  --normalization-file /path/to/normalization.json \
  --output-root /path/to/reports/closed_loop \
  --config-root /path/to/repo/baseline/config \
  --nuplan-overrides-json '["scenario_builder=nuplan", "scenario_filter=..."]'
```

闭环入口会对多个 `rho` 分别启动 NuPlan simulation，使用
`stylelora/config/planner/lora_diffusion_planner.yaml`，最后汇总碰撞、可行驶区域、TTC 和舒适性指标。

## 5. 删除旧目录前的检查

先运行：

```bash
python -m compileall -q stylelora
python -m pytest stylelora/tests -q
```

然后检查代码导入：

```bash
rg -n "^(from|import) (research_lora_2|research_lora|research_v1|research_v2)" stylelora
```

该命令应无输出。文档或历史格式字符串里出现旧目录名不代表运行依赖；真正必须为零的是 Python import 和 Hydra `_target_`。

## 6. 迁移来源

| StyleLoRA 目录 | 原始来源 |
|---|---|
| `data/`、`model/`、`training/`、主要偏好脚本 | `research_lora_2/` |
| `lora/`、闭环桥接与 LoRA planner 配置 | `research_lora/` |
| `pipeline/data_pipeline/`、`pipeline/scene_data/` | `research_v1/` |
| `scripts/build_preference_source_manifest.py` | 新增的 scene split → 无标签偏好接口 |

`research_v2/style_prototype_residual` 没有进入当前 StyleLoRA 调用链，因此没有混入正式包。它属于另一条 prototype residual 实验线。
