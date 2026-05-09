# NuPlan Diffusion Baseline

> 文档更新时间：**2026-05-09 (UTC+8)**

## 1. 项目简介

这是一个面向 NuPlan 的研究基线工程，当前支持两种可切换模型：

- `diffusion-planner`
- `wayformer`

本仓库已经将训练与仿真入口改为**注册制**，便于后续继续扩展新架构并保持主流程稳定。

## 2. 目录结构

```text
baseline/
  config/                    # 训练与仿真配置
    method/                  # 训练方法配置 (diffusion_planner / wayformer)
    planner/                 # 仿真 planner 配置 (diffusion_planner / wayformer)
  common/                    # 通用数据增强/数据集工具
  core/                      # 注册器与核心编排工具
  data_process/              # NuPlan 场景转训练样本
  model/
    diff_planner/            # Diffusion Planner 模型
    wayformer/               # Wayformer 模型
  simulation/                # 闭环仿真封装、渲染、评估
  train/                     # 训练工具模块
  utils/                     # 配置、归一化、日志、DDP 工具
  resources/                 # baseline 内置 JSON 资源（含 mini/boston 划分）

  train.py                   # 训练主入口（注册制）
  process_data.py            # 数据处理主入口
  run_simulation.py          # 闭环仿真主入口（注册制）
  run_nuboard_viewer.py      # nuBoard 可视化入口
```

## 3. 环境准备

```bash
conda env create -f environment.yml
conda activate testlocal39
```

确保 `nuplan-devkit` 位于仓库根目录（推荐）或可被 `PYTHONPATH` 访问。

## 4. 快速开始

### 4.1 数据处理

先做前置划分（只做一次，后续会复用已存在 test 列表）：

```bash
python baseline/prepare_data_splits.py
```

再做缓存生成：

```bash
python baseline/process_data.py \
  --data_path <nuplan_db_dir> \
  --map_path <nuplan_map_dir> \
  --save_path <npz_output_dir>
```

说明：
- `process_data.py` 默认读取 `baseline/resources/mini/splits/mini_cache_logs.json`
- 处理完成后默认生成样本列表：`baseline/resources/mini/training_mini.json`

### 4.2 训练（注册制）

默认配置文件：`baseline/config/train.yaml`

- 训练 Diffusion（默认）：

```bash
python baseline/train.py
```

- 训练 Wayformer：

```bash
python baseline/train.py method=wayformer
```

说明：

- `train.py` 会按 `method.name` 自动从注册表选择模型与 train/val 循环。
- 当前已注册：`diffusion-planner`、`wayformer`。
- 在线日志默认使用 `swanlab`，可在 `baseline/config/train.yaml` 切换为 `wandb` 或 `disabled`。

### 4.3 闭环仿真（注册制）

脚本：`baseline/run_simulation.py`

运行前修改：

- `PLANNER`（`diffusion_planner` 或 `wayformer`）
- `CHECKPOINT_DIR / ARGS_FILE / CKPT_FILE`
- 数据与地图路径常量
- `ONLINE_LOGGER`（默认 `swanlab`）

运行：

```bash
python baseline/run_simulation.py
```

说明：

- 脚本会动态注入 `planner.<name>.*` Hydra 覆盖参数；
- 当 `SPLIT="mini"` 且 `baseline/resources/mini/splits/mini_test_logs.json` 存在时，
  会自动覆盖 `scenario_filter.log_names` 为这份 test 列表；
- `mini.yaml` 里的其他过滤参数（如 `limit_total_scenarios`）保持生效；
- 默认同时保存：
  - 视频：`simulation_video/`
  - step 级原始导出：`raw_step_data/`

### 4.4 nuBoard 可视化

```bash
python baseline/run_nuboard_viewer.py
```

## 5. resources 资源说明

`baseline/resources/` 当前重点如下：

- `normalization.json`
- `normalization_train.json`
- `mini/nuplan_scenarios_mini.json`
- `mini/splits/mini_cache_logs.json`
- `mini/splits/mini_test_logs.json`
- `mini/training_mini.json`（由 `process_data.py` 处理后生成）

## 6. 注意事项

- 部分脚本包含本地绝对路径示例，请按机器环境修改。
- 首次运行建议先用 mini 数据做链路冒烟测试。
- 本地运行配置细节见：`baseline/docs/main_scripts_local_run.md`
- 若你继续增加新模型，建议仅做两件事：
  1. 在训练入口注册模型与 train/val 函数；
  2. 在仿真入口注册 planner 并补对应 `config/planner/*.yaml`。
