# Nuplan Diffusion Baseline

> 文档更新时间：**2026-05-08 (UTC+8)**

## 1. 项目简介

这是一个面向 NuPlan 的 **Diffusion Planner baseline** 工程包，目标是提供一套可复现、可整理、可共享的研究基线。

当前版本聚焦：

- `diffusion-planner` 训练主线
- `diffusion-planner` 闭环仿真主线
- 数据处理与标签校验脚本
- 统一后的 `baseline/` 目录结构

当前不包含（按当前整理策略有意去除）：

- Cognitive / Wayformer / Alpha 的运行入口

## 2. 目录结构

```text
baseline/
  config/                    # 训练与仿真配置
  common/                    # 通用数据增强/数据集工具
  core/                      # 预留核心编排模块
  data_process/              # NuPlan 场景转训练样本
  model/diff_planner/        # Diffusion Planner 模型
  simulation/                # 闭环仿真 planner / render / metrics
  train/                     # 训练工具模块（manage / train_utils / train_val_epoch）
  utils/                     # 配置、归一化、日志、DDP 工具
  resources/                 # baseline 内置 JSON 资源

  train.py                   # 训练主入口
  process_data.py            # 数据处理主入口
  run_simulation.py          # 闭环仿真入口
  run_nuboard_viewer.py      # nuBoard 可视化入口
  verify_codebook_labels.py  # 标签质量检查
  build_data_index_json.py   # 数据索引 JSON 生成
```

## 3. 环境准备

建议使用仓库内环境文件：

```bash
conda env create -f environment.yml
conda activate testlocal39
```

## 4. 快速开始

### 4.1 数据处理

```bash
python baseline/process_data.py \
  --data_path <nuplan_db_dir> \
  --map_path <nuplan_map_dir> \
  --save_path <npz_output_dir>
```

说明：

- `process_data.py` 会优先加载当前工程下的 `nuplan-devkit`
- 场景列表可以用 `--scenario_log_json` 指定

### 4.2 训练

```bash
python baseline/train.py
```

默认读取：

- `baseline/config/train.yaml`
- `baseline/config/method/diffusion_planner.yaml`

### 4.3 闭环仿真

```bash
python baseline/run_simulation.py
```

运行前请先检查脚本顶部路径（checkpoint、数据集、地图、输出目录）。

### 4.4 结果可视化（nuBoard）

```bash
python baseline/run_nuboard_viewer.py
```

运行前请先检查 `.nuboard` 路径和地图/数据目录路径。

### 4.5 标签分布校验

```bash
python baseline/verify_codebook_labels.py --data_path <npz_dir>
```

## 5. resources 资源说明

`baseline/resources/` 当前包含：

- `normalization.json`
- `normalization_train.json`
- `nuplan_scenarios_mini.json`
- `training_mini.json`

这些文件用于 baseline 默认配置和脚本启动兜底，不替代你自己的大规模数据集配置。

## 6. 输出目录（典型）

- 训练日志与权重：`<save_dir>/train_log/...`
- 仿真输出：`<save_dir>/simulation/...`
- 视频输出：`simulation_video/`
- step 级原始导出（若启用）：`raw_step_data/`

## 7. 注意事项

- 仓库中部分入口脚本包含本地绝对路径示例，请按你的机器环境修改。
- 首次运行建议先用小规模数据（mini）做完整链路冒烟测试。
- 若仅用于 baseline 复现，请保持 `diffusion-planner` 主线配置不变。

## 8. 许可证与引用

如需开源发布，建议补充：

- `LICENSE`
- 论文/项目引用格式（BibTeX）

