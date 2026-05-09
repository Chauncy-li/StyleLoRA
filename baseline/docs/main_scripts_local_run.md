# Baseline 主要脚本与本地运行配置指南

> 更新时间：2026-05-09 (UTC+8)

## 1. 主要运行脚本

0. `baseline/prepare_data_splits.py`
- 作用：生成并维护 mini 数据的固定 test/cache 日志列表。
- 常用命令：
```bash
python baseline/prepare_data_splits.py
```
- 输出：
  - `baseline/resources/mini/splits/mini_test_logs.json`
  - `baseline/resources/mini/splits/mini_cache_logs.json`

1. `baseline/process_data.py`
- 作用：将 nuPlan 原始场景处理为训练用 `.npz` 缓存，并导出样本列表 JSON。
- 常用命令：
```bash
python baseline/process_data.py \
  --data_path <nuplan_db_dir> \
  --map_path <nuplan_map_dir> \
  --save_path <cache_dir>
```
- 默认行为：
  - 读取 `baseline/resources/mini/splits/mini_cache_logs.json`
  - 生成 `baseline/resources/mini/training_mini.json`（npz 文件名列表）

2. `baseline/train.py`
- 作用：训练入口（注册制），按 `method.name` 自动加载 `diffusion-planner` 或 `wayformer`。
- 常用命令：
```bash
python baseline/train.py
python baseline/train.py method=wayformer
```

3. `baseline/run_simulation.py`
- 作用：闭环仿真入口（注册制），按 `PLANNER` 选择 planner 并自动拼接 Hydra 参数。
- 常用命令：
```bash
python baseline/run_simulation.py
```
- mini 默认行为：
  - 当 `SPLIT="mini"` 且 `mini_test_logs.json` 存在时，自动把该列表注入 `scenario_filter.log_names`
  - `baseline/config/scenario_filter/mini.yaml` 的其他参数（如 `limit_total_scenarios`）保持生效

4. `baseline/run_nuboard_viewer.py`
- 作用：启动 nuBoard 结果可视化。


## 2. 本地运行时优先修改的位置

1. 训练配置：`baseline/config/train.yaml`
- `save_dir`：训练日志/权重输出目录
- `data.train_set` / `data.train_set_list`：训练数据与列表
- `data.normalization_file_path`：归一化参数文件
- `training.batch_size` / `training.train_epochs` / `training.learning_rate`
- 日志后端：
  - `online_logger`: `swanlab` / `wandb` / `disabled`
  - `use_online_logger`: `true` / `false`
  - 默认已设为 `swanlab`

2. 方法配置：
- `baseline/config/method/diffusion_planner.yaml`
- `baseline/config/method/wayformer.yaml`
- 作用：控制模型结构参数（层数、hidden dim、mode 数等）

3. 仿真脚本常量：`baseline/run_simulation.py`
- `PLANNER`：`diffusion_planner` / `wayformer`
- `CHECKPOINT_DIR` / `ARGS_FILE` / `CKPT_FILE`
- `NUPLAN_DATA_ROOT` / `NUPLAN_MAPS_ROOT` / `NUPLAN_EXP_ROOT`
- `ONLINE_LOGGER`：`swanlab` / `wandb` / `disabled`（默认 `swanlab`）

4. 仿真 planner 配置模板：
- `baseline/config/planner/diffusion_planner.yaml`
- `baseline/config/planner/wayformer.yaml`
- 主要由 `run_simulation.py` 动态覆盖 `args_file/ckpt_path/render_save_dir/raw_data_save_dir`


## 3. 日志后端切换示例

1. 使用默认 swanlab（推荐）
- `baseline/config/train.yaml` 中保持：
```yaml
online_logger: "swanlab"
use_online_logger: true
```

2. 切换到 wandb
- 方式 A：改配置文件
```yaml
online_logger: "wandb"
use_online_logger: true
```
- 方式 B：命令行覆盖
```bash
python baseline/train.py online_logger=wandb
```

3. 完全关闭在线日志（仅本地 tensorboard）
```bash
python baseline/train.py use_online_logger=false
```


## 4. 一个推荐的本地启动顺序

1. 先执行 `prepare_data_splits.py`，冻结并复用 mini test 列表。  
2. 再跑小规模数据处理，确认 `npz` 和 `training_mini.json` 输出正常。  
3. 用 `diffusion-planner` 做一次短轮数训练冒烟。  
4. 再切到 `method=wayformer` 做短轮数训练。  
5. 最后用对应 checkpoint 运行 `run_simulation.py` 做闭环验证。  
