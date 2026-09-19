# 模型横向对比（StyleLoRA · DiffPlanner · AD-MLP · StageVec）

本文档说明如何在 StyleLoRA 项目中接入并横向对比**四个**规划模型：

- **StyleLoRA**：项目提出的方法（DiffPlanner + LoRA 适配器 + 偏好路由，连续 `rho` 个性化）；
- **DiffPlanner**：项目已有的扩散规划器 baseline；
- **AD-MLP**：仅依赖自车状态的 MLP 规划器（对齐 navsim `ego_status_mlp_agent`）；
- **StageVec**：STAGE 的向量消融版（ACT/CVAE-DETR 去掉图像与 ray-cast，仅 256 维向量输入）。

四者共用同一份 NuPlan boston 数据、同一份 train/val 场景划分、**同一训练样本量**、同一个闭环 token 名单；训练阶段只保留模型本身的差异（输入维度、预测时域、损失函数），其余变量（数据、划分、样本量、监督来源、归一化口径、随机种子）全部对齐，作为论文的横向对比实验。

所有命令都在仓库根目录运行。`/path/to/...` 是需要换成当前机器实际位置的占位符；路径优先从 `stylelora/config/paths.local.json` 读取，也可用 `NUPLAN_*` 环境变量临时覆盖。

---

## 0. 目录结构

```text
comparison/
├── README.md               # 本文档
├── splits.json             # 唯一 train/val（+ 可选 test）划分，四模型共享
├── make_splits.py          # 复用学长 raw_split/cache_split 的 train/val/test 日志划分（不自行切分）
├── extract.py              # 统一抽取入口（一次读 scenario → 统一 .npz）
├── train_diff_planner.py   # DiffPlanner 训练入口（薄封装，调 baseline/train.py）
├── train_admlp.py          # AD-MLP 训练入口
├── train_stage_vec.py      # StageVec 训练入口
├── run_closed_loop.sh      # 闭环仿真（按 --model 选模型，跑学长的固定 token 名单）
└── aggregate_results.py    # 汇总四模型跑分为一张表
```

> StyleLoRA 是项目本身，其训练与闭环**复用学长既有管线**（`stylelora/scripts/train_*.py`、`stylelora/scripts/evaluate_closed_loop.py`），本模块不重写。

模型本体、特征定义、闭环封装仍按项目分层放在 `baseline/` 下：

```text
baseline/model/diff_planner/      # 已有，不动
baseline/model/admlp/             # 新增：EgoStatusMLP
baseline/model/stage_vec/         # 新增：DETRVAEVec + vendor 的 detr/ 依赖
baseline/data_process/featurizers.py        # 新增：三个对比模型的“特征定义公式”（唯一实现）
baseline/data_process/unified_extractor.py  # 新增：统一快照抽取器
baseline/simulation/planner.py    # 注册 ego_status_planner / stage_vector_planner
baseline/config/method/*.yaml     # 每模型一份结构配置
```

---

## 1. 模型详细信息

### 1.1 StyleLoRA（本项目方法）

| 项 | 值 |
|---|---|
| 类型 | DiffPlanner 基座 + 低秩适配器（high/low 纵向响应）+ 条件路由器 + 偏好编码器 |
| 输入 | 同 DiffPlanner 特征字典，外加连续 `rho` ∈ [-1,1] 个性化强度 |
| 输出 | `[B, P, T, 4]` 轨迹（同 DiffPlanner） |
| 个性化体现 | `rho` 控制纵向激进/保守风格，训练与闭环一致 |
| 训练/闭环 | 复用学长既有 `stylelora/scripts/` 管线（`train_*`、`evaluate_closed_loop.py`），不改 |

> StyleLoRA 训练样本量、闭环 token 名单也要与另外三个模型对齐（见第 2 节），这是横向对比成立的前提。

### 1.2 DiffPlanner（baseline）

| 项 | 值 |
|---|---|
| 类型 | 扩散规划器：Transformer Encoder + DiT Decoder + SDE（`x_start`） |
| 输入 | 特征字典：`ego_current_state`[10]、`ego_agent_past`[21,7]、`neighbor_agents_past`[32,21,11]、`static_objects`[5,10]、`lanes`[70,20,12]、`route_lanes`[25,20,12] 及各 mask |
| 输出 | `[B, P, T, 4]` 的 (x, y, cos_h, sin_h)，`P = 1 + predicted_neighbor_num(10)` |
| 时域 | 历史 2s / 21 帧；未来 8s / 80 帧 |
| 结构 | encoder_depth=3, decoder_depth=3, num_heads=6, hidden_dim=192 |
| 损失 | 扩散损失（planning loss）+ meta action loss |

### 1.3 AD-MLP（仅自车状态）

| 项 | 值 |
|---|---|
| 类型 | 4 层 MLP：`8 → 512 → 512 → 512 → 24`，ReLU |
| 输入 | `[8]` = 自车速度 `(vx, vy)` + 加速度 `(ax, ay)` + 导航指令 4 维 one-hot（0=左转 / 1=右转 / 2=直行）；`--with-style` 时 `[11]` = 8 + 3 维风格 one-hot |
| 输出 | 8 个相对位姿 `(x, y, heading)`，未来 4s @ 0.5s |
| 时域 | 未来 4s / 8 帧（`num_poses=8`） |
| 损失 | L1 |
| 参数量 | 542,232 |

导航指令由地图推导：沿最近车道中心线向前看 15m，与自车航向夹角 >0.35 rad 记左转、<-0.35 rad 记右转，否则直行。该口径在训练抽取与闭环推理中完全一致。

**风格机制（可选，`--with-style`）**：对齐 StyleDrive `EgoStatusMLPAgent`——输入从 8 维变为 11 维（8 自车状态 + 3 维风格 one-hot）。风格标签由规则代理分（速度 + 油门 + 跟车距离，口径同 STAGE `rule_style_preference`）按训练集 1/3、2/3 分位分箱为 A(激进)/N(正常)/C(保守)，one-hot 与 StyleDrive `STYLE_MAP={A:0,N:1,C:2}` 一致。推理侧强制塞 one-hot A/C 测可控性（不复现场景自身风格）。

### 1.4 StageVec（STAGE-向量消融版）

| 项 | 值 |
|---|---|
| 类型 | ACT(CVAE-DETR)，去掉 ResNet 图像主干与 ray-cast，仅保留向量分支 |
| 输入 | `[256]` = `ego_state(6)` + `lane_detector(40)` + `navi_info(10)` + `history_info(200)` |
| 输出 | `traj_action`[8,2]（局部未来路点）+ `steer_throttle`[1,2] + `style` |
| 时域 | 未来 4s / 8 帧（`num_queries=8`） |
| 结构 | hidden_dim=256, dim_feedforward=1024, nheads=8, enc_layers=4, dec_layers=1, latent_dim=32 |
| 损失 | traj L1(×0.5) + steer L1(×1.0) + KL(×10) + style(×10) |
| 参数量 | 7.53M |

256 维向量各段含义：

| 段 | 维度 | 含义 |
|---|---|---|
| ego_state | 6 | heading_error, speed, long_accel, yaw_rate, lateral_offset, steering |
| lane_detector | 40 | 20 个前视弧长 × (left_offset, right_offset) 车道半宽 |
| navi_info | 10 | 5 个前视路点 × (dx, dy)（最近车道中心线，ego 局部系） |
| history_info | 200 | 5 主体（自车 + 4 最近车）× 10 帧 × (dx, dy, cos_h, sin_h) |

> 说明：这是「STAGE-向量消融版」，分数只代表 STAGE 能力的下界，不是 STAGE 本身。历史轨迹用匀速外推近似，导航用最近车道中心线替代 route roadblock。

### 1.5 四者对照

| | StyleLoRA | DiffPlanner | AD-MLP | StageVec |
|---|---|---|---|---|
| 输入信息量 | 全量 + rho | 全量（自车+邻居+静态+地图+路线） | 仅自车状态 + 命令 | 向量（自车+车道+导航+邻居） |
| 输入维度 | 特征字典 | 特征字典 | 8 | 256 |
| 预测时域 | 8s / 80 帧 | 8s / 80 帧 | 4s / 8 帧 | 4s / 8 帧 |
| 生成方式 | 扩散 + LoRA 个性化 | 扩散去噪 | 回归 | CVAE-DETR |
| 参数量 | — | — | 0.54M | 7.53M |

---

## 2. 可调参数（使用方需要自己调整的地方）

以下参数**全部由使用方按机器/实验需求调整**，脚本不写死；四模型对比时，标 ⚠️ 的参数必须**四个模型一致**。

### 2.1 机器路径：`stylelora/config/paths.local.json`

沿用项目约定。首次在新机器（本地或服务器）上先复制并修改：

```bash
if [ ! -f stylelora/config/paths.local.json ]; then
  cp stylelora/config/paths.example.json stylelora/config/paths.local.json
fi
python stylelora/config/runtime_paths.py --show
```

本对比模块会用到以下键：

| 键 | 含义 | 用途 |
|---|---|---|
| `data_root` | NuPlan boston DB 目录 | `make_splits` / `extract` / 闭环 `--data-root` |
| `maps_root` | NuPlan 地图目录 | 同上 `--maps-root` |
| `record_root` | 实验记录根目录 | 统一 `.npz` 缓存与训练权重 |
| `output_root` | 结果输出根目录 | 闭环仿真结果 |
| `tokens_file` | **闭环固定 token 名单**（学长 `select_closed_loop_scenarios` 产出） | `run_closed_loop.sh` |
| `cache_train_log_names_path` | 学长 `cache_split` 产出的 train 日志名单 | `make_splits.py` |
| `cache_val_log_names_path` | 学长 `cache_split` 产出的 val 日志名单 | `make_splits.py` |
| `test_simu_log_names_path` | 学长 `raw_split` 产出的留出闭环日志名单 | `make_splits.py` |

> 也可用环境变量 `NUPLAN_DATA_ROOT` / `NUPLAN_MAPS_ROOT` / `COMPARISON_TOKENS_FILE` 临时覆盖，不落盘。

### 2.2 训练样本量（⚠️ 四模型必须一致）

三个对比模型（DiffPlanner/AD-MLP/StageVec）通过同一个参数 `--train-samples N --val-samples M` 控制训练/验证样本数；`None` 表示用满 train/val split。**StyleLoRA 的训练脚本也要用同样的 N/M**。

| 模型 | 入口 | 样本量参数 |
|---|---|---|
| DiffPlanner | `train_diff_planner.py` | `--train-samples N --val-samples M` |
| AD-MLP | `train_admlp.py` | `--train-samples N --val-samples M` |
| StageVec | `train_stage_vec.py` | `--train-samples N --val-samples M` |
| StyleLoRA | 学长 `train_*.py` | 需手动对齐到同样的 N/M |

> 子采样口径：AD-MLP / StageVec 用**固定随机种子**（`SUBSAMPLE_SEED = 0`，在 `train_admlp.py` / `train_stage_vec.py` 顶部）做无放回随机抽样，而非「取前 N」。因为三个对比模型读的是同一份 `train_tokens`（顺序一致），同一种子下抽出的就是**同一批样本**，可复现且四模型对齐。DiffPlanner 的 `train_num_samples` 走 `baseline/train.py` 自带的采样（Hydra 内同 seed），需确认其采样种子与此对齐（见第 6 节「待对齐」）。

### 2.3 各模型优化超参（允许声明范围内微调）

| 模型 | lr | batch | epochs | 备注 |
|---|---|---|---|---|
| DiffPlanner | 5e-4 | 200 | 100 | 复用 `baseline/train.py`，`--extra` 传 hydra override |
| AD-MLP | 1e-4 | 16 | 20 | `train_admlp.py --lr --batch --epochs`（风格版加 `--with-style`） |
| StageVec | 1e-4 | 16 | 20 | `train_stage_vec.py --lr --batch --epochs`（另有 `--kl-weight --style-weight`） |
| StyleLoRA | 学长管线 | 学长管线 | 学长管线 | 保持学长原样 |

> `lr`/`batch` 允许每模型在声明范围内微调（属模型调优而非实验变量）；但 **`seed`、`样本量`、`样本选择`、`归一化口径` 必须四者一致**。

### 2.4 闭环场景集合（⚠️ 四模型必须一致）

闭环**不重新切分**，而是复用学长 `select_closed_loop_scenarios.py` 产出的固定 token 名单（`paths.local.json` 的 `tokens_file`，如 `closed_loop_tokens_balanced_50.json`），四模型在**同一份 token 名单**上跑，注入口径与学长 `evaluate_closed_loop.py` 一致（`scenario_filter.scenario_tokens` + `limit_total_scenarios`）。

| 想换闭环场景 | 改哪里 |
|---|---|
| 换 token 名单 | `paths.local.json` 的 `tokens_file`，或 `run_closed_loop.sh --tokens-file` |
| 重新生成名单 | `python stylelora/scripts/select_closed_loop_scenarios.py ...` |

### 2.5 模型结构：`baseline/config/method/*.yaml`

每个模型一份结构配置，训练与闭环都从这里读取，保证结构一致（StyleLoRA 用学长已有结构，不动）。

### 2.6 可调参数总览

| 想做什么 | 改哪里 |
|---|---|
| 换数据/地图/cache/输出位置 | `stylelora/config/paths.local.json`（或 `NUPLAN_*` 环境变量） |
| 改训练样本量（四模型一致） | 各 `train_*.py --train-samples/--val-samples` |
| 改 lr/batch/epochs | 各 `train_*.py` 参数 |
| 改模型结构 | `baseline/config/method/<模型>.yaml` |
| 换闭环模型 | `run_closed_loop.sh --model` |
| 换闭环场景名单 | `run_closed_loop.sh --tokens-file` |
| 换 DiffPlanner 训练参数文件 | `run_closed_loop.sh --args-file`（默认取 checkpoint 同目录 `args.json`） |
| 强制风格（6 组对比用） | `run_closed_loop.sh --style aggressive\|normal\|conservative` |
| StageVec 原始 style_control | `run_closed_loop.sh --style-control <float>`（未传 `--style` 时生效） |

---

## 3. 对齐数据集接口的详细方式

### 3.1 学长项目的数据输出口是什么

本项目喂给模型的方式（即「数据输出口」）有两条路径，本质是同一套特征：

1. **离线（训练）**：`baseline.data_process.data_processor.DataProcessor.work(scenarios)` 把每个 scenario 抽成一份 `.npz`（含自车/邻居/静态/车道/路线特征 + GT）。
2. **闭环（推理）**：`DataProcessor.observation_adapter(...)` 在每帧把 `PlannerInput` 转成同一套特征张量，喂给 planner。

### 3.2 统一快照：一次读、多模型共用

为保证「三个对比模型的输入来自同一次观测」，对比模块引入一个统一抽取器，每个 scenario **只查询一次**地图/自车/邻居，落盘一份统一 `.npz`，字段如下：

| 字段 | shape | 归属 |
|---|---|---|
| `token` / `map_name` | — | 公共 |
| `ego_future_abs` | [80,3] | 公共 GT 源（绝对 x,y,heading，8s@0.1s） |
| `ego_current_state` / `ego_agent_past` | [10] / [21,7] | DiffPlanner |
| `ego_agent_future` | [80,3] | DiffPlanner（由 `ego_future_abs` 转相对坐标） |
| `neighbor_agents_past`(+mask) / `neighbor_agents_future`(+mask) | [32,21,11] / [10,80,3] | DiffPlanner |
| `static_objects` | [5,10] | DiffPlanner |
| `lanes` / `route_lanes`(+mask/限速) | [70,20,12] / [25,20,12] | DiffPlanner |
| `code_lat` / `code_lon` / `code_rho` | — | DiffPlanner |
| `admlp_x` / `admlp_y` | [8] / [8,3] | AD-MLP |
| `admlp_style_score` | [1] | AD-MLP 风格打分（规则代理分，`--with-style` 训练时用于 A/N/C 分箱） |
| `stage_vec_x` | [256] | StageVec（未归一化） |
| `stage_vec_y_traj` / `stage_vec_y_steer` | [8,2] / [1,2] | StageVec |
| `stage_vec_prefer` | [4] | StageVec 风格偏好损失所需（速度 km/h、油门代理、是否 20m 近车、最近车距） |

DiffPlanner 的字段与 `DataProcessor.work()` 现有输出**完全一致**，因此它的训练/推理零改动。StyleLoRA 复用 DiffPlanner 的特征字典，也不走本抽取器（用学长既有 cache）。

### 3.3 各模型输入如何从同一快照派生

`baseline/data_process/featurizers.py` 是唯一实现，抽取器（离线造数据）与闭环 planner（在线推理）都 import 它，保证 train/inference 口径一字不差：

| 模型 | 派生方式 |
|---|---|
| DiffPlanner / StyleLoRA | 直接使用 `DataProcessor.work()` 字段 |
| AD-MLP | 自车 `dynamic_car_state` 的 vel/acc + `compute_driving_command`（查 `map_api` 最近车道 15m 前视） |
| StageVec | `build_vector`：自车状态 + 车道半宽 + 导航路点 + 邻居历史，均来自同一 map/ego/邻居快照 |

### 3.4 GT 与归一化

- **GT 单一来源**：`ego_future_abs` 是唯一监督源。DiffPlanner 转相对坐标取 80 帧，AD-MLP/StageVec 从同一轨迹抽 8 帧 @ 0.5s 各自转相对/局部坐标。
- **归一化同口径**：各模型各自的归一化统计都在**同一 train 划分**上计算一次，写成独立文件：
  - DiffPlanner：复用 `normalization.json`（若需严格一致，在对比 train 子集上重算）；
  - StageVec：`vec_mean/std`、`traj_mean/std`、`steer_mean/std`；
  - AD-MLP：无输入归一化（模型设计如此，保持）。

### 3.5 训练场景划分

`comparison/splits.json` 是训练的唯一划分来源，**不自行切分**，而是由 `make_splits.py` 复用学长既有的三份日志名单（`raw_split` → `test_simu_log_names`；`cache_split` → `cache_train_log_names` / `cache_val_log_names`），仅把日志名展开成 scenario token：

```json
{
  "train_logs": ["..."], "val_logs": ["..."], "test_logs": ["..."],
  "train_tokens": ["..."], "val_tokens": ["..."], "test_tokens": ["..."]
}
```

- 这样 train/val 与学长 DiffPlanner / StyleLoRA 训练用的 cache 日志完全同源，`test` 为学长留出的闭环日志，**从根上消除训练集泄漏**；
- 训练按 token 采样（而非按 log 随机），保证四模型监督的样本完全一致；
- `test_tokens` 仅供 AD-MLP 开环评测使用；**闭环评测不使用它**，而是用第 2.4 节学长的 `tokens_file`。

---

## 4. 跑通流程：训练 → 闭环 → 跑分

### 步骤 0：准备数据与路径

```bash
python stylelora/config/runtime_paths.py --show
```

服务器数据、地图、token 名单按 `paths.local.json` 就位即可；本地机器按实际位置修改。

### 步骤 1：生成训练划分 + 统一抽取

```bash
# 1a. 复用学长划分生成 splits.json（不传三份日志名单时，回退读 paths.local.json 的
#     cache_train_log_names_path / cache_val_log_names_path / test_simu_log_names_path）
python comparison/make_splits.py \
  --data-root "$(python stylelora/config/runtime_paths.py --get data_root)" \
  --maps-root "$(python stylelora/config/runtime_paths.py --get maps_root)" \
  --output comparison/splits.json

# 1b. 统一抽取（train/val/test 三个 split 都要跑）
for split in train val test; do
  python comparison/extract.py --split "$split" \
    --splits comparison/splits.json \
    --data-root "$(python stylelora/config/runtime_paths.py --get data_root)" \
    --maps-root "$(python stylelora/config/runtime_paths.py --get maps_root)" \
    --output-root "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/cache"
done
```

### 步骤 2：训练（四模型，样本量一致）

```bash
# 统一训练量（四模型一致；N/M 按实际需求调整，服务器通常很大）
TRAIN_N=60000; VAL_N=14000

# DiffPlanner
python comparison/train_diff_planner.py --train-samples $TRAIN_N --val-samples $VAL_N

# AD-MLP（无风格基线）
python comparison/train_admlp.py \
  --npz-root "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/cache" \
  --out-dir "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/MODELS/admlp" \
  --train-samples $TRAIN_N --val-samples $VAL_N

# AD-MLP（风格版，激进/保守两组共用同一份 with_style 权重，仅推理时塞不同 one-hot）
python comparison/train_admlp.py \
  --npz-root "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/cache" \
  --out-dir "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/MODELS/admlp_with_style" \
  --train-samples $TRAIN_N --val-samples $VAL_N --with-style

# StageVec
python comparison/train_stage_vec.py \
  --npz-root "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/cache" \
  --out-dir "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/MODELS/stage_vec" \
  --train-samples $TRAIN_N --val-samples $VAL_N

# StyleLoRA：用学长 train_*.py，样本量对齐到 $TRAIN_N / $VAL_N
```

### 步骤 3：闭环仿真（四模型，同一 token 名单）

```bash
# 三个对比模型（复用学长的 tokens_file 名单）
# DiffPlanner 需训练时的 args.json：默认取 --ckpt 同目录 args.json，可用 --args-file 覆盖
bash comparison/run_closed_loop.sh --model diff_planner \
  --ckpt <diffusion 权重.pth> [--args-file <该训练的 args.json>]

# AD-MLP：无风格基线（默认 with_style=false）
bash comparison/run_closed_loop.sh --model ego_status_planner \
  --ckpt "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/MODELS/admlp/admlp_checkpoint.pt"
# AD-MLP 激进/保守（with_style=true，拼接 one-hot）
bash comparison/run_closed_loop.sh --model ego_status_planner \
  --ckpt <admlp_with_style 权重.pt> --style aggressive   # 或 conservative

# StageVec：无风格基线（默认 style_control=0.0）
bash comparison/run_closed_loop.sh --model stage_vector_planner \
  --ckpt "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/MODELS/stage_vec/stage_vec_checkpoint.pt" \
  --stats "$(python stylelora/config/runtime_paths.py --get record_root)/COMPARISON/MODELS/stage_vec/stage_vec_stats.npz"
# StageVec 激进/保守（style_control = ±σ，σ 来自 stats 的 style_value_std）
bash comparison/run_closed_loop.sh --model stage_vector_planner \
  --ckpt <stage_vec 权重.pt> --stats <stage_vec_stats.npz> --style aggressive   # 或 conservative

# StyleLoRA：用学长 stylelora/scripts/evaluate_closed_loop.py（同一 tokens_file），
# 6 组中 rho=-0.5(保守)/+0.5(激进) 两组的 rho 注入仍在学长管线侧（见第 6 节）。
```

**6 组风格对照口径**：`stylelora(rho=-0.5)`、`stylelora(rho=+0.5)` 走学长管线；`stagevec/admlp 激进/保守` 用本脚本 `--style`。三者风格方向语义一致（激进=更快/更高油门/更近跟车），见各模型 §1 说明。

### 步骤 4：汇总跑分

```bash
python comparison/aggregate_results.py \
  --style <stylelora 结果目录> \
  --diff  <diffusion_planner 结果目录> \
  --admlp <ego_status_planner 结果目录> \
  --stage <stage_vector_planner 结果目录> \
  --out comparison/summary.json
```

输出一张四模型对照表（`final_score` + 无责碰撞、TTC、舒适度、可行驶区、方向合规、限速合规、专家路线进度等子指标），并落一份 `comparison/summary.json`。

---

## 5. 注意事项

- `paths.local.json`、`comparison/splits.json`、抽取的 `.npz`、训练权重与跑分结果都**不提交 Git**（与项目现有 `.gitignore` 约定一致）。
- StageVec 依赖 `detr` 模块，已 vendor 进 `baseline/model/stage_vec/detr/`，无需外部 STAGE 仓库。
- 各模型时域不同（DiffPlanner/StyleLoRA 8s，另两个 4s）是模型自身属性，论文里显式声明即可，不强行拉平。
- StyleLoRA 训练/闭环复用学长既有管线，本模块只保证它与另三个模型共享**同一训练样本量**与**同一闭环 token 名单**。

---

## 6. 待对齐事项（TODO，尚未落实到代码）

**已落实到代码（本次 6 组改造）**：AD-MLP 增加 `with_style` 风格拼接（对齐 StyleDrive），StageVec 增加 `style_control=±σ` 注入，`run_closed_loop.sh` 增加 `--style/--style-control`，训练侧补 `admlp_style_score` 字段与 `stage_vec` 的 `style_value_mean/std` 统计。风格方向语义三模型一致（激进=更快/更高油门/更近跟车）。

### 6.1 已解决：StyleLoRA `rho=±0.5` 的取值

学长代码已明确指示，无需新写：`stylelora/scripts/evaluate_closed_loop.py` 定义 `FORMAL_RHOS = (-1.0, -0.5, 0.0, 0.5, 1.0)`（`:21`）、`--rhos` 默认 `-1,-0.5,0,0.5,1`（`:472`），并逐 `rho` 注入 `planner.lora_diffusion_planner.config.lora_rho={rho}`（`:683`；config 默认见 `stylelora/config/planner/lora_diffusion_planner.yaml:15` 的 `lora_rho: 0.0`）。方向映射 `_STYLE_ORDINAL={"conservative":-1,"normal":0,"aggressive":1}`、`high=(0.8,1.0,"aggressive")` / `low=(0.0,0.2,"conservative")`，故 `rho=+0.5`→激进、`rho=-0.5`→保守，与本模块 `--style aggressive/conservative` 方向一致。

**StyleLoRA 两组**沿用学长脚本即可（不在 `run_closed_loop.sh` 里重写）：传 token 名单时正式评测要求完整五点网格 `--rhos=-1,-0.5,0,0.5,1`（或 `--allow-rho-shard --rhos=-0.5,0.5` 拆开跑），再取 ±0.5 两组。

### 6.2 已确定：样本口径（固定种子 + 同源划分即可）

结论：**不追求四模型抽到同一批具体样本，只要求 train/val 划分同源 + 固定种子可复现**。

- **划分同源（已满足）**：`comparison/splits.json` 直接复用学长 `cache_split` 的 train/val 日志（`make_splits.py`），DiffPlanner / StyleLoRA 走学长同一 cache。四模型"训练集 / 验证集是哪批场景"从源头一致，无泄漏。
- **固定种子可复现（已满足）**：AD-MLP / StageVec 用 `SUBSAMPLE_SEED=0`（`RandomState(0).permutation()[:N]`），重跑抽同一批样本。
- **种子序号不追求一致**：0 与 StyleLoRA 的 17、DiffPlanner 的确定性「取前 N」互不比较，各自可复现即可。
- **更省事**：`--train-samples` 不传（`None`）即全量训练，此时无子采样差异，四模型各自全量、天然同一批场景。
