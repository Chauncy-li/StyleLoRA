# Baseline 数据处理产物说明

更新时间：2026-05-09 (UTC+8)

## 1. 数据处理会产出什么

运行入口：`baseline/process_data.py`

mini 默认输入输出：
- 输入日志列表：`baseline/resources/mini/splits/mini_cache_logs.json`
- 输出样本列表：`baseline/resources/mini/training_mini.json`

核心产出有 2 类：

1. 每帧一个 `.npz` 样本文件  
命名规则：`{map_name}_{token}.npz`  
生成位置：`--save_path`

2. 一个样本清单 JSON  
内容是 `.npz` 文件名列表（非绝对路径）  
生成位置：`--npz_list_output_json`

## 2. 单个 `.npz` 的字段总览

### 2.1 元信息

| 字段名 | 说明 | 类型 |
|---|---|---|
| `map_name` | 地图名 | 字符串标量 |
| `token` | 场景 token | 字符串标量 |

### 2.2 轨迹与参与体特征

| 字段名 | 说明 | 典型 shape（默认参数） | 类型 |
|---|---|---|---|
| `ego_current_state` | 自车当前扩展状态 `(x,y,cos,sin,vx,vy,ax,ay,steer,yaw_rate)` | `[10]` | float |
| `ego_agent_past` | 自车历史轨迹 | `[21, 7]` | float32 |
| `ego_agent_future` | 自车未来 GT（相对坐标） | `[80, 3]` | float |
| `neighbor_agents_past` | 邻居历史特征（含 one-hot type） | `[N, 21, 11]` | float32 |
| `neighbor_agents_past_mask` | 邻居历史有效 mask | `[N, 21]` | bool |
| `neighbor_agents_future` | 邻居未来 GT（相对坐标） | `[N, 80, 3]` | float32 |
| `neighbor_agents_future_mask` | 邻居未来有效 mask | `[N, 80]` | bool |
| `static_objects` | 静态物体特征（含 one-hot type） | `[S, 10]` | float32 |

注：
- `N = agent_num`（默认 32）
- `S = static_objects_num`（默认 5）
- 历史长度 21 来自 `2s * 10Hz + 当前帧`
- 未来长度 80 来自 `8s * 10Hz`

### 2.3 地图特征

| 字段名 | 说明 | 典型 shape（默认参数） | 类型 |
|---|---|---|---|
| `lanes` | 车道特征（中心线+向量+左右边界相对量+红绿灯编码） | `[L, P, 12]` | float32 |
| `lanes_mask` | 车道点有效 mask | `[L, P]` | bool |
| `lanes_speed_limit` | 车道限速 | `[L, 1]` | float32 |
| `lanes_has_speed_limit` | 是否有限速 | `[L, 1]` | bool |
| `route_lanes` | 路由车道特征 | `[R, P, 12]` | float32 |
| `route_lanes_mask` | 路由车道点有效 mask | `[R, P]` | bool |
| `route_lanes_speed_limit` | 路由车道限速 | `[R, 1]` | float32 |
| `route_lanes_has_speed_limit` | 路由车道是否有限速 | `[R, 1]` | bool |

注：
- `L = lane_num`（默认 70）
- `R = route_num`（默认 25）
- `P = lane_len`（默认 20；route 也按该点数对齐）

### 2.4 Codebook 额外标签（用不到）

| 字段名 | 说明 | 类型 |
|---|---|---|
| `code_lat` | 横向分支标签（a_lat） | int64 |
| `code_lon` | 纵向分箱标签（a_lon） | int64 |
| `code_rho` | 纵向连续进度值（rho） | float32 |

这 3 个字段由 `baseline/data_process/codebook_labeler.py` 在离线处理阶段生成并写入。

## 3. 当前主流程中哪些字段被实际使用

以当前 baseline 的 diffusion 主线为准：

1. `baseline/train.py` 训练时主要使用：
- `ego_current_state`
- `ego_agent_future`
- `neighbor_agents_past`
- `neighbor_agents_future`
- `lanes` / `lanes_speed_limit` / `lanes_has_speed_limit`
- `route_lanes` / `route_lanes_speed_limit` / `route_lanes_has_speed_limit`
- `static_objects`

2. `code_lat` / `code_lon` / `code_rho` 不参与当前 `train.py` 的损失计算  

## 4. 快速自检一个 `.npz` 是否完整

```bash
python - <<'PY'
import numpy as np
path = "<your_npz_path>"
with np.load(path, allow_pickle=True) as d:
    print("keys:", sorted(d.files))
    for k in sorted(d.files):
        v = d[k]
        print(f"{k:30s} shape={v.shape} dtype={v.dtype}")
PY
```

如果你只关心 codebook 标签：

```bash
python baseline/verify_codebook_labels.py --data_path <npz_dir>
```
