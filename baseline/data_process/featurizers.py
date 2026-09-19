"""三个对比模型的“特征定义公式”（唯一实现）。

训练数据抽取（unified_extractor）与闭环推理（planner）都 import 本模块，
保证 train/inference 口径一字不差。

内容：
  - AD-MLP：compute_driving_command / build_admlp_input
  - StageVec：build_vector（256 维向量）+ 派生助手 + 归一化统计
"""
import math

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def _norm_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# AD-MLP（仅自车状态）
# ============================================================

def compute_driving_command(ego_state, map_api):
    """Route-based command：沿最近车道中心线向前看 15m。0=left, 1=right, 2=straight。"""
    try:
        x, y = float(ego_state.rear_axle.x), float(ego_state.rear_axle.y)
        heading = float(ego_state.rear_axle.heading)
        layers = map_api.get_proximal_map_objects(
            Point2D(x, y), 25, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )
        lanes = list(layers[SemanticMapLayer.LANE]) + list(layers[SemanticMapLayer.LANE_CONNECTOR])
        best_pts = None
        best_d = 1e18
        for ln in lanes:
            pts = [(p.x, p.y) for p in ln.baseline_path.discrete_path]
            if len(pts) < 2:
                continue
            arr = np.array(pts, dtype=np.float64)
            d = ((arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2).min()
            if d < best_d:
                best_d = d
                best_pts = arr
        if best_pts is None or len(best_pts) < 2:
            return 2
        d = np.hypot(best_pts[:, 0] - x, best_pts[:, 1] - y)
        i = int(np.argmin(d))
        j = i
        for k in range(i, len(best_pts)):
            if np.hypot(best_pts[k, 0] - best_pts[i, 0], best_pts[k, 1] - best_pts[i, 1]) >= 15.0:
                j = k
                break
        if j == i:
            j = min(len(best_pts) - 1, i + 1)
        dx, dy = best_pts[j, 0] - best_pts[i, 0], best_pts[j, 1] - best_pts[i, 1]
        if math.hypot(dx, dy) < 0.5:
            return 2
        diff = _norm_angle(math.atan2(dy, dx) - heading)
        if diff > 0.35:
            return 0
        if diff < -0.35:
            return 1
        return 2
    except Exception:
        return 2


def build_admlp_input(ego_state, map_api):
    """自车速度(2) + 加速度(2) + 导航指令(4 one-hot) -> [8]。"""
    v = ego_state.dynamic_car_state.rear_axle_velocity_2d
    a = ego_state.dynamic_car_state.rear_axle_acceleration_2d
    cmd = compute_driving_command(ego_state, map_api)
    cmd_vec = np.zeros(4, dtype=np.float32)
    cmd_vec[cmd] = 1.0
    return np.concatenate(
        [[float(v.x), float(v.y), float(a.x), float(a.y)], cmd_vec]
    ).astype(np.float32)


# ============================================================
# StageVec（STAGE-向量消融版，256 维）
# ============================================================

EGO_STATE_DIM = 6
LANE_DETECTOR_DIM = 40
NAVI_INFO_DIM = 10
HISTORY_INFO_DIM = 200
INPUT_DIM = EGO_STATE_DIM + LANE_DETECTOR_DIM + NAVI_INFO_DIM + HISTORY_INFO_DIM  # 256

LANE_LOOKAHEAD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50]  # 20
NAVI_LOOKAHEAD = [10, 20, 30, 40, 50]  # 5
HISTORY_N = 10
HISTORY_DT = 0.1

_WHEEL_BASE = 2.845  # nuPlan 默认林肯 MKZ 轴距 [m]

_LANE_LAYERS = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]


def derive_yaw_rate(cur_ego, prev_ego, dt):
    """由航向变化推导横摆角速度 (rad/s)。DB 不存 angular_velocity。"""
    if prev_ego is None or dt <= 0:
        return 0.0
    return float(_norm_angle(float(cur_ego.rear_axle.heading) - float(prev_ego.rear_axle.heading)) / dt)


def _nearest_lane(map_api, x, y):
    """返回 (center_pts [N,3] (x,y,heading), lane)。无则 (None, None)。"""
    try:
        layers = map_api.get_proximal_map_objects(Point2D(x, y), 40, _LANE_LAYERS)
    except Exception:
        return None, None
    lanes = list(layers[SemanticMapLayer.LANE]) + list(layers[SemanticMapLayer.LANE_CONNECTOR])
    best_pts = None
    best_lane = None
    best_d = 1e18
    for ln in lanes:
        try:
            pts = [(p.x, p.y, p.heading) for p in ln.baseline_path.discrete_path]
        except Exception:
            continue
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.float64)
        d = float(np.hypot(arr[:, 0] - x, arr[:, 1] - y).min())
        if d < best_d:
            best_d = d
            best_pts = arr
            best_lane = ln
    return best_pts, best_lane


def _boundary_points(lane, side):
    """返回左/右车道边界折线点 [M,2]。无则 None。"""
    try:
        poly = lane.left_boundary if side == "left" else lane.right_boundary
        return np.array([(p.x, p.y) for p in poly.discrete_path], dtype=np.float64)
    except Exception:
        return None


def _project_arclen(pts):
    """返回 (s0, cum)。cum 为逐段弧长累计。"""
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum


def _sample_at_arclen(pts, cum, s_target):
    """中心线上弧长 s_target 处的点 [x,y,heading]。"""
    if s_target <= cum[0]:
        return pts[0]
    if s_target >= cum[-1]:
        return pts[-1]
    i = int(np.searchsorted(cum, s_target)) - 1
    i = max(0, min(i, len(pts) - 2))
    denom = cum[i + 1] - cum[i] + 1e-9
    frac = float(np.clip((s_target - cum[i]) / denom, 0.0, 1.0))
    x = pts[i, 0] + frac * (pts[i + 1, 0] - pts[i, 0])
    y = pts[i, 1] + frac * (pts[i + 1, 1] - pts[i, 1])
    h = pts[i, 2]
    return np.array([x, y, h], dtype=np.float64)


def _to_local(pts_global, ego_x, ego_y, ego_h):
    """全局 [N,2] -> ego 局部 (x 前向, y 左向)。"""
    pts_global = np.asarray(pts_global, dtype=np.float64)
    dx = pts_global[:, 0] - ego_x
    dy = pts_global[:, 1] - ego_y
    c, s = math.cos(ego_h), math.sin(ego_h)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return local_x, local_y


def _ego_state(ego, map_api, yaw_rate):
    x, y, h = float(ego.rear_axle.x), float(ego.rear_axle.y), float(ego.rear_axle.heading)
    # heading_error / lateral_offset: 相对最近车道
    center_pts, _ = _nearest_lane(map_api, x, y)
    heading_error = 0.0
    lateral_offset = 0.0
    if center_pts is not None:
        d = np.hypot(center_pts[:, 0] - x, center_pts[:, 1] - y)
        i = int(np.argmin(d))
        lane_h = center_pts[i, 2]
        heading_error = float(_norm_angle(lane_h - h))
        # 有符号横向偏移（左为正），指向车道左法向
        dx = x - center_pts[i, 0]
        dy = y - center_pts[i, 1]
        lateral_offset = float(dx * (-math.sin(lane_h)) + dy * math.cos(lane_h))

    speed = float(ego.dynamic_car_state.speed)
    acc = ego.dynamic_car_state.rear_axle_acceleration_2d
    long_accel = float(acc.x * math.cos(h) + acc.y * math.sin(h))
    # 转向角由自行车模型反推（DB 不存 tire_steering_angle）
    steering = math.atan(_WHEEL_BASE * yaw_rate / max(speed, 1.0))
    return np.array([heading_error, speed, long_accel, yaw_rate, lateral_offset, steering], dtype=np.float32)


def _lane_detector(ego, map_api):
    x, y = float(ego.rear_axle.x), float(ego.rear_axle.y)
    center_pts, lane = _nearest_lane(map_api, x, y)
    out = np.zeros(LANE_DETECTOR_DIM, dtype=np.float32)
    if center_pts is None:
        return out
    cum = _project_arclen(center_pts)
    d = np.hypot(center_pts[:, 0] - x, center_pts[:, 1] - y)
    s0 = float(cum[int(np.argmin(d))])
    left_b = _boundary_points(lane, "left")
    right_b = _boundary_points(lane, "right")
    for j, ahead in enumerate(LANE_LOOKAHEAD):
        c = _sample_at_arclen(center_pts, cum, s0 + ahead)
        lo = ro = 1.75  # 缺省半宽
        if left_b is not None and len(left_b):
            lo = float(np.hypot(left_b[:, 0] - c[0], left_b[:, 1] - c[1]).min())
        if right_b is not None and len(right_b):
            ro = float(np.hypot(right_b[:, 0] - c[0], right_b[:, 1] - c[1]).min())
        out[2 * j] = lo
        out[2 * j + 1] = ro
    return out


def _navi_info(ego, map_api):
    x, y, h = float(ego.rear_axle.x), float(ego.rear_axle.y), float(ego.rear_axle.heading)
    center_pts, _ = _nearest_lane(map_api, x, y)
    out = np.zeros(NAVI_INFO_DIM, dtype=np.float32)
    if center_pts is None:
        return out
    cum = _project_arclen(center_pts)
    d = np.hypot(center_pts[:, 0] - x, center_pts[:, 1] - y)
    s0 = float(cum[int(np.argmin(d))])
    pts = np.stack([_sample_at_arclen(center_pts, cum, s0 + a)[:2] for a in NAVI_LOOKAHEAD])
    lx, ly = _to_local(pts, x, y, h)
    out[0::2] = lx.astype(np.float32)
    out[1::2] = ly.astype(np.float32)
    return out


def _history_info(ego, neighbor_agents):
    """5 主体 x 10 帧 x (dx,dy,cos_h,sin_h)，匀速外推。"""
    ego_x, ego_y, ego_h = float(ego.rear_axle.x), float(ego.rear_axle.y), float(ego.rear_axle.heading)
    ev = ego.dynamic_car_state.rear_axle_velocity_2d

    # 主体列表: (pos_x, pos_y, heading, vel_x, vel_y)
    subjects = [(ego_x, ego_y, ego_h, float(ev.x), float(ev.y))]

    # 4 最近车
    neighbors = []
    for a in neighbor_agents:
        try:
            bx, by, bh = float(a.box.center.x), float(a.box.center.y), float(a.box.center.heading)
        except Exception:
            continue
        dist = math.hypot(bx - ego_x, by - ego_y)
        vx = float(a.velocity.x)
        vy = float(a.velocity.y)
        neighbors.append((dist, bx, by, bh, vx, vy))
    neighbors.sort(key=lambda t: t[0])
    for t in neighbors[:4]:
        subjects.append((t[1], t[2], t[3], t[4], t[5]))

    hist = np.zeros((5, HISTORY_N, 4), dtype=np.float32)
    for i, (px, py, ph, vx, vy) in enumerate(subjects):
        for k in range(HISTORY_N):
            t = -k * HISTORY_DT
            gx = px + vx * t
            gy = py + vy * t
            dx = gx - ego_x
            dy = gy - ego_y
            c, s = math.cos(ego_h), math.sin(ego_h)
            hist[i, k, 0] = c * dx + s * dy
            hist[i, k, 1] = -s * dx + c * dy
            hist[i, k, 2] = math.cos(ph)
            hist[i, k, 3] = math.sin(ph)
    return hist


def build_vector(ego_state, neighbor_agents, map_api, yaw_rate=0.0):
    """返回原始 256 维向量（未归一化）。yaw_rate 由调用方从航向变化推导。"""
    eg = _ego_state(ego_state, map_api, yaw_rate)
    lane = _lane_detector(ego_state, map_api)
    navi = _navi_info(ego_state, map_api)
    hist = _history_info(ego_state, neighbor_agents)
    return np.concatenate([eg, lane, navi, hist.reshape(-1)]).astype(np.float32)


def compute_stats(vec_list):
    """vec_list: list of [256] arrays。返回 {mean:[256], std:[256]}。"""
    arr = np.stack(vec_list).astype(np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.clip(std, 1e-2, np.inf)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize(vec, stats):
    return (vec - stats["mean"]) / stats["std"]


def build_stage_vec_target(ego_state, future_states, yaw_rate):
    """由 8 个未来状态（4s@0.5s）推导 StageVec 目标：traj [8,2] + steer [1,2]。"""
    x, y, h = float(ego_state.rear_axle.x), float(ego_state.rear_axle.y), float(ego_state.rear_axle.heading)
    pts = np.array([[s.rear_axle.x, s.rear_axle.y] for s in future_states], dtype=np.float64)
    lx, ly = _to_local(pts, x, y, h)
    traj = np.stack([lx, ly], axis=1).astype(np.float32)
    speed = float(ego_state.dynamic_car_state.speed)
    steering = math.atan(_WHEEL_BASE * yaw_rate / max(speed, 1.0))
    steer = np.array([[steering, speed]], dtype=np.float32)
    return traj, steer


def build_prefer(ego_state, neighbor_agents):
    """StageVec 风格偏好损失所需的偏好代理量 [speed_kmh, throttle_proxy, has_nearest, nearest_dist]。

    speed_kmh / has_nearest / nearest_dist 与 STAGE 原始 extract（utils.py `_parse_ego_states`）同口径；
    油门 STAGE 用真实 throttle（`ego_control[1]` ∈ [-1,1]，反归一化得到），NuPlan 无此原始值，
    故用 speed/40 代理（与速度单调一致）。仅用于 StageVec 训练的风格偏好排名损失，不参与推理。
    """
    speed = float(ego_state.dynamic_car_state.speed)
    speed_kmh = speed * 3.6
    throttle_proxy = speed / 40.0
    x, y = float(ego_state.rear_axle.x), float(ego_state.rear_axle.y)
    nearest_dist = 1e9
    for a in neighbor_agents:
        try:
            bx, by = float(a.box.center.x), float(a.box.center.y)
        except Exception:
            continue
        d = math.hypot(bx - x, by - y)
        if d < nearest_dist:
            nearest_dist = d
    has_nearest = 1.0 if nearest_dist < 20.0 else 0.0
    return np.array([speed_kmh, throttle_proxy, has_nearest, nearest_dist], dtype=np.float32)


def build_admlp_style_score(prefer):
    """由 build_prefer 的代理量推导连续激进分，供 AD-MLP 风格标签分箱（A/N/C）。

    速度分(km/h/30)、跟车距离分(20m 内近车时 1/nearest_dist) 与 STAGE rule_style_preference 一致；
    油门分用 throttle_proxy/0.5（speed/40 代理，STAGE 用真实 throttle/0.5）。
    分数越高越激进。仅用于 AD-MLP 训练侧风格标签，不参与推理。
    """
    speed_kmh, throttle_proxy, has_nearest, nearest_dist = (float(x) for x in prefer)
    speed_score = speed_kmh / 30.0
    throttle_score = throttle_proxy / 0.5
    distance_score = 0.0
    if bool(has_nearest) and nearest_dist < 20.0:
        distance_score = 1.0 / max(nearest_dist, 1e-6)
    return float(speed_score + throttle_score + distance_score)
