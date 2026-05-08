import numpy as np
import warnings
from shapely.geometry import LineString, Point
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class CodebookLabeler:
    def __init__(self, map_api: AbstractMap, lookahead_dist: float = 50.0, num_bins: int = 8, custom_bins=None):
        self.map_api = map_api
        self.lookahead_dist = lookahead_dist

        # 纵向分箱配置
        if custom_bins is not None:
            self.lon_quantiles = np.array(custom_bins)
            self.num_bins = len(self.lon_quantiles) - 1
        else:
            self.num_bins = num_bins
            self.lon_quantiles = np.linspace(0, 3.0, num_bins + 1)

    def get_labels(self, ego_state, gt_future_traj, current_speed, future_horizon_time, route_roadblock_ids=None,
                   route_lanes=None, route_lanes_mask=None, lat_match_dist_thresh=5.0):

        """
        全场景意图标签生成 (On-Route 增强版 + 终极鲁棒版)
        Args:
            route_roadblock_ids: List[str] 导航路径包含的 roadblock ID 列表 (从 scenario 获取)
        """
        # --- 输入数据 NaN 检查 ---
        if gt_future_traj is None or len(gt_future_traj) == 0 or not np.isfinite(gt_future_traj).all():
            return self._get_empty_label()

        # --- 统一定义：GT 终点（供 lat/lon 计算共用） ---
        gt_end_point_coords = gt_future_traj[-1, :2]
        gt_end_point = Point(gt_end_point_coords)

        # 预置变量，避免 route_lanes 分支下未定义
        matched_branch = None
        unique_candidates = []
        unique_intents = []

        # --- [Phase-1 最小修复] 语义对齐：若提供 route_lanes，则直接在 route_lanes(on-route) 上编号 a_lat ---
        use_route_lanes_lat = (route_lanes is not None)

        if use_route_lanes_lat:
            best_lat_idx, min_lat_dist, matched_s_H = self._match_gt_endpoint_to_route_lanes(
                gt_future_traj=gt_future_traj,
                route_lanes=route_lanes,
                route_lanes_mask=route_lanes_mask,
                dist_thresh=lat_match_dist_thresh,
                yaw_thresh_deg=60.0,
                ego_state=ego_state
            )

            # ====== route_lanes 编号体系下，直接在这里完成 lon 并 return，避免后续 DFS 覆盖 ======
            # matched_s_H 是 GT 终点在该 route_lanes 上的沿线弧长（由 _project_point_to_polyline_s 计算）
            epsilon = 1.0
            s_nom = max(current_speed * future_horizon_time, epsilon)
            rho = matched_s_H / s_nom
            rho = np.clip(rho, 0.0, 3.0)

            lon_bin_idx = np.digitize(rho, self.lon_quantiles) - 1
            lon_bin_idx = np.clip(lon_bin_idx, 0, self.num_bins - 1)

            return {
                "a_lat": int(best_lat_idx),
                "a_lon": int(lon_bin_idx),
                "rho": float(rho),
                "lat_intent": 0,  # route_lanes 分支下不再依赖 DFS 的 intent，可先置 0
                "num_candidates": int(route_lanes.shape[0]) if hasattr(route_lanes, "shape") else 0
            }

        if not use_route_lanes_lat:
            # 1. 获取起始车道 (Neighbor Search)
            starting_lanes = self._get_candidate_starting_lanes(ego_state)

            all_candidates = []
            candidate_intents = []

            # 2. 对每个起始车道进行 DFS 搜索 (集成 On-Route 逻辑)
            for lane in starting_lanes:
                intent = self._classify_lane_intent(ego_state, lane)

                # [核心修改] 传入 route_ids 进行搜索剪枝
                branches = self._find_all_candidate_routes(
                    lane,
                    max_length=self.lookahead_dist,
                    route_roadblock_ids=route_roadblock_ids
                )

                for branch_points in branches:
                    if len(branch_points) < 2: continue

                    # --- [鲁棒性 2] 坐标点清洗 ---
                    # 过滤掉 NaN 或 Inf 的点
                    valid_points = [(p.x, p.y) for p in branch_points if np.isfinite(p.x) and np.isfinite(p.y)]
                    if len(valid_points) < 2: continue

                    # 检查首尾距离，防止重合点构成的线
                    p_start, p_end = np.array(valid_points[0]), np.array(valid_points[-1])
                    if np.linalg.norm(p_start - p_end) < 0.1: continue

                    try:
                        line = LineString(valid_points)
                        # 只有极短的线才会被视为无效 (1mm)
                        if not line.is_valid or line.length < 1e-3: continue
                    except Exception:
                        continue

                    all_candidates.append(line)
                    candidate_intents.append(intent)

        # 3. 路径去重
        unique_candidates, unique_intents = self._deduplicate_paths(all_candidates, candidate_intents)

        # 兜底
        if not unique_candidates:
            unique_candidates = [self._generate_dummy_branch(ego_state)]
            unique_intents = [0]

        # 4. 匹配 GT (Lat Label)
        gt_end_point_coords = gt_future_traj[-1, :2]
        gt_end_point = Point(gt_end_point_coords)

        best_lat_idx = -1
        min_lat_dist = float('inf')
        matched_branch = None

        for idx, poly in enumerate(unique_candidates):
            dist = poly.distance(gt_end_point)
            if dist < min_lat_dist:
                min_lat_dist = dist
                best_lat_idx = idx
                matched_branch = poly

        # 5. 计算纵向进度 (Lon Label) - [鲁棒性 3] 安全投影
        s_H = 0.0
        if matched_branch:
            # 使用 context manager 屏蔽 Shapely 的 RuntimeWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:
                    s_H = matched_branch.project(gt_end_point)
                except Exception:
                    # 任何几何计算失败，回退到欧式距离
                    s_H = np.linalg.norm(gt_end_point_coords - gt_future_traj[0, :2])
        else:
            s_H = np.linalg.norm(gt_end_point_coords - gt_future_traj[0, :2])

        # 双重保险
        if np.isnan(s_H) or np.isinf(s_H):
            s_H = np.linalg.norm(gt_end_point_coords - gt_future_traj[0, :2])

        epsilon = 1.0
        s_nom = max(current_speed * future_horizon_time, epsilon)
        rho = s_H / s_nom
        rho = np.clip(rho, 0.0, 3.0)

        lon_bin_idx = np.digitize(rho, self.lon_quantiles) - 1
        lon_bin_idx = np.clip(lon_bin_idx, 0, self.num_bins - 1)

        return {
            "a_lat": best_lat_idx,
            "a_lon": lon_bin_idx,
            "rho": rho,
            "lat_intent": unique_intents[best_lat_idx] if best_lat_idx != -1 else 0,
            "num_candidates": len(unique_candidates)
        }



    def _find_all_candidate_routes(self, start_lane, max_length, route_roadblock_ids=None):
        """
        DFS 搜索，集成 Pluto 的 On-Route 过滤逻辑
        """
        results = []
        # 将 route_ids 转为 set 加速查询
        route_set = set(route_roadblock_ids) if route_roadblock_ids else set()

        init_pts = [p for p in start_lane.baseline_path.discrete_path]
        init_len = start_lane.baseline_path.length
        stack = [(start_lane, init_pts, init_len)]

        while stack:
            curr_lane, curr_pts, curr_len = stack.pop()

            if curr_len >= max_length:
                results.append(curr_pts)
                continue

            next_lanes = curr_lane.outgoing_edges
            if not next_lanes:
                results.append(curr_pts)
                continue

            # === [核心逻辑] On-Route 过滤 ===
            valid_next_lanes = []
            if route_set:
                # 策略：优先找在导航路径上的后继
                on_route_successors = [l for l in next_lanes if l.get_roadblock_id() in route_set]

                if on_route_successors:
                    # 如果有路可走，就只走正确的路 (Pluto 逻辑)
                    valid_next_lanes = on_route_successors
                else:
                    # 如果所有后继都不在导航上 (比如车已经偏航，或者导航断了)
                    # 兜底策略：搜索所有物理连接，保证不断路
                    valid_next_lanes = next_lanes
            else:
                # 没有导航信息，全搜索
                valid_next_lanes = next_lanes
            # ==============================

            for next_l in valid_next_lanes:
                next_l_pts = [p for p in next_l.baseline_path.discrete_path]
                new_pts = curr_pts + next_l_pts
                new_len = curr_len + next_l.baseline_path.length
                stack.append((next_l, new_pts, new_len))

        return results



    def _match_gt_endpoint_to_route_lanes(self,
                                          gt_future_traj: np.ndarray,
                                          route_lanes: np.ndarray,
                                          route_lanes_mask: np.ndarray = None,
                                          dist_thresh: float = 5.0,
                                          yaw_thresh_deg=60.0,
                                          ego_state=None
                                          ):
        """Match GT end point to route_lanes (on-route candidates) by nearest distance.
        Returns (best_lat_idx, min_dist, s_H_along_lane).
        """
        if gt_future_traj is None or len(gt_future_traj) == 0:
            return -1, float('inf'), 0.0
        if route_lanes is None:
            return -1, float('inf'), 0.0

        route_lanes = np.asarray(route_lanes)
        if route_lanes.ndim != 3:
            return -1, float('inf'), 0.0

        end_xy = np.asarray(gt_future_traj[-1, :2], dtype=np.float32)

        # === 方向一致性：参考朝向（优先 GT 末段，否则 ego heading） ===
        ref_yaw = self._get_ref_yaw(ego_state, gt_future_traj)
        yaw_thresh = float(np.deg2rad(yaw_thresh_deg))

        M = route_lanes.shape[0]
        lane_valid = self._lane_valid_mask_1d(route_lanes_mask)
        if lane_valid is None:
            lane_valid = np.ones((M,), dtype=bool)

        best_i = -1
        best_d2 = float('inf')
        best_sH = 0.0

        for i in range(M):
            if not lane_valid[i]:
                continue

            pts = route_lanes[i, :, :2].astype(np.float32)
            if pts.shape[0] < 2:
                continue

            diff = pts - end_xy[None, :]
            d2_all = np.sum(diff ** 2, axis=1)
            j = int(np.argmin(d2_all))
            d2 = float(d2_all[j])

            # ---- 估计 lane 切向方向 lane_yaw（用最近点附近的一段） ----
            # 取邻近点构造方向，尽量避免越界
            if j < pts.shape[0] - 1:
                v = pts[j + 1] - pts[j]
            else:
                v = pts[j] - pts[j - 1]

            v2 = float(v[0] * v[0] + v[1] * v[1])
            if v2 < 1e-8:
                # 退化：再尝试另一侧
                if 0 < j < pts.shape[0] - 1:
                    v = pts[j] - pts[j - 1]
                    v2 = float(v[0] * v[0] + v[1] * v[1])
            if v2 < 1e-8:
                continue

            lane_yaw = float(np.arctan2(v[1], v[0]))

            # ---- 方向一致性过滤：对向/反向 lane 直接跳过 ----
            if abs(self._angle_diff(lane_yaw, ref_yaw)) > yaw_thresh:
                continue

            # ---- 通过过滤后，再参与最小距离选择 ----
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_sH = self._project_point_to_polyline_s(pts, end_xy)

        min_dist = float(np.sqrt(best_d2)) if np.isfinite(best_d2) else float('inf')
        if best_i < 0:
            return -1, min_dist, 0.0
        if dist_thresh is not None and min_dist > float(dist_thresh):
            return -1, min_dist, 0.0
        return int(best_i), min_dist, float(best_sH)


    def _lane_valid_mask_1d(self, route_lanes_mask: np.ndarray) -> np.ndarray:
        """Convert route_lanes_mask to per-lane valid mask [M]."""
        if route_lanes_mask is None:
            return None
        route_lanes_mask = np.asarray(route_lanes_mask)
        if route_lanes_mask.ndim == 2:
            return (route_lanes_mask.sum(axis=1) > 0)
        if route_lanes_mask.ndim == 1:
            return route_lanes_mask.astype(bool)
        return np.ones((route_lanes_mask.shape[0],), dtype=bool)


    def _project_point_to_polyline_s(self, polyline_xy: np.ndarray, point_xy: np.ndarray) -> float:
        """Approximate arc-length s along polyline at the orthogonal projection of point."""
        pts = np.asarray(polyline_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return 0.0
        p = np.asarray(point_xy, dtype=np.float32).reshape(-1)[:2]

        seg = pts[1:] - pts[:-1]
        seg_len2 = (seg[:, 0] ** 2 + seg[:, 1] ** 2)
        seg_len2 = np.maximum(seg_len2, 1e-6)

        v = p - pts[:-1]
        t = (v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2
        t = np.clip(t, 0.0, 1.0)

        proj = pts[:-1] + seg * t[:, None]
        d2 = np.sum((proj - p[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))

        seg_lens = np.sqrt(seg_len2)
        s = float(np.sum(seg_lens[:k]) + seg_lens[k] * t[k])
        if not np.isfinite(s):
            return 0.0
        return max(s, 0.0)


    def _get_candidate_starting_lanes(self, ego_state):
        # 保持 v3.1 逻辑
        objs = self.map_api.get_proximal_map_objects(
            ego_state.rear_axle.point, radius=6.0, layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )
        candidates = objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.LANE_CONNECTOR]
        valid_lanes = []
        for lane in candidates:
            if lane.baseline_path.length < 2.0: continue
            path_points = np.array([[p.x, p.y, p.heading] for p in lane.baseline_path.discrete_path])
            dists = np.linalg.norm(path_points[:, :2] - [ego_state.rear_axle.x, ego_state.rear_axle.y], axis=1)
            nearest_idx = np.argmin(dists)
            lane_heading = path_points[nearest_idx, 2]
            heading_diff = np.abs(normalize_angle(lane_heading - ego_state.rear_axle.heading))
            if heading_diff < np.pi / 2:
                valid_lanes.append(lane)
        return valid_lanes


    def _classify_lane_intent(self, ego_state, lane):
        # 保持 v3.1 逻辑
        path_points = np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path])
        ego_pos = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y])
        dists = np.linalg.norm(path_points - ego_pos, axis=1)
        if np.min(dists) < 1.0: return 0
        nearest_pt = path_points[np.argmin(dists)]
        ego_vec = np.array([np.cos(ego_state.rear_axle.heading), np.sin(ego_state.rear_axle.heading)])
        to_lane_vec = nearest_pt - ego_pos
        cross_prod = ego_vec[0] * to_lane_vec[1] - ego_vec[1] * to_lane_vec[0]
        return 1 if cross_prod > 0 else 2


    def _deduplicate_paths(self, candidates, intents):
        # 保持 v3.1 逻辑
        if not candidates: return [], []
        keep_indices = []
        for i in range(len(candidates)):
            is_duplicate = False
            curr_line = candidates[i]
            for j in keep_indices:
                kept_line = candidates[j]
                if intents[i] == intents[j]:
                    dist = curr_line.distance(kept_line)
                    if dist < 1.0:
                        is_duplicate = True
                        break
            if not is_duplicate:
                keep_indices.append(i)
        return [candidates[i] for i in keep_indices], [intents[i] for i in keep_indices]


    def _generate_dummy_branch(self, ego_state):
        x, y = ego_state.rear_axle.x, ego_state.rear_axle.y
        h = ego_state.rear_axle.heading
        p_start = Point(x, y)
        p_end = Point(x + np.cos(h) * self.lookahead_dist, y + np.sin(h) * self.lookahead_dist)
        return LineString([p_start, p_end])


    def _get_empty_label(self):
        return {"a_lat": -1, "a_lon": -1, "rho": 0.0, "lat_intent": 0, "num_candidates": 0}


    def _get_ref_yaw(self, ego_state, gt_future_traj):
        """
        计算用于匹配车道的参考航向角 (Ref Yaw)。

        注意：输入的 gt_future_traj 已经是局部坐标系 (Local Frame) 下的轨迹。
        局部坐标系定义：Ego 当前位置为 (0,0)，Ego 当前朝向为 0.0 rad。
        """
        # 1) 尝试利用 GT 轨迹的末段计算动态航向
        if gt_future_traj is not None and len(gt_future_traj) >= 2:
            # 取最后两个点计算向量
            p1 = gt_future_traj[-2, :2]
            p2 = gt_future_traj[-1, :2]
            v = p2 - p1

            # 只有当位移足够大时，计算出的角度才可靠
            if np.isfinite(v).all() and (v[0] ** 2 + v[1] ** 2) > 1e-4:  # 稍微调大一点阈值增加鲁棒性
                return float(np.arctan2(v[1], v[0]))

        # 2) 兜底逻辑 (车辆静止或微小蠕动)
        # 错误写法 (原代码): return float(ego_state.rear_axle.heading) -> 这是 Global Yaw!
        # 正确写法: 在局部坐标系下，自车的当前朝向永远是 0.0
        return 0.0


    def _angle_diff(self, a, b):
        """wrap 到 [-pi, pi]"""
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi





