import matplotlib

# 设置后端为 Agg，这样不需要图形界面支持（适合服务器/Docker环境）
matplotlib.use('Agg')

from typing import Dict, List, Set, Deque, Tuple, Optional
from collections import deque
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import numpy as np
import shapely.geometry
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)

AGENT_COLOR_MAPPING = {
    TrackedObjectType.VEHICLE: "#001eff",  # 蓝色
    TrackedObjectType.PEDESTRIAN: "#9500ff",  # 紫色
    TrackedObjectType.BICYCLE: "#ff0059",  # 红色
}

TRAFFIC_LIGHT_COLOR_MAPPING = {
    TrafficLightStatusType.GREEN: "#2ca02c",
    TrafficLightStatusType.YELLOW: "#ff7f0e",
    TrafficLightStatusType.RED: "#d62728",
}


class NuplanScenarioRender:
    """
    [增强版] NuPlan 场景渲染器
    新增功能：决策仪表盘、车道热力图、轨迹调试标记
    """

    def __init__(
            self,
            future_horizon: float = 8,
            sample_interval: float = 0.1,
            bounds=60,
            offset=20,
            disable_agent=False,
    ) -> None:

        self.future_horizon = future_horizon
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval

        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width

        self.bounds = bounds
        self.offset = offset

        self.disable_agent = disable_agent

        self._history_trajectory = []
        self._expert_history_trajectory = []
        self.agent_histories: Dict[str, Deque[Tuple[float, float]]] = {}

        self.road_elements = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ]

        # 颜色映射用于车道热力图 (Blue -> Red)
        self.cmap = get_cmap("coolwarm")

    def reset(self):
        """重置历史轨迹，通常在场景切换时调用"""
        self._history_trajectory = []
        self._expert_history_trajectory = []
        self.agent_histories = {}

    def render_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
            route_roadblock_ids: List[str] = None,
            scenario=None,
            iteration=None,
            planning_trajectory=None,
            # [新增] 接收无引导的基线轨迹
            planning_trajectory_unguided=None,
            predictions=None,
            # --- 新增调试接口 ---
            candidate_lanes: Optional[np.ndarray] = None,  # [M, P, 2] 模型看到的车道几何
            lat_logits: Optional[np.ndarray] = None,  # [M] 横向得分
            lon_logits: Optional[np.ndarray] = None,  # [K] 纵向得分
    ):
        """
        专为 Simulation 循环设计的渲染接口
        """
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        tracked_objects = current_input.history.observations[-1]
        traffic_light_status = list(current_input.traffic_light_data)
        mission_goal = initialization.mission_goal

        if route_roadblock_ids is None:
            route_roadblock_ids = initialization.route_roadblock_ids

        gt_state = None
        if scenario is not None and iteration is not None:
            try:
                gt_state = scenario.get_ego_state_at_iteration(iteration)
            except:
                pass

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            planning_trajectory=planning_trajectory,
            # [新增] 往下传递给 render 函数
            planning_trajectory_unguided=planning_trajectory_unguided,
            predictions=predictions,
            candidate_lanes=candidate_lanes,
            lat_logits=lat_logits,
            lon_logits=lon_logits,
            return_img=True,
        )

    def render(
            self,
            map_api: AbstractMap,
            ego_state: EgoState,
            route_roadblock_ids: List[str],
            tracked_objects: TrackedObjects,
            traffic_light_status: List[TrafficLightStatusData],
            mission_goal: StateSE2,
            gt_state=None,
            planning_trajectory=None,
            # [新增] 接收参数
            planning_trajectory_unguided=None,
            predictions=None,
            candidate_lanes=None,
            lat_logits=None,
            lon_logits=None,
            return_img=False,
    ):
        # 创建画布，关闭交互模式
        # 增加 DPI 以获得更清晰的文字
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

        # 记录历史
        self._history_trajectory.append(ego_state.rear_axle.array)
        if gt_state is not None:
            self._expert_history_trajectory.append(gt_state.rear_axle.array)

        # 维护 Agent 历史
        current_tokens = set()
        for track in tracked_objects.tracked_objects:
            token = track.track_token
            current_tokens.add(token)
            if token not in self.agent_histories:
                self.agent_histories[token] = deque(maxlen=40)
            self.agent_histories[token].append(track.center.array)

        # 清除消失的 Agent
        for token in list(self.agent_histories.keys()):
            if token not in current_tokens:
                del self.agent_histories[token]

        # 建立局部坐标系变换参数 (以自车为原点)
        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

        # 1. 绘制地图 (背景层)
        tls_dict = {tl.lane_connector_id: tl.status for tl in traffic_light_status}
        self._plot_map(
            ax,
            map_api,
            ego_state.center.point,
            tls_dict,
            set(route_roadblock_ids) if route_roadblock_ids else set(),
        )

        # 2. [新增] 绘制模型候选车道热力图 (覆盖在地图之上)
        if candidate_lanes is not None and lat_logits is not None:
            self._plot_candidate_lanes(ax, candidate_lanes, lat_logits)

        # 3. 绘制自车
        self._plot_ego(ax, ego_state)
        if gt_state is not None:
            self._plot_ego(ax, gt_state, gt=True)

        # 4. 绘制 Agents
        if not self.disable_agent:
            for track in tracked_objects.tracked_objects:
                self._plot_tracked_object(ax, track)
                self._plot_agent_history(ax, track.track_token)

        # 5. 绘制预测/多模态轨迹 (如果有)
        if predictions is not None:
            self._plot_prediction(ax, predictions)

        # 6. 绘制最终规划轨迹 (含折线调试标记)
        if planning_trajectory is not None:
            self._plot_planning(ax, planning_trajectory)

        # 7. 绘制目标和自车历史
        self._plot_mission_goal(ax, mission_goal)
        self._plot_history(ax)

        # [新增] 调用绘制无引导对比轨迹的函数
        if planning_trajectory_unguided is not None:
            self._plot_unguided_trajectory(fig.axes[0], planning_trajectory_unguided)

        # 8. [新增] 绘制决策仪表盘 (画中画)
        if lat_logits is not None or lon_logits is not None:
            self._plot_dashboard(fig, lat_logits, lon_logits)

        # 设置视野 (Ego-Centric)
        ax.axis("equal")
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
        ax.axis("off")
        plt.tight_layout(pad=0)

        # 输出图像
        if return_img:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            img = img.reshape(height, width, 3)
            plt.close(fig)
            return img
        else:
            plt.show()

    # --- 绘图辅助函数 ---

    def _plot_unguided_trajectory(self, ax, trajectory_states):
        """
        [新增] 绘制关闭引导力时的模型原始预测轨迹 (灰色粗虚线)
        用于在论文视频中与加入了势场的轨迹进行 A/B 对比。
        """
        if not trajectory_states:
            return

        # 1. 提取全局坐标 (x, y)
        points = np.array([[st.x, st.y] for st in trajectory_states])

        # 2. 核心：将全局坐标转换到当前 BEV 画布的局部坐标系
        # 利用 render.py 之前初始化好的原点和平移旋转矩阵
        points_local = np.matmul(points - self.origin, self.rot_mat)

        # 3. 绘制带有科技感的灰色虚线
        ax.plot(
            points_local[:, 0],
            points_local[:, 1],
            color='gray',  # 灰色代表“被舍弃的/原始的”
            linestyle='--',  # 虚线代表“虚拟的预测”
            linewidth=3.0,  # 加粗，使其在视频中显眼
            alpha=0.8,  # 微微透明，不遮挡主路网
            zorder=19,  # 层级设置在黄色主轨迹(zorder=20)的下方一点
            label="Unguided Baseline"
        )

    def _plot_dashboard(self, fig, lat_logits, lon_logits):
        """
        绘制悬浮仪表盘：显示横纵向概率分布
        """
        # 定义子图位置 [left, bottom, width, height] (归一化坐标)
        # 放在左上角
        ax_lat = fig.add_axes([0.05, 0.82, 0.25, 0.12])
        ax_lon = fig.add_axes([0.32, 0.82, 0.15, 0.12])

        # 通用样式设置
        for ax in [ax_lat, ax_lon]:
            ax.patch.set_alpha(0.6)  # 半透明背景
            ax.tick_params(axis='both', which='major', labelsize=8)

        # 1. 横向 (Lateral)
        if lat_logits is not None:
            probs = torch.softmax(torch.tensor(lat_logits), dim=0).numpy() if not isinstance(lat_logits,
                                                                                             np.ndarray) else np.exp(
                lat_logits) / np.sum(np.exp(lat_logits))

            # 只显示 Top-10 以防太拥挤
            top_k = min(len(probs), 10)
            indices = np.argsort(probs)[::-1][:top_k]
            top_probs = probs[indices]

            # 使用颜色区分：第一名绿色，其他红色
            colors = ['green' if i == 0 else 'red' for i in range(top_k)]

            ax_lat.bar(range(top_k), top_probs, color=colors, alpha=0.7)
            ax_lat.set_title(f"Lat Prob (Top-{top_k})", fontsize=10, pad=2)
            ax_lat.set_ylim(0, 1.0)
            # 在柱子上标 Lane ID
            ax_lat.set_xticks(range(top_k))
            ax_lat.set_xticklabels([str(i) for i in indices])

        # 2. 纵向 (Longitudinal)
        if lon_logits is not None:
            probs = torch.softmax(torch.tensor(lon_logits), dim=0).numpy() if not isinstance(lon_logits,
                                                                                             np.ndarray) else np.exp(
                lon_logits) / np.sum(np.exp(lon_logits))

            # 假设 Lon Mode 0=停止/慢, 7=快
            # 使用渐变色表示速度等级
            x = range(len(probs))
            ax_lon.bar(x, probs, color='orange', alpha=0.7)
            ax_lon.set_title("Lon Prob", fontsize=10, pad=2)
            ax_lon.set_ylim(0, 1.0)
            ax_lon.set_xticks(x)
            # 标记当前最大概率
            max_idx = np.argmax(probs)
            ax_lon.get_xticklabels()[max_idx].set_color('red')
            ax_lon.get_xticklabels()[max_idx].set_weight('bold')

    def _plot_candidate_lanes(self, ax, candidate_lanes, lat_logits):
        """
        绘制带有置信度颜色的候选车道
        candidate_lanes: [M, P, 2] (Local Frame)
        """
        import torch
        if not isinstance(lat_logits, torch.Tensor):
            lat_logits = torch.tensor(lat_logits)

        # Softmax 归一化用于颜色映射
        probs = torch.softmax(lat_logits, dim=0).numpy()

        # 排序：先画低分的，后画高分的（防止高分被遮挡）
        indices = np.argsort(probs)

        for i in indices:
            lane = candidate_lanes[i]
            score = probs[i]

            # 过滤无效点 (0,0)
            valid_mask = np.abs(lane).sum(axis=1) > 0.1
            if valid_mask.sum() < 2: continue

            lane = lane[valid_mask]

            # 颜色映射：分数越高越红，越低越蓝
            color = self.cmap(score)

            # 线宽和透明度也随分数变化
            alpha = 0.3 + 0.7 * score  # 0.3 ~ 1.0
            lw = 1.0 + 3.0 * score  # 1.0 ~ 4.0

            # 绘制 (注意：这里假设 candidate_lanes 已经是 Local Frame，如果不是需转换)
            # 根据 planner.py 的逻辑，inputs['route_lanes'] 通常是 Relative (Local) 的
            # 但 render 里的 ax 也是 Local Frame (以 Ego 为原点)
            # 唯一的问题是 render 这里的 origin 和 planner 的 origin 是否完全一致
            # 我们直接画，因为 render 的 rot_mat 是 Identity (如果在 plot_candidate_lanes 外部已经处理过)
            # 等等，render 的其他函数都在做 transform。
            # 这里假设 candidate_lanes 已经是 Ego-Centric 的 (x向前, y向左)
            # 而 Render 的坐标系也是 Ego-Centric。
            # 所以不需要额外 transform。

            # 稍微做个简单的旋转修正以防万一 (如果 candidate lanes 是基于 map 坐标系的)
            # 通常 planner input 里的 route_lanes 是已经转正的 (Heading=0)

            ax.plot(lane[:, 0], lane[:, 1], color=color, alpha=alpha, linewidth=lw, zorder=15)

            # 在起点标上 Lane ID
            if score > 0.1:  # 只标概率大于 10% 的
                ax.text(lane[0, 0], lane[0, 1], str(i), color=color, fontsize=8, fontweight='bold', zorder=16)

    def _plot_map(self, ax, map_api, query_point, tls_dict, route_ids):
        # 获取附近道路元素
        road_objects = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, self.road_elements
        )
        lanes = road_objects.get(SemanticMapLayer.LANE, []) + road_objects.get(SemanticMapLayer.LANE_CONNECTOR, [])

        for obj in lanes:
            obj_id = str(obj.id)
            color = "lightgray"
            alpha = 0.4
            zorder = 0

            # 导航路径高亮
            rb_id = obj.get_roadblock_id()
            if rb_id in route_ids:
                color = "dodgerblue"
                alpha = 0.2
                zorder = 1

            # 绘制多边形
            patch = self._polygon_to_patch(obj.polygon, color=color, alpha=alpha, ec=None, zorder=zorder)
            ax.add_artist(patch)

            # 绘制中心线
            cl_color = "gray"
            # 如果是红绿灯关联车道
            if obj_id in tls_dict:
                status = tls_dict[obj_id]
                cl_color = TRAFFIC_LIGHT_COLOR_MAPPING.get(status, "gray")
            elif zorder == 1:
                cl_color = "royalblue"  # 导航线

            if hasattr(obj, 'baseline_path'):
                cl = np.array([[s.x, s.y] for s in obj.baseline_path.discrete_path])
                if len(cl) > 0:
                    cl = np.matmul(cl - self.origin, self.rot_mat)
                    ax.plot(cl[:, 0], cl[:, 1], color=cl_color, alpha=0.6, linestyle="--", zorder=zorder + 1,
                            linewidth=1)

    def _plot_ego(self, ax, ego_state: EgoState, gt=False):
        kwargs = {"lw": 1.5}
        if gt:
            ax.add_patch(
                self._polygon_to_patch(ego_state.car_footprint.geometry, color="gray", alpha=0.3, zorder=9, **kwargs))
        else:
            ax.add_patch(
                self._polygon_to_patch(ego_state.car_footprint.geometry, ec="#ff7f0e", fill=False, zorder=10, **kwargs))

    def _plot_tracked_object(self, ax, track: TrackedObject):
        center = track.center.array
        angle = track.center.heading

        # 坐标变换
        center_local = np.matmul(center - self.origin, self.rot_mat)
        angle_local = angle - self.angle

        velocity = track.velocity.magnitude()
        vis_len = max(velocity * 2.0, track.box.length)

        color = AGENT_COLOR_MAPPING.get(track.tracked_object_type, "k")

        # 绘制包围盒
        ax.add_patch(self._polygon_to_patch(track.box.geometry, ec=color, fill=False, alpha=1.0, zorder=4, lw=1.5))

        # 绘制速度方向
        if color != "k":
            direct = np.array([np.cos(angle_local), np.sin(angle_local)]) * vis_len
            start = center_local
            end = center_local + direct
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1, zorder=4, linestyle="--")

    def _plot_agent_history(self, ax, token: str):
        if token not in self.agent_histories or len(self.agent_histories[token]) < 2:
            return
        points = np.array(self.agent_histories[token])
        points = np.matmul(points - self.origin, self.rot_mat)
        ax.plot(points[:, 0], points[:, 1], color="gray", alpha=0.4, linewidth=1, zorder=3, linestyle=":")

    def _plot_planning(self, ax, planning_trajectory: np.ndarray):
        # 假设输入是 [T, 3] (x, y, h) 全局坐标
        if planning_trajectory is None or len(planning_trajectory) == 0: return

        # 提取 XY
        if isinstance(planning_trajectory, list):  # List[InterpolatableState]
            traj = np.array([s.rear_axle.array for s in planning_trajectory])
        else:
            traj = planning_trajectory[:, :2]  # ndarray

        # 变换到局部坐标系
        traj_local = np.matmul(traj - self.origin, self.rot_mat)

        # 绘制轨迹线
        ax.plot(traj_local[:, 0], traj_local[:, 1], color="#ff7f0e", linewidth=3, alpha=0.8, zorder=20)

        # === [新增] 调试标记：绘制轨迹起点 ===
        # 如果起点 (0,0) 和车身中心偏差很大，这里会显现出来
        start_pt = traj_local[0]
        ax.plot(start_pt[0], start_pt[1], 'rx', markersize=10, markeredgewidth=2, zorder=21, label="Plan Start")

        # 也可以画一条虚线连接车身原点(0,0)和轨迹起点，看偏差
        ax.plot([0, start_pt[0]], [0, start_pt[1]], 'r:', linewidth=1, zorder=21)

    def _plot_prediction(self, ax, predictions):
        """
        绘制多模态预测轨迹 (如果提供)
        Predictions format: [K, T, 2] or similar
        """
        if predictions is None: return

        # 简单的处理逻辑，假设 predictions 是 list of arrays
        # 如果是 Tensor 需要先转 numpy
        # 这里只做框架性支持
        pass

    def _plot_mission_goal(self, ax, mission_goal: StateSE2):
        if mission_goal is None: return
        point = np.matmul(mission_goal.point.array - self.origin, self.rot_mat)
        ax.plot(point[0], point[1], marker="*", markersize=8, color="gold", zorder=6, markeredgecolor='black')

    def _plot_history(self, ax):
        if len(self._history_trajectory) > 1:
            history = np.array(self._history_trajectory)
            history = np.matmul(history - self.origin, self.rot_mat)
            ax.plot(history[:, 0], history[:, 1], color="#ff7f0e", alpha=0.5, zorder=6, linewidth=2)

    def _polygon_to_patch(self, polygon, **kwargs):
        if not isinstance(polygon, shapely.geometry.Polygon):
            return patches.Polygon(np.array([]), **kwargs)
        polygon_pts = np.array(polygon.exterior.xy).T
        # 对多边形所有点做坐标变换
        polygon_pts = np.matmul(polygon_pts - self.origin, self.rot_mat)
        return patches.Polygon(polygon_pts, **kwargs)