"""Build style-scene split outputs directly from raw NuPlan scenarios."""

from __future__ import annotations

import inspect
import json
import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from research_v1.paths import DEFAULT_NUM_WORKERS, ensure_repo_on_path

ensure_repo_on_path()

from nuplan.common.actor_state.state_representation import Point2D

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

from baseline.data_process.agent_process import (
    agent_future_process,
    agent_past_process,
    sampled_static_objects_to_array_list,
    sampled_tracked_objects_to_array_list,
)
from baseline.data_process.codebook_labeler import CodebookLabeler
from baseline.data_process.data_processor import DataProcessor
from baseline.data_process.ego_process import (
    calculate_additional_ego_states,
    get_ego_future_array_from_scenario,
    get_ego_past_array_from_scenario,
)
from baseline.data_process.map_process import get_neighbor_vector_set_map, map_process
from baseline.data_process.roadblock_utils import route_roadblock_correction
from baseline.train.train_utils import openjson, opendata
from research_v1.scene_data.legacy.schema import (
    SCENE_BUCKET_ID_TO_NAME,
    STYLE_LABEL_ID_TO_NAME,
    STYLE_SCENE_SPLIT_SCHEMA_VERSION,
    TOPOLOGY_BUCKET_ID_TO_NAME,
)
from research_v1.scene_data.legacy.splitter import StyleSceneSplitter

SCHEMA_VERSION = STYLE_SCENE_SPLIT_SCHEMA_VERSION


def get_filter_parameters(
    num_scenarios_per_type: Optional[int] = None,
    limit_total_scenarios: Optional[int] = None,
    shuffle: bool = True,
    scenario_tokens: Optional[Sequence[str]] = None,
    log_names: Optional[Sequence[str]] = None,
):
    """Mirror the raw scenario filtering used by the top-level data preprocessing script."""

    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def load_optional_json_list(path: str) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"json list path not found: {path}")
    payload = openjson(path)
    if not isinstance(payload, list):
        raise ValueError(f"json list path must contain a list: {path}")
    return [str(item) for item in payload]


def build_raw_scenarios(
    data_path: str,
    map_path: str,
    map_version: str,
    scenarios_per_type: Optional[int],
    total_scenarios: Optional[int],
    shuffle_scenarios: bool,
    log_names_path: str = "",
    scenario_tokens_path: str = "",
) -> List[object]:
    """Materialize NuPlan scenarios directly from the raw dataset."""

    sensor_root = None
    db_files = None
    log_names = load_optional_json_list(log_names_path)
    scenario_tokens = load_optional_json_list(scenario_tokens_path)

    builder = NuPlanScenarioBuilder(data_path, map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            num_scenarios_per_type=scenarios_per_type,
            limit_total_scenarios=total_scenarios,
            shuffle=shuffle_scenarios,
            scenario_tokens=scenario_tokens,
            log_names=log_names,
        )
    )
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    del worker, builder, scenario_filter
    scenarios = list(scenarios)
    scenarios.sort(key=lambda item: (str(getattr(item, "log_name", "")), str(getattr(item, "scenario_name", "")), str(getattr(item, "token", ""))))
    return scenarios


def _create_codebook_labeler(processor: DataProcessor):
    """
    Prefer the helper on newer DataProcessor versions.
    Fall back to local construction so style_scene_split can still run on
    older remote branches.
    """
    if hasattr(processor, "create_codebook_labeler"):
        return processor.create_codebook_labeler()

    custom_bins = [0.40, 0.80, 1.00, 1.35, 2.30, 3.00, 3.50]
    init_signature = inspect.signature(CodebookLabeler.__init__)
    init_kwargs = {
        "map_api": None,
        "lookahead_dist": 50.0,
        "num_bins": 8,
    }
    if "custom_bins" in init_signature.parameters:
        init_kwargs["custom_bins"] = custom_bins
    return CodebookLabeler(**init_kwargs)


def _get_labels_compat(labeler, ego_state, ego_future_rel_np: np.ndarray, current_speed: float, future_horizon_time: float,
                       route_roadblock_ids, route_lanes, route_lanes_mask):
    """
    Call CodebookLabeler.get_labels() against both new and older signatures.
    """
    get_labels_signature = inspect.signature(labeler.get_labels)
    kwargs = {}
    if "route_roadblock_ids" in get_labels_signature.parameters:
        kwargs["route_roadblock_ids"] = route_roadblock_ids
    if "route_lanes" in get_labels_signature.parameters:
        kwargs["route_lanes"] = route_lanes
    if "route_lanes_mask" in get_labels_signature.parameters:
        kwargs["route_lanes_mask"] = route_lanes_mask
    if "lat_match_dist_thresh" in get_labels_signature.parameters:
        kwargs["lat_match_dist_thresh"] = 5.0
    return labeler.get_labels(
        ego_state,
        ego_future_rel_np,
        current_speed,
        future_horizon_time,
        **kwargs,
    )


def _extract_scenario_data(processor: DataProcessor, scenario: object, labeler):
    """
    Reuse DataProcessor.extract_scenario_data() when available.
    Otherwise reconstruct the same sample dict locally.
    """
    if hasattr(processor, "extract_scenario_data"):
        return processor.extract_scenario_data(scenario, labeler=labeler)

    map_name = getattr(scenario, "_map_name", "unknown")
    token = scenario.token
    map_api = scenario.map_api
    labeler.map_api = map_api

    ego_state = scenario.initial_ego_state
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    anchor_ego_state = np.array(
        [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
        dtype=np.float64,
    )

    ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
        scenario,
        processor.num_past_poses,
        processor.past_time_horizon,
    )

    present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
    past_tracked_objects = [
        tracked_objects.tracked_objects
        for tracked_objects in scenario.get_past_tracked_objects(
            iteration=0,
            time_horizon=processor.past_time_horizon,
            num_samples=processor.num_past_poses,
        )
    ]
    sampled_past_observations = past_tracked_objects + [present_tracked_objects]

    neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(sampled_past_observations)
    static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

    ego_agent_past, neighbor_agents_past, neighbor_agents_past_mask, _, static_objects, track_tokens = agent_past_process(
        ego_agent_past,
        neighbor_agents_past,
        neighbor_agents_types,
        processor.num_agents,
        static_objects,
        static_objects_types,
        processor.num_static,
        processor.max_ped_bike,
        anchor_ego_state,
        sampled_past_observations,
    )

    route_roadblock_ids = scenario.get_route_roadblock_ids()
    traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
    if route_roadblock_ids != ['']:
        route_roadblock_ids = route_roadblock_correction(ego_state, map_api, route_roadblock_ids)

    coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
        map_api,
        processor._map_features,
        ego_coords,
        processor._radius,
        traffic_light_data,
    )
    vector_map = map_process(
        route_roadblock_ids,
        anchor_ego_state,
        coords,
        traffic_light_data,
        speed_limit,
        lane_route,
        processor._map_features,
        processor._max_elements,
        processor._max_points,
    )

    ego_agent_future = get_ego_future_array_from_scenario(
        scenario,
        ego_state,
        processor.num_future_poses,
        processor.future_time_horizon,
    )
    future_tracked_objects = [
        tracked_objects.tracked_objects
        for tracked_objects in scenario.get_future_tracked_objects(
            iteration=0,
            time_horizon=processor.future_time_horizon,
            num_samples=processor.num_future_poses,
        )
    ]
    neighbor_agents_future, neighbor_agents_future_mask = agent_future_process(
        anchor_ego_state,
        future_tracked_objects,
        processor.num_agents,
        track_tokens,
    )

    ego_future_abs_states = list(
        scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=processor.num_future_poses,
            time_horizon=processor.future_time_horizon,
        )
    )
    if len(ego_future_abs_states) > 0:
        ego_future_abs_np = np.array(
            [[state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading] for state in ego_future_abs_states],
            dtype=np.float32,
        )
        ax, ay, ah = float(anchor_ego_state[0]), float(anchor_ego_state[1]), float(anchor_ego_state[2])
        c, s = np.cos(-ah), np.sin(-ah)
        xy = ego_future_abs_np[:, :2].copy()
        xy[:, 0] -= ax
        xy[:, 1] -= ay
        x_rel = xy[:, 0] * c - xy[:, 1] * s
        y_rel = xy[:, 0] * s + xy[:, 1] * c
        h_rel = ego_future_abs_np[:, 2] - ah
        h_rel = (h_rel + np.pi) % (2 * np.pi) - np.pi
        ego_future_rel_np = np.stack([x_rel, y_rel, h_rel], axis=1).astype(np.float32)
        labels = _get_labels_compat(
            labeler,
            ego_state,
            ego_future_rel_np,
            ego_state.dynamic_car_state.speed,
            processor.future_time_horizon,
            route_roadblock_ids,
            vector_map.get("route_lanes", None),
            vector_map.get("route_lanes_mask", None),
        )
    else:
        labels = {"a_lat": -1, "a_lon": -1, "rho": 0.0}

    ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

    data = {
        "map_name": map_name,
        "token": token,
        "ego_current_state": ego_current_state,
        "ego_agent_future": ego_agent_future,
        "ego_agent_past": ego_agent_past,
        "neighbor_agents_past": neighbor_agents_past,
        "neighbor_agents_past_mask": neighbor_agents_past_mask,
        "neighbor_agents_future": neighbor_agents_future,
        "neighbor_agents_future_mask": neighbor_agents_future_mask,
        "static_objects": static_objects,
        "code_lat": np.array(labels["a_lat"], dtype=np.int64),
        "code_lon": np.array(labels["a_lon"], dtype=np.int64),
        "code_rho": np.array(labels["rho"], dtype=np.float32),
    }
    data.update(vector_map)
    return data


def _chunk_scenarios(scenarios: Sequence[object], chunk_size: int) -> List[List[object]]:
    return [list(scenarios[idx: idx + chunk_size]) for idx in range(0, len(scenarios), chunk_size)]


def _process_scenario_chunk(worker_payload: Dict[str, object]) -> Dict[str, object]:
    scenarios = worker_payload["scenarios"]
    builder = RawScenarioStyleSceneSplitBuilder(
        output_dir=worker_payload["output_dir"],
        time_delta=float(worker_payload["time_delta"]),
        skip_existing=bool(worker_payload["skip_existing"]),
        log_interval=int(worker_payload["log_interval"]),
        agent_num=int(worker_payload["agent_num"]),
        static_objects_num=int(worker_payload["static_objects_num"]),
        lane_len=int(worker_payload["lane_len"]),
        lane_num=int(worker_payload["lane_num"]),
        route_len=int(worker_payload["route_len"]),
        route_num=int(worker_payload["route_num"]),
        num_workers=1,
    )

    processed = 0
    skipped = 0
    failed = 0
    failure_preview: List[Dict[str, str]] = []

    for scenario in scenarios:
        status, detail = builder.process_single_scenario(scenario)
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            if len(failure_preview) < 10:
                failure_preview.append(
                    {
                        "scenario_name": builder._safe_str(getattr(scenario, "scenario_name", "")),
                        "token": builder._safe_str(getattr(scenario, "token", "")),
                        "reason": str(detail),
                    }
                )

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "count": len(scenarios),
        "failure_preview": failure_preview,
    }


class RawScenarioStyleSceneSplitBuilder:
    """
    Build a style split dataset from raw NuPlan scenarios while emitting
    data_processor-compatible `.npz` files for direct downstream retrieval.
    """

    def __init__(
        self,
        output_dir: str,
        time_delta: float = 0.1,
        skip_existing: bool = False,
        log_interval: int = 200,
        agent_num: int = 32,
        static_objects_num: int = 5,
        lane_len: int = 20,
        lane_num: int = 70,
        route_len: int = 20,
        route_num: int = 25,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self.output_dir = output_dir
        self.style_cache_dir = os.path.join(output_dir, "style_cache")
        self.sidecar_dir = os.path.join(output_dir, "sidecar")
        self.list_dir = os.path.join(output_dir, "subset_lists")
        self.report_dir = os.path.join(output_dir, "reports")
        self.index_path = os.path.join(output_dir, "split_index.jsonl")
        self.cache_list_path = os.path.join(output_dir, "style_cache_list.json")
        self.valid_cache_list_path = os.path.join(output_dir, "valid_style_cache_list.json")
        self.skip_existing = bool(skip_existing)
        self.log_interval = int(log_interval)
        self.num_workers = max(int(num_workers), 1)
        self.agent_num = int(agent_num)
        self.static_objects_num = int(static_objects_num)
        self.lane_len = int(lane_len)
        self.lane_num = int(lane_num)
        self.route_len = int(route_len)
        self.route_num = int(route_num)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.style_cache_dir, exist_ok=True)
        os.makedirs(self.sidecar_dir, exist_ok=True)
        os.makedirs(self.list_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        processor_cfg = SimpleNamespace(
            save_path=self.style_cache_dir,
            agent_num=self.agent_num,
            static_objects_num=self.static_objects_num,
            lane_len=self.lane_len,
            lane_num=self.lane_num,
            route_len=self.route_len,
            route_num=self.route_num,
        )
        self.processor = DataProcessor(processor_cfg)
        self.labeler = _create_codebook_labeler(self.processor)
        self.splitter = StyleSceneSplitter(time_delta=time_delta)

    def process_scenarios(
        self,
        scenarios: Sequence[object],
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, object]:
        sliced = self._slice_scenarios(scenarios, start_index, end_index)
        total = len(sliced)
        start_time = time.time()

        if self.num_workers <= 1 or total <= 1:
            processed, skipped, failed, failure_preview = self._process_scenarios_serial(sliced, start_time)
        else:
            processed, skipped, failed, failure_preview = self._process_scenarios_parallel(sliced, start_time)

        reindex_stats = self.rebuild_global_index()
        elapsed_seconds = time.time() - start_time
        summary = {
            "style_scene_split_schema_version": SCHEMA_VERSION,
            "source_mode": "raw_scenario",
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "elapsed_seconds": elapsed_seconds,
            "scenarios_per_second": total / max(elapsed_seconds, 1e-6),
            "output_dir": self.output_dir,
            "style_cache_dir": self.style_cache_dir,
            "index_path": self.index_path,
            "indexed_total": reindex_stats["indexed_total"],
            "indexed_valid_total": reindex_stats["indexed_valid_total"],
            "scene_distribution": reindex_stats["scene_distribution"],
            "style_distribution": reindex_stats["style_distribution"],
            "subset_distribution": reindex_stats["subset_distribution"],
            "valid_subset_distribution": reindex_stats["valid_subset_distribution"],
            "failure_preview": failure_preview,
        }
        self._write_summary(summary)
        return summary

    def _process_scenarios_serial(
        self,
        scenarios: Sequence[object],
        start_time: float,
    ) -> Tuple[int, int, int, List[Dict[str, str]]]:
        processed = 0
        skipped = 0
        failed = 0
        failure_preview: List[Dict[str, str]] = []

        with tqdm(scenarios, desc="Build raw style_scene_split", unit="scenario") as progress:
            for idx, scenario in enumerate(progress, start=1):
                status, detail = self.process_single_scenario(scenario)
                if status == "processed":
                    processed += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    if len(failure_preview) < 20:
                        failure_preview.append(
                            {
                                "scenario_name": self._safe_str(getattr(scenario, "scenario_name", "")),
                                "token": self._safe_str(getattr(scenario, "token", "")),
                                "reason": str(detail),
                            }
                        )

                if idx % self.log_interval == 0 or idx == len(scenarios):
                    elapsed = time.time() - start_time
                    speed = idx / max(elapsed, 1e-6)
                    progress.set_postfix(processed=processed, skipped=skipped, failed=failed, speed=f"{speed:.2f}/s")

        return processed, skipped, failed, failure_preview

    def _process_scenarios_parallel(
        self,
        scenarios: Sequence[object],
        start_time: float,
    ) -> Tuple[int, int, int, List[Dict[str, str]]]:
        total = len(scenarios)
        worker_count = min(self.num_workers, total)
        chunk_size = max(1, math.ceil(total / worker_count))
        scenario_chunks = _chunk_scenarios(scenarios, chunk_size)

        processed = 0
        skipped = 0
        failed = 0
        failure_preview: List[Dict[str, str]] = []

        worker_payloads: List[Dict[str, object]] = []
        for chunk in scenario_chunks:
            worker_payloads.append(
                {
                    "scenarios": chunk,
                    "output_dir": self.output_dir,
                    "time_delta": self.splitter.time_delta,
                    "skip_existing": self.skip_existing,
                    "log_interval": self.log_interval,
                    "agent_num": self.agent_num,
                    "static_objects_num": self.static_objects_num,
                    "lane_len": self.lane_len,
                    "lane_num": self.lane_num,
                    "route_len": self.route_len,
                    "route_num": self.route_num,
                }
            )

        start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        ctx = multiprocessing.get_context(start_method)
        with ctx.Pool(processes=worker_count) as pool:
            with tqdm(total=total, desc="Build raw style_scene_split", unit="scenario") as progress:
                for result in pool.imap_unordered(_process_scenario_chunk, worker_payloads):
                    processed += int(result["processed"])
                    skipped += int(result["skipped"])
                    failed += int(result["failed"])
                    progress.update(int(result["count"]))

                    for failure in result.get("failure_preview", []):
                        if len(failure_preview) < 20:
                            failure_preview.append(failure)

                    elapsed = time.time() - start_time
                    speed = progress.n / max(elapsed, 1e-6)
                    progress.set_postfix(
                        workers=worker_count,
                        processed=processed,
                        skipped=skipped,
                        failed=failed,
                        speed=f"{speed:.2f}/s",
                    )

        return processed, skipped, failed, failure_preview

    def process_single_scenario(self, scenario: object) -> Tuple[str, str]:
        metadata = self._scenario_metadata(scenario)
        filename = self._filename_from_metadata(metadata)
        style_cache_path = os.path.join(self.style_cache_dir, filename)
        sidecar_path = os.path.join(self.sidecar_dir, filename)

        if self.skip_existing and os.path.exists(style_cache_path) and os.path.exists(sidecar_path):
            return "skipped", filename

        try:
            sample = _extract_scenario_data(self.processor, scenario, labeler=self.labeler)
            sample.update(self._metadata_to_numpy(metadata))
            split_result = self.splitter.split(sample)
            style_payload = dict(sample)
            style_payload.update(split_result.to_numpy_dict())
            style_payload.update(self._metadata_to_numpy(metadata))
            style_payload["style_scene_split_schema_version"] = np.array(SCHEMA_VERSION, dtype=np.int64)
            style_payload["source_mode"] = np.array("raw_scenario")
            sidecar_payload = split_result.to_numpy_dict()
            sidecar_payload.update(self._metadata_to_numpy(metadata))
            sidecar_payload["style_scene_split_schema_version"] = np.array(SCHEMA_VERSION, dtype=np.int64)
            sidecar_payload["source_mode"] = np.array("raw_scenario")

            self._safe_save_npz(style_cache_path, style_payload)
            self._safe_save_npz(sidecar_path, sidecar_payload)
            return "processed", filename
        except Exception as exc:
            return "failed", str(exc)

    def rebuild_global_index(self) -> Dict[str, object]:
        sidecar_files = sorted(filename for filename in os.listdir(self.sidecar_dir) if filename.endswith(".npz"))
        index_records: List[Dict[str, object]] = []
        subset_lists: Dict[str, List[str]] = defaultdict(list)
        scene_counter: Counter[str] = Counter()
        style_counter: Counter[str] = Counter()
        subset_counter: Counter[str] = Counter()
        valid_subset_counter: Counter[str] = Counter()
        valid_cache_filenames: List[str] = []

        with tqdm(sidecar_files, desc="Rebuild raw global index", unit="file") as progress:
            for idx, filename in enumerate(progress, start=1):
                sidecar_path = os.path.join(self.sidecar_dir, filename)
                sidecar_data = None
                try:
                    sidecar_data = opendata(sidecar_path)
                    record = self._compose_index_record(filename, sidecar_path, sidecar_data)
                    index_records.append(record)
                    scene_counter[str(record["scene_bucket_name"])] += 1
                    style_counter[str(record["style_label_name"])] += 1
                    subset_counter[str(record["subset_id"])] += 1
                    if bool(record["split_valid"]):
                        valid_subset_counter[str(record["subset_id"])] += 1
                        subset_lists[str(record["subset_id"])].append(filename)
                        valid_cache_filenames.append(filename)
                finally:
                    if sidecar_data is not None:
                        sidecar_data.close()

                if idx % self.log_interval == 0 or idx == len(sidecar_files):
                    progress.set_postfix(indexed=len(index_records), valid=len(valid_cache_filenames))

        self._write_index(index_records)
        self._write_subset_lists(subset_lists)
        self._write_json_list(self.cache_list_path, sidecar_files)
        self._write_json_list(self.valid_cache_list_path, valid_cache_filenames)
        return {
            "indexed_total": len(index_records),
            "indexed_valid_total": len(valid_cache_filenames),
            "scene_distribution": dict(scene_counter),
            "style_distribution": dict(style_counter),
            "subset_distribution": dict(subset_counter),
            "valid_subset_distribution": dict(valid_subset_counter),
        }

    def _compose_index_record(self, filename: str, sidecar_path: str, sidecar_data) -> Dict[str, object]:
        style_cache_path = os.path.join(self.style_cache_dir, filename)
        scene_bucket_id = self._scalar_int(sidecar_data, "scene_bucket", default=0)
        style_label_id = self._scalar_int(sidecar_data, "style_label", default=0)
        topology_bucket_id = self._scalar_int(sidecar_data, "topology_bucket", default=0)
        primary_bucket_id = self._scalar_int(sidecar_data, "primary_bucket", default=0)
        secondary_bucket_id = self._scalar_int(sidecar_data, "secondary_bucket", default=0)

        return {
            "style_scene_split_schema_version": SCHEMA_VERSION,
            "source_mode": "raw_scenario",
            "sample_id": os.path.splitext(filename)[0],
            "filename": filename,
            "planner_cache_path": style_cache_path,
            "style_cache_path": style_cache_path,
            "cache_path": style_cache_path,
            "sidecar_path": sidecar_path,
            "token": self._scalar_str(sidecar_data, "token", default=os.path.splitext(filename)[0]),
            "map_name": self._scalar_str(sidecar_data, "map_name", default="unknown"),
            "log_name": self._scalar_str(sidecar_data, "log_name", default=""),
            "scenario_name": self._scalar_str(sidecar_data, "scenario_name", default=""),
            "scenario_type": self._scalar_str(sidecar_data, "scenario_type", default=""),
            "scene_bucket_name": SCENE_BUCKET_ID_TO_NAME.get(scene_bucket_id, "none"),
            "style_label_name": STYLE_LABEL_ID_TO_NAME.get(style_label_id, "unknown"),
            "subset_id": self._scalar_str(sidecar_data, "subset_id", default="invalid"),
            "topology_bucket_name": TOPOLOGY_BUCKET_ID_TO_NAME.get(topology_bucket_id, "unknown"),
            "primary_bucket_name": SCENE_BUCKET_ID_TO_NAME.get(primary_bucket_id, "none"),
            "secondary_bucket_name": SCENE_BUCKET_ID_TO_NAME.get(secondary_bucket_id, "none"),
            "scene_confidence": self._scalar_float(sidecar_data, "scene_confidence", default=0.0),
            "style_confidence": self._scalar_float(sidecar_data, "style_confidence", default=0.0),
            "split_confidence": self._scalar_float(sidecar_data, "split_confidence", default=0.0),
            "split_valid": bool(self._scalar_int(sidecar_data, "split_valid", default=0)),
            "primary_score": self._scalar_float(sidecar_data, "primary_score", default=0.0),
            "secondary_score": self._scalar_float(sidecar_data, "secondary_score", default=0.0),
            "dominant_neighbor_idx": self._scalar_int(sidecar_data, "dominant_neighbor_idx", default=-1),
            "scene_reason": self._scalar_str(sidecar_data, "scene_reason", default=""),
            "style_reason": self._scalar_str(sidecar_data, "style_reason", default=""),
            "scene_score_vec": self._array_to_float_list(sidecar_data, "scene_score_vec"),
            "style_score_vec": self._array_to_float_list(sidecar_data, "style_score_vec"),
            "global_min_distance": self._scalar_float(sidecar_data, "global_min_distance", default=0.0),
            "following_min_gap": self._scalar_float(sidecar_data, "following_min_gap", default=0.0),
            "following_min_thw": self._optional_float_from_npz(sidecar_data, "following_min_thw"),
            "crossing_min_distance": self._scalar_float(sidecar_data, "crossing_min_distance", default=0.0),
            "crossing_time_offset": self._optional_float_from_npz(sidecar_data, "crossing_time_offset"),
            "merge_min_gap": self._scalar_float(sidecar_data, "merge_min_gap", default=0.0),
            "merge_lateral_closure": self._scalar_float(sidecar_data, "merge_lateral_closure", default=0.0),
            "ego_mean_speed": self._scalar_float(sidecar_data, "ego_mean_speed", default=0.0),
            "ego_accel_peak": self._scalar_float(sidecar_data, "ego_accel_peak", default=0.0),
            "ego_brake_peak": self._scalar_float(sidecar_data, "ego_brake_peak", default=0.0),
            "ego_jerk_peak": self._scalar_float(sidecar_data, "ego_jerk_peak", default=0.0),
            "ego_jerk_p90": self._scalar_float(sidecar_data, "ego_jerk_p90", default=0.0),
            "ego_progress": self._scalar_float(sidecar_data, "ego_progress", default=0.0),
            "ego_speed_ratio_to_limit": self._optional_float_from_npz(sidecar_data, "ego_speed_ratio_to_limit"),
            "route_speed_limit_mps": self._optional_float_from_npz(sidecar_data, "route_speed_limit_mps"),
            "event_speed_drop_ratio": self._scalar_float(sidecar_data, "event_speed_drop_ratio", default=0.0),
            "event_brake_peak": self._scalar_float(sidecar_data, "event_brake_peak", default=0.0),
            "ego_lateral_disp": self._scalar_float(sidecar_data, "ego_lateral_disp", default=0.0),
            "ego_lateral_speed_peak": self._scalar_float(sidecar_data, "ego_lateral_speed_peak", default=0.0),
            "ego_lateral_onset_step": self._optional_float_from_npz(sidecar_data, "ego_lateral_onset_step"),
            "ego_heading_change": self._scalar_float(sidecar_data, "ego_heading_change", default=0.0),
            "route_lane_count": self._scalar_int(sidecar_data, "route_lane_count", default=0),
            "nearby_agent_count": self._scalar_int(sidecar_data, "nearby_agent_count", default=0),
            "lead_vehicle_present": bool(self._scalar_int(sidecar_data, "lead_vehicle_present", default=0)),
            "route_has_control": bool(self._scalar_int(sidecar_data, "route_has_control", default=0)),
        }

    def _scenario_metadata(self, scenario: object) -> Dict[str, str]:
        map_name = self._safe_str(getattr(scenario, "_map_name", "unknown"))
        return {
            "token": self._safe_str(getattr(scenario, "token", "")),
            "map_name": map_name,
            "log_name": self._safe_str(getattr(scenario, "log_name", "")),
            "scenario_name": self._safe_str(getattr(scenario, "scenario_name", "")),
            "scenario_type": self._safe_str(getattr(scenario, "scenario_type", "")),
        }

    def _metadata_to_numpy(self, metadata: Dict[str, str]) -> Dict[str, np.ndarray]:
        return {key: np.array(value) for key, value in metadata.items()}

    def _filename_from_metadata(self, metadata: Dict[str, str]) -> str:
        return f"{metadata['map_name']}_{metadata['token']}.npz"

    def _slice_scenarios(
        self,
        scenarios: Sequence[object],
        start_index: int,
        end_index: Optional[int],
    ) -> List[object]:
        total = len(scenarios)
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        if end_index is not None and end_index < start_index:
            raise ValueError(f"end_index must be >= start_index, got {end_index}")
        start = min(start_index, total)
        end = total if end_index is None else min(end_index, total)
        return list(scenarios[start:end])

    def _write_index(self, index_records: Sequence[Dict[str, object]]) -> None:
        tmp_path = f"{self.index_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            for record in index_records:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.index_path)

    def _write_subset_lists(self, subset_lists: Dict[str, List[str]]) -> None:
        existing_files = [
            os.path.join(self.list_dir, filename)
            for filename in os.listdir(self.list_dir)
            if filename.endswith(".json")
        ]
        for subset_id, filenames in subset_lists.items():
            save_path = os.path.join(self.list_dir, f"{subset_id}.json")
            self._write_json_list(save_path, sorted(filenames))
        active_paths = {os.path.join(self.list_dir, f"{subset_id}.json") for subset_id in subset_lists}
        for stale_path in existing_files:
            if stale_path not in active_paths and os.path.exists(stale_path):
                os.remove(stale_path)

    def _write_summary(self, summary: Dict[str, object]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.report_dir, f"style_scene_split_summary_{timestamp}.json")
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    def _write_json_list(self, save_path: str, payload: Sequence[str]) -> None:
        tmp_path = f"{save_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(list(payload), file_obj, ensure_ascii=False, indent=2)
        os.replace(tmp_path, save_path)

    def _safe_save_npz(self, save_path: str, payload: Dict[str, np.ndarray]) -> None:
        tmp_path = f"{save_path}.tmp"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(tmp_path, "wb") as file_obj:
            np.savez(file_obj, **payload)
        os.replace(tmp_path, save_path)

    def _safe_str(self, value: object) -> str:
        return "" if value is None else str(value)

    def _scalar_str(self, npz_data, key: str, default: str) -> str:
        if not (hasattr(npz_data, "files") and key in npz_data.files):
            return default
        value = np.asarray(npz_data[key]).reshape(-1)
        if value.size == 0:
            return default
        return str(value[0])

    def _scalar_int(self, npz_data, key: str, default: int) -> int:
        if not (hasattr(npz_data, "files") and key in npz_data.files):
            return int(default)
        value = np.asarray(npz_data[key]).reshape(-1)
        if value.size == 0:
            return int(default)
        return int(value[0])

    def _scalar_float(self, npz_data, key: str, default: float) -> float:
        if not (hasattr(npz_data, "files") and key in npz_data.files):
            return float(default)
        value = np.asarray(npz_data[key]).reshape(-1)
        if value.size == 0:
            return float(default)
        value = float(value[0])
        return float(default) if not np.isfinite(value) else value

    def _array_to_float_list(self, npz_data, key: str) -> List[float]:
        if not (hasattr(npz_data, "files") and key in npz_data.files):
            return []
        value = np.asarray(npz_data[key], dtype=np.float32).reshape(-1)
        return [float(item) for item in value.tolist() if np.isfinite(item)]

    def _optional_float_from_npz(self, npz_data, key: str) -> Optional[float]:
        if not (hasattr(npz_data, "files") and key in npz_data.files):
            return None
        value = np.asarray(npz_data[key]).reshape(-1)
        if value.size == 0:
            return None
        value = float(value[0])
        if not np.isfinite(value) or value >= 1e5 or value <= -0.5:
            return None
        return value
