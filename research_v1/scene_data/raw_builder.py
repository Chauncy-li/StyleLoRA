"""Build v2 style caches and sidecars directly from raw NuPlan scenarios.

The builder uses memory eligibility as the admission rule while retaining the
legacy raw-scenario infrastructure and split semantics.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from baseline.train.train_utils import opendata
from research_v1.paths import DEFAULT_NUM_WORKERS
from research_v1.scene_data.legacy.raw_builder import (
    RawScenarioStyleSceneSplitBuilder,
    _chunk_scenarios,
    _extract_scenario_data,
    _create_codebook_labeler,
    build_raw_scenarios,
)
from research_v1.scene_data.schema import (
    CURVATURE_LEVEL_ID_TO_NAME,
    DENSITY_LEVEL_ID_TO_NAME,
    SCENE_BUCKET_ID_TO_NAME,
    SPEED_REGIME_ID_TO_NAME,
    STYLE_LABEL_ID_TO_NAME,
    STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION,
    TOPOLOGY_BUCKET_ID_TO_NAME,
    style_axis_names_for_scene,
)
from research_v1.scene_data.splitter import StyleSceneSplitterV2


SCHEMA_VERSION = STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION


def _process_scenario_chunk_v2(worker_payload: Dict[str, object]) -> Dict[str, object]:
    """V2 worker entry point that instantiates the enhanced builder."""

    scenarios = worker_payload["scenarios"]
    builder = RawScenarioStyleSceneSplitBuilderV2(
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


class RawScenarioStyleSceneSplitBuilderV2(RawScenarioStyleSceneSplitBuilder):
    """Raw-scenario builder with v2 splitting and eligibility metadata."""

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
        super().__init__(
            output_dir=output_dir,
            time_delta=time_delta,
            skip_existing=skip_existing,
            log_interval=log_interval,
            agent_num=agent_num,
            static_objects_num=static_objects_num,
            lane_len=lane_len,
            lane_num=lane_num,
            route_len=route_len,
            route_num=route_num,
            num_workers=num_workers,
        )
        # Replace only the splitter; retain all other parent infrastructure.
        self.labeler = _create_codebook_labeler(self.processor)
        self.splitter = StyleSceneSplitterV2(time_delta=time_delta)

    def process_scenarios(
        self,
        scenarios: Sequence[object],
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, object]:
        """Process raw scenarios and return the enhanced summary."""

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
            "source_mode": "raw_scenario_v2",
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
            "indexed_memory_total": reindex_stats["indexed_memory_total"],
            "indexed_quality_valid_total": reindex_stats["indexed_quality_valid_total"],
            "scene_distribution": reindex_stats["scene_distribution"],
            "style_distribution": reindex_stats["style_distribution"],
            "subset_distribution": reindex_stats["subset_distribution"],
            "memory_subset_distribution": reindex_stats["memory_subset_distribution"],
            "failure_preview": failure_preview,
        }
        self._write_summary(summary)
        return summary

    def _process_scenarios_parallel(
        self,
        scenarios: Sequence[object],
        start_time: float,
    ) -> Tuple[int, int, int, List[Dict[str, str]]]:
        """Run the parent chunking logic with the v2 worker."""

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
            with tqdm(total=total, desc="Build raw style_scene_split_v2", unit="scenario") as progress:
                for result in pool.imap_unordered(_process_scenario_chunk_v2, worker_payloads):
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
        """Process one raw scenario and write its v2 cache and sidecar."""

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
            style_payload["source_mode"] = np.array("raw_scenario_v2")

            sidecar_payload = split_result.to_numpy_dict()
            sidecar_payload.update(self._metadata_to_numpy(metadata))
            sidecar_payload["style_scene_split_schema_version"] = np.array(SCHEMA_VERSION, dtype=np.int64)
            sidecar_payload["source_mode"] = np.array("raw_scenario_v2")

            self._safe_save_npz(style_cache_path, style_payload)
            self._safe_save_npz(sidecar_path, sidecar_payload)
            return "processed", filename
        except Exception as exc:
            return "failed", str(exc)

    def rebuild_global_index(self) -> Dict[str, object]:
        """Rebuild the v2 global index from memory-eligible samples."""

        sidecar_files = sorted(filename for filename in os.listdir(self.sidecar_dir) if filename.endswith(".npz"))
        index_records: List[Dict[str, object]] = []
        subset_lists: Dict[str, List[str]] = defaultdict(list)
        scene_counter: Counter[str] = Counter()
        style_counter: Counter[str] = Counter()
        subset_counter: Counter[str] = Counter()
        memory_subset_counter: Counter[str] = Counter()
        quality_valid_total = 0
        memory_valid_total = 0
        valid_cache_filenames: List[str] = []

        with tqdm(sidecar_files, desc="Rebuild raw global index v2", unit="file") as progress:
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

                    if bool(record["sample_quality_valid"]):
                        quality_valid_total += 1

                    if bool(record["memory_eligible"]):
                        memory_subset_counter[str(record["subset_id"])] += 1
                        subset_lists[str(record["subset_id"])].append(filename)
                        valid_cache_filenames.append(filename)
                        memory_valid_total += 1
                finally:
                    if sidecar_data is not None:
                        sidecar_data.close()

                if idx % self.log_interval == 0 or idx == len(sidecar_files):
                    progress.set_postfix(indexed=len(index_records), memory_eligible=memory_valid_total)

        self._write_index(index_records)
        self._write_subset_lists(subset_lists)
        self._write_json_list(self.cache_list_path, sidecar_files)
        self._write_json_list(self.valid_cache_list_path, valid_cache_filenames)
        return {
            "indexed_total": len(index_records),
            "indexed_memory_total": memory_valid_total,
            "indexed_quality_valid_total": quality_valid_total,
            "scene_distribution": dict(scene_counter),
            "style_distribution": dict(style_counter),
            "subset_distribution": dict(subset_counter),
            "memory_subset_distribution": dict(memory_subset_counter),
        }

    def _compose_index_record(self, filename: str, sidecar_path: str, sidecar_data) -> Dict[str, object]:
        """Flatten a v2 sidecar into a readable index record."""

        record = super()._compose_index_record(filename, sidecar_path, sidecar_data)
        scene_bucket_name = str(record["scene_bucket_name"])
        axis_names = style_axis_names_for_scene(scene_bucket_name)

        record.update(
            {
                "style_scene_split_schema_version": SCHEMA_VERSION,
                "source_mode": "raw_scenario_v2",
                "sample_quality_valid": bool(self._scalar_int(sidecar_data, "sample_quality_valid", default=0)),
                "memory_eligible": bool(self._scalar_int(sidecar_data, "memory_eligible", default=0)),
                "quality_score": self._scalar_float(sidecar_data, "quality_score", default=0.0),
                "quality_reason": self._scalar_str(sidecar_data, "quality_reason", default=""),
                "quality_vec": self._array_to_float_list(sidecar_data, "quality_vec"),
                "quality_component_names": [
                    "traj_completeness",
                    "neighbor_consistency",
                    "route_metadata",
                    "metric_sanity",
                ],
                "style_performance_vec": self._array_to_float_list(sidecar_data, "style_performance_vec"),
                "style_performance_confidence": self._scalar_float(
                    sidecar_data,
                    "style_performance_confidence",
                    default=0.0,
                ),
                "style_axis_names": list(axis_names),
                "condition_density_level": DENSITY_LEVEL_ID_TO_NAME.get(
                    self._scalar_int(sidecar_data, "condition_density_level", default=0),
                    "unknown",
                ),
                "condition_speed_regime": SPEED_REGIME_ID_TO_NAME.get(
                    self._scalar_int(sidecar_data, "condition_speed_regime", default=0),
                    "unknown",
                ),
                "condition_curvature_level": CURVATURE_LEVEL_ID_TO_NAME.get(
                    self._scalar_int(sidecar_data, "condition_curvature_level", default=0),
                    "unknown",
                ),
            }
        )
        # In raw mode, planner_cache_path refers to the generated style cache.
        record["style_cache_path"] = record["planner_cache_path"]
        return record


__all__ = [
    "RawScenarioStyleSceneSplitBuilderV2",
    "build_raw_scenarios",
]
