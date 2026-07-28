"""Build simplified style-scene split datasets from planner caches or raw NuPlan scenarios."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research_v1.paths import DEFAULT_NUM_WORKERS, ensure_repo_on_path

ensure_repo_on_path()

from baseline.train.train_utils import openjson, opendata
from research_v1.scene_data.paths import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_LIST_PATH,
    DEFAULT_LOG_NAMES_PATH,
    DEFAULT_MAP_PATH,
    DEFAULT_PLANNER_CACHE_DIR,
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_STYLE_SCENE_SPLIT_V1_DIR,
)
from research_v1.scene_data.legacy.raw_builder import (
    RawScenarioStyleSceneSplitBuilder,
    build_raw_scenarios,
)
from research_v1.scene_data.legacy.schema import (
    SCENE_BUCKET_ID_TO_NAME,
    STYLE_LABEL_ID_TO_NAME,
    STYLE_SCENE_SPLIT_SCHEMA_VERSION,
    TOPOLOGY_BUCKET_ID_TO_NAME,
)
from research_v1.scene_data.legacy.splitter import StyleSceneSplitter

DEFAULT_OUTPUT_DIR = DEFAULT_STYLE_SCENE_SPLIT_V1_DIR
SCHEMA_VERSION = STYLE_SCENE_SPLIT_SCHEMA_VERSION


def _chunk_file_list(file_list: Sequence[str], chunk_size: int) -> List[List[str]]:
    return [list(file_list[start : start + chunk_size]) for start in range(0, len(file_list), chunk_size)]


def _process_planner_cache_chunk(worker_payload: Dict[str, object]) -> Dict[str, object]:
    builder_module = importlib.import_module(str(worker_payload["builder_module"]))
    builder_class = getattr(builder_module, str(worker_payload["builder_class"]))
    builder = builder_class(
        planner_cache_dir=str(worker_payload["planner_cache_dir"]),
        output_dir=str(worker_payload["output_dir"]),
        time_delta=float(worker_payload["time_delta"]),
        skip_existing=bool(worker_payload["skip_existing"]),
        log_interval=int(worker_payload["log_interval"]),
        num_workers=1,
    )

    counters = builder._empty_file_counters()
    failure_preview: List[Dict[str, str]] = []
    filenames = [str(filename) for filename in worker_payload["filenames"]]
    for filename in filenames:
        status, record_or_reason = builder.process_single_file(filename)
        builder._accumulate_file_result(
            counters=counters,
            failure_preview=failure_preview,
            filename=filename,
            status=status,
            record_or_reason=record_or_reason,
        )

    return {
        "total": len(filenames),
        "processed": counters["processed"],
        "skipped": counters["skipped"],
        "missing": counters["missing"],
        "failed": counters["failed"],
        "failure_preview": failure_preview,
    }


class StyleSceneSplitBuilder:
    """Materialize per-sample sidecars and retrieval-ready subset lists."""


    def __init__(
        self,
        planner_cache_dir: str,
        output_dir: str,
        time_delta: float = 0.1,
        skip_existing: bool = False,
        log_interval: int = 500,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:

        self.planner_cache_dir = planner_cache_dir
        self.output_dir = output_dir

        self.sidecar_dir = os.path.join(output_dir, "sidecar")
        self.list_dir = os.path.join(output_dir, "subset_lists")
        self.report_dir = os.path.join(output_dir, "reports")
        self.index_path = os.path.join(output_dir, "split_index.jsonl")
        self.skip_existing = bool(skip_existing)
        self.log_interval = int(log_interval)


        self.splitter = StyleSceneSplitter(time_delta=time_delta)


        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sidecar_dir, exist_ok=True)
        os.makedirs(self.list_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def __init__(
        self,
        planner_cache_dir: str,
        output_dir: str,
        time_delta: float = 0.1,
        skip_existing: bool = False,
        log_interval: int = 500,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self.planner_cache_dir = planner_cache_dir
        self.output_dir = output_dir
        self.time_delta = float(time_delta)
        self.sidecar_dir = os.path.join(output_dir, "sidecar")
        self.list_dir = os.path.join(output_dir, "subset_lists")
        self.report_dir = os.path.join(output_dir, "reports")
        self.index_path = os.path.join(output_dir, "split_index.jsonl")
        self.skip_existing = bool(skip_existing)
        self.log_interval = int(log_interval)
        self.num_workers = max(1, int(num_workers))

        self.splitter = StyleSceneSplitter(time_delta=self.time_delta)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sidecar_dir, exist_ok=True)
        os.makedirs(self.list_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def process_from_json_list(
        self,
        data_list_path: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, object]:
        """Load tasks from a JSON file list and process the selected slice."""
        self._validate_inputs(data_list_path)
        full_file_list = openjson(data_list_path)
        sliced = self._slice_file_list(full_file_list, start_index, end_index)

        return self.process_files(
            sliced,
            data_list_path=data_list_path,
            index_file_list=full_file_list,
        )

    def process_files(
        self,
        file_list: Iterable[str],
        data_list_path: Optional[str] = None,
        index_file_list: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        """Process planner-cache files and return a summary."""
        file_list = list(file_list)
        total = len(file_list)
        start_time = time.time()


        processed = 0
        skipped = 0
        missing = 0
        failed = 0
        failure_preview: List[Dict[str, str]] = []


        with tqdm(file_list, desc="Build style_scene_split", unit="file") as progress:
            for idx, filename in enumerate(progress, start=1):

                status, record_or_reason = self.process_single_file(filename)


                if status == "processed":
                    processed += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "missing":
                    missing += 1
                    failed += 1
                    if len(failure_preview) < 20:
                        failure_preview.append({"filename": filename, "reason": str(record_or_reason)})
                else: # failed
                    failed += 1
                    if len(failure_preview) < 20:
                        failure_preview.append({"filename": filename, "reason": str(record_or_reason)})


                if idx % self.log_interval == 0 or idx == total:
                    elapsed = time.time() - start_time
                    speed = idx / max(elapsed, 1e-6)
                    progress.set_postfix(
                        processed=processed,
                        skipped=skipped,
                        missing=missing,
                        failed=failed,
                        speed=f"{speed:.2f}/s",
                    )


        reindex_stats = self.rebuild_global_index(index_file_list if index_file_list is not None else file_list)

        elapsed_seconds = time.time() - start_time

        summary = {
            "style_scene_split_schema_version": SCHEMA_VERSION,
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "missing": missing,
            "failed": failed,
            "elapsed_seconds": elapsed_seconds,
            "files_per_second": total / max(elapsed_seconds, 1e-6),
            "planner_cache_dir": self.planner_cache_dir,
            "data_list_path": data_list_path,
            "output_dir": self.output_dir,
            "sidecar_dir": self.sidecar_dir,
            "index_path": self.index_path,
            "indexed_total": reindex_stats["indexed_total"],
            "indexed_valid_total": reindex_stats["indexed_valid_total"],
            "index_missing_sidecar": reindex_stats["index_missing_sidecar"],
            "index_failed_load": reindex_stats["index_failed_load"],
            "scene_distribution": reindex_stats["scene_distribution"],
            "style_distribution": reindex_stats["style_distribution"],
            "subset_distribution": reindex_stats["subset_distribution"],
            "valid_subset_distribution": reindex_stats["valid_subset_distribution"],
            "failure_preview": failure_preview,
        }
        self._write_summary(summary)
        return summary

    def process_files(
        self,
        file_list: Iterable[str],
        data_list_path: Optional[str] = None,
        index_file_list: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        """Process planner-cache files and return a summary."""

        file_list = list(file_list)
        total = len(file_list)
        start_time = time.time()

        process_stats = self._process_file_list(
            file_list=file_list,
            progress_desc="Build style_scene_split",
            start_time=start_time,
        )
        processed = int(process_stats["processed"])
        skipped = int(process_stats["skipped"])
        missing = int(process_stats["missing"])
        failed = int(process_stats["failed"])
        failure_preview = list(process_stats["failure_preview"])

        reindex_stats = self.rebuild_global_index(index_file_list if index_file_list is not None else file_list)
        elapsed_seconds = time.time() - start_time
        summary = {
            "style_scene_split_schema_version": SCHEMA_VERSION,
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "missing": missing,
            "failed": failed,
            "elapsed_seconds": elapsed_seconds,
            "files_per_second": total / max(elapsed_seconds, 1e-6),
            "planner_cache_dir": self.planner_cache_dir,
            "data_list_path": data_list_path,
            "output_dir": self.output_dir,
            "sidecar_dir": self.sidecar_dir,
            "index_path": self.index_path,
            "indexed_total": reindex_stats["indexed_total"],
            "indexed_valid_total": reindex_stats["indexed_valid_total"],
            "index_missing_sidecar": reindex_stats["index_missing_sidecar"],
            "index_failed_load": reindex_stats["index_failed_load"],
            "scene_distribution": reindex_stats["scene_distribution"],
            "style_distribution": reindex_stats["style_distribution"],
            "subset_distribution": reindex_stats["subset_distribution"],
            "valid_subset_distribution": reindex_stats["valid_subset_distribution"],
            "failure_preview": failure_preview,
        }
        self._write_summary(summary)
        return summary

    def _process_file_list(
        self,
        file_list: Sequence[str],
        progress_desc: str,
        start_time: float,
    ) -> Dict[str, object]:
        if self.num_workers <= 1 or len(file_list) <= 1:
            return self._process_files_serial(
                file_list=file_list,
                progress_desc=progress_desc,
                start_time=start_time,
            )
        return self._process_files_parallel(
            file_list=file_list,
            progress_desc=progress_desc,
            start_time=start_time,
        )

    def _process_files_serial(
        self,
        file_list: Sequence[str],
        progress_desc: str,
        start_time: float,
    ) -> Dict[str, object]:
        counters = self._empty_file_counters()
        failure_preview: List[Dict[str, str]] = []
        total = len(file_list)

        with tqdm(file_list, desc=progress_desc, unit="file") as progress:
            for idx, filename in enumerate(progress, start=1):
                status, record_or_reason = self.process_single_file(filename)
                self._accumulate_file_result(
                    counters=counters,
                    failure_preview=failure_preview,
                    filename=filename,
                    status=status,
                    record_or_reason=record_or_reason,
                )
                if idx % self.log_interval == 0 or idx == total:
                    self._update_file_progress(
                        progress=progress,
                        completed=idx,
                        total=total,
                        start_time=start_time,
                        counters=counters,
                    )

        return {
            "processed": counters["processed"],
            "skipped": counters["skipped"],
            "missing": counters["missing"],
            "failed": counters["failed"],
            "failure_preview": failure_preview,
        }

    def _process_files_parallel(
        self,
        file_list: Sequence[str],
        progress_desc: str,
        start_time: float,
    ) -> Dict[str, object]:
        total = len(file_list)
        worker_count = min(self.num_workers, total)
        target_chunks = max(worker_count * 256, worker_count)
        chunk_size = max(64, min(256, math.ceil(total / target_chunks)))
        file_chunks = _chunk_file_list(file_list, chunk_size)
        worker_payloads = [self._build_chunk_worker_payload(chunk) for chunk in file_chunks]

        counters = self._empty_file_counters()
        failure_preview: List[Dict[str, str]] = []
        start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        ctx = multiprocessing.get_context(start_method)

        with ctx.Pool(processes=worker_count) as pool:
            with tqdm(total=total, desc=progress_desc, unit="file") as progress:
                for result in pool.imap_unordered(_process_planner_cache_chunk, worker_payloads):
                    self._merge_chunk_result(counters, failure_preview, result)
                    progress.update(int(result["total"]))
                    self._update_file_progress(
                        progress=progress,
                        completed=progress.n,
                        total=total,
                        start_time=start_time,
                        counters=counters,
                        workers=worker_count,
                    )

        return {
            "processed": counters["processed"],
            "skipped": counters["skipped"],
            "missing": counters["missing"],
            "failed": counters["failed"],
            "failure_preview": failure_preview,
        }

    def _build_chunk_worker_payload(self, filenames: Sequence[str]) -> Dict[str, object]:
        return {
            "builder_module": self.__class__.__module__,
            "builder_class": self.__class__.__name__,
            "planner_cache_dir": self.planner_cache_dir,
            "output_dir": self.output_dir,
            "time_delta": self.time_delta,
            "skip_existing": self.skip_existing,
            "log_interval": self.log_interval,
            "filenames": list(filenames),
        }

    def _empty_file_counters(self) -> Dict[str, int]:
        return {"processed": 0, "skipped": 0, "missing": 0, "failed": 0}

    def _accumulate_file_result(
        self,
        *,
        counters: Dict[str, int],
        failure_preview: List[Dict[str, str]],
        filename: str,
        status: str,
        record_or_reason: object,
    ) -> None:
        if status == "processed":
            counters["processed"] += 1
            return
        if status == "skipped":
            counters["skipped"] += 1
            return
        if status == "missing":
            counters["missing"] += 1
        counters["failed"] += 1
        if len(failure_preview) < 20:
            failure_preview.append({"filename": filename, "reason": str(record_or_reason)})

    def _merge_chunk_result(
        self,
        counters: Dict[str, int],
        failure_preview: List[Dict[str, str]],
        result: Dict[str, object],
    ) -> None:
        counters["processed"] += int(result["processed"])
        counters["skipped"] += int(result["skipped"])
        counters["missing"] += int(result["missing"])
        counters["failed"] += int(result["failed"])
        for failure in result.get("failure_preview", []):
            if len(failure_preview) < 20:
                failure_preview.append(failure)

    def _update_file_progress(
        self,
        *,
        progress,
        completed: int,
        total: int,
        start_time: float,
        counters: Dict[str, int],
        workers: Optional[int] = None,
    ) -> None:
        elapsed = time.time() - start_time
        speed = completed / max(elapsed, 1e-6)
        postfix = {
            "processed": counters["processed"],
            "skipped": counters["skipped"],
            "missing": counters["missing"],
            "failed": counters["failed"],
            "speed": f"{speed:.2f}/s",
        }
        if workers is not None:
            postfix["workers"] = workers
        if completed >= total or completed % self.log_interval == 0:
            progress.set_postfix(**postfix)

    def process_single_file(self, filename: str) -> Tuple[str, object]:
        """Process one planner-cache file and write its sidecar payload."""
        planner_cache_path = os.path.join(self.planner_cache_dir, filename)
        sidecar_path = os.path.join(self.sidecar_dir, filename)


        if not os.path.exists(planner_cache_path):
            return "missing", f"missing planner cache: {planner_cache_path}"

        if self.skip_existing and os.path.exists(sidecar_path):
            return "skipped", "exists"

        try:

            cache_data = opendata(planner_cache_path)

            split_result = self.splitter.split(cache_data)

            payload = split_result.to_numpy_dict()

            payload["token"] = np.array(self._scalar_str(cache_data, "token", default=os.path.splitext(filename)[0]))
            payload["map_name"] = np.array(self._scalar_str(cache_data, "map_name", default="unknown"))

            self._safe_save_npz(sidecar_path, payload)
            return "processed", "ok"
        except Exception as exc:
            return "failed", str(exc)
        finally:

            if "cache_data" in locals() and cache_data is not None:
                cache_data.close()

    def rebuild_global_index(self, index_file_list: Iterable[str]) -> Dict[str, object]:
        """Rebuild the global split index and subset lists from sidecars."""

        index_records: List[Dict[str, object]] = []
        subset_lists: Dict[str, List[str]] = defaultdict(list)

        scene_counter: Counter[str] = Counter()
        style_counter: Counter[str] = Counter()
        subset_counter: Counter[str] = Counter()
        valid_subset_counter: Counter[str] = Counter()
        missing_sidecar = 0
        failed_load = 0

        index_file_list = list(index_file_list)
        with tqdm(index_file_list, desc="Rebuild global index", unit="file") as progress:
            for idx, filename in enumerate(progress, start=1):
                sidecar_path = os.path.join(self.sidecar_dir, filename)
                if not os.path.exists(sidecar_path):
                    missing_sidecar += 1
                    continue

                sidecar_data = None
                try:
                    sidecar_data = opendata(sidecar_path)

                    record = self._compose_index_record_from_sidecar(
                        filename=filename,
                        sidecar_path=sidecar_path,
                        sidecar_data=sidecar_data,
                    )
                    index_records.append(record)

                    scene_counter[str(record["scene_bucket_name"])] += 1
                    style_counter[str(record["style_label_name"])] += 1
                    subset_counter[str(record["subset_id"])] += 1

                    if bool(record["split_valid"]):
                        valid_subset_counter[str(record["subset_id"])] += 1
                        subset_lists[str(record["subset_id"])].append(filename)
                except Exception:
                    failed_load += 1
                finally:
                    if "sidecar_data" in locals() and sidecar_data is not None:
                        sidecar_data.close()

                if idx % self.log_interval == 0 or idx == len(index_file_list):
                    progress.set_postfix(
                        indexed=len(index_records),
                        missing_sidecar=missing_sidecar,
                        failed_load=failed_load,
                    )


        self._write_index(index_records)
        self._write_subset_lists(subset_lists)
        return {
            "indexed_total": len(index_records),
            "indexed_valid_total": int(sum(valid_subset_counter.values())),
            "index_missing_sidecar": missing_sidecar,
            "index_failed_load": failed_load,
            "scene_distribution": dict(scene_counter),
            "style_distribution": dict(style_counter),
            "subset_distribution": dict(subset_counter),
            "valid_subset_distribution": dict(valid_subset_counter),
        }

    def _compose_index_record_from_sidecar(
        self,
        filename: str,
        sidecar_path: str,
        sidecar_data,
    ) -> Dict[str, object]:
        """Convert a sidecar NPZ payload into a plain Python index record."""
        planner_cache_path = os.path.join(self.planner_cache_dir, filename)


        token = self._scalar_str(sidecar_data, "token", default=os.path.splitext(filename)[0])
        map_name = self._scalar_str(sidecar_data, "map_name", default="unknown")
        scene_bucket_id = self._scalar_int(sidecar_data, "scene_bucket", default=0)
        style_label_id = self._scalar_int(sidecar_data, "style_label", default=0)
        topology_bucket_id = self._scalar_int(sidecar_data, "topology_bucket", default=0)
        primary_bucket_id = self._scalar_int(sidecar_data, "primary_bucket", default=0)
        secondary_bucket_id = self._scalar_int(sidecar_data, "secondary_bucket", default=0)


        return {
            "style_scene_split_schema_version": SCHEMA_VERSION,
            "sample_id": os.path.splitext(filename)[0],
            "filename": filename,
            "planner_cache_path": planner_cache_path,
            "sidecar_path": sidecar_path,
            "token": token,
            "map_name": map_name,

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


    def _write_index(self, index_records: Sequence[Dict[str, object]]) -> None:
        """Atomically write index records to the JSONL index file."""
        tmp_path = f"{self.index_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            for record in index_records:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.index_path)

    def _write_subset_lists(self, subset_lists: Dict[str, List[str]]) -> None:
        """Write subset lists grouped by subset_id and clean stale files."""
        existing_files = [
            os.path.join(self.list_dir, filename)
            for filename in os.listdir(self.list_dir)
            if filename.endswith(".json")
        ]

        for subset_id, filenames in subset_lists.items():
            save_path = os.path.join(self.list_dir, f"{subset_id}.json")
            tmp_path = f"{save_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as file_obj:
                json.dump(sorted(filenames), file_obj, ensure_ascii=False, indent=2)
            os.replace(tmp_path, save_path)

        active_paths = {os.path.join(self.list_dir, f"{subset_id}.json") for subset_id in subset_lists}
        for stale_path in existing_files:
            if stale_path not in active_paths and os.path.exists(stale_path):
                os.remove(stale_path)

    def _write_summary(self, summary: Dict[str, object]) -> None:
        """Write a timestamped run summary under the report directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.report_dir, f"style_scene_split_summary_{timestamp}.json")
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    def _safe_save_npz(self, save_path: str, payload: Dict[str, np.ndarray]) -> None:
        """Atomically save an NPZ file to avoid partial writes."""
        tmp_path = f"{save_path}.tmp"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(tmp_path, "wb") as file_obj:
            np.savez(file_obj, **payload)
        os.replace(tmp_path, save_path)

    def _slice_file_list(
        self,
        file_list: Sequence[str],
        start_index: int,
        end_index: Optional[int],
    ) -> List[str]:
        """Safely slice a file list for chunked or distributed processing."""
        total = len(file_list)
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        if end_index is not None and end_index < start_index:
            raise ValueError(f"end_index must be >= start_index, got {end_index}")
        start = min(start_index, total)
        end = total if end_index is None else min(end_index, total)
        return list(file_list[start:end])

    def _validate_inputs(self, data_list_path: str) -> None:
        """Validate that required input paths exist before processing."""
        if not os.path.isdir(self.planner_cache_dir):
            raise FileNotFoundError(f"planner_cache_dir not found: {self.planner_cache_dir}")
        if not os.path.exists(data_list_path):
            raise FileNotFoundError(f"data_list_path not found: {data_list_path}")
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be > 0, got {self.log_interval}")





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

    def _optional_float(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        return None if not np.isfinite(value) else value

    def _optional_float_from_npz(self, npz_data, key: str) -> Optional[float]:
        """Read an optional float field and return None for placeholder values."""
        if not (hasattr(npz_data, "files") and key in npz_data.files):
            return None
        value = np.asarray(npz_data[key]).reshape(-1)
        if value.size == 0:
            return None
        value = float(value[0])
        if not np.isfinite(value) or value >= 1e5 or value <= -0.5:
            return None
        return value



def get_args():
    parser = argparse.ArgumentParser(description="Build simplified style-scene split datasets")

    parser.add_argument("--source_mode", type=str, default="raw_scenario", choices=["planner_cache", "raw_scenario"])

    parser.add_argument("--planner_cache_dir", type=str, default=DEFAULT_PLANNER_CACHE_DIR)
    parser.add_argument("--data_list_path", type=str, default=DEFAULT_DATA_LIST_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--time_delta", type=float, default=0.1)
    parser.add_argument("--skip_existing", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--rebuild_index_only", type=int, default=0)

    parser.add_argument("--data_path", type=str, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--map_path", type=str, default=DEFAULT_MAP_PATH)
    parser.add_argument("--map_version", type=str, default="nuplan-maps-v1.0")
    parser.add_argument("--log_names_path", type=str, default=DEFAULT_LOG_NAMES_PATH)
    parser.add_argument("--scenario_tokens_path", type=str, default="")
    parser.add_argument("--scenarios_per_type", type=int, default=None)
    parser.add_argument("--total_scenarios", type=int, default=500000)
    parser.add_argument("--shuffle_scenarios", type=int, default=0)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_num", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()



def main():
    args = get_args()
    end_index = None if args.end_index < 0 else args.end_index


    print("[StyleSceneSplitBuilder] start")
    print(f"[StyleSceneSplitBuilder] source_mode={args.source_mode}")
    print(f"[StyleSceneSplitBuilder] output_dir={args.output_dir}")
    print(f"[StyleSceneSplitBuilder] time_delta={args.time_delta}")
    print(f"[StyleSceneSplitBuilder] skip_existing={bool(args.skip_existing)}")
    print(f"[StyleSceneSplitBuilder] log_interval={args.log_interval}")
    print(f"[StyleSceneSplitBuilder] start_index={args.start_index}")
    print(f"[StyleSceneSplitBuilder] end_index={end_index}")
    print(f"[StyleSceneSplitBuilder] rebuild_index_only={bool(args.rebuild_index_only)}")


    if args.source_mode == "raw_scenario":
        print(f"[StyleSceneSplitBuilder] data_path={args.data_path}")
        print(f"[StyleSceneSplitBuilder] map_path={args.map_path}")
        print(f"[StyleSceneSplitBuilder] map_version={args.map_version}")
        print(f"[StyleSceneSplitBuilder] log_names_path={args.log_names_path}")
        print(f"[StyleSceneSplitBuilder] scenario_tokens_path={args.scenario_tokens_path}")
        print(f"[StyleSceneSplitBuilder] scenarios_per_type={args.scenarios_per_type}")
        print(f"[StyleSceneSplitBuilder] total_scenarios={args.total_scenarios}")
        print(f"[StyleSceneSplitBuilder] shuffle_scenarios={bool(args.shuffle_scenarios)}")
        print(f"[StyleSceneSplitBuilder] num_workers={args.num_workers}")


        builder = RawScenarioStyleSceneSplitBuilder(
            output_dir=args.output_dir,
            time_delta=args.time_delta,
            skip_existing=bool(args.skip_existing),
            log_interval=args.log_interval,
            agent_num=args.agent_num,
            static_objects_num=args.static_objects_num,
            lane_len=args.lane_len,
            lane_num=args.lane_num,
            route_len=args.route_len,
            route_num=args.route_num,
            num_workers=args.num_workers,
        )


        if bool(args.rebuild_index_only):
            stats = builder.rebuild_global_index()
            print(
                "[StyleSceneSplitBuilder] raw reindex finished | "
                f"indexed_total={stats['indexed_total']} indexed_valid_total={stats['indexed_valid_total']}"
            )
        else:

            print("[StyleSceneSplitBuilder] building raw scenario list...")

            scenarios = build_raw_scenarios(
                data_path=args.data_path,
                map_path=args.map_path,
                map_version=args.map_version,
                scenarios_per_type=args.scenarios_per_type,
                total_scenarios=args.total_scenarios,
                shuffle_scenarios=bool(args.shuffle_scenarios),
                log_names_path=args.log_names_path,
                scenario_tokens_path=args.scenario_tokens_path,
            )
            print(f"[StyleSceneSplitBuilder] raw scenario count={len(scenarios)}")

            stats = builder.process_scenarios(
                scenarios=scenarios,
                start_index=args.start_index,
                end_index=end_index,
            )
            print(
                "[StyleSceneSplitBuilder] raw build finished | "
                f"total={stats['total']} processed={stats['processed']} skipped={stats['skipped']} "
                f"failed={stats['failed']} indexed_total={stats['indexed_total']}"
            )


    else:
        print(f"[StyleSceneSplitBuilder] planner_cache_dir={args.planner_cache_dir}")
        print(f"[StyleSceneSplitBuilder] data_list_path={args.data_list_path}")

        builder = StyleSceneSplitBuilder(
            planner_cache_dir=args.planner_cache_dir,
            output_dir=args.output_dir,
            time_delta=args.time_delta,
            skip_existing=bool(args.skip_existing),
            log_interval=args.log_interval,
            num_workers=args.num_workers,
        )

        if bool(args.rebuild_index_only):
            file_list = openjson(args.data_list_path)
            stats = builder.rebuild_global_index(file_list)
            print(
                "[StyleSceneSplitBuilder] reindex finished | "
                f"indexed_total={stats['indexed_total']} indexed_valid_total={stats['indexed_valid_total']} "
                f"missing_sidecar={stats['index_missing_sidecar']} failed_load={stats['index_failed_load']}"
            )
        else:

            stats = builder.process_from_json_list(
                data_list_path=args.data_list_path,
                start_index=args.start_index,
                end_index=end_index,
            )
            print(
                "[StyleSceneSplitBuilder] finished | "
                f"total={stats['total']} processed={stats['processed']} skipped={stats['skipped']} "
                f"missing={stats['missing']} failed={stats['failed']} "
                f"indexed_total={stats['indexed_total']}"
            )


if __name__ == "__main__":
    main()
