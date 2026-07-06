"""Build the primary v2 straight-scene split outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _repo_root_str = str(_REPO_ROOT)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from research._runtime import DEFAULT_NUM_WORKERS, ensure_repo_on_path

ensure_repo_on_path()

from baseline.train.train_utils import openjson, opendata
from research.style_scene_split.build_style_scene_split import StyleSceneSplitBuilder
from research.style_scene_split.defaults import (
    DEFAULT_DATA_LIST_PATH,
    DEFAULT_LOG_NAMES_PATH,
    DEFAULT_MAP_PATH,
    DEFAULT_PLANNER_CACHE_DIR,
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_STYLE_SCENE_SPLIT_V2_DIR,
)
from research.style_scene_split.raw_scenario_builder_v2 import (
    RawScenarioStyleSceneSplitBuilderV2,
    build_raw_scenarios,
)
from research.style_scene_split.schema_v2 import (
    CURVATURE_LEVEL_ID_TO_NAME,
    DENSITY_LEVEL_ID_TO_NAME,
    SCENE_BUCKET_ID_TO_NAME,
    SPEED_REGIME_ID_TO_NAME,
    STYLE_LABEL_ID_TO_NAME,
    STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION,
    TOPOLOGY_BUCKET_ID_TO_NAME,
    style_axis_names_for_scene,
)
from research.style_scene_split.splitter_v2 import StyleSceneSplitterV2


DEFAULT_OUTPUT_DIR = DEFAULT_STYLE_SCENE_SPLIT_V2_DIR
SCHEMA_VERSION = STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION


class StyleSceneSplitBuilderV2(StyleSceneSplitBuilder):
    """Planner-cache builder for the primary v2 straight-scene split."""

    def __init__(
        self,
        planner_cache_dir: str,
        output_dir: str,
        time_delta: float = 0.1,
        skip_existing: bool = False,
        log_interval: int = 500,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        super().__init__(
            planner_cache_dir=planner_cache_dir,
            output_dir=output_dir,
            time_delta=time_delta,
            skip_existing=skip_existing,
            log_interval=log_interval,
            num_workers=num_workers,
        )
        self.splitter = StyleSceneSplitterV2(time_delta=self.time_delta)

    def process_from_json_list(
        self,
        data_list_path: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, object]:
        """Load a planner-cache file list and process the requested slice."""

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
        """Process planner-cache files and write the v2 summaries."""

        file_list = list(file_list)
        total = len(file_list)
        start_time = time.time()

        process_stats = self._process_file_list(
            file_list=file_list,
            progress_desc="Build style_scene_split_v2",
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
            "source_mode": "planner_cache_v2",
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
            "indexed_memory_total": reindex_stats["indexed_memory_total"],
            "indexed_quality_valid_total": reindex_stats["indexed_quality_valid_total"],
            "index_missing_sidecar": reindex_stats["index_missing_sidecar"],
            "index_failed_load": reindex_stats["index_failed_load"],
            "scene_distribution": reindex_stats["scene_distribution"],
            "style_distribution": reindex_stats["style_distribution"],
            "subset_distribution": reindex_stats["subset_distribution"],
            "memory_subset_distribution": reindex_stats["memory_subset_distribution"],
            "failure_preview": failure_preview,
        }
        self._write_summary(summary)
        return summary

    def rebuild_global_index(self, index_file_list: Iterable[str]) -> Dict[str, object]:
        """Rebuild the v2 global index and memory-eligible subset lists."""

        index_records: List[Dict[str, object]] = []
        subset_lists: Dict[str, List[str]] = defaultdict(list)
        scene_counter: Counter[str] = Counter()
        style_counter: Counter[str] = Counter()
        subset_counter: Counter[str] = Counter()
        memory_subset_counter: Counter[str] = Counter()
        quality_valid_total = 0
        memory_valid_total = 0
        missing_sidecar = 0
        failed_load = 0
        index_file_list = list(index_file_list)
        with tqdm(index_file_list, desc="Rebuild global index v2", unit="file") as progress:
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

                    if bool(record["sample_quality_valid"]):
                        quality_valid_total += 1
                    if bool(record["memory_eligible"]):
                        memory_subset_counter[str(record["subset_id"])] += 1
                        subset_lists[str(record["subset_id"])].append(filename)
                        memory_valid_total += 1
                except Exception:
                    failed_load += 1
                finally:
                    if "sidecar_data" in locals() and sidecar_data is not None:
                        sidecar_data.close()

                if idx % self.log_interval == 0 or idx == len(index_file_list):
                    progress.set_postfix(
                        indexed=len(index_records),
                        quality_valid=quality_valid_total,
                        memory_eligible=memory_valid_total,
                        missing_sidecar=missing_sidecar,
                        failed_load=failed_load,
                    )

        self._write_index(index_records)
        self._write_subset_lists(subset_lists)
        return {
            "indexed_total": len(index_records),
            "indexed_memory_total": memory_valid_total,
            "indexed_quality_valid_total": quality_valid_total,
            "index_missing_sidecar": missing_sidecar,
            "index_failed_load": failed_load,
            "scene_distribution": dict(scene_counter),
            "style_distribution": dict(style_counter),
            "subset_distribution": dict(subset_counter),
            "memory_subset_distribution": dict(memory_subset_counter),
        }

    def _compose_index_record_from_sidecar(
        self,
        filename: str,
        sidecar_path: str,
        sidecar_data,
    ) -> Dict[str, object]:
        """Convert a v2 sidecar file into an analysis-friendly index record."""

        record = super()._compose_index_record_from_sidecar(filename, sidecar_path, sidecar_data)
        scene_bucket_name = str(record["scene_bucket_name"])
        axis_names = style_axis_names_for_scene(scene_bucket_name)

        record.update(
            {
                "style_scene_split_schema_version": SCHEMA_VERSION,
                "source_mode": "planner_cache_v2",
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
        return record


def get_args():
    """Parse CLI arguments for the primary v2 builder."""

    parser = argparse.ArgumentParser(description="Build enhanced style-scene split datasets (v2)")
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
    parser.add_argument("--total_scenarios", type=int, default=1000000)
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
    """Run the primary v2 builder in planner-cache or raw-scenario mode."""

    args = get_args()
    end_index = None if args.end_index < 0 else args.end_index

    print("[StyleSceneSplitBuilderV2] start")
    print(f"[StyleSceneSplitBuilderV2] source_mode={args.source_mode}")
    print(f"[StyleSceneSplitBuilderV2] output_dir={args.output_dir}")
    print(f"[StyleSceneSplitBuilderV2] time_delta={args.time_delta}")
    print(f"[StyleSceneSplitBuilderV2] skip_existing={bool(args.skip_existing)}")
    print(f"[StyleSceneSplitBuilderV2] log_interval={args.log_interval}")
    print(f"[StyleSceneSplitBuilderV2] start_index={args.start_index}")
    print(f"[StyleSceneSplitBuilderV2] end_index={end_index}")
    print(f"[StyleSceneSplitBuilderV2] rebuild_index_only={bool(args.rebuild_index_only)}")

    if args.source_mode == "raw_scenario":
        builder = RawScenarioStyleSceneSplitBuilderV2(
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
                "[StyleSceneSplitBuilderV2] raw reindex finished | "
                f"indexed_total={stats['indexed_total']} indexed_memory_total={stats['indexed_memory_total']}"
            )
        else:
            print("[StyleSceneSplitBuilderV2] building raw scenario list...")
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
            print(f"[StyleSceneSplitBuilderV2] raw scenario count={len(scenarios)}")
            stats = builder.process_scenarios(
                scenarios=scenarios,
                start_index=args.start_index,
                end_index=end_index,
            )
            print(
                "[StyleSceneSplitBuilderV2] raw build finished | "
                f"total={stats['total']} processed={stats['processed']} skipped={stats['skipped']} "
                f"failed={stats['failed']} indexed_total={stats['indexed_total']}"
            )
    else:
        builder = StyleSceneSplitBuilderV2(
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
                "[StyleSceneSplitBuilderV2] reindex finished | "
                f"indexed_total={stats['indexed_total']} indexed_memory_total={stats['indexed_memory_total']} "
                f"missing_sidecar={stats['index_missing_sidecar']} failed_load={stats['index_failed_load']}"
            )
        else:
            stats = builder.process_from_json_list(
                data_list_path=args.data_list_path,
                start_index=args.start_index,
                end_index=end_index,
            )
            print(
                "[StyleSceneSplitBuilderV2] finished | "
                f"total={stats['total']} processed={stats['processed']} skipped={stats['skipped']} "
                f"missing={stats['missing']} failed={stats['failed']} indexed_total={stats['indexed_total']}"
            )


if __name__ == "__main__":
    main()
