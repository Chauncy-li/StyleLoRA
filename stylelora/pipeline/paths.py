"""Shared runtime helpers for research-side scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from stylelora.config.runtime_paths import get_config_value

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "stylelora"
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
SERVER_PROGRAM_ROOT = Path(
    os.environ.get("NUPLAN_SERVER_PROGRAM_ROOT", str(get_config_value("program_root", "/home/lisw/programs")))
)
_SERVER_REPO_DEFAULT = (
    SERVER_PROGRAM_ROOT / "Nuplan-Diffusion-Baseline"
    if os.environ.get("NUPLAN_SERVER_PROGRAM_ROOT", "").strip()
    else get_config_value("repo_root", SERVER_PROGRAM_ROOT / "Nuplan-Diffusion-Baseline")
)
SERVER_REPO_ROOT = Path(
    os.environ.get(
        "NUPLAN_SERVER_REPO_ROOT",
        str(_SERVER_REPO_DEFAULT),
    )
)
_SERVER_DEVKIT_DEFAULT = (
    SERVER_REPO_ROOT / "nuplan-devkit"
    if any(
        os.environ.get(name, "").strip()
        for name in ("NUPLAN_SERVER_PROGRAM_ROOT", "NUPLAN_SERVER_REPO_ROOT")
    )
    else get_config_value("devkit_root", SERVER_REPO_ROOT / "nuplan-devkit")
)
SERVER_DEVKIT_ROOT = Path(
    os.environ.get(
        "NUPLAN_SERVER_DEVKIT_ROOT",
        str(_SERVER_DEVKIT_DEFAULT),
    )
)
SERVER_STYLE_RECORD_ROOT = Path(
    os.environ.get(
        "NUPLAN_SERVER_STYLE_RECORD_ROOT",
        str(get_config_value("style_record_root", "/mnt/mydata/lishangwen/NuplanBaselinesRecord")),
    )
)
SERVER_CACHE_RECORD_ROOT = Path(
    os.environ.get(
        "NUPLAN_SERVER_CACHE_RECORD_ROOT",
        str(get_config_value("record_root", "/mnt/mydata/lishangwen/Nuplan-Baseline-Record")),
    )
)
SERVER_DATA_ROOT = Path(
    os.environ.get(
        "NUPLAN_SERVER_DATA_ROOT",
        str(get_config_value("data_root", "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston")),
    )
)
SERVER_MAP_ROOT = Path(
    os.environ.get(
        "NUPLAN_SERVER_MAP_ROOT",
        str(get_config_value("maps_root", "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps")),
    )
)
SERVER_LOG_NAMES_PATH = Path(
    os.environ.get(
        "NUPLAN_SERVER_LOG_NAMES_PATH",
        str(
            SERVER_STYLE_RECORD_ROOT / "nuplan_scenarios_boston.json"
            if os.environ.get("NUPLAN_SERVER_STYLE_RECORD_ROOT", "").strip()
            else get_config_value(
                "log_names_path",
                SERVER_STYLE_RECORD_ROOT / "nuplan_scenarios_boston.json",
            )
        ),
    )
)


def ensure_repo_on_path() -> None:
    """Expose the repo root and nuplan-devkit to local research scripts."""

    preferred_paths = [REPO_ROOT, DEVKIT_ROOT]
    fallback_paths = [SERVER_REPO_ROOT, SERVER_DEVKIT_ROOT]

    for path in reversed(preferred_paths):
        path_str = str(path)
        if not path.exists():
            continue
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    for path in fallback_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)


def _env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value)
    return Path(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


DEFAULT_RECORD_ROOT = _env_path("NUPLAN_RECORD_ROOT", SERVER_CACHE_RECORD_ROOT)
DEFAULT_CACHE_ROOT = _env_path("NUPLAN_CACHE_ROOT", DEFAULT_RECORD_ROOT / "CACHE")
DEFAULT_DATA_SPLITS_ROOT = _env_path(
    "NUPLAN_DATA_SPLITS_ROOT",
    DEFAULT_RECORD_ROOT / "DATA_SPLITS_CONFIG" / "boston_raw_seed3407",
)
DEFAULT_SCENARIO_FILTER_ROOT = _env_path(
    "NUPLAN_SCENARIO_FILTER_ROOT",
    RESEARCH_ROOT / "config" / "scenario_filter",
)

DEFAULT_NUPLAN_DATA_PATH = os.environ.get("NUPLAN_DATA_PATH", str(SERVER_DATA_ROOT)).strip()
DEFAULT_NUPLAN_MAP_PATH = os.environ.get("NUPLAN_MAP_PATH", str(SERVER_MAP_ROOT)).strip()
DEFAULT_NUPLAN_LOG_NAMES_PATH = os.environ.get("NUPLAN_LOG_NAMES_PATH", str(SERVER_LOG_NAMES_PATH)).strip()
_CACHE_ROOT_ENV_OVERRIDES = any(
    os.environ.get(name, "").strip()
    for name in (
        "NUPLAN_SERVER_CACHE_RECORD_ROOT",
        "NUPLAN_RECORD_ROOT",
        "NUPLAN_CACHE_ROOT",
    )
)
_DEFAULT_CACHE_TRAIN_VAL_DIR = (
    DEFAULT_CACHE_ROOT / "boston_cache_train_val"
    if _CACHE_ROOT_ENV_OVERRIDES
    else get_config_value("cache_root", DEFAULT_CACHE_ROOT / "boston_cache_train_val")
)
_DEFAULT_CACHE_TRAIN_VAL_LIST_PATH = (
    DEFAULT_CACHE_ROOT / "boston_cache_train_val_list.json"
    if _CACHE_ROOT_ENV_OVERRIDES
    else get_config_value(
        "cache_train_val_list_path",
        DEFAULT_CACHE_ROOT / "boston_cache_train_val_list.json",
    )
)
_DEFAULT_CACHE_TRAIN_VAL_MANIFEST_PATH = (
    DEFAULT_CACHE_ROOT / "boston_cache_train_val_manifest.json"
    if _CACHE_ROOT_ENV_OVERRIDES
    else get_config_value(
        "cache_train_val_manifest_path",
        DEFAULT_CACHE_ROOT / "boston_cache_train_val_manifest.json",
    )
)
DEFAULT_CACHE_TRAIN_VAL_DIR = _env_path(
    "NUPLAN_CACHE_TRAIN_VAL_DIR",
    _DEFAULT_CACHE_TRAIN_VAL_DIR,
)
DEFAULT_CACHE_TRAIN_VAL_LIST_PATH = _env_path(
    "NUPLAN_CACHE_TRAIN_VAL_LIST_PATH",
    _DEFAULT_CACHE_TRAIN_VAL_LIST_PATH,
)
DEFAULT_CACHE_TRAIN_VAL_MANIFEST_PATH = _env_path(
    "NUPLAN_CACHE_TRAIN_VAL_MANIFEST_PATH",
    _DEFAULT_CACHE_TRAIN_VAL_MANIFEST_PATH,
)
DEFAULT_PLANNER_CACHE_DIR = os.environ.get(
    "NUPLAN_PLANNER_CACHE_DIR",
    str(DEFAULT_CACHE_TRAIN_VAL_DIR),
).strip()
DEFAULT_PLANNER_CACHE_LIST_PATH = os.environ.get(
    "NUPLAN_PLANNER_CACHE_LIST_PATH",
    str(DEFAULT_CACHE_TRAIN_VAL_LIST_PATH),
).strip()
DEFAULT_NUM_WORKERS = max(1, _env_int("NUPLAN_DEFAULT_NUM_WORKERS", 54))
