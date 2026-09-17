"""Machine-specific path configuration for StyleLoRA.

优先读取被 Git 忽略的本机路径配置，并保留原有环境变量名作为覆盖选项，
便于本地和服务器分别使用各自的数据与输出目录。

任何 StyleLoRA CLI 的默认数据/输出路径都应从这里取，避免硬编码路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from stylelora.config.runtime_paths import get_config_value

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "stylelora"
DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"

# ---- 服务器路径（与历史数据流水线环境变量名保持一致）----
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
    """把仓库根目录与 nuplan-devkit 注入 sys.path（本地/服务器双路径优先）。"""
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
    return Path(value) if value else Path(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


# ---- 数据/缓存根目录（环境变量优先，服务器默认值兜底）----
DEFAULT_RECORD_ROOT = _env_path("NUPLAN_RECORD_ROOT", SERVER_CACHE_RECORD_ROOT)
DEFAULT_CACHE_ROOT = _env_path("NUPLAN_CACHE_ROOT", DEFAULT_RECORD_ROOT / "CACHE")
DEFAULT_DATA_SPLITS_ROOT = _env_path(
    "NUPLAN_DATA_SPLITS_ROOT",
    DEFAULT_RECORD_ROOT / "DATA_SPLITS_CONFIG" / "boston_raw_seed3407",
)
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
DEFAULT_CACHE_TRAIN_VAL_DIR = _env_path(
    "NUPLAN_CACHE_TRAIN_VAL_DIR",
    _DEFAULT_CACHE_TRAIN_VAL_DIR,
)
DEFAULT_CACHE_TRAIN_VAL_LIST_PATH = _env_path(
    "NUPLAN_CACHE_TRAIN_VAL_LIST_PATH",
    _DEFAULT_CACHE_TRAIN_VAL_LIST_PATH,
)
DEFAULT_PLANNER_CACHE_DIR = os.environ.get(
    "NUPLAN_PLANNER_CACHE_DIR",
    str(DEFAULT_CACHE_TRAIN_VAL_DIR),
).strip()
DEFAULT_PLANNER_CACHE_LIST_PATH = os.environ.get(
    "NUPLAN_PLANNER_CACHE_LIST_PATH",
    str(DEFAULT_CACHE_TRAIN_VAL_LIST_PATH),
).strip()

# StyleLoRA source manifest 目录约定：按拆分输出 train/val/test.jsonl
DEFAULT_MANIFESTS_ROOT = _env_path(
    "NUPLAN_STYLE_MANIFESTS_ROOT",
    DEFAULT_RECORD_ROOT / "STYLE_LORA_MANIFESTS",
)
DEFAULT_TRAIN_MANIFEST = _env_path(
    "NUPLAN_STYLE_TRAIN_MANIFEST",
    DEFAULT_MANIFESTS_ROOT / "train.jsonl",
)
DEFAULT_VAL_MANIFEST = _env_path(
    "NUPLAN_STYLE_VAL_MANIFEST",
    DEFAULT_MANIFESTS_ROOT / "val.jsonl",
)
DEFAULT_TEST_MANIFEST = _env_path(
    "NUPLAN_STYLE_TEST_MANIFEST",
    DEFAULT_MANIFESTS_ROOT / "test.jsonl",
)

# StyleLoRA 弱排序偏好产物目录
DEFAULT_PREFERENCE_ROOT = _env_path(
    "NUPLAN_PREFERENCE_ROOT",
    DEFAULT_RECORD_ROOT / "STYLE_LORA_PREFERENCE",
)
DEFAULT_PREFERENCE_MANIFEST = _env_path(
    "NUPLAN_PREFERENCE_MANIFEST",
    DEFAULT_PREFERENCE_ROOT / "weak_preference_train.jsonl",
)
DEFAULT_PREFERENCE_VAL_MANIFEST = _env_path(
    "NUPLAN_PREFERENCE_VAL_MANIFEST",
    DEFAULT_PREFERENCE_ROOT / "weak_preference_val.jsonl",
)
DEFAULT_FEATURE_NPY = _env_path(
    "NUPLAN_PREFERENCE_FEATURE_NPY",
    DEFAULT_PREFERENCE_ROOT / "scene_features_train.npy",
)
DEFAULT_FEATURE_INDEX = _env_path(
    "NUPLAN_PREFERENCE_FEATURE_INDEX",
    DEFAULT_PREFERENCE_ROOT / "scene_features_train_index.jsonl",
)
DEFAULT_FEATURE_VAL_NPY = _env_path(
    "NUPLAN_PREFERENCE_FEATURE_VAL_NPY",
    DEFAULT_PREFERENCE_ROOT / "scene_features_val.npy",
)
DEFAULT_FEATURE_VAL_INDEX = _env_path(
    "NUPLAN_PREFERENCE_FEATURE_VAL_INDEX",
    DEFAULT_PREFERENCE_ROOT / "scene_features_val_index.jsonl",
)
DEFAULT_PREFERENCE_AUDIT = _env_path(
    "NUPLAN_PREFERENCE_AUDIT",
    DEFAULT_PREFERENCE_ROOT / "audit_preference.json",
)

# CSPQ 偏好编码器产物目录
DEFAULT_ENCODER_ROOT = _env_path(
    "NUPLAN_ENCODER_ROOT",
    DEFAULT_RECORD_ROOT / "STYLE_LORA_ENCODER",
)
DEFAULT_ENCODER_CHECKPOINT = _env_path(
    "NUPLAN_ENCODER_CHECKPOINT",
    DEFAULT_ENCODER_ROOT / "preference_encoder.pt",
)
DEFAULT_ENCODER_LAST_CHECKPOINT = _env_path(
    "NUPLAN_ENCODER_LAST_CHECKPOINT",
    DEFAULT_ENCODER_ROOT / "preference_encoder_last.pt",
)
DEFAULT_ENCODER_LATENT_BANK = _env_path(
    "NUPLAN_ENCODER_LATENT_BANK",
    DEFAULT_ENCODER_ROOT / "latent_bank_train.npy",
)
DEFAULT_ENCODER_LATENT_BANK_INDEX = _env_path(
    "NUPLAN_ENCODER_LATENT_BANK_INDEX",
    DEFAULT_ENCODER_ROOT / "latent_bank_train_index.jsonl",
)
DEFAULT_ENCODER_EVAL_REPORT = _env_path(
    "NUPLAN_ENCODER_EVAL_REPORT",
    DEFAULT_ENCODER_ROOT / "evaluate_encoder.json",
)
DEFAULT_ENCODER_VAL_LATENT = _env_path(
    "NUPLAN_ENCODER_VAL_LATENT",
    DEFAULT_ENCODER_ROOT / "latent_bank_val.npy",
)
DEFAULT_ENCODER_VAL_LATENT_INDEX = _env_path(
    "NUPLAN_ENCODER_VAL_LATENT_INDEX",
    DEFAULT_ENCODER_ROOT / "latent_bank_val_index.jsonl",
)

# 默认并行度
DEFAULT_NUM_WORKERS = max(1, _env_int("NUPLAN_DEFAULT_NUM_WORKERS", 54))
