"""Cache-pipeline entrypoint for train/val splits or held-out test cache generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from research._runtime import DEFAULT_DATA_SPLITS_ROOT, DEFAULT_RECORD_ROOT, ensure_repo_on_path

ensure_repo_on_path()

DEFAULT_SPLIT_ROOT = Path(DEFAULT_DATA_SPLITS_ROOT)
DEFAULT_CACHE_ROOT = Path(DEFAULT_RECORD_ROOT) / "CACHE"

SPLIT_NAME_CHOICES = ("cache_train_val", "test_simu")
DEFAULT_SPLIT_NAME = "test_simu"


def _option_present(argv: Sequence[str], option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in argv)


def _prepend_default_option(argv: Sequence[str], option_name: str, value: str) -> list[str]:
    if _option_present(argv, option_name):
        return list(argv)
    return [option_name, value, *argv]


def _run_with_argv(main_fn: Callable[[], None], argv: Sequence[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        main_fn()
    finally:
        sys.argv = original_argv


def _forward_cache_train_val(argv: Sequence[str]) -> None:
    from research.data.cache_train_val_split import main as split_main

    print("[CachePipelineEntry] split_name=cache_train_val")
    _run_with_argv(split_main, argv)


def _forward_test_simu(argv: Sequence[str]) -> None:
    if _option_present(argv, "--train_log_names_path") or _option_present(argv, "--val_log_names_path"):
        raise ValueError(
            "split_name=test_simu should use a single --log_names_path. "
            "Do not pass --train_log_names_path/--val_log_names_path."
        )

    from research.data.cache_pipeline import process_cache_split as process_module

    test_log_names_path = DEFAULT_SPLIT_ROOT / "test_simu_log_names.json"
    test_save_path = DEFAULT_CACHE_ROOT / "boston_cache_test_simu"
    test_output_list_path = DEFAULT_CACHE_ROOT / "boston_cache_test_simu_list.json"
    test_manifest_path = DEFAULT_CACHE_ROOT / "boston_cache_test_simu_manifest.json"

    forwarded = list(argv)
    forwarded = _prepend_default_option(forwarded, "--log_names_path", str(test_log_names_path))
    forwarded = _prepend_default_option(forwarded, "--train_log_names_path", "")
    forwarded = _prepend_default_option(forwarded, "--val_log_names_path", "")
    forwarded = _prepend_default_option(forwarded, "--save_path", str(test_save_path))
    forwarded = _prepend_default_option(forwarded, "--output_list_path", str(test_output_list_path))
    forwarded = _prepend_default_option(forwarded, "--manifest_path", str(test_manifest_path))

    print("[CachePipelineEntry] split_name=test_simu")
    print(f"[CachePipelineEntry] test_log_names_path={test_log_names_path}")
    print(f"[CachePipelineEntry] save_path={test_save_path}")
    print(f"[CachePipelineEntry] output_list_path={test_output_list_path}")
    print(f"[CachePipelineEntry] manifest_path={test_manifest_path}")
    _run_with_argv(process_module.main, forwarded)


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Cache pipeline wrapper for train/val split planning or held-out test cache generation"
    )
    parser.add_argument("--split_name", choices=SPLIT_NAME_CHOICES, default=DEFAULT_SPLIT_NAME)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = get_args()
    if args.split_name == "test_simu":
        _forward_test_simu(passthrough)
        return
    _forward_cache_train_val(passthrough)


if __name__ == "__main__":
    main()
