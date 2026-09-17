"""Read machine-local StyleLoRA paths from a small JSON config file.

The local config is intentionally ignored by Git. New paths can be added to
its ``paths`` object and consumed by Python or shell scripts without changing
this resolver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG = CONFIG_DIR / "paths.local.json"
EXAMPLE_CONFIG = CONFIG_DIR / "paths.example.json"
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _config_file() -> Path:
    configured = os.environ.get("STYLELORA_PATHS_FILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path
    return LOCAL_CONFIG if LOCAL_CONFIG.is_file() else EXAMPLE_CONFIG


def _load_paths() -> dict[str, Any]:
    path = _config_file()
    if not path.is_file():
        raise FileNotFoundError(
            f"StyleLoRA path config not found: {path}. Copy "
            f"{EXAMPLE_CONFIG.name} to {LOCAL_CONFIG.name} and edit machine-specific paths."
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    paths = payload.get("paths", payload) if isinstance(payload, dict) else None
    if not isinstance(paths, dict):
        raise ValueError(f"Expected a JSON object (or an object under 'paths') in {path}")
    return paths


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a config value, expanding ``${other_key}`` references recursively."""

    paths = _load_paths()
    if key not in paths:
        if default is not None:
            return default
        raise KeyError(f"Path key {key!r} is not defined in {_config_file()}")

    def resolve(current_key: str, stack: tuple[str, ...]) -> Any:
        if current_key in stack:
            chain = " -> ".join((*stack, current_key))
            raise ValueError(f"Circular path reference in {_config_file()}: {chain}")
        if current_key not in paths:
            raise KeyError(
                f"Path reference {current_key!r} is not defined in {_config_file()}"
            )
        value = paths[current_key]
        if not isinstance(value, str):
            return value

        def replace(match: re.Match[str]) -> str:
            referenced = resolve(match.group(1), (*stack, current_key))
            return str(referenced)

        return _REFERENCE.sub(replace, value)

    return resolve(key, ())


def get_path(key: str, default: str | Path | None = None) -> Path:
    """Get a configured filesystem path as :class:`Path`."""

    value = get_config_value(key, default)
    if value is None:
        raise KeyError(f"Path key {key!r} has no configured value")
    return Path(str(value)).expanduser()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--get", metavar="KEY", help="print one configured value")
    group.add_argument("--show", action="store_true", help="print all resolved paths")
    args = parser.parse_args()

    if args.get:
        print(get_config_value(args.get))
        return
    resolved = {key: get_config_value(key) for key in _load_paths()}
    print(json.dumps(resolved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
