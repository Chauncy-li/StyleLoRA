"""Small explicit YAML default loader used by the research entry points."""

from __future__ import annotations

from pathlib import Path


def _merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def _load_yaml(path: Path):
    import yaml
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = {}
    for default in payload.pop("defaults", []):
        if isinstance(default, str):
            default_path = path.parent / (default if default.endswith(".yaml") else f"{default}.yaml")
            merged = _merge(merged, _load_yaml(default_path))
    return _merge(merged, payload)


def apply_yaml_defaults(parser, args):
    path = getattr(args, "config", None)
    if not path:
        return args
    payload = _load_yaml(Path(path))
    mapping = {
        "rank": ("model", "rank"), "alpha": ("model", "alpha"), "dropout": ("model", "dropout"),
        "steps": ("training", "steps"), "batch_size": ("training", "batch_size"),
        "lr": ("training", "learning_rate"), "neighbor_weight": ("training", "neighbor_weight"),
        "lora_reg_weight": ("training", "lora_reg_weight"), "workers": ("data", "num_workers"),
        "prototype_weight": ("training", "prototype_weight"),
        "prototype_margin_weight": ("training", "prototype_margin_weight"),
        "prototype_margin": ("training", "prototype_margin"),
    }
    for argument, keys in mapping.items():
        if not hasattr(args, argument) or getattr(args, argument) != parser.get_default(argument):
            continue
        value = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            setattr(args, argument, value)
    return args
