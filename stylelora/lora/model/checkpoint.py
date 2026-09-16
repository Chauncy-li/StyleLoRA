"""Small, self-describing adapter-only checkpoints with baseline compatibility checks."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from stylelora.lora.model.style_lora_planner import StyleLoRAPlanner


@lru_cache(maxsize=16)
def _sha256_file_cached(resolved_path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash an immutable experiment artifact once per process/stat signature."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_file_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def save_adapter_checkpoint(path: str | Path, planner: StyleLoRAPlanner, *, style: str,
                            baseline_checkpoint: str | Path, manifest_hash: str,
                            normalization_file: str | Path, training_config: Mapping[str, Any],
                            best_validation: Mapping[str, Any] | None = None,
                            baseline_sha256: str | None = None,
                            normalization_sha256: str | None = None) -> None:
    """保存仅含一个风格分支 LoRA 权重的 checkpoint。

    周期保存时可传入预先计算的两个哈希，避免对同一基线大权重反复做磁盘扫描；
    不传时保持原有行为并在此处计算哈希。
    """
    style = {"aggr": "aggressive", "cons": "conservative"}.get(style, style)
    if style not in {"aggressive", "conservative"}:
        raise ValueError("Only aggressive or conservative adapter branches can be checkpointed")
    baseline_path, normalization_path = Path(baseline_checkpoint), Path(normalization_file)
    adapter_layers = list(iter_adapter_layer_metadata(planner))
    payload = {
        "format": "research_lora.adapter.v2", "adapter_state": planner.adapter_state_dict(style),
        "metadata": {"style": style, "injection_layers": list(planner.report.layers), "adapter_layers": adapter_layers,
                     "baseline_checkpoint": str(baseline_path), "baseline_sha256": baseline_sha256 or sha256_file(baseline_path),
                     "manifest_hash": manifest_hash, "normalization_file": str(normalization_path),
                     "normalization_sha256": normalization_sha256 or sha256_file(normalization_path),
                     "training_config": dict(training_config), "best_validation": dict(best_validation or {})},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def iter_adapter_layer_metadata(planner: StyleLoRAPlanner):
    for name, module in planner.baseline.named_modules():
        if hasattr(module, "aggressive") and hasattr(module.aggressive, "rank"):
            yield {"name": name, "rank": module.aggressive.rank, "alpha": module.aggressive.alpha,
                   "dropout": module.aggressive.dropout.p}


def load_adapter_checkpoint(path: str | Path, planner: StyleLoRAPlanner, *, baseline_checkpoint: str | Path,
                            normalization_file: str | Path | None = None, strict_hash: bool = True) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "research_lora.adapter.v2":
        raise ValueError(f"Unsupported adapter checkpoint: {path}")
    metadata = payload["metadata"]
    if strict_hash and metadata["baseline_sha256"] != sha256_file(baseline_checkpoint):
        raise RuntimeError("Baseline checkpoint hash mismatch; refusing to apply LoRA to a different base model")
    if normalization_file is not None and strict_hash and metadata["normalization_sha256"] != sha256_file(normalization_file):
        raise RuntimeError("Normalization file hash mismatch")
    if tuple(metadata["injection_layers"]) != planner.report.layers:
        raise RuntimeError("Injection-layer mismatch between adapter checkpoint and current wrapper")
    planner.load_adapter_state_dict(payload["adapter_state"], metadata["style"])
    return metadata

