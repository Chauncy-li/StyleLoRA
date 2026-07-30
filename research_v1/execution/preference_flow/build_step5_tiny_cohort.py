"""Build a fixed, label-balanced Step-5 smoke cohort from a V6 JSONL index.

The exported cohort deliberately keeps only causal runtime metadata and a
training-only scalar endpoint.  Future-derived axis targets are used here to
choose that endpoint, then discarded; the Step-5 adapter never receives them
as an inference condition.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
_SCENES = ("straight_free_drive", "straight_car_follow")
_SCENE_INDEX = {scene: index for index, scene in enumerate(_SCENES)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the fixed 8--16 sample Step-5 free-drive/car-follow cohort."
    )
    parser.add_argument("--conditioning-index", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--per-scene", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _records(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"JSONL row {line_number} is not an object")
            yield payload


def _nearest_grid(value: float) -> float:
    return min(_GRID, key=lambda candidate: abs(candidate - float(value)))


def _candidate(record: Mapping[str, Any], cache_root: Path) -> Dict[str, Any] | None:
    scene = str(record.get("causal_scene_bucket", record.get("scene_bucket", "none")))
    if scene not in _SCENES:
        return None
    filename = str(record.get("filename", "")).strip()
    if not filename:
        return None
    path = Path(filename)
    cache_path = path.resolve() if path.is_absolute() else (cache_root / path).resolve()
    try:
        cache_path.relative_to(cache_root)
    except ValueError as error:
        raise ValueError(f"conditioning filename escapes --cache-root: {filename}") from error
    if not cache_path.is_file() or cache_path.suffix.lower() != ".npz":
        return None

    style = [float(value) for value in record.get("style_value_condition", [])]
    if len(style) != 12:
        return None
    targets, causal_mask = style[:3], style[3:6]
    active_targets = [targets[index] for index, active in enumerate(causal_mask) if active > 0.5]
    if not active_targets:
        return None
    # All V6 percentile axes use the same conservative(-)/aggressive(+)
    # orientation.  This is a training endpoint only, never a model input.
    training_rho = _nearest_grid(2.0 * sum(active_targets) / len(active_targets) - 1.0)
    one_hot = style[6:9]
    gate = style[9:12]
    expected_index = _SCENE_INDEX[scene]
    if one_hot[expected_index] <= 0.5:
        return None
    task_features = one_hot + gate + [1.0 if value > 0.5 else 0.0 for value in causal_mask]
    return {
        "filename": str(cache_path.relative_to(cache_root)),
        "causal_scene_bucket": scene,
        "training_rho": float(training_rho),
        "task_features": [float(value) for value in task_features],
        "rho_source": "future_axis_targets_for_training_only",
    }


def _select(candidates: List[Dict[str, Any]], per_scene: int, rng: random.Random) -> List[Dict[str, Any]]:
    buckets: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[float(candidate["training_rho"])].append(candidate)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: List[Dict[str, Any]] = []
    for rho in _GRID:
        if buckets[rho] and len(selected) < per_scene:
            selected.append(buckets[rho].pop())
    remaining = [item for bucket in buckets.values() for item in bucket]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, per_scene - len(selected))])
    if len(selected) != per_scene:
        raise RuntimeError(f"needed {per_scene} samples, found only {len(selected)}")
    rhos = [float(item["training_rho"]) for item in selected]
    if not any(value < 0.0 for value in rhos) or not any(value > 0.0 for value in rhos):
        raise RuntimeError(
            "selected cohort lacks either negative or positive training rho; "
            "use a conditioning index with both conservative and aggressive rows"
        )
    return selected


def run(args: argparse.Namespace) -> Path:
    index_path = Path(args.conditioning_index).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_path = Path(args.output_file).expanduser()
    if not index_path.is_file():
        raise FileNotFoundError(f"--conditioning-index does not exist: {index_path}")
    if not cache_root.is_dir():
        raise NotADirectoryError(f"--cache-root is not a directory: {cache_root}")
    if int(args.per_scene) <= 0:
        raise ValueError("--per-scene must be positive")
    total = 2 * int(args.per_scene)
    if total < 8 or total > 16:
        raise ValueError("Step-5 cohort must contain 8--16 total samples")
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"refusing to overwrite cohort: {output_path}")

    grouped: Dict[str, List[Dict[str, Any]]] = {scene: [] for scene in _SCENES}
    for record in _records(index_path):
        candidate = _candidate(record, cache_root)
        if candidate is not None:
            grouped[str(candidate["causal_scene_bucket"])].append(candidate)
    rng = random.Random(int(args.seed))
    samples = []
    for scene in _SCENES:
        samples.extend(_select(grouped[scene], int(args.per_scene), rng))
    rng.shuffle(samples)
    report = {
        "schema_version": "preference_flow_step5_tiny_cohort_v1",
        "conditioning_index": str(index_path),
        "cache_root": str(cache_root),
        "seed": int(args.seed),
        "sample_count": len(samples),
        "scene_counts": dict(Counter(item["causal_scene_bucket"] for item in samples)),
        "rho_counts": {str(rho): sum(item["training_rho"] == rho for item in samples) for rho in _GRID},
        "inference_condition": "[causal_scene_one_hot, causal_scene_gate, signed_causal_axis_mask]",
        "future_axis_targets_in_inference_condition": False,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def main() -> None:
    path = run(_parser().parse_args())
    print(f"Step-5 tiny cohort written: {path}")


if __name__ == "__main__":
    main()
