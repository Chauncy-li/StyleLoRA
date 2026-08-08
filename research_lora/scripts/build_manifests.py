from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from research_lora.data.manifest import iter_style_index
from research_lora.evaluation.reports import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build versioned style manifests from a JSON/JSONL style index.")
    parser.add_argument("--style-index", default=None, help="Legacy single index; it must contain split or use --split-override.")
    parser.add_argument("--train-index", default=None, help="Index whose rows are externally assigned to train.")
    parser.add_argument("--val-index", default=None, help="Index whose rows are externally assigned to val.")
    parser.add_argument("--test-index", default=None, help="Index whose rows are externally assigned to test.")
    parser.add_argument("--split-override", choices=("train", "val", "test"), default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()
    inputs = [(args.style_index, args.split_override), (args.train_index, "train"), (args.val_index, "val"), (args.test_index, "test")]
    if not any(path for path, _ in inputs):
        parser.error("Provide --style-index or one or more of --train-index/--val-index/--test-index")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    paths = {split: output / f"{split}.jsonl" for split in ("train", "val", "test")}
    counts, skipped, quality_flags = Counter(), Counter(), Counter()
    metric_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf")})
    # Stream indices directly to disk: source train indexes can exceed available RAM.
    with paths["train"].open("w", encoding="utf-8", newline="\n") as train_file, \
         paths["val"].open("w", encoding="utf-8", newline="\n") as val_file, \
         paths["test"].open("w", encoding="utf-8", newline="\n") as test_file:
        handles = {"train": train_file, "val": val_file, "test": test_file}
        for path, split in inputs:
            if not path:
                continue
            source_skipped = Counter()
            for sample in iter_style_index(path, split_override=split, skipped=source_skipped):
                if not sample.is_usable:
                    skipped["quality_or_split_invalid"] += 1; continue
                if sample.label_confidence < args.min_confidence:
                    skipped["low_label_confidence"] += 1; continue
                handles[sample.split].write(json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                counts[(sample.split, sample.scene_type, sample.style)] += 1
                for flag, value in sample.quality_flags.items():
                    if value: quality_flags[flag] += 1
                for name, value in sample.metrics.items():
                    if sample.metric_mask.get(name, True):
                        stats = metric_stats[name]; numeric = float(value)
                        stats["count"] += 1; stats["sum"] += numeric; stats["min"] = min(stats["min"], numeric); stats["max"] = max(stats["max"], numeric)
            skipped.update(source_skipped)
    audit = {"sample_count": sum(counts.values()), "counts": {"|".join(key): value for key, value in sorted(counts.items())},
             "metric_distribution": {name: {"count": stats["count"], "min": stats["min"], "max": stats["max"], "mean": stats["sum"] / stats["count"]}
                                     for name, stats in metric_stats.items() if stats["count"]},
             "quality_flag_counts": dict(quality_flags), "skipped_rows": dict(skipped),
             "source_indexes": [str(Path(path).resolve()) for path, _ in inputs if path],
             "manifests": {key: str(value) for key, value in paths.items()}}
    write_json(Path(args.output_dir).parent / "reports" / "data_audit.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
