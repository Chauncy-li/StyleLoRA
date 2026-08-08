from __future__ import annotations

import argparse
import json
from collections import defaultdict

from research_lora.evaluation.reports import write_json


def _monotonic_by_magnitude(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("target_style") in {"aggressive", "conservative"}:
            grouped[(row["scene"], row["target_style"])].append((abs(float(row["rho"])), float(row["cluster_hit_rate"])))
    if not grouped: return False
    return all(all(next_score >= score - 1e-6 for (_, score), (_, next_score) in zip(sorted(points), sorted(points)[1:]))
               for points in grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate denoising, open-loop and closed-loop LoRA evidence.")
    parser.add_argument("inputs", nargs="+"); parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-hit-threshold", type=float, default=0.5)
    args = parser.parse_args(); denoising, scene_rows, closed_loop = [], [], []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle: payload = json.load(handle)
        if "scene_style_summary" in payload: scene_rows.extend(payload["scene_style_summary"])
        elif "correct_beats_opposite" in payload: denoising.append(payload)
        else: closed_loop.append(payload)
    target_reached = {
        style: all(any(row["target_style"] == style and row["scene"] == scene and row["cluster_hit_rate"] >= args.cluster_hit_threshold
                       for row in scene_rows) for scene in ("straight_free_drive", "straight_car_follow"))
        for style in ("aggressive", "conservative")
    }
    normal_identity = bool(denoising) and all(float(row.get("identity_error", float("inf"))) == 0.0 for row in denoising)
    correct_direction = bool(denoising) and all(bool(row.get("correct_beats_opposite", False)) for row in denoising)
    monotonic = _monotonic_by_magnitude(scene_rows)
    closed_loop_pass = bool(closed_loop) and all(bool(payload.get("closed_loop_pass", False)) for payload in closed_loop)
    summary = {"aggressive_target_cluster_reached_both_scenes": target_reached["aggressive"],
               "conservative_target_cluster_reached_both_scenes": target_reached["conservative"],
               "normal_preserves_baseline": normal_identity, "correct_style_beats_opposite_denoising": correct_direction,
               "rho_style_response_monotonic": monotonic, "closed_loop_reports": len(closed_loop), "closed_loop_pass": closed_loop_pass,
               "supports_hyperlora_next_stage": all(target_reached.values()) and normal_identity and correct_direction and monotonic and closed_loop_pass,
               "scene_style_summary": scene_rows}
    write_json(args.output, summary); print(summary)


if __name__ == "__main__": main()
