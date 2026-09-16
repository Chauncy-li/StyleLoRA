"""Select a fixed, distribution-aware token set for every closed-loop rho run."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from stylelora.closed_loop_constants import ROUTE_FAILURE_TOKENS


SCENES = ("straight_free_drive", "straight_car_follow")


def _proportional_quotas(candidate_counts: dict[str, int], total: int) -> dict[str, int]:
    """按候选分布分配固定总数，并保证总和严格等于 total。"""
    available = sum(candidate_counts.values())
    if total <= 0:
        raise ValueError("total 必须为正整数")
    if available < total:
        raise ValueError(f"有效候选仅有 {available} 条，少于 --total={total}")

    raw = {scene: total * candidate_counts[scene] / available for scene in SCENES}
    quotas = {scene: min(candidate_counts[scene], int(raw[scene])) for scene in SCENES}
    remaining = total - sum(quotas.values())
    order = sorted(
        SCENES,
        key=lambda scene: (raw[scene] - int(raw[scene]), candidate_counts[scene]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for scene in order:
            if quotas[scene] < candidate_counts[scene]:
                quotas[scene] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("无法为固定总数分配场景配额")
    return quotas


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc


def _load_allowed_tokens(path: str | None) -> set[str] | None:
    """读取场景预筛查输出；未提供时不限制候选集合。"""
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("scenario_tokens", payload.get("tokens"))
    if not isinstance(payload, list) or not all(isinstance(item, str) and item for item in payload):
        raise ValueError("--allowed-tokens-file 必须是 JSON 字符串列表或包含 tokens 的对象")
    if len(payload) != len(set(payload)):
        raise ValueError("--allowed-tokens-file 包含重复 token")
    return set(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="为闭环 rho 扫描生成固定、分布感知的 scenario token 列表")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="weak_preference_val.jsonl（兼容旧用法）")
    source.add_argument("--split-index", help="test scene split 的 split_index.jsonl（闭环正式实验推荐）")
    parser.add_argument("--output", required=True, help="输出 JSON token 列表")
    parser.add_argument("--per-scene", type=int, default=25)
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="按候选分布确定性抽取固定总数；提供后覆盖 --per-scene，论文扩展实验使用200",
    )
    parser.add_argument(
        "--exclude-known-route-failures",
        action="store_true",
        help="在抽样前排除已确认的 NuPlan 地图路由失败 token",
    )
    parser.add_argument(
        "--allowed-tokens-file",
        default=None,
        help="仅从场景预筛查确认可执行的 JSON token 列表中抽样",
    )
    parser.add_argument("--min-rank-confidence", type=float, default=0.6)
    parser.add_argument("--min-scene-confidence", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.per_scene <= 0:
        parser.error("--per-scene 必须为正数")
    if args.total is not None and args.total <= 0:
        parser.error("--total 必须为正数")
    if not 0 <= args.min_rank_confidence <= 1:
        parser.error("--min-rank-confidence 必须位于 [0,1]")
    if not 0 <= args.min_scene_confidence <= 1:
        parser.error("--min-scene-confidence 必须位于 [0,1]")

    candidates: dict[str, list[dict]] = defaultdict(list)
    rejected = Counter()
    seen_tokens: set[str] = set()
    allowed_tokens = _load_allowed_tokens(args.allowed_tokens_file)
    source_path = Path(args.manifest or args.split_index)
    source_mode = "preference_manifest" if args.manifest else "scene_split_index"
    for raw_row in _iter_jsonl(source_path):
        row = dict(raw_row)
        scene = str(row.get("scene_type", row.get("scene_bucket", row.get("scene_bucket_name", ""))))
        token = str(row.get("token", "") or "")
        if source_mode == "preference_manifest":
            valid = any(bool(value) for value in row.get("axis_valid", []))
            confidence = float(row.get("rank_confidence", 0.0))
            confidence_threshold = args.min_rank_confidence
            order_value = float(row.get("preference_rank", 0.5))
        else:
            valid = bool(row.get("split_valid", False))
            confidence = float(row.get("scene_confidence", 0.0))
            confidence_threshold = args.min_scene_confidence
            # 覆盖不同驾驶上下文：free 按速度、follow 按最小时距排序。
            if scene == "straight_free_drive":
                order_value = float(row.get("ego_mean_speed", 0.0))
            else:
                value = row.get("following_min_thw")
                order_value = float(value) if value is not None else float(row.get("following_min_gap", 0.0))
        row["_selection_confidence"] = confidence
        row["_selection_order"] = order_value
        if scene not in SCENES:
            rejected["unsupported_scene"] += 1
        elif not token:
            rejected["missing_token"] += 1
        elif token in seen_tokens:
            rejected["duplicate_token"] += 1
        elif allowed_tokens is not None and token not in allowed_tokens:
            rejected["not_in_allowed_tokens"] += 1
        elif args.exclude_known_route_failures and token in ROUTE_FAILURE_TOKENS:
            rejected["known_route_failure"] += 1
        elif not valid:
            rejected["invalid_scene_or_axis"] += 1
        elif confidence < confidence_threshold:
            rejected["low_confidence"] += 1
        else:
            seen_tokens.add(token)
            candidates[scene].append(row)

    candidate_counts = {scene: len(candidates[scene]) for scene in SCENES}
    quotas = (
        _proportional_quotas(candidate_counts, args.total)
        if args.total is not None
        else {scene: args.per_scene for scene in SCENES}
    )

    selected: dict[str, list[dict]] = {}
    for scene_index, scene in enumerate(SCENES):
        rows = candidates[scene]
        requested = quotas[scene]
        if len(rows) < requested:
            raise ValueError(
                f"{scene} 仅有 {len(rows)} 条候选，少于请求数量 {requested}"
            )
        # 先按 preference rank 排序并切成等宽位置，再在每个位置附近随机抖动；
        # 既覆盖低/中/高参考风格，又避免永远选择同一个规则分位端点。
        rows = sorted(rows, key=lambda item: float(item["_selection_order"]))
        rng = random.Random(args.seed + scene_index * 1009)
        chosen = []
        for index in range(requested):
            lower = int(index * len(rows) / requested)
            upper = max(lower + 1, int((index + 1) * len(rows) / requested))
            chosen.append(rows[rng.randrange(lower, min(upper, len(rows)))])
        selected[scene] = chosen

    tokens = [str(row["token"]) for scene in SCENES for row in selected[scene]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "source": str(source_path),
        "source_mode": source_mode,
        "output": str(output),
        "seed": args.seed,
        "min_rank_confidence": args.min_rank_confidence,
        "min_scene_confidence": args.min_scene_confidence,
        "selection_mode": "fixed_total" if args.total is not None else "per_scene",
        "requested_total": args.total,
        "per_scene": None if args.total is not None else args.per_scene,
        "exclude_known_route_failures": bool(args.exclude_known_route_failures),
        "allowed_tokens_file": args.allowed_tokens_file,
        "allowed_token_count": len(allowed_tokens) if allowed_tokens is not None else None,
        "total_tokens": len(tokens),
        "candidate_counts": candidate_counts,
        "selection_quotas": quotas,
        "selected": {
            scene: [
                {
                    "token": str(row["token"]),
                    "selection_order": float(row["_selection_order"]),
                    "selection_confidence": float(row["_selection_confidence"]),
                }
                for row in selected[scene]
            ]
            for scene in SCENES
        },
        "rejected": dict(rejected),
    }
    report_path = output.with_name(output.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), "counts": {
        scene: len(selected[scene]) for scene in SCENES
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
