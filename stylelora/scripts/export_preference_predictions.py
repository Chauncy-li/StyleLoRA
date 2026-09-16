"""逐样本导出 CSPQ 偏好编码器预测，并关联原始场景划分置信度。

输出 JSONL 同时保存弱偏好监督、编码器 s/q_hat/z、有效轴信息以及可选
split_index.jsonl 中的 scene_confidence / split_confidence，供分组对齐分析使用。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from stylelora.data.encoder_dataset import PreferenceEncoderDataset
from stylelora.model.preference_encoder import CSPQPreferenceEncoder
from stylelora.paths import ensure_repo_on_path
from stylelora.scripts.evaluate_preference_encoder import _collect


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """逐行读取 JSONL，并在格式错误时给出精确行号。"""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc


def _finite_or_none(value: object) -> float | None:
    """把数值转成有限浮点数；NaN/Inf 输出为 JSON null。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normal_path(value: object) -> str:
    """只做分隔符与冗余前缀归一化，不要求服务器路径当前可访问。"""
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def _row_keys(row: Mapping[str, object]) -> list[str]:
    """为 split-index 行生成由强到弱的关联键。"""
    log_name = str(row.get("log_name", "") or "")
    token = str(row.get("token", row.get("scenario_token", "")) or "")
    sample_id = str(row.get("sample_id", "") or "")
    keys: list[str] = []
    if log_name and token:
        keys.append(f"log_token:{log_name}:{token}")
    if token:
        keys.append(f"token:{token}")
    if sample_id:
        keys.append(f"sample:{sample_id}")
    for field in ("cache_path", "style_cache_path", "planner_cache_path", "filename"):
        path = _normal_path(row.get(field, ""))
        if not path:
            continue
        keys.append(f"path:{path}")
        keys.append(f"basename:{Path(path).name}")
    return list(dict.fromkeys(keys))


def _sample_keys(*, log_name: str, token: str, cache_path: str) -> list[str]:
    """为偏好样本生成关联键，优先使用 log+token。"""
    keys: list[str] = []
    if log_name and token:
        keys.append(f"log_token:{log_name}:{token}")
    if token:
        keys.append(f"token:{token}")
        keys.append(f"sample:{token}")
    path = _normal_path(cache_path)
    if path:
        keys.append(f"path:{path}")
        keys.append(f"basename:{Path(path).name}")
    return list(dict.fromkeys(keys))


class SplitIndexLookup:
    """只接受唯一键的 split-index 查询器，拒绝静默匹配到重复样本。"""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lookup: dict[str, dict | None] = {}
        self.rows = 0
        self.ambiguous_keys = 0
        if self.path is None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"split index 不存在：{self.path}")
        for row in _iter_jsonl(self.path):
            self.rows += 1
            for key in _row_keys(row):
                if key in self._lookup:
                    if self._lookup[key] is not None:
                        self.ambiguous_keys += 1
                    self._lookup[key] = None
                else:
                    self._lookup[key] = row

    def find(self, *, log_name: str, token: str, cache_path: str) -> tuple[dict | None, str]:
        """返回唯一匹配行和命中的键类型。"""
        for key in _sample_keys(log_name=log_name, token=token, cache_path=cache_path):
            row = self._lookup.get(key)
            if row is not None:
                return row, key.split(":", 1)[0]
        return None, ""


def _load_model(checkpoint: str, device: torch.device) -> CSPQPreferenceEncoder:
    """从 checkpoint 的 model_config 重建 CSPQ。"""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = CSPQPreferenceEncoder(
        trajectory_dim=config["trajectory_dim"],
        hc_dim=config["hc_dim"],
        d_model=config["d_model"],
        heads=config["heads"],
        z_dim=config["z_dim"],
        query_rank=config["query_rank"],
    )
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval()


def _scene_from_split(row: Mapping[str, object]) -> str:
    return str(row.get("scene_bucket", row.get("scene_bucket_name", row.get("scene_type", ""))) or "")


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="导出逐样本 CSPQ 预测并关联场景置信度")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-npy", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split-index", default=None,
                        help="可选 scene split_index.jsonl；提供后导出 scene/split confidence。")
    parser.add_argument("--output", required=True, help="逐样本预测 JSONL。")
    parser.add_argument("--summary", default=None, help="关联与导出审计 JSON；默认 output 同名 *_summary.json。")
    parser.add_argument("--min-split-match-rate", type=float, default=0.8,
                        help="提供 split index 时要求的最低关联率；低于该值直接报错。")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.workers < 0 or not 0 <= args.min_split_match_rate <= 1:
        parser.error("--batch-size 必须为正，--workers 必须非负，匹配率阈值必须在 [0,1]")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dataset = PreferenceEncoderDataset(
        args.manifest, args.feature_npy, args.feature_index, args.cache_root,
    )
    if dataset.missing:
        raise ValueError(f"有 {dataset.missing} 条偏好样本缺少 h_c，拒绝静默缩小验证集")
    if len(dataset) == 0:
        raise ValueError("验证数据集为空")
    print(f"[stage 1/4] 数据集就绪：{len(dataset)} 个样本。", flush=True)

    print(f"[stage 2/4] 加载偏好编码器：{args.checkpoint}", flush=True)
    model = _load_model(args.checkpoint, device)
    records = _collect(
        model, dataset, device, batch_size=args.batch_size, workers=args.workers,
        progress_label="偏好编码器前向",
    )
    if len(records["key"]) != len(dataset):
        raise RuntimeError(f"前向输出 {len(records['key'])} 条，但数据集有 {len(dataset)} 条")

    split_lookup = SplitIndexLookup(args.split_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    match_types: Counter[str] = Counter()
    print("[stage 3/4] 写入逐样本预测。", flush=True)
    write_interval = max(1, len(dataset) // 20)

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(dataset.samples):
            if records["key"][index] != sample.key:
                raise RuntimeError(f"第 {index} 行 key 错位：{records['key'][index]!r} != {sample.key!r}")
            split_row, match_type = split_lookup.find(
                log_name=sample.log_name,
                token=sample.token,
                cache_path=sample.cache_path,
            )
            if split_row is None:
                counts["split_unmatched"] += 1
            else:
                counts["split_matched"] += 1
                match_types[match_type] += 1

            split_scene = _scene_from_split(split_row or {})
            scene_match = None if not split_scene else split_scene == sample.scene_type
            if scene_match is True:
                counts["scene_label_match"] += 1
            elif scene_match is False:
                counts["scene_label_mismatch"] += 1

            q_values = [_finite_or_none(value) for value in sample.axis_percentiles]
            q_hat = [_finite_or_none(value) for value in records["q_hat"][index].tolist()]
            valid = [bool(value) for value in sample.axis_valid]
            axis_abs_error = [
                (abs(float(q_hat[i]) - float(q_values[i]))
                 if valid[i] and q_hat[i] is not None and q_values[i] is not None else None)
                for i in range(3)
            ]
            prediction = {
                "row": index,
                "key": sample.key,
                "log_name": sample.log_name,
                "token": sample.token,
                "cache_path": sample.cache_path,
                "scene_type": sample.scene_type,
                "axis_names": list(sample.axis_names),
                "axis_raw": [_finite_or_none(value) for value in sample.axis_raw],
                "q": q_values,
                "q_hat": q_hat,
                "axis_valid": valid,
                "valid_axis_count": sample.valid_axes,
                "axis_abs_error": axis_abs_error,
                "rank": float(sample.preference_rank),
                "rank_confidence": float(sample.rank_confidence),
                "s": float(np.asarray(records["s"][index]).reshape(-1)[0]),
                "s_abs_error": abs(float(np.asarray(records["s"][index]).reshape(-1)[0])
                                   - float(sample.preference_rank)),
                "z": [float(value) for value in records["z"][index].tolist()],
                "split_joined": split_row is not None,
                "split_match_type": match_type,
                "split_scene": split_scene or None,
                "scene_label_match": scene_match,
                "scene_confidence": _finite_or_none((split_row or {}).get("scene_confidence")),
                "split_confidence": _finite_or_none((split_row or {}).get("split_confidence")),
                "split_valid": ((bool(split_row.get("split_valid"))) if split_row is not None
                                and "split_valid" in split_row else None),
                "scene_reason": str((split_row or {}).get("scene_reason", "") or ""),
                "source_mode": str((split_row or {}).get("source_mode", "") or ""),
            }
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
            counts[f"scene:{sample.scene_type}"] += 1
            counts[f"valid_axes:{sample.valid_axes}"] += 1
            counts["written"] += 1
            completed = index + 1
            if completed == 1 or completed == len(dataset) or completed % write_interval == 0:
                print(
                    f"[progress] 写入预测: {completed}/{len(dataset)} "
                    f"({completed / len(dataset):.1%})",
                    flush=True,
                )

    summary_path = (Path(args.summary) if args.summary else
                    output.with_name(f"{output.stem}_summary.json"))
    split_match_rate = counts["split_matched"] / max(1, counts["written"])
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "feature_npy": str(Path(args.feature_npy).resolve()),
        "feature_index": str(Path(args.feature_index).resolve()),
        "split_index": str(split_lookup.path.resolve()) if split_lookup.path else "",
        "output": str(output.resolve()),
        "dataset_samples": len(dataset),
        "split_index_rows": split_lookup.rows,
        "split_index_ambiguous_keys": split_lookup.ambiguous_keys,
        "split_match_rate": split_match_rate,
        "counts": dict(counts),
        "split_match_types": dict(match_types),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[stage 4/4] 汇总与审计文件写入完成。", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[done] predictions -> {output}")
    print(f"[done] summary -> {summary_path}")
    if split_lookup.path is not None and split_match_rate < args.min_split_match_rate:
        raise RuntimeError(
            f"split-index 关联率仅 {split_match_rate:.2%}，低于要求的 "
            f"{args.min_split_match_rate:.2%}；请核对 train/val 索引是否与偏好 manifest 同源"
        )


if __name__ == "__main__":
    main()
