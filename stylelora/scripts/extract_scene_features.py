"""Extract frozen scene context features h_c from the untrained DiffPlanner encoder.

从冻结 DiffPlanner（不注入 LoRA）提取场景上下文特征 h_c：

- 使用 ``load_plain_baseline`` 加载原始 baseline（不含任何 LoRA 适配器），
  全程 no_grad + eval，确保特征与冻结基座完全一致；
- 复用统一场景表征函数捕获 ``Encoder.fusion`` 的 key_padding_mask；
- 对 encoder 输出的场景 token 特征 ``encoding`` [B, token_num, D] 做
  masked mean pooling，得到每条样本的 ``h_c ∈ R^{D}``；
- 输出 ``features.npy``（N×D）+ ``feature_index.jsonl``（行序对齐），
  并可选择把场景特征引用 id 写回弱偏好 manifest 的 ``scene_feature`` 字段。

不改动任何 baseline / research_lora 源文件；只读复用 research_lora 的数据集与 runtime。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stylelora.lora.data.dataset import style_collate
from stylelora.lora.runtime import load_plain_baseline, prepare_diffusion_batch

from stylelora.data.loader import CacheOnlyDataset
from stylelora.model.scene_context import encode_scene_context
from stylelora.paths import (
    DEFAULT_FEATURE_INDEX,
    DEFAULT_FEATURE_NPY,
    DEFAULT_PREFERENCE_MANIFEST,
    ensure_repo_on_path,
)

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256（支持大文件，不一次性读入内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Extract frozen DiffPlanner scene features h_c without modifying baseline.")
    parser.add_argument("--args-file", required=True, help="Baseline args.json.")
    parser.add_argument("--baseline-checkpoint", required=True, help="Untouched DiffPlanner checkpoint.")
    parser.add_argument("--manifest", default=str(DEFAULT_PREFERENCE_MANIFEST),
                        help="Weak-preference manifest (or any manifest with cache_path rows).")
    parser.add_argument("--cache-root", required=True, help="Cache root for relative cache_path entries.")
    parser.add_argument("--output", default=str(DEFAULT_FEATURE_NPY), help="Output features.npy (N x D).")
    parser.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX), help="Output feature index JSONL (row-aligned).")
    parser.add_argument("--update-manifest", default=None,
                        help="Optional weak-preference manifest to annotate each row with its feature id (scene_feature).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size 必须为正，--workers 必须非负")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for a small smoke test")

    device = torch.device(args.device)
    model, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    model = model.to(device).eval()

    dataset = CacheOnlyDataset(args.manifest, root=args.cache_root,
                               predicted_neighbor_num=config.predicted_neighbor_num)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), collate_fn=style_collate)

    features: list[np.ndarray] = []
    index_rows: list[dict] = []
    total = 0
    with torch.no_grad():
        for batch in loader:
            metadata = batch.get("metadata") or []
            prepared = prepare_diffusion_batch(batch, device, config.observation_normalizer)
            model_inputs = prepared[0]
            _, feature = encode_scene_context(model, model_inputs)
            feature = feature.detach().cpu().to(dtype=torch.float32).numpy()
            for i, meta in enumerate(metadata):
                index_rows.append({
                    "fid": str(total + i),
                    "cache_path": str(meta.get("cache_path", "")),
                    "scene_type": str(meta.get("scene_type", "")),
                    "split": str(meta.get("split", "")),
                    "log_name": str(meta.get("log_name", "")),
                    "token": str(meta.get("token", "")),
                })
                features.append(feature[i])
            total += feature.shape[0]

    if total == 0:
        raise ValueError("No features were extracted; check manifest / cache coverage")
    if len(features) != total:
        raise ValueError(f"Feature count {len(features)} != manifest rows {total}")

    # 对齐校验：feature index 与输入 manifest 按行序 (log_name, token) 严格一致，防静默错位
    for i, raw_row in enumerate(dataset.rows):
        m_key = (str(raw_row.get("log_name", "")), str(raw_row.get("token", "")))
        i_key = (str(index_rows[i]["log_name"]), str(index_rows[i]["token"]))
        if any(m_key):
            if m_key != i_key:
                raise ValueError(f"Feature/manifest misalignment at row {i}: manifest key {m_key} vs index {i_key}")
        else:
            if str(raw_row.get("cache_path", "")) != str(index_rows[i]["cache_path"]):
                raise ValueError(f"Feature/manifest misalignment at row {i}: cache_path mismatch")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.stack(features, axis=0))
    index_path = Path(args.feature_index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # 保存版本元信息，供下游核对特征与偏好 manifest 的对应关系
    meta = {
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": _file_sha256(Path(args.manifest)),
        "feature_index_path": str(index_path.resolve()),
        "feature_index_sha256": _file_sha256(index_path),
        "feature_count": total,
    }
    meta_path = index_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote feature meta (hashes) -> {meta_path}")

    # 可选：把 scene_feature 引用写回偏好 manifest（生成一份新文件，不改 research_lora）
    if args.update_manifest:
        update_path = Path(args.update_manifest)
        if not update_path.exists():
            raise ValueError(f"--update-manifest target does not exist: {update_path}")
        annotated = update_path.with_name(f"{update_path.stem}_with_feature{update_path.suffix}")
        with annotated.open("w", encoding="utf-8", newline="\n") as handle:
            for row, index in zip(_iter_jsonl(update_path), index_rows):
                row["scene_feature"] = index["fid"]
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"Annotated {total} rows with scene_feature ids -> {annotated}")

    print(f"Extracted {total} scene features [{np.stack(features, axis=0).shape}] -> {output_path}")
    print(f"Wrote feature index -> {index_path}")


if __name__ == "__main__":
    main()
