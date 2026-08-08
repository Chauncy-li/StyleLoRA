"""Evaluate the trained CSPQ preference encoder.

评测输出：
- 标量 s 与弱排序 r 的 Spearman/MAE（各场景分开 + 全局）；
- 三因子 q_hat 与 q_vec 的 MAE/Spearman（按 valid_mask）；
- 同场景最近邻 rank 误差、跨场景最近邻 rank 误差（对比随机检索基线）；
- 隐空间维度分析：各隐维标准差、协方差特征值、有效维数；
- 导出 train/val latent bank（embeddings + index），供后续 LoRA 使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from research_lora_2.data.encoder_dataset import (
    PreferenceEncoderDataset,
    SceneBalancedBatchSampler,
    encoder_collate,
)
from research_lora_2.model.preference_encoder import CSPQPreferenceEncoder
from research_lora_2.paths import (
    DEFAULT_ENCODER_CHECKPOINT,
    DEFAULT_ENCODER_EVAL_REPORT,
    DEFAULT_ENCODER_LATENT_BANK,
    DEFAULT_ENCODER_LATENT_BANK_INDEX,
    DEFAULT_ENCODER_VAL_LATENT,
    DEFAULT_ENCODER_VAL_LATENT_INDEX,
    DEFAULT_FEATURE_INDEX,
    DEFAULT_FEATURE_NPY,
    DEFAULT_FEATURE_VAL_INDEX,
    DEFAULT_FEATURE_VAL_NPY,
    DEFAULT_PREFERENCE_MANIFEST,
    DEFAULT_PREFERENCE_VAL_MANIFEST,
    ensure_repo_on_path,
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """平均秩 Spearman（并列组取组内均值秩）。"""
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]
        ranks = np.empty_like(values, dtype=np.float64)
        start = 0
        while start < values.shape[0]:
            end = start + 1
            while end < values.shape[0] and sorted_vals[end] == sorted_vals[start]:
                end += 1
            avg = (start + end - 1) / 2.0
            ranks[start:end] = avg
            start = end
        result = np.empty_like(ranks)
        result[order] = ranks
        return result
    ra, rb = average_ranks(a), average_ranks(b)
    da, db = ra - ra.mean(), rb - rb.mean()
    denom = np.linalg.norm(da) * np.linalg.norm(db)
    if denom == 0:
        return float("nan")
    return float((da * db).sum() / denom)


def _spearman_mae(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {"spearman": _spearman(pred, target), "mae": float(np.mean(np.abs(pred - target)))}


@torch.no_grad()
def _collect(model: CSPQPreferenceEncoder, ds: PreferenceEncoderDataset, device: torch.device,
             batch_size: int, workers: int) -> Dict[str, np.ndarray]:
    """全量顺序遍历数据集（不使用平衡采样器），前向收集指标与 latent。"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=encoder_collate)
    model.eval()
    records = {"s": [], "q_hat": [], "z": [], "rank": [], "q_vec": [], "valid": [], "scene": [], "key": []}
    for batch in loader:
        traj = batch["trajectory"].to(device)
        hc = batch["h_c"].to(device)
        out = model(traj, hc)
        records["s"].append(out["s"].cpu().numpy())
        records["q_hat"].append(out["q_hat"].cpu().numpy())
        records["z"].append(out["z"].cpu().numpy())
        records["rank"].append(batch["rank"].cpu().numpy())
        records["q_vec"].append(batch["q_vec"].cpu().numpy())
        records["valid"].append(batch["valid_mask"].cpu().numpy())
        records["scene"].append(batch["scene_id"].cpu().numpy())
        records["key"].extend(batch["key"])
    return {k: (np.concatenate(v, axis=0) if k != "key" else v) for k, v in records.items()}


def _block_nn_rank_error(z: np.ndarray, rank: np.ndarray, scene: np.ndarray, same_scene: bool,
                         *, block_size: int = 512) -> float:
    """同/跨场景最近邻 rank 绝对误差（分块计算，避免 O(N^2) 内存）。

    对每个 query 块 z_q，与所有参考向量 z_ref 分块计算余弦相似度，
    在候选（同/跨场景）中取相似度最高的邻居的 |rank 差| 均值。
    """
    n = z.shape[0]
    errs = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        z_q = z[start:end]
        # 分块计算余弦相似度（z 已归一化）
        sim_rows = []
        for ref_start in range(0, n, block_size):
            ref_end = min(ref_start + block_size, n)
            sim_rows.append(z_q @ z[ref_start:ref_end].T)
        sim = np.concatenate(sim_rows, axis=1)  # [block, n]
        for local_i, i in enumerate(range(start, end)):
            # 排除自身：置为极小值
            sim[local_i, i] = -np.inf
            if same_scene:
                candidates = np.where(scene == scene[i])[0]
            else:
                candidates = np.where(scene != scene[i])[0]
            candidates = candidates[candidates != i]
            if candidates.size == 0:
                continue
            j = candidates[np.argmax(sim[local_i, candidates])]
            errs.append(abs(rank[i] - rank[j]))
    return float(np.mean(errs)) if errs else float("nan")


def _select_retrieval_subset(scene: np.ndarray, max_per_scene: int, rng: np.random.Generator) -> np.ndarray:
    """从全量样本中按每场景最多 max_per_scene 条取平衡检索子集（固定种子可复现）。"""
    free_idx = np.where(scene == 0)[0]
    car_idx = np.where(scene == 1)[0]
    sel = []
    for group in (free_idx, car_idx):
        if group.size == 0:
            continue
        pick = rng.choice(group, size=min(max_per_scene, group.size), replace=False)
        sel.extend(pick.tolist())
    return np.asarray(sel, dtype=np.int64)


def _cached_scene_indices(scene: np.ndarray) -> dict:
    """预缓存两个场景的索引，避免随机基线反复 np.where。"""
    return {"free": np.where(scene == 0)[0], "car": np.where(scene == 1)[0]}


def _random_rank_error(rank: np.ndarray, scene: np.ndarray, same_scene: bool, rng: np.random.Generator,
                       scene_cache: dict | None = None) -> float:
    """随机检索基线：随机取同/跨场景邻居的 rank 误差（场景索引预缓存，避免重复扫描）。"""
    idx = rank.shape[0]
    cache = scene_cache if scene_cache is not None else _cached_scene_indices(scene)
    errs = []
    for i in range(idx):
        if same_scene:
            candidates = cache["free"] if scene[i] == 0 else cache["car"]
        else:
            candidates = cache["car"] if scene[i] == 0 else cache["free"]
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            continue
        j = rng.choice(candidates)
        errs.append(abs(rank[i] - rank[j]))
    return float(np.mean(errs)) if errs else float("nan")


def _latent_dims(z: np.ndarray) -> Dict[str, object]:
    """隐空间维度分析：各维标准差、协方差特征值、累计能量、有效维数。"""
    z_centered = z - z.mean(axis=0, keepdims=True)
    std = z_centered.std(axis=0)
    cov = np.cov(z_centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    cum = np.cumsum(eigvals) / max(total, 1e-12) if total > 1e-12 else np.zeros_like(eigvals)
    eff_dim = float(np.sum(cum < 0.95) + 1) if total > 1e-12 else 0.0
    return {
        "per_dim_std": std.tolist(),
        "eigenvalues": eigvals.tolist(),
        "cumulative_energy_ratio": cum.tolist(),
        "effective_dim_95": eff_dim,
    }


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Evaluate CSPQ preference encoder.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_ENCODER_CHECKPOINT))
    parser.add_argument("--train-manifest", default=str(DEFAULT_PREFERENCE_MANIFEST))
    parser.add_argument("--val-manifest", default=str(DEFAULT_PREFERENCE_VAL_MANIFEST))
    parser.add_argument("--train-feature-npy", default=str(DEFAULT_FEATURE_NPY))
    parser.add_argument("--train-feature-index", default=str(DEFAULT_FEATURE_INDEX))
    parser.add_argument("--val-feature-npy", default=str(DEFAULT_FEATURE_VAL_NPY))
    parser.add_argument("--val-feature-index", default=str(DEFAULT_FEATURE_VAL_INDEX))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-report", default=str(DEFAULT_ENCODER_EVAL_REPORT))
    parser.add_argument("--latent-bank-train", default=str(DEFAULT_ENCODER_LATENT_BANK))
    parser.add_argument("--latent-bank-train-index", default=str(DEFAULT_ENCODER_LATENT_BANK_INDEX))
    parser.add_argument("--latent-bank-val", default=str(DEFAULT_ENCODER_VAL_LATENT))
    parser.add_argument("--latent-bank-val-index", default=str(DEFAULT_ENCODER_VAL_LATENT_INDEX))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    model = CSPQPreferenceEncoder(
        trajectory_dim=config["trajectory_dim"], hc_dim=config["hc_dim"], d_model=config["d_model"],
        heads=config["heads"], z_dim=config["z_dim"], query_rank=config["query_rank"],
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model = model.to(device).eval()

    # 评测使用全量数据集（顺序遍历，不经过平衡采样器），导出全部有效样本的 latent
    train_ds = PreferenceEncoderDataset(args.train_manifest, args.train_feature_npy,
                                        args.train_feature_index, args.cache_root)
    val_ds = PreferenceEncoderDataset(args.val_manifest, args.val_feature_npy,
                                      args.val_feature_index, args.cache_root)
    if train_ds.missing > 0 or val_ds.missing > 0:
        raise ValueError(
            f"存在特征缺失样本（train missing={train_ds.missing}, val missing={val_ds.missing}）；"
            "请先生成完整的对齐特征索引，避免评测数据静默减少"
        )

    rng = np.random.default_rng(args.seed)
    report: Dict[str, object] = {"checkpoint": args.checkpoint}
    latent_banks: Dict[str, Dict[str, object]] = {}

    for split, ds in (("train", train_ds), ("val", val_ds)):
        rec = _collect(model, ds, device, batch_size=args.batch_size, workers=0)
        print(f"[collect] {split}: {len(rec['key'])} samples, "
              f"free={int((rec['scene'] == 0).sum())}, car={int((rec['scene'] == 1).sum())}")
        s_flat = rec["s"].reshape(-1)
        q_hat = rec["q_hat"]
        q_vec = rec["q_vec"]
        valid = rec["valid"]
        rank = rec["rank"].reshape(-1)
        scene = rec["scene"].reshape(-1)

        # 1) s vs rank（全局 + 分场景）
        s_metrics = {"all": _spearman_mae(s_flat, rank)}
        for scene_name, sid in (("straight_free_drive", 0), ("straight_car_follow", 1)):
            mask = scene == sid
            if mask.sum() >= 4:
                s_metrics[scene_name] = _spearman_mae(s_flat[mask], rank[mask])

        # 2) 三因子 q_hat vs q_vec（按 valid_mask）
        axis_metrics = {}
        for axis_idx in range(3):
            mask = valid[:, axis_idx]
            if mask.sum() >= 4:
                axis_metrics[f"axis_{axis_idx}"] = _spearman_mae(q_hat[mask, axis_idx], q_vec[mask, axis_idx])

        # 3) 最近邻 rank 误差 + 随机基线（固定、场景平衡的检索子集：每场景最多 5000）
        z_norm = rec["z"] / np.linalg.norm(rec["z"], axis=-1, keepdims=True)
        subset = _select_retrieval_subset(scene, max_per_scene=5000, rng=rng)
        z_sub, rank_sub, scene_sub = z_norm[subset], rank[subset], scene[subset]
        scene_cache = _cached_scene_indices(scene_sub)
        nn_same = _block_nn_rank_error(z_sub, rank_sub, scene_sub, same_scene=True)
        nn_cross = _block_nn_rank_error(z_sub, rank_sub, scene_sub, same_scene=False)
        rand_same = _random_rank_error(rank_sub, scene_sub, same_scene=True, rng=rng, scene_cache=scene_cache)
        rand_cross = _random_rank_error(rank_sub, scene_sub, same_scene=False, rng=rng, scene_cache=scene_cache)

        # 4) 隐空间维度
        dims = _latent_dims(rec["z"])

        report[split] = {
            "s_vs_rank": s_metrics,
            "q_hat_vs_q": axis_metrics,
            "nearest_rank_error": {
                "same_scene": nn_same,
                "cross_scene": nn_cross,
                "random_same_scene": rand_same,
                "random_cross_scene": rand_cross,
                "retrieval_samples": int(subset.shape[0]),
            },
            "latent_dims": dims,
        }
        latent_banks[split] = {"z": rec["z"], "key": rec["key"], "rank": rank, "scene": scene}

    # 写出 latent bank + index
    for split, bank_path, index_path in (
        ("train", args.latent_bank_train, args.latent_bank_train_index),
        ("val", args.latent_bank_val, args.latent_bank_val_index),
    ):
        Path(bank_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(bank_path, latent_banks[split]["z"])
        with Path(index_path).open("w", encoding="utf-8", newline="\n") as handle:
            for i, key in enumerate(latent_banks[split]["key"]):
                handle.write(json.dumps({
                    "fid": str(i),
                    "key": key,
                    "rank": float(latent_banks[split]["rank"][i]),
                    "scene": "free" if latent_banks[split]["scene"][i] == 0 else "car",
                }, ensure_ascii=False) + "\n")

    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_report).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote eval report -> {args.output_report}")
    print(f"Wrote latent banks -> {args.latent_bank_train} / {args.latent_bank_val}")


if __name__ == "__main__":
    main()