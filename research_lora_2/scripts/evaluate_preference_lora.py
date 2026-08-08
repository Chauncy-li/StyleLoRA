"""Evaluate continuous-preference LoRA via fixed-seed rho scanning (full metrics).

4b 完整指标：
- 双独立 baseline：planner(注入LoRA) 与 baseline_plain(未注入) 各自 load_plain_baseline；
- identity：adapter 关闭 vs 独立 baseline，同 batch+seed；
- rho 扫描：固定 batch，内层遍历 rho（同 seed），prediction 物理轨迹过冻结 CSPQ；
- 分场景 s/z 统计（straight_free_drive / straight_car_follow 分开）；
- z 目标分布 MMD：用 latent bank 对应 rank 区间（high/low）做参考分布；
- rho 单调性检查（s 随 rho 是否单调增）与正确/相反方向对比；
- 三轴物理指标（复用 scene_style_vector）、ADE/FDE、邻车变化、rollout 耗时。
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from research_lora.evaluation.rollout import rollout_with_rho
from research_lora.evaluation.style_metrics import ade_fde, mmd_rbf, scene_style_vector
from research_lora.model.checkpoint import load_adapter_checkpoint
from research_lora.model.style_lora_planner import StyleLoRAPlanner
from research_lora.runtime import load_plain_baseline, prepare_diffusion_batch
from research_lora.training.losses import build_noisy_inputs

from research_lora_2.data.preference_lora_dataset import (
    PreferenceLoRADataset, SceneBalancedLoRASampler, preference_lora_collate,
)
from research_lora_2.paths import (
    DEFAULT_ENCODER_CHECKPOINT, DEFAULT_FEATURE_INDEX, DEFAULT_FEATURE_NPY,
    DEFAULT_PREFERENCE_MANIFEST, ensure_repo_on_path,
)
from research_lora_2.training.preference_lora import load_frozen_cspq

SCENE_NAMES = ("straight_free_drive", "straight_car_follow")
HIGH_RANK = (0.8, 1.0)
LOW_RANK = (0.0, 0.2)


def _featurize_ego(pred_phys: torch.Tensor) -> torch.Tensor:
    """prediction 已是物理轨迹（[B,T,4]: x,y,cosθ,sinθ），直接构 token（不回 normalizer.inverse）。

    解码器/缓存管线的物理状态约定为 [x, y, cosθ, sinθ]：第 2/3 维已经是
    归一化方向向量，不是 heading 弧度，因此只做 L2 归一化后拼接，
    绝不能再当成 heading 去算 cos/sin（会产生错误的二次编码）。
    """
    pos = pred_phys[:, :, :2]
    delta = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        delta[:, 1:] = pos[:, 1:] - pos[:, :-1]
    heading_vec = torch.nn.functional.normalize(pred_phys[:, :, 2:4], dim=-1, eps=1e-6)
    return torch.cat((pos, delta, heading_vec), dim=-1)


def _ego_physical(pred: torch.Tensor) -> torch.Tensor:
    """取出 ego token 的物理轨迹 [B,T,4]；兼容 [B,P,T,4]（token=0 为 ego）与 [B,T,4]。"""
    return pred[:, 0] if pred.ndim == 4 else pred


def _prep(batch, device, obs_normalizer):
    tensors = {k: v.to(device) for k, v in batch["tensors"].items()}
    # 用真实 scene_id 推断场景类型（不要假设 batch 内第 0 个样本一定是 free_drive；
    # SceneBalancedLoRASampler 只保证每场景各半，不保证顺序）。
    meta = [{"scene_type": SCENE_NAMES[int(sid)]} for sid in batch["scene_id"].cpu().tolist()]
    return prepare_diffusion_batch({"tensors": tensors, "metadata": meta}, device, obs_normalizer,
                                   return_style_context=False)  # 只返回 2 值


def _style_vector_for_item(tensors: dict, prediction: torch.Tensor, index: int, scene: str):
    """单个样本的三轴物理风格向量（预测轨迹 -> scene_style_vector）。"""
    return scene_style_vector(
        scene=scene,
        ego_future=_ego_physical(prediction[index:index + 1])[0],
        ego_current=tensors["ego_current_state"][index],
        neighbors_past=tensors["neighbor_agents_past"][index],
        neighbors_future=tensors["neighbors_future_gt"][index],
        route_limits=tensors["route_lanes_speed_limit"][index],
        route_has_limits=tensors["route_lanes_has_speed_limit"][index],
        lane_limits=tensors["lanes_speed_limit"][index],
        lane_has_limits=tensors["lanes_has_speed_limit"][index],
    )


def _latent_roi(ds: PreferenceLoRADataset) -> torch.Tensor:
    """收集 latent bank 中与 dataset 样本行对应的 z（作为目标分布参考）。

    复用 dataset 已加载的 latent bank，避免重复读取 npy。
    """
    if ds.latent_rows and hasattr(ds, "_latent"):
        return torch.as_tensor(np.asarray(ds._latent)[ds.latent_rows], dtype=torch.float32)
    return torch.stack([ds[i]["z_target"] for i in range(len(ds))])


def _mean_std(values: list) -> dict:
    return {"mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values)) if values else float("nan")}


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-file", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter-high", required=True)
    parser.add_argument("--adapter-low", required=True)
    parser.add_argument("--cspq-checkpoint", default=str(DEFAULT_ENCODER_CHECKPOINT))
    parser.add_argument("--manifest", default=str(DEFAULT_PREFERENCE_MANIFEST))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--feature-npy", default=str(DEFAULT_FEATURE_NPY))
    parser.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX))
    parser.add_argument("--latent-bank", required=True)
    parser.add_argument("--latent-bank-index", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-eval-batches", type=int, default=10)
    parser.add_argument("--rho-min", type=float, default=-1.0)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--rho-steps", type=int, default=9)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None,
                        help="LoRA alpha；与训练 --alpha 保持一致，否则 rho 强度缩放不一致。")
    parser.add_argument("--identity-tolerance", type=float, default=1e-4,
                        help="identity_max_mismatch 超过该容差时直接报错（默认 1e-4）。")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    cspq = load_frozen_cspq(args.cspq_checkpoint, args.device)
    rho_list = [float(round(x, 3)) for x in np.linspace(args.rho_min, args.rho_max, args.rho_steps)]

    ds = PreferenceLoRADataset(args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
                               args.feature_npy, args.feature_index, direction="high", rank_low=0.0, rank_high=1.0)
    if len(ds) == 0:
        raise ValueError("评测 manifest 无样本")
    sampler = SceneBalancedLoRASampler(ds, args.batch_size, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=preference_lora_collate)
    fixed_batches = [b for _, b in zip(range(args.n_eval_batches), loader)]

    # 参考 latent 分布：high/low rank 区间各自作为目标 z 分布
    ref_high = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction="high", rank_low=HIGH_RANK[0], rank_high=HIGH_RANK[1])
    ref_low = PreferenceLoRADataset(
        args.manifest, args.cache_root, args.latent_bank, args.latent_bank_index,
        args.feature_npy, args.feature_index, direction="low", rank_low=LOW_RANK[0], rank_high=LOW_RANK[1])
    ref_z = {"high": _latent_roi(ref_high), "low": _latent_roi(ref_low)}
    for name, ref in ref_z.items():
        if ref.numel() == 0:
            raise ValueError(f"参考 latent 分布 {name} 为空，请检查 manifest 的 rank 覆盖")

    # 双独立 baseline
    baseline_plain, config = load_plain_baseline(args.args_file, args.baseline_checkpoint, args.device)
    baseline_plain = baseline_plain.to(device).eval()
    # Keep an independent object while guaranteeing an exactly identical base
    # state, including any state absent from a non-strict baseline checkpoint.
    model_lora = copy.deepcopy(baseline_plain)
    planner = StyleLoRAPlanner(model_lora, rank=args.rank, alpha=args.alpha, dropout=0.0).to(device)
    for ckpt in (args.adapter_high, args.adapter_low):
        load_adapter_checkpoint(ckpt, planner, baseline_checkpoint=args.baseline_checkpoint,
                                normalization_file=config.normalization_file_path, strict_hash=True)
    planner.eval()

    report = {"adapter_high": args.adapter_high, "adapter_low": args.adapter_low, "rho_grid": rho_list}

    # ---------- rho=0 identity + 邻居基准预测（同 batch+seed） ----------
    # 说明：rollout_with_rho 在 fork_rng 内同时设置 torch.manual_seed 与
    # torch.cuda.manual_seed_all；identity 阶段必须用同样的 seed 设定，
    # 才能让邻居基准预测与 rho 扫描共享完全相同的采样噪声。
    fork_devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ] if device.type == "cuda" else []
    max_diff = 0.0
    for batch in fixed_batches:
        prepped, futures = _prep(batch, device, config.observation_normalizer)
        ego_future, neighbors_future, _ = futures
        future = torch.cat((ego_future[:, None], neighbors_future), dim=1)
        fixed_time = torch.full((future.shape[0],), 0.5, device=device, dtype=future.dtype)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(args.seed)
            if fork_devices:
                torch.cuda.manual_seed_all(args.seed)
            fixed_noise = torch.randn_like(future)
        fixed_inputs, _, _ = build_noisy_inputs(
            prepped, futures, planner.sde.marginal_prob, config.state_normalizer,
            time=fixed_time, noise=fixed_noise,
        )
        planner.disable_adapter()
        with torch.no_grad():
            # 两个前向分别重置同一 seed：保证二者共享完全相同的初始采样噪声，
            # 否则第二次前向的 RNG 已前进，identity_max_mismatch 会混入采样随机性。
            _, out_wrap = planner(fixed_inputs)
            _, out_plain = baseline_plain(fixed_inputs)
        for key in set(out_wrap) & set(out_plain):
            a, b = out_wrap[key], out_plain[key]
            if torch.is_tensor(a) and tuple(a.shape) == tuple(b.shape):
                max_diff = max(max_diff, float((a - b).abs().max()))
    # identity 完成后必须重新启用适配器，否则后续 rho 扫描的 set_strength
    # 仍按 _enabled=False 路由，所有 rho 实际都是关闭适配器的 baseline。
    planner.enable_adapter()
    report["identity_max_mismatch"] = float(max_diff)
    if max_diff > args.identity_tolerance:
        raise RuntimeError(
            f"identity_max_mismatch={max_diff:.3e} 超过容差 {args.identity_tolerance:.3e}；"
            "LoRA 注入非恒等，评测不可信，请检查注入层/适配器加载。")

    # ---------- rho 扫描（固定 batch，内层 rho 同 seed） ----------
    # Full rho=0 rollouts are trajectory references, not the injection identity
    # test. They use the same wrapped model and seed as the subsequent rho scan.
    baseline_preds = []
    for batch in fixed_batches:
        prepped, _ = _prep(batch, device, config.observation_normalizer)
        with torch.no_grad():
            base_out, _ = rollout_with_rho(planner, prepped, 0.0, seed=args.seed)
        base_pred = base_out.get("prediction", base_out.get("x_start"))
        if base_pred is None or base_pred.ndim not in (3, 4):
            raise RuntimeError(
                f"Decoded prediction shape unexpected: "
                f"{None if base_pred is None else tuple(base_pred.shape)}"
            )
        baseline_preds.append(base_pred.detach())

    records = []
    for rho in rho_list:
        planner.set_strength(rho)
        for bi, batch in enumerate(fixed_batches):
            prepped, futures = _prep(batch, device, config.observation_normalizer)
            with torch.no_grad():
                out, seconds = rollout_with_rho(planner, prepped, rho, seed=args.seed)
            pred = out.get("prediction", out.get("x_start"))
            if pred is None or pred.ndim not in (3, 4):
                raise RuntimeError(f"rho={rho} prediction missing or shape {None if pred is None else tuple(pred.shape)}")
            pred = pred.detach()
            pref = cspq(_featurize_ego(_ego_physical(pred)), batch["h_c"].to(device))
            s_out = pref["s"].squeeze(-1)      # [B]
            z_out = pref["z"]                  # [B, z_dim]
            scenes = [SCENE_NAMES[int(sid)] for sid in batch["scene_id"].cpu().tolist()]
            base_pred = baseline_preds[bi]
            # 邻车变化：自适应 vs baseline(rho=0) 邻居 token 预测差异
            neighbor_change = None
            if pred.ndim == 4 and base_pred.ndim == 4:
                neighbor_change = (pred[:, 1:] - base_pred[:, 1:]).abs().mean().item()
            elif pred.ndim == 4:
                neighbor_change = pred[:, 1:].abs().mean().item()
            # 三轴指标：batch["tensors"] 来自 DataLoader 是 CPU 张量，必须用 CPU 预测，
            # 否则 scene_style_vector 内部 torch.cat 会 device mismatch；
            # CSPQ/ADE 仍使用 GPU pred。
            pred_cpu = pred.detach().cpu()
            n_samples = pred.shape[0]  # [B,T,D] 与 [B,P,T,D] 都按 batch 遍历
            for i in range(n_samples):
                phys_ego = _ego_physical(pred[i:i + 1])[0]
                gt_ego = futures[0][i:i + 1]
                ade_fde_dict = ade_fde(phys_ego.unsqueeze(0), gt_ego)
                scene = scenes[i]
                vector, valid = _style_vector_for_item(batch["tensors"], pred_cpu, i, scene)
                records.append({
                    "rho": float(rho), "batch": bi, "sample": i, "scene": scene,
                    "s": float(s_out[i]), "z": [float(x) for x in z_out[i].cpu().tolist()],
                    "style_vector": [float(x) for x in vector.cpu().tolist()],
                    "style_valid": bool(valid.all()),
                    "ade": ade_fde_dict["ade"], "fde": ade_fde_dict["fde"],
                    "neighbor_change": neighbor_change,
                    "seconds_per_batch": seconds,
                })

    # ---------- 跨 rho 共同有效样本集 ----------
    # car-follow 的前车有效性会随生成 ego 轨迹与 rho 一起变化；若每个 rho 各自
    # 取 valid 样本，不同 rho 的三轴均值会来自不同样本集，比较不公平。
    # 因此只保留"所有 rho 下都 style_valid"的样本做三轴均值比较
    # （沿用 evaluate_open_loop._mark_common_style_validity 的思想）。
    valid_by_rho: dict[str, dict[tuple[int, int], set[float]]] = defaultdict(lambda: defaultdict(set))
    for r in records:
        if r["style_valid"]:
            valid_by_rho[r["scene"]][(r["batch"], r["sample"])].add(r["rho"])
    expected_rhos = set(rho_list)
    common_valid: dict[str, set[tuple[int, int]]] = {
        scene: {key for key, valid_set in valid_by_rho[scene].items() if valid_set == expected_rhos}
        for scene in SCENE_NAMES
    }

    # ---------- 聚合 per-rho ----------
    per_rho = {}
    for rho in rho_list:
        rows = [r for r in records if abs(r["rho"] - rho) < 1e-9]
        s_all = [r["s"] for r in rows]
        z_all = torch.tensor([r["z"] for r in rows], dtype=torch.float32) if rows else torch.zeros(0)
        entry = {
            "count": len(rows),
            "s": _mean_std(s_all),
            "ade": _mean_std([r["ade"] for r in rows]),
            "fde": _mean_std([r["fde"] for r in rows]),
            "neighbor_change_mean": float(np.mean([r["neighbor_change"] for r in rows])) if rows and rows[0]["neighbor_change"] is not None else None,
            "seconds_per_batch": _mean_std([r["seconds_per_batch"] for r in rows]),
            "by_scene": {},
        }
        for scene in SCENE_NAMES:
            scene_rows = [r for r in rows if r["scene"] == scene]
            zs = torch.tensor([r["z"] for r in scene_rows], dtype=torch.float32) if scene_rows else torch.zeros(0)
            common_keys = common_valid[scene]
            # 三轴均值只统计跨 rho 共同有效样本（free 恒有效；car-follow 受前车有效性影响）
            comp_rows = [r for r in scene_rows if (r["batch"], r["sample"]) in common_keys]
            vecs = [r["style_vector"] for r in comp_rows]
            entry["by_scene"][scene] = {
                "count": len(scene_rows),
                "s": _mean_std([r["s"] for r in scene_rows]),
                "z_mean": zs.mean(dim=0).tolist() if zs.numel() else [],
                "z_std": zs.std(dim=0).tolist() if zs.numel() else [],
                "style": {
                    "valid_rate": float(np.mean([r["style_valid"] for r in scene_rows])) if scene_rows else float("nan"),
                    "comparison_valid_rate": float(np.mean([(r["batch"], r["sample"]) in common_keys for r in scene_rows])) if scene_rows else float("nan"),
                    "comparison_valid_count": len(comp_rows),
                    "mean_axis_vector": np.mean(vecs, axis=0).tolist() if vecs else [],
                },
            }
        if z_all.numel():
            entry["mmd_z_high_ref"] = float(mmd_rbf(z_all, ref_z["high"]))
            entry["mmd_z_low_ref"] = float(mmd_rbf(z_all, ref_z["low"]))
        per_rho[f"rho_{rho:.2f}"] = entry
    report["per_rho"] = per_rho
    report["common_valid_by_scene"] = {scene: sorted(keys) for scene, keys in common_valid.items()}

    # ---------- rho 单调性（s 随 rho 单调不减） ----------
    ascending = sorted(rho_list)
    s_seq = [per_rho[f"rho_{r:.2f}"]["s"]["mean"] for r in ascending]
    eps = 1e-3
    violations = sum(1 for i in range(1, len(s_seq)) if s_seq[i] < s_seq[i - 1] - eps)
    report["monotonicity"] = {
        "rho_ascending": ascending, "s_sequence": s_seq,
        "monotonic_increasing": violations == 0, "decreasing_violations": violations,
    }

    # ---------- 正确方向 vs 相反方向 ----------
    by_sample: dict[tuple[int, int], dict[float, float]] = defaultdict(dict)
    for r in records:
        by_sample[(r["batch"], r["sample"])][r["rho"]] = r["s"]
    correct_zero, zero_total = 0, 0
    plus_minus_correct, plus_minus_total = 0, 0
    for key, s_by in by_sample.items():
        s0 = s_by.get(0.0)
        for rho in sorted(s_by):
            if abs(rho) < 1e-9 or s0 is None:
                continue
            zero_total += 1
            if np.sign(s_by[rho] - s0) == np.sign(rho):
                correct_zero += 1
            if rho > 0 and -rho in s_by:
                plus_minus_total += 1
                plus_minus_correct += int(s_by[rho] > s_by[-rho])
    report["direction_check"] = {
        "sign_correct_vs_zero_rate": (correct_zero / zero_total) if zero_total else float("nan"),
        "samples_compared_vs_zero": zero_total,
        "plus_minus_rate": (plus_minus_correct / plus_minus_total) if plus_minus_total else float("nan"),
        "samples_compared_plus_minus": plus_minus_total,
    }

    report["records"] = records
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_report).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    summary = {k: v for k, v in report.items() if k != "records"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
