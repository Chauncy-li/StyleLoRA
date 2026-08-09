"""Self-contained unit tests for the CSPQ encoder stage (no real data files needed).

测试项（覆盖专家验收条目）：
- 场景采样严格 1:1；
- confidence=0 不参与 RNC/标量排序；
- 跨场景近排名样本能形成有效正对；
- 数据集按稳定键对齐；
- 模型输入输出形状和梯度正常。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from stylelora.data.encoder_dataset import (
    PreferenceEncoderDataset,
    SceneBalancedBatchSampler,
    encoder_collate,
)
from stylelora.model.preference_encoder import CSPQPreferenceEncoder
from stylelora.training.rnc_loss import (
    axis_loss,
    cross_scene_rnc_loss,
    rank_huber_loss,
    rnc_loss,
)


def _make_manifest_and_features(path: Path, n_free: int = 40, n_car: int = 40) -> Path:
    """构造临时偏好 manifest + 特征 npy/index（全部有效三轴）。"""
    rows = []
    keys = []
    free_scene = "straight_free_drive"
    car_scene = "straight_car_follow"
    for i in range(n_free):
        keys.append(f"free_{i}")
        rows.append({
            "scene_type": free_scene,
            "axis_names": ["speed_preference", "longitudinal_intensity", "smoothness"],
            "axis_raw": [float(i), 1.0, 1.0 / (float(i) + 1.0)],
            "axis_valid": [True, True, True],
            "axis_percentiles": [i / max(n_free - 1, 1), 0.5, 0.5],
            "preference_rank": i / max(n_free - 1, 1),
            "rank_confidence": 1.0,
            "cache_path": f"cache_free_{i}.npz",
            "log_name": "log_free",
            "token": str(i),
            "source_index": "test",
        })
    for i in range(n_car):
        keys.append(f"car_{i}")
        rows.append({
            "scene_type": car_scene,
            "axis_names": ["headway_margin", "response_decisiveness", "response_smoothness"],
            "axis_raw": [float(i), 1.0, 1.0 / (float(i) + 1.0)],
            "axis_valid": [True, True, True],
            "axis_percentiles": [i / max(n_car - 1, 1), 0.5, 0.5],
            "preference_rank": i / max(n_car - 1, 1),
            "rank_confidence": 1.0,
            "cache_path": f"cache_car_{i}.npz",
            "log_name": "log_car",
            "token": str(i),
            "source_index": "test",
        })
    manifest_path = path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    features = np.random.RandomState(0).randn(n_free + n_car, 192).astype(np.float32)
    npy_path = path / "features.npy"
    np.save(npy_path, features)
    index_path = path / "features_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for fid, key in enumerate(keys):
            # 与 manifest 的 log_name/token 保持一致：free -> log_free, car -> log_car
            prefix, suffix = key.split("_")
            log_name = f"log_{prefix}" if prefix in ("free", "car") else prefix
            handle.write(json.dumps({"fid": fid, "log_name": log_name, "token": suffix,
                                     "cache_path": f"cache_{fid}.npz"}, ensure_ascii=False) + "\n")
    return manifest_path


def test_balanced_sampler_1to1() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest = _make_manifest_and_features(root)
        ds = PreferenceEncoderDataset(manifest, root / "features.npy", root / "features_index.jsonl", root)
        sampler = SceneBalancedBatchSampler(ds, batch_size=16)
        for batch in sampler:
            scenes = [ds.samples[i].scene_type for i in batch]
            assert scenes.count("straight_free_drive") == scenes.count("straight_car_follow") == 8


def test_confidence_zero_excluded() -> None:
    """confidence=0 的样本不参与标量排序；全零置信度 RNC 不崩。"""
    torch.manual_seed(0)
    z = torch.randn(8, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    rank = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    s = torch.randn(8, 1)
    conf_zero = torch.zeros(8)
    conf_half = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    l0 = rank_huber_loss(s, rank, conf_zero)
    assert float(l0) == 0.0
    l1 = rank_huber_loss(s, rank, conf_half)
    assert float(l1) >= 0.0
    r, _ = rnc_loss(z, rank, conf_zero, temperature=0.1)
    assert torch.isfinite(r)


def test_cross_scene_positive_pairs() -> None:
    """跨场景 RNC：两场景同排名样本应构成有效正对（loss 可计算且有限）。"""
    torch.manual_seed(0)
    z = torch.randn(8, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    rank = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    conf = torch.ones(8)
    scene_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 前 4 free, 后 4 car
    loss, pairs = cross_scene_rnc_loss(z, rank, conf, scene_ids, temperature=0.1)
    assert torch.isfinite(loss)
    assert pairs > 0


def test_axis_loss_valid_mask() -> None:
    """axis_loss：无效目标为真 NaN 时 loss 与梯度均有限（先清 NaN 再平方）。"""
    q_hat = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    # 无效轴目标用真正 NaN，验证不会污染损失/梯度
    q_vec = torch.tensor([[0.1, 0.9, float("nan")]])
    valid_mask = torch.tensor([[True, True, False]])
    l = axis_loss(q_hat, q_vec, valid_mask)
    expected = (0.4 ** 2 + 0.4 ** 2) / 2.0
    assert abs(float(l) - expected) < 1e-6
    l.backward()
    assert q_hat.grad is not None and torch.isfinite(q_hat.grad).all(), "axis_loss 梯度出现 NaN"


def test_model_shapes_and_grad() -> None:
    """模型输入输出形状与反向梯度正常。"""
    torch.manual_seed(0)
    model = CSPQPreferenceEncoder(trajectory_dim=6, hc_dim=192, d_model=32, heads=2, z_dim=8,
                                  query_rank=2)
    traj = torch.randn(4, 20, 6)
    hc = torch.randn(4, 192)
    out = model(traj, hc)
    assert out["s"].shape == (4, 1)
    assert out["q_hat"].shape == (4, 3)
    assert out["z"].shape == (4, 8)
    # 必须包含 q_hat，否则 q_hat_head 无梯度
    loss = out["z"].sum() + out["s"].sum() + out["q_hat"].sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_dataset_alignment() -> None:
    """数据集按稳定键对齐：特征 fid 行必须与偏好行对应。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest = _make_manifest_and_features(root)
        ds = PreferenceEncoderDataset(manifest, root / "features.npy", root / "features_index.jsonl", root)
        assert ds.missing == 0
        assert len(ds) == 80
        assert len(ds.free_indices) == 40
        assert len(ds.car_indices) == 40


def test_sampler_no_duplicate_within_epoch() -> None:
    """一个 epoch 内场景样本不重复（free/car 各自被使用约一次）。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest = _make_manifest_and_features(root, n_free=40, n_car=40)
        ds = PreferenceEncoderDataset(manifest, root / "features.npy", root / "features_index.jsonl", root)
        # batch_size=16 -> half=8, 每个场景 40//8=5 批，恰好覆盖各自全量 40 样本一次
        sampler = SceneBalancedBatchSampler(ds, batch_size=16)
        seen_free, seen_car = set(), set()
        for batch in sampler:
            for i in batch:
                if ds.samples[i].scene_type == "straight_free_drive":
                    assert i not in seen_free, "同一 epoch 内 free 样本重复出现"
                    seen_free.add(i)
                else:
                    assert i not in seen_car, "同一 epoch 内 car 样本重复出现"
                    seen_car.add(i)
        assert len(seen_free) == 40
        assert len(seen_car) == 40


def test_rnc_ordered_better_than_scrambled() -> None:
    """RNC 应偏好"有序圆弧嵌入"而非"同场景内随机打乱嵌入"。

    构造：
    - 有序：z_i 沿单位圆弧按 rank_i 排列（cos/sin 归一化后不会坍缩），
      相近 rank -> 弧上夹角小 -> 余弦相似度高；
    - 打乱：保持同一圆弧嵌入集合与同一 rank 张量，仅在同场景内随机交换
      样本的 embedding 行，使"相近 rank"对应的相似度随机化。

    说明：完全反序不改变 pairwise rank 距离，RNC 无法区分方向（方向由
    Huber(s, rank) 决定）；因此用"同场景内打乱"验证 RNC 确实在加强邻近性。
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_per_scene = 8
    rank = torch.linspace(0.1, 0.9, n_per_scene * 2)
    conf = torch.ones(n_per_scene * 2)
    scene_ids = torch.tensor([0] * n_per_scene + [1] * n_per_scene, dtype=torch.long)

    # 弧角范围在 (0, π) 内：余弦相似度 = cos(夹角) 随 rank 距离单调递减
    alpha = 0.9 * np.pi
    ang = alpha * rank
    z_ordered = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)
    z_ordered = torch.cat((z_ordered, torch.zeros(n_per_scene * 2, 6)), dim=-1)  # pad 到 8 维
    z_ordered = z_ordered / z_ordered.norm(dim=-1, keepdim=True)

    # 同场景内随机交换 embedding 行；rank 张量不变 -> pairwise rank 距离完全一致
    perm = torch.cat((
        torch.from_numpy(rng.permutation(n_per_scene)),
        torch.from_numpy(n_per_scene + rng.permutation(n_per_scene)),
    ))
    z_scrambled = z_ordered[perm]

    loss_ordered, _ = rnc_loss(z_ordered, rank, conf, temperature=0.1,
                               scene_id=scene_ids, cross_scene=False)
    loss_scrambled, _ = rnc_loss(z_scrambled, rank, conf, temperature=0.1,
                                 scene_id=scene_ids, cross_scene=False)
    assert float(loss_ordered) < float(loss_scrambled), (
        f"有序圆弧 RNC 应小于同场景打乱 RNC: {loss_ordered:.4f} vs {loss_scrambled:.4f}"
    )


def test_rnc_zero_confidence_no_pairs() -> None:
    """confidence=0 时 RNC loss 和有效配对数均为 0。"""
    torch.manual_seed(0)
    z = torch.randn(8, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    rank = torch.linspace(0.1, 0.9, 8)
    conf_zero = torch.zeros(8)
    scene_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss, pairs = rnc_loss(z, rank, conf_zero, temperature=0.1, scene_id=scene_ids, cross_scene=False)
    assert float(loss) == 0.0
    assert pairs == 0


def _reference_rnc_loss(z: torch.Tensor, rank: torch.Tensor, confidence: torch.Tensor,
                        scene_id: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """循环参考版 RNC（仅用于一致性测试，与张量化版逐 pair 定义相同）。

    L_ij = -log[ exp(sim_ij/τ) / Σ_{k: |r_i-r_k|>=|r_i-r_j|, k!=i, valid_ik} exp(sim_ik/τ) ]
    分母/正对均限制在同场景且 c>0 的配对内；对分母为空的 pair 直接跳过。
    """
    sim = z @ z.T / temperature
    rank_dist = (rank.unsqueeze(1) - rank.unsqueeze(0)).abs()
    active = confidence > 0
    pair_active = active.unsqueeze(1) & active.unsqueeze(0)
    same_scene = scene_id.unsqueeze(0) == scene_id.unsqueeze(1)
    valid_pair = pair_active & same_scene & ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    b = sim.shape[0]
    losses, weights = [], []
    for i in range(b):
        for j in torch.nonzero(valid_pair[i]).squeeze(-1).tolist():
            d_ij = rank_dist[i, j]
            k_mask = valid_pair[i] & (rank_dist[i] >= d_ij)
            k_idx = torch.nonzero(k_mask).squeeze(-1)
            if k_idx.numel() == 0:
                continue
            log_max = sim[i].max()
            num = torch.exp(sim[i, j] - log_max)
            den = torch.exp(sim[i, k_idx] - log_max).sum()
            losses.append(-torch.log(num / den.clamp_min(1e-12)))
            weights.append(confidence[i] * confidence[j])
    if not losses:
        return sim.new_zeros(())
    w = torch.stack(weights)
    return (torch.stack(losses) * w).sum() / w.sum()


def test_rnc_tensorized_matches_reference() -> None:
    """张量化 RNC 与循环参考版在 loss 和梯度上一致（误差 <= 1e-5）。"""
    torch.manual_seed(0)
    z = torch.randn(12, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    rank = torch.rand(12)
    conf = torch.rand(12)  # 含连续置信度（非全 1）
    scene_ids = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)

    loss_vec, _ = rnc_loss(z, rank, conf, temperature=0.1, scene_id=scene_ids, cross_scene=False)
    loss_ref = _reference_rnc_loss(z, rank, conf, scene_ids, temperature=0.1)
    assert abs(float(loss_vec) - float(loss_ref)) < 1e-5, (
        f"张量化与参考 RNC loss 不一致: {loss_vec:.8f} vs {loss_ref:.8f}"
    )

    # 梯度一致性：对 z 求导后比较（各自建图不相干）
    zv = z.clone().requires_grad_(True)
    lv, _ = rnc_loss(zv, rank, conf, temperature=0.1, scene_id=scene_ids, cross_scene=False)
    lv.backward()
    g_vec = zv.grad.clone()

    zr = z.clone().requires_grad_(True)
    lr = _reference_rnc_loss(zr, rank, conf, scene_ids, temperature=0.1)
    lr.backward()
    g_ref = zr.grad.clone()

    grad_diff = (g_vec - g_ref).abs().max().item()
    assert grad_diff < 1e-5, f"张量化与参考 RNC 梯度不一致，max diff={grad_diff:.2e}"


if __name__ == "__main__":
    test_balanced_sampler_1to1()
    test_confidence_zero_excluded()
    test_cross_scene_positive_pairs()
    test_axis_loss_valid_mask()
    test_model_shapes_and_grad()
    test_dataset_alignment()
    test_sampler_no_duplicate_within_epoch()
    test_rnc_ordered_better_than_scrambled()
    test_rnc_zero_confidence_no_pairs()
    test_rnc_tensorized_matches_reference()
    print("All encoder stage tests passed.")


