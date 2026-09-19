"""训练 StageVec（STAGE 向量消融版，256 维向量 -> 8 路点轨迹 + steer）。

读取统一抽取产物 <npz_root>/<split>/<token>.npz 中的
stage_vec_x / stage_vec_y_traj / stage_vec_y_steer / stage_vec_prefer，
复用 baseline.model.stage_vec.model.build_vec_model（与闭环 planner 同一份模型定义）。

损失 = traj L1(0.5) + steer L1(1.0) + KL*kl_weight + style*style_weight（保留 ACT CVAE + 偏好头）。

用法：
  python comparison/train_stage_vec.py --npz-root <cache_root> --out-dir <out> \
      [--train-samples N --val-samples M]   # 默认 None=全量，四个模型需一致
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (REPO_ROOT / "nuplan-devkit", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline.data_process import featurizers  # noqa: E402
from baseline.model.stage_vec.model import build_vec_model  # noqa: E402
from comparison._io import load_splits, load_fields  # noqa: E402

NUM_POSES = 8
SUBSAMPLE_SEED = 0  # 子采样随机种子：四个模型保持一致，抽同一批可复现样本


def kl_divergence(mu, logvar):
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean()


def rule_style_preference(style_value, prefer_dict):
    """与 STAGE policy.py 对齐的风格偏好排名损失。prefer_dict 为 numpy 列表。"""
    prefer_score = []
    for i in range(style_value.size(0)):
        speed_score = prefer_dict["ego_vel_kmh"][i].mean() / 30.0
        throttle_score = prefer_dict["ego_control"][i][1] / 0.5
        if bool(prefer_dict["has_nearest"][i]):
            if prefer_dict["nearest_distance"][i][0] < 20:
                distance_score = 1.0 / prefer_dict["nearest_distance"][i].mean()
            else:
                distance_score = 0.0
        else:
            distance_score = 0.0
        prefer_score.append(speed_score + throttle_score + distance_score)

    loss = []
    for i in range(0, style_value.size(0) - 1, 2):
        if prefer_score[i] > prefer_score[i + 1]:
            loss.append(-F.logsigmoid(style_value[i] - style_value[i + 1]))
        elif prefer_score[i + 1] > prefer_score[i]:
            loss.append(-F.logsigmoid(style_value[i + 1] - style_value[i]))
    if len(loss) == 0:
        return torch.zeros((), device=style_value.device)
    return torch.stack(loss).mean()


def make_prefer_dict(prefer_batch):
    return {
        "ego_vel_kmh": [np.array([p[0]], dtype=np.float32) for p in prefer_batch],
        "ego_control": [np.array([0.0, p[1]], dtype=np.float32) for p in prefer_batch],
        "has_nearest": [bool(p[2]) for p in prefer_batch],
        "nearest_distance": [np.array([p[3]], dtype=np.float32) for p in prefer_batch],
    }


def compute_style_value_stats(model, X_n, Yt_n, Ys_n, device, batch):
    """训练态前向（带 actions）收集 style_value，返回 (mean, std)。

    推理态 style_value 恒为 style_control 或 0，拿不到学到的分布，故必须走训练态分支。
    """
    model.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, X_n.shape[0], batch):
            sl = slice(i, i + batch)
            vec = torch.from_numpy(X_n[sl]).to(device)
            actions = {
                "steer_throttle": torch.from_numpy(Ys_n[sl]).to(device),
                "traj_action": torch.from_numpy(Yt_n[sl]).to(device),
            }
            B = vec.shape[0]
            is_pad = torch.zeros(B, NUM_POSES + 1, dtype=torch.bool, device=device)
            _, _, _, style_value = model(vec, actions=actions, is_pad=is_pad)
            vals.append(style_value.detach().cpu().numpy().reshape(-1))
    model.train()
    arr = np.concatenate(vals)
    return float(arr.mean()), float(arr.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(SCRIPT_DIR / "splits.json"))
    ap.add_argument("--npz-root", required=True, help="extract.py 的 --output-root")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--style-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-samples", type=int, default=None, help="训练样本数；None=用满 train split（四个模型需一致）")
    ap.add_argument("--val-samples", type=int, default=None, help="验证样本数；None=用满 val split")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(16)

    splits = load_splits(args.splits)
    fields = ["stage_vec_x", "stage_vec_y_traj", "stage_vec_y_steer", "stage_vec_prefer"]
    train = load_fields(os.path.join(args.npz_root, "train"), splits["train_tokens"], fields)
    val = load_fields(os.path.join(args.npz_root, "val"), splits["val_tokens"], fields)
    print(f"[stage_vec] train {len(train['stage_vec_x'])}, val {len(val['stage_vec_x'])}")

    # 固定 seed 的随机子采样（而非取前 N），保证四个模型抽到同一批可复现样本。
    if args.train_samples is not None:
        idx = np.random.RandomState(SUBSAMPLE_SEED).permutation(train["stage_vec_x"].shape[0])[: args.train_samples]
        for k in fields:
            train[k] = train[k][idx]
    if args.val_samples is not None:
        idx = np.random.RandomState(SUBSAMPLE_SEED).permutation(val["stage_vec_x"].shape[0])[: args.val_samples]
        for k in fields:
            val[k] = val[k][idx]
    print(f"[stage_vec] 训练样本: train {len(train['stage_vec_x'])}, val {len(val['stage_vec_x'])}")

    X_tr = train["stage_vec_x"].astype(np.float32)
    Yt_tr = train["stage_vec_y_traj"].astype(np.float32)
    Ys_tr = train["stage_vec_y_steer"].astype(np.float32)
    P_tr = train["stage_vec_prefer"].astype(np.float32)
    X_va = val["stage_vec_x"].astype(np.float32)
    Yt_va = val["stage_vec_y_traj"].astype(np.float32)
    Ys_va = val["stage_vec_y_steer"].astype(np.float32)
    P_va = val["stage_vec_prefer"].astype(np.float32)

    # 归一化统计：仅 train 划分上计算（与 DiffPlanner normalization 同口径，val 不参与；
    # 与 planner 共用，保存为 stage_vec_stats.npz）。val/test 仍用 train 的统计归一化。
    vec_stats = featurizers.compute_stats(X_tr)
    vec_mean, vec_std = vec_stats["mean"], vec_stats["std"]
    traj_mean = Yt_tr.mean(axis=0).astype(np.float32)
    traj_std = np.clip(Yt_tr.std(axis=0), 1e-2, None).astype(np.float32)
    steer_mean = Ys_tr.mean(axis=0).astype(np.float32)
    steer_std = np.clip(Ys_tr.std(axis=0), 1e-2, None).astype(np.float32)

    X_tr_n = (X_tr - vec_mean) / vec_std
    X_va_n = (X_va - vec_mean) / vec_std
    Yt_tr_n = (Yt_tr - traj_mean) / traj_std
    Yt_va_n = (Yt_va - traj_mean) / traj_std
    Ys_tr_n = (Ys_tr - steer_mean) / steer_std
    Ys_va_n = (Ys_va - steer_mean) / steer_std

    model = build_vec_model(num_queries=NUM_POSES)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[stage_vec] model params: {nparam} ({nparam / 1e6:.2f}M)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward_batch(X_n, Yt_n, Ys_n, P):
        vec = torch.from_numpy(X_n).to(device)
        actions = {
            "steer_throttle": torch.from_numpy(Ys_n).to(device),
            "traj_action": torch.from_numpy(Yt_n).to(device),
        }
        B = vec.shape[0]
        is_pad = torch.zeros(B, NUM_POSES + 1, dtype=torch.bool, device=device)
        a_hat, _, (mu, logvar), style_value = model(vec, actions=actions, is_pad=is_pad)
        steer_l1 = F.l1_loss(actions["steer_throttle"], a_hat["steer_throttle"], reduction="none") * 1.0
        traj_l1 = F.l1_loss(actions["traj_action"], a_hat["traj_action"], reduction="none") * 0.5
        all_l1 = torch.cat([steer_l1, traj_l1], dim=1)
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        kl = kl_divergence(mu, logvar)
        style = rule_style_preference(style_value, make_prefer_dict(P))
        loss = l1 + kl * args.kl_weight + style * args.style_weight
        return loss, {"l1": l1.item(), "kl": kl.item(), "style": style.item(), "loss": loss.item()}

    def evaluate(X_n, Yt_n, Ys_n, P):
        model.eval()
        total, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, X_n.shape[0], args.batch):
                sl = slice(i, i + args.batch)
                bx, by, bs, bp = X_n[sl], Yt_n[sl], Ys_n[sl], P[sl]
                if bx.shape[0] % 2 == 1:
                    bx, by, bs, bp = bx[:-1], by[:-1], bs[:-1], bp[:-1]
                if bx.shape[0] == 0:
                    continue
                loss, _ = forward_batch(bx, by, bs, bp)
                total += loss.item() * bx.shape[0]
                nb += bx.shape[0]
        model.train()
        return total / max(nb, 1)

    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(X_tr_n.shape[0])
        total, nb = 0.0, 0
        for i in range(0, X_tr_n.shape[0], args.batch):
            idx = perm[i:i + args.batch]
            bx, by, bs, bp = X_tr_n[idx], Yt_tr_n[idx], Ys_tr_n[idx], P_tr[idx]
            if bx.shape[0] % 2 == 1:
                bx, by, bs, bp = bx[:-1], by[:-1], bs[:-1], bp[:-1]
            if bx.shape[0] == 0:
                continue
            opt.zero_grad()
            loss, _ = forward_batch(bx, by, bs, bp)
            loss.backward()
            opt.step()
            total += loss.item() * bx.shape[0]
            nb += bx.shape[0]
        train_loss = total / max(nb, 1)
        val_loss = evaluate(X_va_n, Yt_va_n, Ys_va_n, P_va)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[stage_vec] ep {ep:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time() - t0:.0f}s")

    # 用 best checkpoint 计算风格统计，保证与保存的权重（best_state）严格一致。
    model.load_state_dict(best_state)

    # 风格可控性：存训练集 style_value 的 mean/std，推理激进=+σ、保守=-σ（用户约定）。
    style_mean, style_std = compute_style_value_stats(model, X_tr_n, Yt_tr_n, Ys_tr_n, device, args.batch)
    print(f"[stage_vec] style_value mean={style_mean:.4f} std={style_std:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"state_dict": best_state, "nparam": nparam},
               os.path.join(args.out_dir, "stage_vec_checkpoint.pt"))
    np.savez(
        os.path.join(args.out_dir, "stage_vec_stats.npz"),
        vec_mean=vec_mean, vec_std=vec_std,
        traj_mean=traj_mean, traj_std=traj_std,
        steer_mean=steer_mean, steer_std=steer_std,
        style_value_mean=style_mean, style_value_std=style_std,
    )
    with open(os.path.join(args.out_dir, "stage_vec_metrics.json"), "w") as f:
        json.dump({"best_val_loss": best_val, "epochs": args.epochs, "nparam": nparam,
                   "train_samples": int(X_tr.shape[0]), "val_samples": int(X_va.shape[0]),
                   "kl_weight": args.kl_weight, "style_weight": args.style_weight,
                   "style_value_mean": style_mean, "style_value_std": style_std}, f, indent=2)
    print(f"[stage_vec] saved stage_vec_checkpoint.pt, stage_vec_stats.npz, stage_vec_metrics.json")
    print(f"[stage_vec] BEST val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
