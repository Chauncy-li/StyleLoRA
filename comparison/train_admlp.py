"""训练 AD-MLP（8 维自车状态 -> 8 帧未来位姿）。

读取统一抽取产物 <npz_root>/<split>/<token>.npz 中的 admlp_x / admlp_y，
复用 baseline.model.admlp.model.EgoStatusMLP（与闭环 planner 同一份模型定义）。

用法：
  python comparison/train_admlp.py --npz-root <cache_root> --out-dir <out> \
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
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (REPO_ROOT / "nuplan-devkit", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline.model.admlp.model import STYLE_DIM, EgoStatusMLP  # noqa: E402
from comparison._io import load_splits, load_fields  # noqa: E402

NUM_POSES = 8
SUBSAMPLE_SEED = 0  # 子采样随机种子：四个模型保持一致，抽同一批可复现样本


def ade_fde(pred, gt):
    d = torch.norm(pred[..., :2] - gt[..., :2], dim=-1)  # [N,8]
    return d.mean(dim=-1).mean().item(), d[:, -1].mean().item()


def _style_onehot(scores, q33, q67):
    """连续激进分按 (q33, q67) 分箱为 A(0)/N(1)/C(2)，返回 [N, STYLE_DIM] one-hot。

    与 StyleDrive STYLE_MAP = {"A": 0, "N": 1, "C": 2} 对齐：高分激进=A，低分保守=C。
    """
    scores = np.asarray(scores, dtype=np.float32)
    idx = np.full(scores.shape[0], 1, dtype=np.int64)  # 默认 N
    idx[scores >= q67] = 0  # A 激进
    idx[scores < q33] = 2  # C 保守
    return torch.from_numpy(np.eye(STYLE_DIM, dtype=np.float32)[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(SCRIPT_DIR / "splits.json"))
    ap.add_argument("--npz-root", required=True, help="extract.py 的 --output-root")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-samples", type=int, default=None, help="训练样本数；None=用满 train split（四个模型需一致）")
    ap.add_argument("--val-samples", type=int, default=None, help="验证样本数；None=用满 val split")
    ap.add_argument("--with-style", action="store_true", help="按 StyleDrive 口径拼接 3 维风格 one-hot（input 8->11）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(16)

    splits = load_splits(args.splits)
    fields = ["admlp_x", "admlp_y"] + (["admlp_style_score"] if args.with_style else [])
    train = load_fields(os.path.join(args.npz_root, "train"), splits["train_tokens"], fields)
    val = load_fields(os.path.join(args.npz_root, "val"), splits["val_tokens"], fields)
    test = load_fields(os.path.join(args.npz_root, "test"), splits["test_tokens"], fields)
    print(f"[admlp] train {len(train['admlp_x'])}, val {len(val['admlp_x'])}, test {len(test['admlp_x'])}")

    Xt = torch.from_numpy(train["admlp_x"]).float()
    Yt = torch.from_numpy(train["admlp_y"]).float()
    Xv = torch.from_numpy(val["admlp_x"]).float()
    Yv = torch.from_numpy(val["admlp_y"]).float()
    Xte = torch.from_numpy(test["admlp_x"]).float()
    Yte = torch.from_numpy(test["admlp_y"]).float()

    # 固定 seed 的随机子采样（而非取前 N），保证四个模型抽到同一批可复现样本。
    train_idx = val_idx = None
    if args.train_samples is not None:
        train_idx = np.random.RandomState(SUBSAMPLE_SEED).permutation(Xt.shape[0])[: args.train_samples]
        Xt, Yt = Xt[train_idx], Yt[train_idx]
    if args.val_samples is not None:
        val_idx = np.random.RandomState(SUBSAMPLE_SEED).permutation(Xv.shape[0])[: args.val_samples]
        Xv, Yv = Xv[val_idx], Yv[val_idx]
    print(f"[admlp] 训练样本: train {Xt.shape[0]}, val {Xv.shape[0]}")

    # AD-MLP 风格标签：规则代理分 -> 训练集 1/3、2/3 分位分箱 A/N/C -> 3 维 one-hot 拼到输入。
    style_quantiles = None
    if args.with_style:
        style_tr = train["admlp_style_score"].reshape(-1)
        style_va = val["admlp_style_score"].reshape(-1)
        style_te = test["admlp_style_score"].reshape(-1)
        if train_idx is not None:
            style_tr = style_tr[train_idx]
        if val_idx is not None:
            style_va = style_va[val_idx]
        q33 = float(np.quantile(style_tr, 1 / 3))
        q67 = float(np.quantile(style_tr, 2 / 3))
        style_quantiles = (q33, q67)
        Xt = torch.cat([Xt, _style_onehot(style_tr, q33, q67)], dim=1)
        Xv = torch.cat([Xv, _style_onehot(style_va, q33, q67)], dim=1)
        Xte = torch.cat([Xte, _style_onehot(style_te, q33, q67)], dim=1)
        print(f"[admlp] 风格分箱 A/N/C 阈值 q33={q33:.4f}, q67={q67:.4f}")

    model = EgoStatusMLP(hidden_dim=args.hidden, num_poses=NUM_POSES, with_style=args.with_style)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[admlp] model params: {nparam}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.L1Loss()

    best_val_l1 = float("inf")
    best_state = None
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        total, nb = 0.0, 0
        for i in range(0, Xt.shape[0], args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xt[idx].to(device), Yt[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        train_l1 = total / nb

        model.eval()
        with torch.no_grad():
            pv = model(Xv.to(device))
            val_l1 = lossf(pv, Yv.to(device)).item()
            val_ade, val_fde = ade_fde(pv, Yv.to(device))
        history.append((ep, train_l1, val_l1, val_ade, val_fde))
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[admlp] ep {ep:3d} | train_l1 {train_l1:.4f} | val_l1 {val_l1:.4f} "
                  f"val_ade {val_ade:.3f} val_fde {val_fde:.3f} | {time.time() - t0:.0f}s")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(Xte.to(device))
        test_l1 = lossf(pt, Yte.to(device)).item()
        test_ade, test_fde = ade_fde(pt, Yte.to(device))

    print("\n===== FINAL TEST (held-out logs) =====")
    print(f"[admlp] test_l1 {test_l1:.4f} | test_ADE {test_ade:.4f} m | test_FDE {test_fde:.4f} m")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "admlp_checkpoint.pt")
    torch.save({"state_dict": best_state, "nparam": nparam, "history": history,
                "hidden_dim": args.hidden, "input_dim": STYLE_DIM + 8 if args.with_style else 8,
                "with_style": args.with_style, "style_quantiles": style_quantiles,
                "num_poses": NUM_POSES,
                "test_ade": test_ade, "test_fde": test_fde, "test_l1": test_l1}, ckpt_path)
    with open(os.path.join(args.out_dir, "admlp_metrics.json"), "w") as f:
        json.dump({"test_ADE_m": test_ade, "test_FDE_m": test_fde, "test_L1": test_l1,
                   "nparam": nparam, "train_samples": int(Xt.shape[0]),
                   "val_samples": int(Xv.shape[0]), "test_samples": int(Xte.shape[0]),
                   "epochs": args.epochs, "best_val_l1": best_val_l1,
                   "with_style": args.with_style, "style_quantiles": style_quantiles}, f, indent=2)
    print(f"[admlp] saved {ckpt_path}")


if __name__ == "__main__":
    main()
