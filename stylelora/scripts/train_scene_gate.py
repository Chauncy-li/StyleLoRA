"""训练独立的场景上下文强度上限门控；不更新 baseline 与 LoRA。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from stylelora.data.encoder_dataset import _stable_key
from stylelora.model.scene_gate import SceneStrengthGate, save_scene_gate_checkpoint
from stylelora.paths import ensure_repo_on_path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class SceneGateDataset(Dataset):
    """按稳定 key 对齐冻结 ``h_c`` 与门控上限标签。"""

    def __init__(self, targets: str, feature_npy: str, feature_index: str) -> None:
        self.features = np.load(feature_npy, allow_pickle=False)
        key_to_fid: dict[str, int] = {}
        for row in _iter_jsonl(Path(feature_index)):
            key = str(row.get("key") or _stable_key(row))
            if key in key_to_fid:
                raise ValueError(f"feature index 存在重复 key: {key}")
            key_to_fid[key] = int(row["fid"])
        self.rows: list[tuple[int, float, float, float, str]] = []
        missing: list[str] = []
        for row in _iter_jsonl(Path(targets)):
            key = str(row["key"])
            fid = key_to_fid.get(key)
            if fid is None:
                missing.append(key)
                continue
            self.rows.append((
                fid,
                float(row["c_low_target"]),
                float(row["c_high_target"]),
                max(0.0, float(row.get("target_confidence", 1.0))),
                key,
            ))
        if missing:
            raise ValueError(f"有 {len(missing)} 个门控标签无法对齐 h_c，例如 {missing[:3]}")
        if not self.rows:
            raise ValueError("场景门控数据集为空")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        fid, low, high, confidence, key = self.rows[index]
        return {
            "h_c": torch.as_tensor(self.features[fid], dtype=torch.float32),
            "target": torch.tensor((low, high), dtype=torch.float32),
            "confidence": torch.tensor(confidence, dtype=torch.float32),
            "key": key,
        }

    def target_std(self) -> tuple[float, float]:
        target = np.asarray([(row[1], row[2]) for row in self.rows], dtype=np.float64)
        return float(target[:, 0].std()), float(target[:, 1].std())


def _weighted_gate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    overestimate_weight: float = 4.0,
) -> torch.Tensor:
    """非对称 Huber：强度上限预测过大比保守低估受到更强惩罚。"""
    element = F.huber_loss(prediction, target, reduction="none")
    element = element * torch.where(
        prediction > target,
        torch.full_like(element, overestimate_weight),
        torch.ones_like(element),
    )
    per_sample = element.mean(dim=1)
    weight = confidence.clamp_min(0.0)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-12)


@torch.no_grad()
def _evaluate(
    gate: SceneStrengthGate,
    loader: DataLoader,
    device: torch.device,
    *,
    overestimate_weight: float,
) -> dict[str, float]:
    gate.eval()
    predictions, targets, weights = [], [], []
    for batch in loader:
        h_c = batch["h_c"].to(device)
        target = batch["target"].to(device)
        confidence = batch["confidence"].to(device)
        predictions.append(gate(h_c))
        targets.append(target)
        weights.append(confidence)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    confidence = torch.cat(weights)
    error = prediction - target
    return {
        "loss": float(_weighted_gate_loss(
            prediction, target, confidence, overestimate_weight=overestimate_weight)),
        "mae_low": float(error[:, 0].abs().mean()),
        "mae_high": float(error[:, 1].abs().mean()),
        "overestimate_low_rate": float((error[:, 0] > 0).float().mean()),
        "overestimate_high_rate": float((error[:, 1] > 0).float().mean()),
        "pred_low_std": float(prediction[:, 0].std(unbiased=False)),
        "pred_high_std": float(prediction[:, 1].std(unbiased=False)),
    }


def main() -> None:
    ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="训练独立连续场景门控（baseline/LoRA 均不参与训练）")
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--train-feature-npy", required=True)
    parser.add_argument("--train-feature-index", required=True)
    parser.add_argument("--val-targets", required=True)
    parser.add_argument("--val-feature-npy", required=True)
    parser.add_argument("--val-feature-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--minimum-target-std", type=float, default=0.02)
    parser.add_argument("--overestimate-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0 or args.val_every <= 0 or args.workers < 0:
        parser.error("steps/batch-size/val-every 必须为正，workers 必须非负")
    if args.overestimate_weight < 1:
        parser.error("--overestimate-weight 必须不小于 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = SceneGateDataset(args.train_targets, args.train_feature_npy, args.train_feature_index)
    val_ds = SceneGateDataset(args.val_targets, args.val_feature_npy, args.val_feature_index)
    low_std, high_std = train_ds.target_std()
    if low_std < args.minimum_target_std and high_std < args.minimum_target_std:
        raise ValueError(
            f"门控标签几乎为常数：low_std={low_std:.4f}, high_std={high_std:.4f}；"
            "当前数据没有可学习的场景强度差异"
        )
    hc_dim = int(train_ds[0]["h_c"].numel())
    if int(val_ds[0]["h_c"].numel()) != hc_dim:
        raise ValueError("训练集与验证集 h_c 维度不一致")
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    gate = SceneStrengthGate(hc_dim=hc_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_state = None
    best_validation: dict[str, float] = {}
    step = 0
    while step < args.steps:
        for batch in train_loader:
            gate.train()
            h_c = batch["h_c"].to(device)
            target = batch["target"].to(device)
            confidence = batch["confidence"].to(device)
            prediction = gate(h_c)
            loss = _weighted_gate_loss(
                prediction, target, confidence,
                overestimate_weight=args.overestimate_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step % 20 == 0 or step == args.steps:
                print(f"step {step}/{args.steps} train_loss={float(loss):.6f}")
            if step % args.val_every == 0 or step == args.steps:
                validation = _evaluate(
                    gate, val_loader, device,
                    overestimate_weight=args.overestimate_weight,
                )
                print(f"  val step={step} {json.dumps(validation, ensure_ascii=False, sort_keys=True)}")
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    best_validation = validation
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in gate.state_dict().items()}
            if step >= args.steps:
                break

    if best_state is None:
        raise RuntimeError("门控训练没有产生有效 checkpoint")
    gate.load_state_dict(best_state, strict=True)
    save_scene_gate_checkpoint(
        args.output, gate, training_config=vars(args), validation=best_validation,
    )
    report = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "target_std": {"low": low_std, "high": high_std},
        "best_validation": best_validation,
    }
    report_path = Path(args.output).with_suffix(".report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"场景门控 checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
