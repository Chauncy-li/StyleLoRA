"""
Codebook 标签质量校验脚本。

功能说明：
1. 扫描目录下 `.npz` 文件；
2. 统计 `code_lat` / `code_lon` / `code_rho` 的分布；
3. 输出匹配失败率、分桶均衡性和 rho 分位点建议。

典型用途：
- 在数据预处理（打标签）后快速做质量验收。
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import numpy as np
from tqdm import tqdm


def ascii_bar_chart(counter_dict, total, title="Distribution"):
    """打印简单的 ASCII 条形图。"""
    print(f"\n--- {title} ---")
    sorted_keys = sorted(counter_dict.keys())
    for key in sorted_keys:
        count = counter_dict[key]
        pct = (count / total) * 100
        bar_len = int(pct / 2)
        bar = "#" * bar_len
        print(f"Label {key:>3}: {count:>5} ({pct:>5.1f}%) | {bar}")


def verify_dataset(data_path):
    """执行标签分布校验。"""
    files = glob.glob(os.path.join(data_path, "*.npz"))
    if not files:
        print(f"Error: No .npz files found in {data_path}")
        return

    print(f"Scanning {len(files)} files for label verification...")

    lat_counts = Counter()
    lon_counts = Counter()
    rho_values = []

    fail_count = 0
    total_valid = 0

    for file_path in tqdm(files):
        try:
            with np.load(file_path, allow_pickle=True) as data:
                if "code_lat" not in data or "code_lon" not in data:
                    print(f"Skipping {file_path}: Missing label keys")
                    continue

                a_lat = int(data["code_lat"])
                a_lon = int(data["code_lon"])
                rho = float(data["code_rho"])

                if a_lat == -1:
                    fail_count += 1
                else:
                    lat_counts[a_lat] += 1
                    total_valid += 1

                if a_lon != -1:
                    lon_counts[a_lon] += 1
                    rho_values.append(rho)
        except Exception as exc:
            print(f"Error reading {file_path}: {exc}")

    total_samples = len(files)

    print("\n========================================")
    print("      CODEBOOK LABEL VERIFICATION       ")
    print("========================================")
    fail_rate = (fail_count / total_samples) * 100
    print(f"Total Samples: {total_samples}")
    print(f"Match Failures: {fail_count}")
    print(f"Failure Rate:   {fail_rate:.2f}%")

    if fail_rate > 5.0:
        print("⚠️  WARNING: Failure rate is high (>5%). Check search radius or map topology.")
    else:
        print("✅  Match rate looks good.")

    ascii_bar_chart(lat_counts, total_valid, title="Lateral Mode Distribution (Branch Index)")
    if len(lat_counts) > 1:
        print("✅  Multiple lateral branches detected (Good for multimodality).")
    else:
        print("⚠️  WARNING: Only one branch index detected. Are you extracting candidates correctly?")

    ascii_bar_chart(lon_counts, len(rho_values), title=f"Longitudinal Bin Distribution ({len(lon_counts)} bins)")

    if len(lon_counts) > 0:
        counts = list(lon_counts.values())
        if max(counts) / (min(counts) + 1e-6) > 5.0:
            print("⚠️  WARNING: Bin imbalance detected. Consider recalculating quantiles.")
        else:
            print("✅  Bins are relatively balanced.")

    if rho_values:
        print("\n--- Rho Statistics (for adjusting quantiles) ---")
        print(f"Min: {np.min(rho_values):.4f}")
        print(f"Max: {np.max(rho_values):.4f}")
        print(f"Mean: {np.mean(rho_values):.4f}")
        percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
        print(f"Suggested Quantiles (Percentiles {percentiles}):")
        print(np.percentile(rho_values, percentiles))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/mydata/lishangwen/NuplanBaselinesRecord/CACHE/cog_cache",
        help="Path to folder containing .npz files",
    )
    args = parser.parse_args()
    verify_dataset(args.data_path)


if __name__ == "__main__":
    main()
