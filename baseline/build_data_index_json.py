"""
数据索引 JSON 构建脚本。

功能说明：
1. 扫描目录下 `.db` 文件并导出名称列表（不带后缀）；
2. 扫描目录下 `.npz` 文件并导出文件名列表；
3. 输出 JSON 供数据处理/训练脚本引用。
"""

from __future__ import annotations

import argparse
import json
import os


def _dump_sorted_unique(items, output_file):
    unique_items = sorted(set(items))
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(unique_items, json_file, indent=4)
    return len(unique_items)


def extract_db_filenames_to_json(data_path, output_file):
    """遍历目录并提取所有 `.db` 文件名（去后缀）。"""
    db_files = []
    for root, _, files in os.walk(data_path):
        for file_name in files:
            if file_name.endswith(".db"):
                db_files.append(file_name[:-3])

    count = _dump_sorted_unique(db_files, output_file)
    print(f"共找到 {count} 个不重复的 .db 文件")
    print(f"结果已保存到 {output_file}")


def extract_npz_filenames_to_json(data_path, output_file):
    """遍历目录并提取所有 `.npz` 文件名（保留后缀）。"""
    npz_files = []
    for root, _, files in os.walk(data_path):
        for file_name in files:
            if file_name.endswith(".npz"):
                npz_files.append(file_name)

    count = _dump_sorted_unique(npz_files, output_file)
    print(f"共找到 {count} 个不重复的 .npz 文件")
    print(f"结果已保存到 {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON index from .db/.npz files")
    parser.add_argument("--mode", choices=["db", "npz"], default="npz", help="Index file type")
    parser.add_argument("--data_path", type=str, default="/media/lsw/Work/ubuntu_system/DATACACHE/cache")
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/lsw/Work/ubuntu_system/DATACACHE/cache_list.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.mode == "db":
        extract_db_filenames_to_json(args.data_path, args.output_file)
    else:
        extract_npz_filenames_to_json(args.data_path, args.output_file)


if __name__ == "__main__":
    main()
