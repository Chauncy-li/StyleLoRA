"""
nuBoard 可视化启动脚本。

功能说明：
1. 启动 NuPlan 官方 `run_nuboard.py`；
2. 注入仿真输出 `.nuboard` 文件、地图路径、数据路径；
3. 用固定端口提供本地 Web 界面查看。

使用建议：
- 先修改本文件中的路径常量；
- 再运行：`python baseline/run_nuboard_viewer.py`。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_nuboard() -> None:
    """启动 nuBoard。"""
    simulation_file = (
        "/mnt/mydata/lishangwen/NuplanBaselinesRecord/nuplan_baseline/simulation/"
        "retrieval_augmented_diffusion_planner/2026-04-08-00-35-09/nuboard_1775579724.nuboard"
    )

    db_files_root = "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/nuplan-v1.1/splits/train_boston"
    map_root = "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/maps"
    exp_root = "/mnt/mydata/lishangwen/TrafficDataSetSource/dataset/exp"

    project_root = Path(__file__).resolve().parents[1]
    nuboard_script = project_root / "nuplan-devkit" / "nuplan" / "planning" / "script" / "run_nuboard.py"

    scenario_builder = "nuplan"
    map_version = "nuplan-maps-v1.0"
    port_number = 8106

    required_paths = {
        ".nuboard 文件": Path(simulation_file),
        "db 目录": Path(db_files_root),
        "地图目录": Path(map_root),
        "nuBoard 启动脚本": nuboard_script,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        print("❌ 以下路径不存在，请先修正脚本里的配置:")
        for item in missing:
            print(f"  - {item}")
        return

    os.environ["NUPLAN_DATA_ROOT"] = db_files_root
    os.environ["NUPLAN_MAPS_ROOT"] = map_root
    os.environ["NUPLAN_EXP_ROOT"] = exp_root
    os.environ["NUPLAN_NUBOARD_AUTO_OPEN"] = "0"

    cmd = [
        sys.executable,
        str(nuboard_script),
        f"simulation_path=['{simulation_file}']",
        f"scenario_builder={scenario_builder}",
        f"scenario_builder.db_files={db_files_root}",
        f"scenario_builder.map_root={map_root}",
        f"scenario_builder.map_version={map_version}",
        f"port_number={port_number}",
    ]

    print("🚀 正在启动 nuBoard...")
    print(f"📂 结果文件: {simulation_file}")
    print(f"🌐 监听端口: {port_number}")
    print(f"🖥️  本地浏览器地址: http://localhost:{port_number}")
    print(f"🔁 远程转发示例: ssh -N -L {port_number}:127.0.0.1:{port_number} <user>@<server>")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 已停止 nuBoard。")
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ 启动失败: {exc}")


def main() -> None:
    run_nuboard()


if __name__ == "__main__":
    main()
