"""生成 raw DB 的 cache / simulation-test 划分名单。

这个脚本只做“名单划分”，不会处理数据，也不会生成 .npz。

输入通常是完整 Boston DB 名称列表：
  remote_files/nuplan_scenarios_boston.json

输出包括：
  - cache_log_names.json：后续允许进入训练/cache 处理的 DB 名单
  - test_simu_log_names.json：只用于 closed-loop simulation test 的 DB 名单
  - boston_cache_raw.yaml / boston_test_simu.yaml：Hydra scenario_filter 配置

什么时候运行：
  - 第一次建立数据划分时运行
  - raw DB 列表变化时运行
  - 想重新调整 simulation test 比例/seed 时运行

平时重新生成 .npz 数据时，不需要运行这个脚本；运行
research.data.cache_pipeline.process_cache_split 即可。
"""

from __future__ import annotations

from research.data.raw_db_split import main


if __name__ == "__main__":
    main()
