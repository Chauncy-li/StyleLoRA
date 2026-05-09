"""
Wayformer 模型子包。

对外暴露：
- `WayFormer`: 推荐使用的模型类名（兼容历史 `Alpha_Planner`）。
"""

from baseline.model.wayformer.wayf_planner import Alpha_Planner, WayFormer

__all__ = ["WayFormer", "Alpha_Planner"]
