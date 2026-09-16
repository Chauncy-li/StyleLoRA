"""闭环正式评测中跨脚本共享的固定常量。"""

from __future__ import annotations


# 这些 token 已在首轮闭环中确认存在地图路由修正失败，正式抽样时固定排除。
ROUTE_FAILURE_TOKENS = frozenset({
    "4985fb5081945e13",
    "7e89cf85f6765900",
    "c046cb464b8c519d",
    "e92894bc67005123",
    "f75e2c472f455b67",
})
