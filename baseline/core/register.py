"""
通用注册器（Registry）工具。

功能说明：
1. 用统一接口注册模型/流程函数/仿真 planner；
2. 支持装饰器注册与显式注册两种方式；
3. 在查询失败时提供可读的错误信息，方便排查配置拼写问题。

示例：
    REG = Registry("demo")

    @REG.register("foo")
    class Foo:
        ...

    REG.register("bar", object())
    FooCls = REG.get("foo")
"""

from __future__ import annotations

from typing import Dict, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """一个轻量、无框架依赖的注册器。"""

    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, value: Optional[T] = None):
        """
        注册一个对象。

        支持两种用法：
        1) 显式注册：registry.register("name", obj)
        2) 装饰器注册：
           @registry.register("name")
           class Obj: ...
        """
        if value is not None:
            self._register_item(key, value)
            return value

        def _decorator(obj: T) -> T:
            self._register_item(key, obj)
            return obj

        return _decorator

    def _register_item(self, key: str, value: T) -> None:
        if key in self._items:
            raise KeyError(f"[{self._name}] duplicate key: '{key}'")
        self._items[key] = value

    def get(self, key: str) -> T:
        if key not in self._items:
            choices = ", ".join(sorted(self._items.keys()))
            raise KeyError(
                f"[{self._name}] key '{key}' is not registered. "
                f"Available: [{choices}]"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def items(self):
        return self._items.items()

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(self._items)
