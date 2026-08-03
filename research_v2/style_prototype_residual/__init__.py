"""Same-state residual style-prototype feasibility study.

This package never changes ``baseline``.  It reuses the frozen StylePlanner and
its existing clean-x0 callback from research_v1 at runtime.
"""

from research_v2.style_prototype_residual.core import STYLE_NAMES

__all__ = ["STYLE_NAMES"]
