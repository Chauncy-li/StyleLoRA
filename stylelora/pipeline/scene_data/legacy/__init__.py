"""Frozen scene-splitting foundations required by the canonical pipeline.

These modules preserve the original decision boundaries. New experiments
should import the canonical modules from ``stylelora.pipeline.scene_data``.
"""

from .schema import StyleSceneSplitResult
from .splitter import StyleSceneSplitter

__all__ = ["StyleSceneSplitResult", "StyleSceneSplitter"]

