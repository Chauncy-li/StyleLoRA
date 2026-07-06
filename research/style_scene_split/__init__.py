"""Straight-scene split utilities kept separate from the baseline package.

`v2` is the primary path for current experiments. Legacy `v1` scripts remain
available only for compatibility with older outputs.
"""

from .defaults import (
    DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_V1_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR,
    PRIMARY_SCENE_BUCKETS,
)
from .schema import (
    SCENE_BUCKET_ID_TO_NAME,
    SCENE_BUCKET_NAME_TO_ID,
    STYLE_LABEL_ID_TO_NAME,
    STYLE_LABEL_NAME_TO_ID,
    TOPOLOGY_BUCKET_ID_TO_NAME,
    TOPOLOGY_BUCKET_NAME_TO_ID,
    StyleSceneSplitResult,
)
from .schema_v2 import STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION
from .splitter import StyleSceneSplitter
from .splitter_v2 import StyleSceneSplitterV2

__all__ = [
    "PRIMARY_SCENE_BUCKETS",
    "DEFAULT_STYLE_SCENE_SPLIT_V1_DIR",
    "DEFAULT_STYLE_SCENE_SPLIT_V2_DIR",
    "DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR",
    "DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR",
    "SCENE_BUCKET_NAME_TO_ID",
    "SCENE_BUCKET_ID_TO_NAME",
    "STYLE_LABEL_NAME_TO_ID",
    "STYLE_LABEL_ID_TO_NAME",
    "TOPOLOGY_BUCKET_NAME_TO_ID",
    "TOPOLOGY_BUCKET_ID_TO_NAME",
    "STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION",
    "StyleSceneSplitResult",
    "StyleSceneSplitter",
    "StyleSceneSplitterV2",
]
