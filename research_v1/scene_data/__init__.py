"""Canonical straight-scene data pipeline.

The public modules expose the current v2 artifact contract without versioned
file names. Frozen foundations remain isolated under ``scene_data.legacy``.
"""

from .paths import (
    DEFAULT_STYLE_SCENE_SPLIT_TRAIN_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_V1_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_V2_DIR,
    DEFAULT_STYLE_SCENE_SPLIT_VAL_V2_DIR,
    PRIMARY_SCENE_BUCKETS,
)
from .legacy.schema import (
    SCENE_BUCKET_ID_TO_NAME,
    SCENE_BUCKET_NAME_TO_ID,
    STYLE_LABEL_ID_TO_NAME,
    STYLE_LABEL_NAME_TO_ID,
    TOPOLOGY_BUCKET_ID_TO_NAME,
    TOPOLOGY_BUCKET_NAME_TO_ID,
    StyleSceneSplitResult,
)
from .schema import STYLE_SCENE_SPLIT_V2_SCHEMA_VERSION
from .legacy.splitter import StyleSceneSplitter
from .splitter import StyleSceneSplitterV2

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
