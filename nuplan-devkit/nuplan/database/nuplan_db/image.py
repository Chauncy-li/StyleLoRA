from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Set

from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow


@dataclass(frozen=True)
class Image(SensorDataTableRow):
    """A row from the NuPlan database ``image`` table."""

    token: Optional[str]
    next_token: Optional[str]
    prev_token: Optional[str]
    ego_pose_token: Optional[str]
    camera_token: Optional[str]
    filename_jpg: Optional[str]
    timestamp: Optional[int]
    channel: Optional[str]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> Image:
        """Convert a SQLite image row to the typed database representation."""

        keys: Set[str] = set(row.keys())  # type: ignore

        def token_value(name: str) -> Optional[str]:
            value = row[name] if name in keys else None
            return value.hex() if value is not None else None

        return cls(
            token=token_value("token"),
            next_token=token_value("next_token"),
            prev_token=token_value("prev_token"),
            ego_pose_token=token_value("ego_pose_token"),
            camera_token=token_value("camera_token"),
            filename_jpg=row["filename_jpg"] if "filename_jpg" in keys else None,
            timestamp=row["timestamp"] if "timestamp" in keys else None,
            channel=row["channel"] if "channel" in keys else None,
        )
