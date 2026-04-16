from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class GnssRecord:
    timestamp: datetime
    message_header: str
    gps_week: int
    gps_time: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    acc_x: float
    acc_y: float
    acc_z: float
    latitude: float
    longitude: float
    altitude_m: float
    vel_east: float
    vel_north: float
    vel_up: float
    vehicle_speed: float
    main_sat_count: int
    aux_sat_count: int
    system_status: int
    diff_age: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class LidarPoint:
    timestamp: datetime
    x: float
    y: float
    z: float
    intensity: int
    ring: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
