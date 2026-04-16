from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..models import GnssRecord


_NUM_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)")


class GnssParser:
    """Parser for parsed GPCHC text lines.

    Input format expected (example):
    2025-09-12T16:56:46.400000（时间戳）,$GPCHC（报文头）,2383（GPS周）,...
    """

    @staticmethod
    def _extract_value(raw: str) -> str:
        return raw.split("（", 1)[0].strip()

    @staticmethod
    def _to_float(raw: str) -> float:
        value = GnssParser._extract_value(raw)
        match = _NUM_RE.match(value)
        if not match:
            raise ValueError(f"cannot parse float from: {raw}")
        return float(match.group(1))

    @staticmethod
    def _to_int(raw: str) -> int:
        return int(round(GnssParser._to_float(raw)))

    @classmethod
    def parse_line(cls, line: str) -> GnssRecord:
        parts = [p.strip() for p in line.strip().split(",") if p.strip()]
        if len(parts) < 24:
            raise ValueError(f"invalid gpchc line field count={len(parts)}")

        ts_text = cls._extract_value(parts[0])
        ts = datetime.fromisoformat(ts_text)

        return GnssRecord(
            timestamp=ts,
            message_header=cls._extract_value(parts[1]),
            gps_week=cls._to_int(parts[2]),
            gps_time=cls._to_float(parts[3]),
            heading_deg=cls._to_float(parts[4]),
            pitch_deg=cls._to_float(parts[5]),
            roll_deg=cls._to_float(parts[6]),
            gyro_x=cls._to_float(parts[7]),
            gyro_y=cls._to_float(parts[8]),
            gyro_z=cls._to_float(parts[9]),
            acc_x=cls._to_float(parts[10]),
            acc_y=cls._to_float(parts[11]),
            acc_z=cls._to_float(parts[12]),
            latitude=cls._to_float(parts[13]),
            longitude=cls._to_float(parts[14]),
            altitude_m=cls._to_float(parts[15]),
            vel_east=cls._to_float(parts[16]),
            vel_north=cls._to_float(parts[17]),
            vel_up=cls._to_float(parts[18]),
            vehicle_speed=cls._to_float(parts[19]),
            main_sat_count=cls._to_int(parts[20]),
            aux_sat_count=cls._to_int(parts[21]),
            system_status=cls._to_int(parts[22]),
            diff_age=cls._to_float(parts[23]),
        )

    @classmethod
    def parse_lines(cls, lines: Iterable[str]) -> list[GnssRecord]:
        records: list[GnssRecord] = []
        for raw in lines:
            if not raw.strip():
                continue
            records.append(cls.parse_line(raw))
        return records

    @classmethod
    def parse_file(cls, path: str | Path, encoding: str = "utf-8") -> list[GnssRecord]:
        file_path = Path(path)
        with file_path.open("r", encoding=encoding) as f:
            return cls.parse_lines(f)

    @staticmethod
    def to_dataframe(records: list[GnssRecord]) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in records])
