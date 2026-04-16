from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..models import LidarPoint


@dataclass(slots=True)
class DecodedMsopPacket:
    timestamp: datetime
    azimuth_deg: float
    points: list[LidarPoint]


class MsopDecoder:
    """LeiShen C32 MSOP decoder (1206-byte payload)."""

    MSOP_PAYLOAD_LENGTH = 1206
    BLOCK_COUNT = 12
    BLOCK_SIZE = 100
    CHANNEL_COUNT = 32
    BLOCK_HEADER = b"\xFF\xEE"

    DEFAULT_VERTICAL_ANGLES_DEG = [
        -16.0,
        -15.0,
        -14.0,
        -13.0,
        -12.0,
        -11.0,
        -10.0,
        -9.0,
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]

    def __init__(
        self,
        vertical_angles_deg: list[float] | None = None,
        distance_resolution_m: float = 0.004,
        min_distance_m: float = 0.05,
        max_distance_m: float = 200.0,
        byteorder: str = "little",
    ) -> None:
        self.vertical_angles_deg = np.asarray(vertical_angles_deg or self.DEFAULT_VERTICAL_ANGLES_DEG, dtype=np.float32)
        if len(self.vertical_angles_deg) != self.CHANNEL_COUNT:
            raise ValueError("vertical_angles_deg must contain 32 entries")
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be 'little' or 'big'")

        self.byteorder = byteorder
        self.distance_resolution_m = float(distance_resolution_m)
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)

        self.vertical_angles_rad = np.deg2rad(self.vertical_angles_deg).astype(np.float32, copy=False)
        self.vertical_sin = np.sin(self.vertical_angles_rad).astype(np.float32, copy=False)
        self.vertical_cos = np.cos(self.vertical_angles_rad).astype(np.float32, copy=False)

    def decode_packet_structured(
        self, payload: bytes, packet_time: datetime
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(payload) != self.MSOP_PAYLOAD_LENGTH:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int16),
            )

        ts_ns = int(packet_time.timestamp() * 1_000_000_000)
        dists: list[float] = []
        intens: list[int] = []
        azis: list[float] = []
        rings: list[int] = []

        blocks = payload[: self.BLOCK_COUNT * self.BLOCK_SIZE]
        for block_idx in range(self.BLOCK_COUNT):
            start = block_idx * self.BLOCK_SIZE
            block = blocks[start : start + self.BLOCK_SIZE]
            if block[0:2] != self.BLOCK_HEADER:
                continue

            azimuth_raw = int.from_bytes(block[2:4], byteorder=self.byteorder, signed=False)
            azimuth_deg = azimuth_raw / 100.0

            channel_data = block[4:]
            for ring in range(self.CHANNEL_COUNT):
                offset = ring * 3
                distance_raw = int.from_bytes(channel_data[offset : offset + 2], byteorder=self.byteorder, signed=False)
                intensity = int(channel_data[offset + 2])

                dists.append(distance_raw * self.distance_resolution_m)
                intens.append(intensity)
                azis.append(azimuth_deg)
                rings.append(ring)

        n = len(dists)
        if n == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int16),
            )

        return (
            np.full(n, ts_ns, dtype=np.int64),
            np.asarray(dists, dtype=np.float32),
            np.asarray(intens, dtype=np.uint8),
            np.asarray(azis, dtype=np.float32),
            np.asarray(rings, dtype=np.int16),
        )


class ProtocolPcapParser:
    """Pure binary protocol parser for LiDAR UDP packets (MSOP-only, GPU-accelerated math)."""

    # Camera / sensor calibration used for visualization alignment.
    DISTORTION_K1 = -0.52939418
    DISTORTION_K2 = 0.37462897
    DISTORTION_P1 = 0.0
    DISTORTION_P2 = 0.0
    DISTORTION_K3 = 0.0

    CAMERA_INTRINSICS = {
        "fx": 2041.23366196,
        "fy": 2038.76906929,
        "cx": 997.68672528,
        "cy": 544.68141109,
    }

    MMWAVE_POS_MM = np.array([0.0, 0.0, 58.5], dtype=np.float64)
    LIDAR_POS_MM = np.array([1.5, -291.8, 187.8], dtype=np.float64)
    CAMERA_POS_MM = np.array([-1.5, -203.8, 169.8], dtype=np.float64)

    FULL_FIELD_COLUMNS = [
        "packet_timestamp",
        "src_port",
        "dst_port",
        "packet_index",
        "block_index",
        "block_flag_hex",
        "azimuth_raw",
        "azimuth_deg",
        "ring",
        "distance_raw",
        "distance_m",
        "intensity",
        "vertical_angle_deg",
        "x",
        "y",
        "z",
    ]
    VALID_FULL_EXPORT_FORMATS = {"parquet", "bin"}

    MSOP_PORT = 2368

    ETHER_HEADER_LEN = 14
    IPV4_HEADER_LEN = 20
    UDP_HEADER_LEN = 8
    UDP_PAYLOAD_OFFSET = ETHER_HEADER_LEN + IPV4_HEADER_LEN + UDP_HEADER_LEN  # 42

    def __init__(
        self,
        decoder: MsopDecoder | None = None,
        payload_byteorder: str = "little",
        debug_stats: bool = False,
        use_cuda: bool = False,
    ) -> None:
        if payload_byteorder not in ("little", "big"):
            raise ValueError("payload_byteorder must be 'little' or 'big'")

        self.decoder = decoder or MsopDecoder(byteorder=payload_byteorder)
        self.debug_stats = debug_stats
        self.use_cuda = use_cuda
        self._stats: dict[str, int] = {}

        self._cuda_available = False
        self._xp = np
        self._cp_module = None
        if self.use_cuda:
            try:
                import cupy as cp  # type: ignore

                self._xp = cp
                self._cp_module = cp
                self._cuda_available = True
            except Exception:
                self._xp = np
                self._cp_module = None
                self._cuda_available = False

        self._stats["cuda_enabled"] = 1 if self._cuda_available else 0
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._stats = {
            "total_udp_packets": 0,
            "hit_2368_packets": 0,
            "payload_len_1206_packets": 0,
            "length_filtered_packets": 0,
            "cuda_enabled": 1 if self._cuda_available else 0,
        }

    def _print_stats(self) -> None:
        print("[ProtocolPcapParser][Stats]")
        print(f"  total UDP packets: {self._stats['total_udp_packets']}")
        print(f"  hit 2368 packets: {self._stats['hit_2368_packets']}")
        print(f"  payload==1206 packets: {self._stats['payload_len_1206_packets']}")
        print(f"  length filtered packets: {self._stats['length_filtered_packets']}")
        print(f"  cuda enabled: {self._stats['cuda_enabled']}")

    @classmethod
    def correct_point_cloud_offset(cls, xyz: np.ndarray) -> np.ndarray:
        """Apply only the sensor-origin offset correction."""
        if len(xyz) == 0:
            return xyz
        xyz = np.asarray(xyz, dtype=np.float64)
        offset_mm = cls.LIDAR_POS_MM - cls.CAMERA_POS_MM
        return xyz - offset_mm / 1000.0

    @classmethod
    def correct_point_cloud_distortion(cls, xyz: np.ndarray, max_iter: int = 5) -> np.ndarray:
        """Apply only lens distortion correction on normalized coordinates."""
        if len(xyz) == 0:
            return xyz
        xyz = np.asarray(xyz, dtype=np.float64)
        corrected = xyz.copy()
        z = corrected[:, 2]
        valid = np.abs(z) > 1e-6
        if not np.any(valid):
            return corrected

        xy_norm = corrected[valid, :2] / z[valid, None]
        x = xy_norm[:, 0]
        y = xy_norm[:, 1]
        x_u = x.copy()
        y_u = y.copy()

        for _ in range(max_iter):
            r2 = x_u * x_u + y_u * y_u
            radial = 1.0 + cls.DISTORTION_K1 * r2 + cls.DISTORTION_K2 * r2 * r2 + cls.DISTORTION_K3 * r2 * r2 * r2
            x_tangential = 2.0 * cls.DISTORTION_P1 * x_u * y_u + cls.DISTORTION_P2 * (r2 + 2.0 * x_u * x_u)
            y_tangential = cls.DISTORTION_P1 * (r2 + 2.0 * y_u * y_u) + 2.0 * cls.DISTORTION_P2 * x_u * y_u
            x_u = (x - x_tangential) / np.maximum(radial, 1e-8)
            y_u = (y - y_tangential) / np.maximum(radial, 1e-8)

        corrected[valid, 0] = x_u * z[valid]
        corrected[valid, 1] = y_u * z[valid]
        return corrected

    @classmethod
    def project_lidar_points_to_image(cls, xyz: np.ndarray) -> np.ndarray:
        """Project LiDAR points into the camera image plane using fx/fy/cx/cy."""
        if len(xyz) == 0:
            return np.empty((0, 2), dtype=np.float64)

        xyz = np.asarray(xyz, dtype=np.float64)
        cam_xyz = cls.correct_point_cloud_offset_distortion(xyz)
        z = cam_xyz[:, 2]
        valid = np.abs(z) > 1e-6
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)

        x = cam_xyz[valid, 0]
        y = cam_xyz[valid, 1]
        z = cam_xyz[valid, 2]
        u = cls.FX * (x / z) + cls.CX
        v = cls.FY * (y / z) + cls.CY
        return np.column_stack((u, v))

    @classmethod
    def correct_point_cloud_offset_distortion(cls, xyz: np.ndarray) -> np.ndarray:
        """Apply offset correction first, then distortion correction."""
        return cls.correct_point_cloud_distortion(cls.correct_point_cloud_offset(xyz))

    @staticmethod
    def _read_exact(f: BinaryIO, size: int) -> bytes:
        data = f.read(size)
        if len(data) != size:
            raise EOFError
        return data

    @staticmethod
    def _detect_pcap_format(global_hdr: bytes) -> tuple[str, bool]:
        magic = global_hdr[:4]

        if magic == b"\x0A\x0D\x0D\x0A":
            raise ValueError("pcapng is not supported; please convert to classic pcap")
        if magic == b"\xD4\xC3\xB2\xA1":
            return "little", False
        if magic == b"\xA1\xB2\xC3\xD4":
            return "big", False
        if magic == b"\x4D\x3C\xB2\xA1":
            return "little", True
        if magic == b"\xA1\xB2\x3C\x4D":
            return "big", True

        raise ValueError("unsupported pcap magic; expected classic pcap file")

    def _parse_udp_from_frame(self, frame: bytes) -> tuple[int, int, bytes] | None:
        if len(frame) < self.UDP_PAYLOAD_OFFSET:
            return None

        if frame[12:14] != b"\x08\x00":     # ipv4协议类型
            return None
        if frame[23] != 17:     # UDP 协议号为 17
            return None

        src_port = int.from_bytes(frame[34:36], "big")
        dst_port = int.from_bytes(frame[36:38], "big")
        udp_len = int.from_bytes(frame[38:40], "big")
        if udp_len < self.UDP_HEADER_LEN:
            return None

        payload_len = udp_len - self.UDP_HEADER_LEN
        payload = frame[self.UDP_PAYLOAD_OFFSET : self.UDP_PAYLOAD_OFFSET + payload_len]

        self._stats["total_udp_packets"] += 1
        return src_port, dst_port, payload

    def iter_udp_packets(self, pcap_path: str | Path) -> Iterator[tuple[datetime, int, int, bytes]]:
        path = Path(pcap_path)
        with path.open("rb") as f:
            global_hdr = f.read(24)
            if len(global_hdr) != 24:
                raise ValueError("invalid pcap file")

            endian, is_nano = self._detect_pcap_format(global_hdr)

            while True:
                pkt_hdr = f.read(16)
                if not pkt_hdr:
                    break
                if len(pkt_hdr) != 16:
                    break

                ts_sec = int.from_bytes(pkt_hdr[0:4], endian, signed=False)
                ts_frac = int.from_bytes(pkt_hdr[4:8], endian, signed=False)
                incl_len = int.from_bytes(pkt_hdr[8:12], endian, signed=False)

                try:
                    frame = self._read_exact(f, incl_len)
                except EOFError:
                    break

                ts_usec = ts_frac / 1_000.0 if is_nano else float(ts_frac)
                pkt_time = datetime.fromtimestamp(ts_sec + ts_usec / 1_000_000, tz=timezone.utc)

                parsed = self._parse_udp_from_frame(frame)
                if parsed is None:
                    continue

                src_port, dst_port, payload = parsed
                yield pkt_time, src_port, dst_port, payload

    def _iter_msop_payloads(self, pcap_path: str | Path) -> Iterator[tuple[datetime, bytes]]:
        for pkt_time, src_port, dst_port, payload in self.iter_udp_packets(pcap_path):
            is_msop = src_port == self.MSOP_PORT or dst_port == self.MSOP_PORT
            if not is_msop:
                continue

            self._stats["hit_2368_packets"] += 1
            if len(payload) == self.decoder.MSOP_PAYLOAD_LENGTH:
                self._stats["payload_len_1206_packets"] += 1
                yield pkt_time, payload
            else:
                self._stats["length_filtered_packets"] += 1

    def _transform_and_filter(
        self,
        ts_ns: np.ndarray,
        distance_m: np.ndarray,
        intensity: np.ndarray,
        azimuth_deg: np.ndarray,
        ring: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xp = self._xp

        d = xp.asarray(distance_m)
        a = xp.asarray(azimuth_deg)
        r = xp.asarray(ring)
        t = xp.asarray(ts_ns)
        it = xp.asarray(intensity)
        v_sin = xp.asarray(self.decoder.vertical_sin)
        v_cos = xp.asarray(self.decoder.vertical_cos)

        mask = (d >= self.decoder.min_distance_m) & (d <= self.decoder.max_distance_m)
        d = d[mask]
        a = a[mask]
        r = r[mask]
        t = t[mask]
        it = it[mask]

        az_rad = a * (math.pi / 180.0)
        cos_v = v_cos[r]
        sin_v = v_sin[r]

        x = d * cos_v * xp.cos(az_rad)
        y = d * cos_v * xp.sin(az_rad)
        z = d * sin_v

        if self._cuda_available and self._cp_module is not None:
            cp = self._cp_module
            return (
                cp.asnumpy(t),
                cp.asnumpy(x),
                cp.asnumpy(y),
                cp.asnumpy(z),
                cp.asnumpy(it),
                cp.asnumpy(r),
            )

        return t, x, y, z, it, r

    def _iter_point_chunks_dataframe(
        self, pcap_path: str | Path, chunk_target_points: int = 2_000_000
    ) -> Iterator[pd.DataFrame]:
        ts_parts: list[np.ndarray] = []
        dist_parts: list[np.ndarray] = []
        inten_parts: list[np.ndarray] = []
        azi_parts: list[np.ndarray] = []
        ring_parts: list[np.ndarray] = []
        chunk_size = 0

        def make_chunk() -> pd.DataFrame | None:
            nonlocal chunk_size
            if not ts_parts:
                return None

            ts_all = np.concatenate(ts_parts)
            dist_all = np.concatenate(dist_parts)
            inten_all = np.concatenate(inten_parts)
            azi_all = np.concatenate(azi_parts)
            ring_all = np.concatenate(ring_parts)

            ts_valid, x, y, z, inten_valid, ring_valid = self._transform_and_filter(
                ts_all, dist_all, inten_all, azi_all, ring_all
            )

            ts_parts.clear()
            dist_parts.clear()
            inten_parts.clear()
            azi_parts.clear()
            ring_parts.clear()
            chunk_size = 0

            if ts_valid.size == 0:
                return None

            return pd.DataFrame(
                {
                    "timestamp_ns": ts_valid,
                    "x": x,
                    "y": y,
                    "z": z,
                    "intensity": inten_valid,
                    "ring": ring_valid,
                }
            )

        for pkt_time, payload in self._iter_msop_payloads(pcap_path):
            ts, dist, inten, azi, ring = self.decoder.decode_packet_structured(payload, pkt_time)
            if ts.size == 0:
                continue

            ts_parts.append(ts)
            dist_parts.append(dist)
            inten_parts.append(inten)
            azi_parts.append(azi)
            ring_parts.append(ring)
            chunk_size += int(ts.size)

            if chunk_size >= chunk_target_points:
                df_chunk = make_chunk()
                if df_chunk is not None:
                    yield df_chunk

        df_chunk = make_chunk()
        if df_chunk is not None:
            yield df_chunk

    def parse_points_dataframe(self, pcap_path: str | Path) -> pd.DataFrame:
        self._reset_stats()
        chunks: list[pd.DataFrame] = []
        for chunk in self._iter_point_chunks_dataframe(pcap_path):
            chunks.append(chunk)

        if not chunks:
            if self.debug_stats:
                self._print_stats()
            return pd.DataFrame(columns=["timestamp_ns", "x", "y", "z", "intensity", "ring"])

        df = pd.concat(chunks, ignore_index=True)
        if self.debug_stats:
            self._print_stats()
        return df

    def parse_points(self, pcap_path: str | Path) -> list[LidarPoint]:
        points: list[LidarPoint] = []
        for df in self._iter_point_chunks_dataframe(pcap_path):
            for row in df.itertuples(index=False):
                points.append(
                    LidarPoint(
                        timestamp=datetime.fromtimestamp(int(row.timestamp_ns) / 1_000_000_000, tz=timezone.utc),
                        x=float(row.x),
                        y=float(row.y),
                        z=float(row.z),
                        intensity=int(row.intensity),
                        ring=int(row.ring),
                    )
                )
        if self.debug_stats:
            self._print_stats()
        return points

    def export_bin_from_points_df(self, df: pd.DataFrame, out_bin: str | Path, mode: str = "kitti") -> Path:
        """Export point dataframe to binary.

        mode='kitti': each point [x, y, z, intensity] float32
        mode='compact': each point [x, y, z] float32 + intensity uint16 + ring uint16
        """
        out = Path(out_bin)
        out.parent.mkdir(parents=True, exist_ok=True)

        if df.empty:
            np.empty((0, 4), dtype=np.float32).tofile(out)
            return out

        x = df["x"].to_numpy(dtype=np.float32, copy=False)
        y = df["y"].to_numpy(dtype=np.float32, copy=False)
        z = df["z"].to_numpy(dtype=np.float32, copy=False)

        if mode == "kitti":
            intensity_u8 = df["intensity"].to_numpy(dtype=np.uint8, copy=False)
            intensity_f32 = (intensity_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)
            arr = np.column_stack((x, y, z, intensity_f32)).astype(np.float32, copy=False)
            arr.tofile(out)
            return out

        if mode == "compact":
            intensity = df["intensity"].to_numpy(dtype=np.uint16, copy=False)
            ring = df["ring"].to_numpy(dtype=np.uint16, copy=False)
            rec = np.rec.fromarrays(
                [x, y, z, intensity, ring],
                names=["x", "y", "z", "intensity", "ring"],
                formats=["<f4", "<f4", "<f4", "<u2", "<u2"],
            )
            rec.tofile(out)
            return out

        raise ValueError("mode must be 'kitti' or 'compact'")

    def _append_columns_to_bin(
        self,
        out_bin_path: Path,
        bin_mode: str,
        x_vals: list[float],
        y_vals: list[float],
        z_vals: list[float],
        intensity_vals: list[int],
        packet_index_vals: list[int],
        block_index_vals: list[int],
        src_port_vals: list[int],
        dst_port_vals: list[int],
        azimuth_raw_vals: list[int],
        azimuth_deg_vals: list[float],
        ring_vals: list[int],
        distance_raw_vals: list[int],
        distance_m_vals: list[float],
        vertical_angle_deg_vals: list[float],
    ) -> None:
        n = len(x_vals)
        if n == 0:
            return

        if bin_mode == "kitti":
            arr = np.empty((n, 4), dtype=np.float32)
            arr[:, 0] = np.asarray(x_vals, dtype=np.float32)
            arr[:, 1] = np.asarray(y_vals, dtype=np.float32)
            arr[:, 2] = np.asarray(z_vals, dtype=np.float32)
            arr[:, 3] = np.asarray(intensity_vals, dtype=np.float32) / 255.0
            with out_bin_path.open("ab") as f:
                arr.tofile(f)
            return

        if bin_mode == "full_float64":
            arr = np.empty((n, 14), dtype=np.float64)
            arr[:, 0] = np.asarray(packet_index_vals, dtype=np.float64)
            arr[:, 1] = np.asarray(block_index_vals, dtype=np.float64)
            arr[:, 2] = np.asarray(src_port_vals, dtype=np.float64)
            arr[:, 3] = np.asarray(dst_port_vals, dtype=np.float64)
            arr[:, 4] = np.asarray(azimuth_raw_vals, dtype=np.float64)
            arr[:, 5] = np.asarray(azimuth_deg_vals, dtype=np.float64)
            arr[:, 6] = np.asarray(ring_vals, dtype=np.float64)
            arr[:, 7] = np.asarray(distance_raw_vals, dtype=np.float64)
            arr[:, 8] = np.asarray(distance_m_vals, dtype=np.float64)
            arr[:, 9] = np.asarray(intensity_vals, dtype=np.float64)
            arr[:, 10] = np.asarray(vertical_angle_deg_vals, dtype=np.float64)
            arr[:, 11] = np.asarray(x_vals, dtype=np.float64)
            arr[:, 12] = np.asarray(y_vals, dtype=np.float64)
            arr[:, 13] = np.asarray(z_vals, dtype=np.float64)
            with out_bin_path.open("ab") as f:
                arr.tofile(f)
            return

        raise ValueError("bin_mode must be 'kitti' or 'full_float64'")

    def export_full_fields(
        self,
        pcap_path: str | Path,
        out_stem: str | Path,
        formats: tuple[str, ...] = ("parquet", "bin"),
        bin_mode: str = "kitti",
        flush_points: int = 500_000,
    ) -> dict[str, Path]:
        fmt_set = {f.lower() for f in formats}
        invalid = fmt_set - self.VALID_FULL_EXPORT_FORMATS
        if invalid:
            raise ValueError(f"unsupported formats: {sorted(invalid)}")

        stem = Path(out_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)

        outputs: dict[str, Path] = {}
        out_parquet_path = stem.with_suffix(".parquet") if "parquet" in fmt_set else None
        out_bin_path = stem.with_suffix(".bin") if "bin" in fmt_set else None

        if out_parquet_path is not None:
            outputs["parquet"] = out_parquet_path
        if out_bin_path is not None:
            outputs["bin"] = out_bin_path
            out_bin_path.write_bytes(b"")

        parquet_writer: pq.ParquetWriter | None = None

        packet_timestamp_ns_vals: list[int] = []
        src_port_vals: list[int] = []
        dst_port_vals: list[int] = []
        packet_index_vals: list[int] = []
        block_index_vals: list[int] = []
        block_flag_hex_vals: list[str] = []
        azimuth_raw_vals: list[int] = []
        azimuth_deg_vals: list[float] = []
        ring_vals: list[int] = []
        distance_raw_vals: list[int] = []
        distance_m_vals: list[float] = []
        intensity_vals: list[int] = []
        vertical_angle_deg_vals: list[float] = []
        x_vals: list[float] = []
        y_vals: list[float] = []
        z_vals: list[float] = []

        def clear_buffers() -> None:
            packet_timestamp_ns_vals.clear()
            src_port_vals.clear()
            dst_port_vals.clear()
            packet_index_vals.clear()
            block_index_vals.clear()
            block_flag_hex_vals.clear()
            azimuth_raw_vals.clear()
            azimuth_deg_vals.clear()
            ring_vals.clear()
            distance_raw_vals.clear()
            distance_m_vals.clear()
            intensity_vals.clear()
            vertical_angle_deg_vals.clear()
            x_vals.clear()
            y_vals.clear()
            z_vals.clear()

        def flush_buffers() -> None:
            nonlocal parquet_writer
            n = len(x_vals)
            if n == 0:
                return

            if out_parquet_path is not None:
                packet_ts_str = pd.to_datetime(np.asarray(packet_timestamp_ns_vals, dtype=np.int64), unit="ns", utc=True).astype(str)
                df_chunk = pd.DataFrame(
                    {
                        "packet_timestamp": packet_ts_str,
                        "src_port": np.asarray(src_port_vals, dtype=np.int32),
                        "dst_port": np.asarray(dst_port_vals, dtype=np.int32),
                        "packet_index": np.asarray(packet_index_vals, dtype=np.int64),
                        "block_index": np.asarray(block_index_vals, dtype=np.int16),
                        "block_flag_hex": block_flag_hex_vals,
                        "azimuth_raw": np.asarray(azimuth_raw_vals, dtype=np.int32),
                        "azimuth_deg": np.asarray(azimuth_deg_vals, dtype=np.float32),
                        "ring": np.asarray(ring_vals, dtype=np.int16),
                        "distance_raw": np.asarray(distance_raw_vals, dtype=np.int32),
                        "distance_m": np.asarray(distance_m_vals, dtype=np.float32),
                        "intensity": np.asarray(intensity_vals, dtype=np.uint8),
                        "vertical_angle_deg": np.asarray(vertical_angle_deg_vals, dtype=np.float32),
                        "x": np.asarray(x_vals, dtype=np.float32),
                        "y": np.asarray(y_vals, dtype=np.float32),
                        "z": np.asarray(z_vals, dtype=np.float32),
                    },
                    columns=self.FULL_FIELD_COLUMNS,
                )
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(out_parquet_path, table.schema)
                parquet_writer.write_table(table)

            if out_bin_path is not None:
                self._append_columns_to_bin(
                    out_bin_path=out_bin_path,
                    bin_mode=bin_mode,
                    x_vals=x_vals,
                    y_vals=y_vals,
                    z_vals=z_vals,
                    intensity_vals=intensity_vals,
                    packet_index_vals=packet_index_vals,
                    block_index_vals=block_index_vals,
                    src_port_vals=src_port_vals,
                    dst_port_vals=dst_port_vals,
                    azimuth_raw_vals=azimuth_raw_vals,
                    azimuth_deg_vals=azimuth_deg_vals,
                    ring_vals=ring_vals,
                    distance_raw_vals=distance_raw_vals,
                    distance_m_vals=distance_m_vals,
                    vertical_angle_deg_vals=vertical_angle_deg_vals,
                )

            clear_buffers()

        packet_index = 0
        try:
            for pkt_time, src_port, dst_port, payload in self.iter_udp_packets(pcap_path):
                packet_index += 1

                is_msop = src_port == self.MSOP_PORT or dst_port == self.MSOP_PORT
                if not is_msop or len(payload) != self.decoder.MSOP_PAYLOAD_LENGTH:
                    continue

                ts_ns = int(pkt_time.timestamp() * 1_000_000_000)
                blocks = payload[: self.decoder.BLOCK_COUNT * self.decoder.BLOCK_SIZE]

                for block_idx in range(self.decoder.BLOCK_COUNT):
                    start = block_idx * self.decoder.BLOCK_SIZE
                    block = blocks[start : start + self.decoder.BLOCK_SIZE]
                    block_flag = block[0:2]
                    if block_flag != self.decoder.BLOCK_HEADER:
                        continue

                    azimuth_raw = int.from_bytes(block[2:4], byteorder=self.decoder.byteorder, signed=False)
                    azimuth_deg = azimuth_raw / 100.0
                    az_rad = math.radians(azimuth_deg)
                    cos_az = math.cos(az_rad)
                    sin_az = math.sin(az_rad)
                    block_flag_hex = block_flag.hex()

                    channel_data = block[4:]
                    for ring in range(self.decoder.CHANNEL_COUNT):
                        offset = ring * 3
                        distance_raw = int.from_bytes(
                            channel_data[offset : offset + 2],
                            byteorder=self.decoder.byteorder,
                            signed=False,
                        )
                        intensity = int(channel_data[offset + 2])
                        distance_m = distance_raw * self.decoder.distance_resolution_m

                        if distance_m < self.decoder.min_distance_m or distance_m > self.decoder.max_distance_m:
                            continue

                        vertical_angle_deg = float(self.decoder.vertical_angles_deg[ring])
                        cos_v = float(self.decoder.vertical_cos[ring])
                        sin_v = float(self.decoder.vertical_sin[ring])

                        x = distance_m * cos_v * cos_az
                        y = distance_m * cos_v * sin_az
                        z = distance_m * sin_v

                        packet_timestamp_ns_vals.append(ts_ns)
                        src_port_vals.append(int(src_port))
                        dst_port_vals.append(int(dst_port))
                        packet_index_vals.append(packet_index)
                        block_index_vals.append(block_idx)
                        block_flag_hex_vals.append(block_flag_hex)
                        azimuth_raw_vals.append(azimuth_raw)
                        azimuth_deg_vals.append(azimuth_deg)
                        ring_vals.append(ring)
                        distance_raw_vals.append(distance_raw)
                        distance_m_vals.append(distance_m)
                        intensity_vals.append(intensity)
                        vertical_angle_deg_vals.append(vertical_angle_deg)
                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)

                if len(x_vals) >= flush_points:
                    flush_buffers()

            flush_buffers()

            if out_parquet_path is not None and parquet_writer is None:
                empty_df = pd.DataFrame(columns=self.FULL_FIELD_COLUMNS)
                empty_df.to_parquet(out_parquet_path, index=False)

        finally:
            if parquet_writer is not None:
                parquet_writer.close()

        return outputs

    def export_parse_points(self, pcap_path: str | Path, out_path: str | Path, fmt: str = "parquet") -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fmt_norm = fmt.lower()

        if fmt_norm != "parquet":
            raise ValueError("fmt must be 'parquet'")

        self._reset_stats()
        writer: pq.ParquetWriter | None = None
        try:
            for df_chunk in self._iter_point_chunks_dataframe(pcap_path):
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
                del df_chunk
            if writer is None:
                empty_df = pd.DataFrame(columns=["timestamp_ns", "x", "y", "z", "intensity", "ring"])
                empty_df.to_parquet(out, index=False)
            if self.debug_stats:
                self._print_stats()
        finally:
            if writer is not None:
                writer.close()

        return out
