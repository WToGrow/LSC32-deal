from __future__ import annotations

"""PyQt 实时视频-LiDAR 投影外参调参工具。

特性
----
1. 读取视频 + parquet 点云，按 frame timestamp 做时间窗口匹配。
2. 主界面实时显示“点云投影到图像”效果。
3. 顶部 6 个滑动条控制外参：平移 tx/ty/tz（cm）+ 旋转 rx/ry/rz（度）。
4. 投影点颜色按距离做亮色渐变（近 -> 亮黄，远 -> 亮蓝）。
5. 可暂停/继续播放，方便精调。

说明
----
- 该工具仿照 `vl_projection.py` 的流程：
  时间对齐 -> 坐标变换 -> 投影 -> 绘制。
- 这里默认旋转矩阵沿用你当前脚本中的 `R_LIDAR_TO_CAM`。
- 通过滑动 tx/ty/tz 可以快速找到更合适的平移外参。
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from PyQt5.QtCore import QSignalBlocker, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# =========================
# 相机参数（与你当前脚本一致）
# =========================
CAMERA_MATRIX = np.array(
    [
        [2041.23366196, 0.0, 997.68672528],
        [0.0, 2038.76906929, 544.68141109],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DIST_COEFFS = np.array([-0.52939418, 0.37462897, 0.0, 0.0, 0.0], dtype=np.float32)

# 设备在世界坐标系中的位置（单位：cm）
# 你给出的相对位置可用于定义更合理的轴过滤阈值：
# - 激光雷达：(1.5, -291.8, 187.8)
# - 正前参考点：(-1.5, -203.8, 169.8)
#
# 由此可得到：
# - X 方向中线约为 x = 0.0
# - Y 方向可用“雷达到正前参考点”的中点作为分界，更接近前后区域
# - Z 方向可用两者中点作为上下区域分界
LIDAR_WORLD_POS = np.array([1.5, -291.8, 187.8], dtype=np.float32)
FRONT_REF_WORLD_POS = np.array([-1.5, -203.8, 169.8], dtype=np.float32)
AXIS_FILTER_THRESHOLDS = {
    "x_mid": -18.0,
    "y_mid": 88.0,
    "z_mid": -3.0,
}

# 轴映射预设（用于“轴对照测试”）
#
# 你现在已经确认：真实装配里并不一定是“单纯的 1 / -1 轴对齐”。
# 因此这里把候选矩阵做成一个可扩展字典，你可以继续往里加你自己的猜测矩阵。
#
# 约定：
# - 左边注释写雷达轴定义
# - 右边注释写相机轴定义
# - 矩阵的每一行表示相机坐标系的一个轴由雷达坐标系哪一轴组成
#
# 例如：
# [[1,0,0],[0,0,-1],[0,1,0]]
# 表示：
#   camera_x = lidar_x
#   camera_y = -lidar_z
#   camera_z = lidar_y
R_AXIS_PRESETS: dict[str, np.ndarray] = {
    # 0) 默认候选：雷达 x 右、y 前、z 上 -> 相机 x 右、y 下、z 前
    "preset_0_x_right_y_front_z_up": np.array(
        [[-0.703847, 0.710352, 0.000000],
        [-0.080507, -0.079770, -0.993557],
        [-0.705775, -0.699312, 0.113334],],
        dtype=np.float32,
    ),
    # 1) 雷达 x 前、y 左、z 上 -> 相机 x 右、y 下、z 前
    "preset_1_x_front_y_left_z_up": np.array(
        [[-0.710413, 0.703785, 0],
         [-0.086109, -0.086920, -0.992487],
         [-0.698497, -0.705076, 0.122351]],
        dtype=np.float32,
    ),
    # 2) 雷达 x 右、y 上、z 前 -> 相机 x 右、y 下、z 前
    "preset_2_x_right_y_up_z_front": np.array(
        [[-0.640407, 0.767508, 0.028454],
         [-0.047300, 0.002436, 0.99878],
         [-0.766578, -0.641035, 0.037864]],
        dtype=np.float32,
    ),
    # 3) 雷达 x 左、y 前、z 上 -> 相机 x 右、y 下、z 前
    "preset_3_x_left_y_front_z_up": np.array(
        [[-0.747909, 0.663802, 0],
         [-0.082236, -0.092656, -0.992296],
         [-0.658688, -0.742147, 0.123886]],
        dtype=np.float32,
    ),
    # 4) x修正水平后
    "preset_4_x_front_y_right_z_up": np.array(
        [[-0.629838, 0.776726, 0.000000],
        [0.004426, 0.003589, -0.999984],
        [-0.776714, -0.629828, -0.005698],],
        dtype=np.float32,
    ),
    # 5) 雷达 x 右、y 前、z 下 -> 相机 x 右、y 上、z 前
    "preset_5_x_right_y_front_z_down": np.array(
        [[ 0.771649, 0.634748, -0.038152],
        [ 0.064297, -0.011576, -0.997853],
        [ 0.632375, -0.772414, 0.052205]],
        dtype=np.float32,
    ),
}

DEFAULT_R_LIDAR_TO_CAM = R_AXIS_PRESETS["preset_0_x_right_y_front_z_up"]


@dataclass
class PointCloudBuffer:
    timestamps_ns: np.ndarray
    xyz: np.ndarray


def load_point_cloud(parquet_path: Path) -> PointCloudBuffer:
    """读取 parquet，并按 timestamp_ns 排序。"""
    pf = pq.ParquetFile(parquet_path)
    parts: list[pd.DataFrame] = []

    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        if len(df) == 0:
            continue
        required = ["timestamp_ns", "x", "y", "z"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"parquet 缺少字段: {missing}")
        parts.append(df[["timestamp_ns", "x", "y", "z"]].copy())

    if not parts:
        raise ValueError(f"parquet 中没有可用点云数据: {parquet_path}")

    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values("timestamp_ns", kind="mergesort").reset_index(drop=True)

    ts = all_df["timestamp_ns"].to_numpy(dtype=np.int64, copy=True)
    xyz = all_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    # 0----z
    # 1----x
    # 2----y
    xyz[:, 1] = -xyz[:, 1]
    # xyz[:, 0] = -xyz[:, 0]
    # xyz[:, 2] = -xyz[:, 2]

    return PointCloudBuffer(timestamps_ns=ts, xyz=xyz)


def frame_time_ns(frame_idx: int, fps: float, video_start_ns: int) -> int:
    """把帧序号换算为时间戳（ns）。"""
    return int(video_start_ns + round(frame_idx * 1_000_000_000.0 / fps))


def points_in_time_window(
    pc: PointCloudBuffer,
    center_ns: int,
    half_window_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按帧中心时间取时间窗口内的点。"""
    left = int(np.searchsorted(pc.timestamps_ns, center_ns - half_window_ns, side="left"))
    right = int(np.searchsorted(pc.timestamps_ns, center_ns + half_window_ns, side="right"))

    if right <= left:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return pc.xyz[left:right], pc.timestamps_ns[left:right]


def axis_angle_deg_to_rotmat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """绕任意轴的轴角旋转矩阵。"""
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or abs(angle_deg) < 1e-8:
        return np.eye(3, dtype=np.float32)

    x, y, z = axis / norm
    theta = np.deg2rad(angle_deg).astype(np.float32)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    one_c = 1.0 - c

    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def body_xyz_deg_to_rotmat(rx_deg: float, ry_deg: float, rz_deg: float, basis: np.ndarray) -> np.ndarray:
    """绕当前外参矩阵定义的三根轴做本体旋转。"""
    basis = np.asarray(basis, dtype=np.float32).reshape(3, 3)
    rx_m = axis_angle_deg_to_rotmat(basis[0], rx_deg)
    ry_m = axis_angle_deg_to_rotmat(basis[1], ry_deg)
    rz_m = axis_angle_deg_to_rotmat(basis[2], rz_deg)
    return rz_m @ ry_m @ rx_m


def apply_axis_filter(points: np.ndarray, axis_mode: str, basis: np.ndarray, frame_mode: str = "camera") -> np.ndarray:
    """按当前外参轴定义的阈值过滤点。

    frame_mode:
    - camera: 使用相机坐标系下的轴向量做过滤
    - lidar: 使用雷达坐标系下的轴向量做过滤
    """
    if len(points) == 0 or axis_mode == "all":
        return points

    basis = np.asarray(basis, dtype=np.float32).reshape(3, 3)
    if frame_mode == "lidar":
        axis_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis_x = basis[0]
        axis_y = basis[1]
        axis_z = basis[2]

    def _signed_distance(axis_vec: np.ndarray) -> np.ndarray:
        axis_vec = axis_vec.reshape(3)
        norm = float(np.linalg.norm(axis_vec))
        if norm < 1e-8:
            return np.zeros((len(points),), dtype=np.float32)
        return (points @ (axis_vec / norm)).astype(np.float32)

    def _keep_by_threshold(axis_vec: np.ndarray, threshold: float, positive: bool) -> np.ndarray:
        scores = _signed_distance(axis_vec)
        return points[scores >= threshold] if positive else points[scores < threshold]

    # 依据雷达与正前参考点的相对位置，给出更贴近实际装配的过滤阈值
    x_mid = AXIS_FILTER_THRESHOLDS["x_mid"]
    y_mid = AXIS_FILTER_THRESHOLDS["y_mid"]
    z_mid = AXIS_FILTER_THRESHOLDS["z_mid"]

    if axis_mode == "x_pos":
        return _keep_by_threshold(axis_x, x_mid, True)
    if axis_mode == "x_neg":
        return _keep_by_threshold(axis_x, x_mid, False)
    if axis_mode == "y_pos":
        return _keep_by_threshold(axis_y, y_mid, True)
    if axis_mode == "y_neg":
        return _keep_by_threshold(axis_y, y_mid, False)
    if axis_mode == "z_pos":
        return _keep_by_threshold(axis_z, z_mid, True)
    if axis_mode == "z_neg":
        return _keep_by_threshold(axis_z, z_mid, False)
    return points


def project_points(
    points_world: np.ndarray,
    tx_cm: float,
    ty_cm: float,
    tz_cm: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    r_lidar_to_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """世界点 -> 相机点 -> 像素。返回 (img_pts, depth)。"""
    if len(points_world) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    t_cam = np.array([[tx_cm], [ty_cm], [tz_cm]], dtype=np.float32)
    basis = np.asarray(r_lidar_to_cam, dtype=np.float32)
    r_tune = body_xyz_deg_to_rotmat(rx_deg, ry_deg, rz_deg, basis)
    # 让 rx/ry/rz 跟随当前外参矩阵定义的三根轴进行旋转
    pts_lidar_rot = (r_tune @ points_world.T).T
    pts_cam = (r_lidar_to_cam @ pts_lidar_rot.T).T + t_cam.reshape(1, 3)
    front_mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[front_mask]
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # 已在相机坐标系，rvec/tvec 设 0
    img_pts, _ = cv2.projectPoints(
        pts_cam,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        CAMERA_MATRIX,
        DIST_COEFFS,
    )
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    depth = pts_cam[:, 2].astype(np.float32)
    return img_pts, depth


def clip_points(points_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(points_uv) == 0:
        return np.empty((0,), dtype=bool)
    return (
        np.isfinite(points_uv).all(axis=1)
        & (points_uv[:, 0] >= 0)
        & (points_uv[:, 0] < width)
        & (points_uv[:, 1] >= 0)
        & (points_uv[:, 1] < height)
    )


def _bright_colormap(values: np.ndarray) -> np.ndarray:
    """Map normalized values to a bright palette for dark backgrounds."""
    palette = np.array(
        [
            [0.000000, 0.850000, 1.000000],  # cyan
            [0.000000, 0.650000, 0.950000],  # sky blue
            [0.000000, 0.900000, 0.450000],  # green
            [0.250000, 0.950000, 0.150000],  # lime
            [0.950000, 0.850000, 0.000000],  # yellow
            [1.000000, 0.560000, 0.000000],  # orange
            [1.000000, 0.250000, 0.000000],  # red-orange
        ],
        dtype=np.float64,
    )
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (len(palette) - 1)
    left = np.floor(scaled).astype(int)
    right = np.clip(left + 1, 0, len(palette) - 1)
    weight = scaled - left
    colors = palette[left] * (1.0 - weight[:, None]) + palette[right] * weight[:, None]
    return np.clip(colors * 1.08, 0.0, 1.0)


def _normalize_with_percentile(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip outliers with percentiles to keep colors stable."""
    if len(values) == 0:
        return values
    v_min, v_max = np.percentile(values, [low, high])
    if np.isclose(v_min, v_max):
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - v_min) / (v_max - v_min), 0.0, 1.0)


def color_by_depth(depth: np.ndarray) -> np.ndarray:
    """按距离使用亮色渐变配色，风格与 visualize_single.py 一致。"""
    if len(depth) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    feature = _normalize_with_percentile(depth)
    colors_rgb = _bright_colormap(feature)
    colors_bgr = (colors_rgb[:, ::-1] * 255.0).astype(np.uint8)
    return colors_bgr


class ProjectionTunerWindow(QMainWindow):
    def __init__(
        self,
        video_path: Path,
        parquet_path: Path,
        time_window_ms: float = 100.0,
        video_start_ns: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR-Video 投影外参调参器 (tx, ty, tz, rx, ry, rz)")

        self.video_path = video_path
        self.parquet_path = parquet_path
        self.time_window_ns = int(time_window_ms * 1_000_000)

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self.pc = load_point_cloud(parquet_path)
        self.video_start_ns = (
            int(video_start_ns)
            if video_start_ns is not None
            else int(self.pc.timestamps_ns[0] + 10_000_000_000)
        )

        self.current_frame_idx = 0
        self.playing = True
        self.follow_playback = True
        self.follow_playback = True

        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0
        self.axis_mode = "all"
        self.r_lidar_to_cam = DEFAULT_R_LIDAR_TO_CAM.copy()
        self.test_point_mode = True
        self.matrix_inputs: list[list[QDoubleSpinBox]] = []
        self.matrix_mode = "preset"
        self.matrix_updating = False
        self.threshold_inputs: dict[str, QDoubleSpinBox] = {}
        self.threshold_frame_mode = "camera"

        self._build_ui()

        interval_ms = max(1, int(round(1000.0 / self.fps)))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(interval_ms)

    def closeEvent(self, event) -> None:
        try:
            self.cap.release()
        finally:
            super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        # 上方外参滑动轴（平移 + 旋转）
        slider_panel = QWidget(root)
        slider_grid = QGridLayout(slider_panel)

        self.tx_slider, self.tx_spin = self._make_slider_row(slider_grid, 0, "tx (cm)", -50000, 50000, -4.91)
        self.ty_slider, self.ty_spin = self._make_slider_row(slider_grid, 1, "ty (cm)", -50000, 50000, 0)
        self.tz_slider, self.tz_spin = self._make_slider_row(slider_grid, 2, "tz (cm)", -50000, 50000, 39.31)

        self.rx_slider, self.rx_spin = self._make_slider_row(slider_grid, 3, "rx 绕雷达 x 轴 (deg)", -18000, 18000, 0)
        self.ry_slider, self.ry_spin = self._make_slider_row(slider_grid, 4, "ry 绕雷达 y 轴 (deg)", -18000, 18000, 0)
        self.rz_slider, self.rz_spin = self._make_slider_row(slider_grid, 5, "rz 绕雷达 z 轴 (deg)", -18000, 18000, 0)

        layout.addWidget(slider_panel)

        # 播放控制
        controls = QHBoxLayout()
        self.play_btn = QPushButton("暂停", root)
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)

        self.follow_btn = QPushButton("跟随播放对齐", root)
        self.follow_btn.setCheckable(True)
        self.follow_btn.setChecked(True)
        self.follow_btn.toggled.connect(self._on_follow_toggled)
        controls.addWidget(self.follow_btn)

        self.reset_btn = QPushButton("重置外参", root)
        self.reset_btn.clicked.connect(self._reset_extrinsic)
        controls.addWidget(self.reset_btn)

        self.open_video_btn = QPushButton("切换视频", root)
        self.open_video_btn.clicked.connect(self._select_video)
        controls.addWidget(self.open_video_btn)

        self.axis_preset = QComboBox(root)
        self.axis_preset.addItems(list(R_AXIS_PRESETS.keys()))
        self.axis_preset.setCurrentText("preset_0_x_right_y_front_z_up")
        self.axis_preset.currentIndexChanged.connect(self._on_axis_preset_change)
        self.axis_preset.setMaximumWidth(280)
        controls.addWidget(self.axis_preset)

        self.axis_filter = QComboBox(root)
        self.axis_filter.addItems([
            "显示全部点",
            f"仅保留 X 正半轴 (阈值 >= {AXIS_FILTER_THRESHOLDS['x_mid']:.2f})",
            f"仅保留 X 负半轴 (阈值 < {AXIS_FILTER_THRESHOLDS['x_mid']:.2f})",
            f"仅保留 Y 正半轴 (阈值 >= {AXIS_FILTER_THRESHOLDS['y_mid']:.2f})",
            f"仅保留 Y 负半轴 (阈值 < {AXIS_FILTER_THRESHOLDS['y_mid']:.2f})",
            f"仅保留 Z 正半轴 (阈值 >= {AXIS_FILTER_THRESHOLDS['z_mid']:.2f})",
            f"仅保留 Z 负半轴 (阈值 < {AXIS_FILTER_THRESHOLDS['z_mid']:.2f})",
        ])
        self.axis_filter.currentIndexChanged.connect(self._on_axis_filter_change)
        self.axis_filter.setMaximumWidth(240)
        controls.addWidget(self.axis_filter)

        self.test_point_btn = QPushButton("切换单点测试", root)
        self.test_point_btn.setCheckable(True)
        self.test_point_btn.setChecked(True)
        self.test_point_btn.toggled.connect(self._on_test_point_mode_change)
        controls.addWidget(self.test_point_btn)

        self.custom_matrix_btn = QPushButton("应用自定义矩阵", root)
        self.custom_matrix_btn.clicked.connect(self._apply_custom_matrix)
        self.custom_matrix_btn.setMaximumWidth(140)
        controls.addWidget(self.custom_matrix_btn)

        self.matrix_lock_btn = QPushButton("锁定正交归一", root)
        self.matrix_lock_btn.setCheckable(True)
        self.matrix_lock_btn.setChecked(True)
        self.matrix_lock_btn.toggled.connect(self._on_matrix_lock_toggled)
        self.matrix_lock_btn.setMaximumWidth(140)
        controls.addWidget(self.matrix_lock_btn)

        self.info_label = QLabel(root)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumWidth(520)
        controls.addWidget(self.info_label, 1)

        layout.addLayout(controls)

        self.image_label = QLabel(root)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setSizePolicy(self.image_label.sizePolicy().horizontalPolicy(), self.image_label.sizePolicy().verticalPolicy())
        layout.addWidget(self.image_label, 1)

        self.setCentralWidget(root)

        self._build_custom_matrix_panel(layout)
        self._build_axis_threshold_panel(layout)
        self._render_current_frame()

    def _build_custom_matrix_panel(self, layout: QVBoxLayout) -> None:
        panel = QWidget(self)
        grid = QGridLayout(panel)
        self.matrix_inputs = []
        for r in range(3):
            row_inputs: list[QDoubleSpinBox] = []
            for c in range(3):
                spin = QDoubleSpinBox(panel)
                spin.setDecimals(6)
                spin.setRange(-10.0, 10.0)
                spin.setSingleStep(0.001)
                spin.setValue(float(self.r_lidar_to_cam[r, c]))
                spin.valueChanged.connect(lambda _v, rr=r, cc=c: self._on_matrix_cell_changed(rr, cc))
                row_inputs.append(spin)
                grid.addWidget(spin, r, c)
            self.matrix_inputs.append(row_inputs)
        self.load_preset_btn = QPushButton("从当前预设加载到输入框", panel)
        self.load_preset_btn.clicked.connect(self._load_current_preset_to_inputs)
        self.apply_custom_btn = QPushButton("应用输入框矩阵", panel)
        self.apply_custom_btn.clicked.connect(self._apply_custom_matrix)
        grid.addWidget(self.load_preset_btn, 3, 0, 1, 2)
        grid.addWidget(self.apply_custom_btn, 3, 2, 1, 1)
        layout.addWidget(panel)

    def _build_axis_threshold_panel(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("轴过滤阈值（可手动修改）", self)
        grid = QGridLayout(group)
        self.threshold_inputs = {}

        self.threshold_mode = QComboBox(group)
        self.threshold_mode.addItems(["按相机坐标系过滤", "按雷达坐标系过滤"])
        self.threshold_mode.setCurrentIndex(0)
        self.threshold_mode.currentIndexChanged.connect(self._on_threshold_mode_changed)
        grid.addWidget(QLabel("过滤模式", group), 0, 0)
        grid.addWidget(self.threshold_mode, 0, 1)

        for row, axis in enumerate(["x", "y", "z"], start=1):
            label = QLabel(f"{axis}_mid", group)
            spin = QDoubleSpinBox(group)
            spin.setDecimals(2)
            spin.setRange(-10000.0, 10000.0)
            spin.setSingleStep(0.5)
            spin.setValue(float(AXIS_FILTER_THRESHOLDS[f"{axis}_mid"]))
            self.threshold_inputs[axis] = spin
            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)

        hint = QLabel("说明：这些阈值用于 X/Y/Z 正负半轴过滤的分界点。", group)
        hint.setWordWrap(True)
        grid.addWidget(hint, 4, 0, 1, 2)

        layout.addWidget(group)

    def _make_slider_row(
        self,
        grid: QGridLayout,
        row: int,
        name: str,
        mn: int,
        mx: int,
        default: float,
    ) -> tuple[QSlider, QDoubleSpinBox]:
        label = QLabel(name)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(int(round(default * 100)))
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setSingleStep(0.01)
        spin.setRange(mn / 100.0, mx / 100.0)
        spin.setValue(float(default))

        slider.valueChanged.connect(lambda v: spin.setValue(v / 100.0))
        spin.valueChanged.connect(lambda v: slider.setValue(int(round(v * 100))))
        slider.valueChanged.connect(self._on_extrinsic_change)

        grid.addWidget(label, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(spin, row, 2)
        return slider, spin

    def _on_axis_preset_change(self) -> None:
        preset_name = self.axis_preset.currentText().strip()
        if not preset_name:
            return
        if preset_name not in R_AXIS_PRESETS:
            return
        self.matrix_mode = "preset"
        self.r_lidar_to_cam = R_AXIS_PRESETS[preset_name].copy()
        self._load_current_preset_to_inputs()
        if not self.playing:
            self._render_current_frame()

    def _load_current_preset_to_inputs(self) -> None:
        self._set_matrix_inputs(self.r_lidar_to_cam)

    def _block_matrix_signals(self, blocked: bool) -> None:
        for row in self.matrix_inputs:
            for spin in row:
                spin.blockSignals(blocked)

    def _set_matrix_inputs(self, matrix: np.ndarray) -> None:
        self._block_matrix_signals(True)
        try:
            for r in range(3):
                for c in range(3):
                    self.matrix_inputs[r][c].setValue(float(matrix[r, c]))
        finally:
            self._block_matrix_signals(False)

    def _orthonormalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        x = matrix[0].astype(np.float32)
        y = matrix[1].astype(np.float32)

        def _norm(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return v if n < 1e-8 else v / n

        x = _norm(x)
        y = y - np.dot(y, x) * x
        y = _norm(y)
        z = np.cross(x, y)
        z = _norm(z)
        y = np.cross(z, x)
        y = _norm(y)
        return np.stack([x, y, z], axis=0).astype(np.float32)

    def _apply_current_matrix_to_state(self) -> None:
        mat = np.array(
            [[self.matrix_inputs[r][c].value() for c in range(3)] for r in range(3)],
            dtype=np.float32,
        )
        if self.matrix_lock_btn.isChecked():
            mat = self._orthonormalize_rows(mat)
            self._set_matrix_inputs(mat)
        self.r_lidar_to_cam = mat

    def _apply_custom_matrix(self) -> None:
        self._apply_current_matrix_to_state()
        self.axis_preset.blockSignals(True)
        try:
            self.axis_preset.setCurrentIndex(0)
        finally:
            self.axis_preset.blockSignals(False)
        if not self.playing:
            self._render_current_frame()

    def _on_axis_filter_change(self) -> None:
        index = self.axis_filter.currentIndex()
        self.axis_mode = [
            "all",
            "x_pos",
            "x_neg",
            "y_pos",
            "y_neg",
            "z_pos",
            "z_neg",
        ][index]
        if not self.playing:
            self._render_current_frame()

    def _on_test_point_mode_change(self, checked: bool) -> None:
        self.test_point_mode = checked
        self.test_point_btn.setText("切换单点测试" if checked else "切换全点显示")
        if not self.playing:
            self._render_current_frame()

    def _on_matrix_lock_toggled(self, checked: bool) -> None:
        if checked:
            self.matrix_mode = "orthonormal"
            self._apply_current_matrix_to_state()
        else:
            self.matrix_mode = "free"

    def _on_matrix_cell_changed(self, row: int, col: int) -> None:
        if self.matrix_updating:
            return
        if self.matrix_lock_btn.isChecked():
            self._apply_current_matrix_to_state()
            self.matrix_updating = True
            try:
                self._load_current_preset_to_inputs()
            finally:
                self.matrix_updating = False
        else:
            self.r_lidar_to_cam = np.array(
                [[self.matrix_inputs[r][c].value() for c in range(3)] for r in range(3)],
                dtype=np.float32,
            )
        if not self.playing:
            self._render_current_frame()

    def _on_threshold_mode_changed(self) -> None:
        self.threshold_frame_mode = "camera" if self.threshold_mode.currentIndex() == 0 else "lidar"
        if not self.playing:
            self._render_current_frame()

    def _sync_axis_thresholds_from_ui(self) -> None:
        for axis in ["x", "y", "z"]:
            spin = self.threshold_inputs.get(axis)
            if spin is not None:
                AXIS_FILTER_THRESHOLDS[f"{axis}_mid"] = float(spin.value())

    def _on_extrinsic_change(self) -> None:
        self.tx = float(self.tx_slider.value()) / 100.0
        self.ty = float(self.ty_slider.value()) / 100.0
        self.tz = float(self.tz_slider.value()) / 100.0
        self.rx = float(self.rx_slider.value()) / 100.0
        self.ry = float(self.ry_slider.value()) / 100.0
        self.rz = float(self.rz_slider.value()) / 100.0
        self._sync_axis_thresholds_from_ui()
        self._render_current_frame()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.setText("暂停" if self.playing else "继续")
        if self.playing and self.follow_playback:
            self._advance_and_render_frame()

    def _on_follow_toggled(self, checked: bool) -> None:
        self.follow_playback = checked

    def _reset_extrinsic(self) -> None:
        self.tx_slider.setValue(0)
        self.ty_slider.setValue(0)
        self.tz_slider.setValue(0)
        self.rx_slider.setValue(0)
        self.ry_slider.setValue(0)
        self.rz_slider.setValue(0)
        self.axis_preset.blockSignals(True)
        self.axis_preset.setCurrentIndex(0)
        self.axis_preset.blockSignals(False)
        self.axis_filter.setCurrentIndex(0)
        self.test_point_btn.setChecked(True)
        self.r_lidar_to_cam = DEFAULT_R_LIDAR_TO_CAM.copy()
        self._load_current_preset_to_inputs()
        self._sync_axis_thresholds_from_ui()

    def _select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(self.video_path.parent), "Video (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        self.cap.release()
        self.video_path = Path(file_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.current_frame_idx = 0

    def _next_frame(self) -> None:
        if not self.playing:
            return
        self._advance_and_render_frame()

    def _advance_and_render_frame(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            ret, frame = self.cap.read()
            if not ret:
                return
        self._render_frame(frame)

    def _render_current_frame(self) -> None:
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or self.current_frame_idx)
        if current_pos <= 0:
            current_pos = self.current_frame_idx
        frame_idx = max(0, current_pos - 1)
        ret, frame = self.cap.read()
        if not ret:
            return
        self._render_frame(frame, frame_idx=frame_idx)

    def _render_frame(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> None:
        if frame_idx is None:
            frame_idx = self.current_frame_idx

        frame_ns = frame_time_ns(frame_idx, self.fps, self.video_start_ns)

        xyz, pc_ts_ns = points_in_time_window(self.pc, frame_ns, self.time_window_ns // 2)

        xyz = apply_axis_filter(xyz, self.axis_mode, self.r_lidar_to_cam, self.threshold_frame_mode)

        if self.test_point_mode:
            if len(xyz) == 0:
                xyz_to_project = xyz
            else:
                distances = np.linalg.norm(xyz, axis=1)
                front_score = np.abs(xyz[:, 0]) + np.abs(xyz[:, 2])
                idx = int(np.lexsort((distances, front_score))[0])
                xyz_to_project = xyz[idx:idx + 1]
            point_mode_text = "single-point"
        else:
            xyz_to_project = xyz
            point_mode_text = "all-points"

        img_pts, depth = project_points(
            xyz_to_project,
            self.tx,
            self.ty,
            self.tz,
            self.rx,
            self.ry,
            self.rz,
            self.r_lidar_to_cam,
        )
        in_mask = clip_points(img_pts, self.width, self.height)
        pts_in = img_pts[in_mask]
        depth_in = depth[in_mask]
        colors = color_by_depth(depth_in)

        for (u, v), c in zip(pts_in.astype(np.int32), colors):
            cv2.circle(frame, (int(u), int(v)), 2, (int(c[0]), int(c[1]), int(c[2])), -1)

        # 时间换算
        video_time_sec = frame_ns / 1e9
        pc_time_sec = pc_ts_ns.mean() / 1e9 if len(pc_ts_ns) > 0 else 0.0
        time_diff_sec = video_time_sec - pc_time_sec

        # 绘制时间对比
        cv2.putText(frame, f"Video Time: {video_time_sec:.3f} s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"LiDAR Time: {pc_time_sec:.3f} s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Time Diff : {time_diff_sec:.3f} s", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(
            frame,
            (
                f"frame={frame_idx} points={len(pts_in)} mode={point_mode_text} "
                f"tx={self.tx:.1f} ty={self.ty:.1f} tz={self.tz:.1f} "
                f"rx(绕x)={self.rx:.1f} ry(绕y)={self.ry:.1f} rz(绕z)={self.rz:.1f}"
            ),
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.info_label.setText(
            f"video: {self.video_path.name} | fps={self.fps:.2f} | frame={frame_idx}/{self.frame_count} | "
            f"window=±{self.time_window_ns / 2 / 1e6:.1f}ms | visible_points={len(pts_in)} | "
            f"axis_preset={self.axis_preset.currentText()} | axis_filter={self.axis_mode} | mode={point_mode_text} | "
            f"t=({self.tx:.2f}, {self.ty:.2f}, {self.tz:.2f})cm rx(绕x)={self.rx:.2f} ry(绕y)={self.ry:.2f} rz(绕z)={self.rz:.2f}deg"
        )

        self.current_frame_idx = frame_idx + 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyQt 实时 LiDAR 投影外参调参器")
    parser.add_argument("--video", required=True, type=Path, help="视频路径")
    parser.add_argument("--parquet", required=True, type=Path, help="点云 parquet 路径")
    parser.add_argument("--time-window-ms", default=100.0, type=float, help="每帧时间窗口（毫秒）")
    parser.add_argument(
        "--video-start-ns",
        default=None,
        type=int,
        help="视频第 0 帧对应时间戳(ns)，默认用 parquet 首时间戳",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    app = QApplication(sys.argv)
    window = ProjectionTunerWindow(
        video_path=args.video,
        parquet_path=args.parquet,
        time_window_ms=args.time_window_ms,
        video_start_ns=args.video_start_ns,
    )
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
