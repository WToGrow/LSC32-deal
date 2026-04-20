from __future__ import annotations

"""Single-frame Video/LiDAR alignment checker.

用途
----
只检查“视频第一帧”与其对应时间段内的点云是否空间对齐，便于快速调参。

你可以手动调的参数
------------------
1. 时间对齐
   - `VIDEO_START_NS`
   - `TIME_WINDOW_MS`
2. 坐标系外参
   - `CAMERA_YAW_DEG / CAMERA_PITCH_DEG / CAMERA_ROLL_DEG`
   - `CAMERA_WORLD_POS_CM`
   - `LIDAR_WORLD_POS_CM`
   - `LIDAR_EXTRA_YAW_DEG / LIDAR_EXTRA_PITCH_DEG / LIDAR_EXTRA_ROLL_DEG`
3. 可视化效果
   - 投影点颜色、大小
   - 是否输出叠加视频

坐标约定
--------
- 默认使用“世界坐标系 -> 相机坐标系”的变换。
- 先做旋转，再做平移。
- 本脚本假定点云 parquet 的字段至少包含：
  `timestamp_ns`, `x`, `y`, `z`。
- 如果你的 parquet 中单位不是 cm，请先统一单位。

说明
----
- 这个脚本只取视频第一帧，并取该帧对应时间窗口内的点云进行投影。
- 会输出：
  1) 相机视场角
  2) 点云在相机坐标系下的角度范围
  3) 投影到图像后的像素范围
  4) 叠加投影点的调试图像 / 视频
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
from datetime import datetime
from typing import Optional, Iterator

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =========================
# 相机参数
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


# =========================
# 坐标系参数（单位：cm）
# =========================
LIDAR_WORLD_POS_CM = np.array([1.5, -291.8, 187.8], dtype=np.float32)
FRONT_REF_WORLD_POS_CM = np.array([-1.5, -203.8, 169.8], dtype=np.float32)
MMWAVE_WORLD_POS_CM = np.array([0.0, 0.0, 58.5], dtype=np.float32)

# 默认把相机放在正前参考点处
CAMERA_WORLD_POS_CM = FRONT_REF_WORLD_POS_CM.copy()

# 你可以在这里直接改旋转，也可以用命令行参数覆盖
CAMERA_YAW_DEG = 0.0
CAMERA_PITCH_DEG = 0.0
CAMERA_ROLL_DEG = 0.0

# 如果雷达和世界坐标系并不完全一致，可以给雷达一个额外旋转
LIDAR_EXTRA_YAW_DEG = 0.0
LIDAR_EXTRA_PITCH_DEG = 0.0
LIDAR_EXTRA_ROLL_DEG = 0.0


# =========================
# 调试参数
# =========================
TIME_WINDOW_MS = 80.0
BATCH_SIZE = 500_000
DRAW_RADIUS = 2
# 颜色约定（OpenCV BGR）
# - LiDAR投影点：红色
# - 相机坐标与基准线：黄色系
# - 雷达坐标与基准线：青色系
LIDAR_POINT_COLOR = (0, 0, 255)
CAM_AXIS_ORIGIN_COLOR = (0, 255, 255)
LIDAR_AXIS_ORIGIN_COLOR = (255, 255, 0)
CAM_FORWARD_COLOR = (0, 220, 220)
LIDAR_FORWARD_COLOR = (220, 220, 0)
TEXT_COLOR = (0, 255, 0)


@dataclass
class ProjectionResult:
    frame_idx: int
    frame_time_ns: int
    total_points: int
    front_points: int
    in_image_points: int
    image_bbox: Optional[tuple[int, int, int, int]]
    camera_angle_range: Optional[tuple[float, float, float, float]]


# =========================
# 基础函数
# =========================
def deg2rad(v: float) -> float:
    return v * math.pi / 180.0


def rotation_matrix_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """由 yaw/pitch/roll 生成旋转矩阵。

    约定：
    - yaw   绕 Z 轴
    - pitch 绕 Y 轴
    - roll  绕 X 轴
    """
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    roll = deg2rad(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return Rz @ Ry @ Rx


def calc_camera_fov(width: int, height: int, camera_matrix: np.ndarray) -> tuple[float, float]:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    fov_x = 2.0 * math.degrees(math.atan(width / (2.0 * fx)))
    fov_y = 2.0 * math.degrees(math.atan(height / (2.0 * fy)))
    return fov_x, fov_y


def calc_angle_range(points_cam: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """返回 (azimuth_min, azimuth_max, elevation_min, elevation_max)。"""
    if len(points_cam) == 0:
        return None
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    azimuth = np.degrees(np.arctan2(x, z))
    elevation = np.degrees(np.arctan2(y, np.sqrt(x * x + z * z)))
    return (
        float(np.min(azimuth)),
        float(np.max(azimuth)),
        float(np.min(elevation)),
        float(np.max(elevation)),
    )


def bbox_int(points_uv: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    if len(points_uv) == 0:
        return None
    return (
        int(np.floor(np.min(points_uv[:, 0]))),
        int(np.floor(np.min(points_uv[:, 1]))),
        int(np.ceil(np.max(points_uv[:, 0]))),
        int(np.ceil(np.max(points_uv[:, 1]))),
    )


def project_world_point(
    point_world_cm: np.ndarray,
    r_world_to_cam: np.ndarray,
    t_world_to_cam: np.ndarray,
) -> Optional[tuple[int, int]]:
    """把一个世界坐标点投影到图像像素坐标。"""
    pt_cam = world_to_cam(np.asarray(point_world_cm, dtype=np.float32).reshape(1, 3), r_world_to_cam, t_world_to_cam)[0]
    if pt_cam[2] <= 0.1:
        return None
    uv, _ = cam_to_pixels(pt_cam.reshape(1, 3))
    if len(uv) == 0:
        return None
    return int(round(float(uv[0, 0]))), int(round(float(uv[0, 1])))


def draw_axis_arrows(
    image: np.ndarray,
    origin_world_cm: np.ndarray,
    r_world_to_cam: np.ndarray,
    t_world_to_cam: np.ndarray,
    axis_length_cm: float,
    label: str,
    origin_color: tuple[int, int, int] = (255, 255, 255),
    axis_colors: Optional[dict[str, tuple[int, int, int]]] = None,
) -> None:
    """在图像上绘制坐标轴箭头。

    约定：
    - X 轴红色
    - Y 轴绿色
    - Z 轴蓝色
    """
    origin_uv = project_world_point(origin_world_cm, r_world_to_cam, t_world_to_cam)
    if origin_uv is None:
        return

    if axis_colors is None:
        axis_colors = {
            "X": (0, 0, 255),
            "Y": (0, 255, 0),
            "Z": (255, 0, 0),
        }

    axes = [
        (np.array([axis_length_cm, 0.0, 0.0], dtype=np.float32), axis_colors.get("X", (0, 0, 255)), "X"),
        (np.array([0.0, axis_length_cm, 0.0], dtype=np.float32), axis_colors.get("Y", (0, 255, 0)), "Y"),
        (np.array([0.0, 0.0, axis_length_cm], dtype=np.float32), axis_colors.get("Z", (255, 0, 0)), "Z"),
    ]
    ox, oy = origin_uv
    cv2.circle(image, (ox, oy), 4, origin_color, -1)
    cv2.putText(image, label, (ox + 6, oy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, origin_color, 2, cv2.LINE_AA)

    origin_cam = world_to_cam(np.asarray(origin_world_cm, dtype=np.float32).reshape(1, 3), r_world_to_cam, t_world_to_cam)[0]
    for axis_vec, color, name in axes:
        end_world = np.asarray(origin_world_cm, dtype=np.float32) + axis_vec
        end_cam = world_to_cam(end_world.reshape(1, 3), r_world_to_cam, t_world_to_cam)[0]
        if origin_cam[2] <= 0.1 or end_cam[2] <= 0.1:
            continue
        end_uv = project_world_point(end_world, r_world_to_cam, t_world_to_cam)
        if end_uv is None:
            continue
        cv2.arrowedLine(image, (ox, oy), end_uv, color, 2, cv2.LINE_AA, 0, 0.15)
        mid = ((ox + end_uv[0]) // 2, (oy + end_uv[1]) // 2)
        cv2.putText(image, name, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_reference_arrow(
    image: np.ndarray,
    origin_world_cm: np.ndarray,
    direction_world_cm: np.ndarray,
    r_world_to_cam: np.ndarray,
    t_world_to_cam: np.ndarray,
    arrow_length_cm: float,
    label: str,
    color: tuple[int, int, int],
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """绘制基准方向箭头，用于显示“正前方向”。

    返回：
    - (origin_uv, end_uv)
    - 若无法投影则返回 None
    """
    origin_uv = project_world_point(origin_world_cm, r_world_to_cam, t_world_to_cam)
    if origin_uv is None:
        return None

    origin_world_cm = np.asarray(origin_world_cm, dtype=np.float32)
    direction_world_cm = np.asarray(direction_world_cm, dtype=np.float32)
    norm = float(np.linalg.norm(direction_world_cm))
    if norm < 1e-6:
        return None
    end_world = origin_world_cm + direction_world_cm / norm * arrow_length_cm
    end_uv = project_world_point(end_world, r_world_to_cam, t_world_to_cam)
    if end_uv is None:
        return None

    ox, oy = origin_uv
    cv2.arrowedLine(image, (ox, oy), end_uv, color, 3, cv2.LINE_AA, 0, 0.2)
    cv2.putText(image, label, (end_uv[0] + 6, end_uv[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return origin_uv, end_uv


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算二维向量夹角，单位度。"""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"parquet 缺少必要字段: {missing}, 当前列: {list(df.columns)}")


def load_video_meta(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, float(fps), width, height, frame_count


def frame_time_ns(frame_idx: int, fps: float, video_start_ns: int) -> int:
    return int(video_start_ns + round(frame_idx * 1_000_000_000.0 / fps))


# =========================
# 坐标变换
# =========================
def build_transforms(
    camera_world_pos_cm: np.ndarray,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    camera_roll_deg: float,
    lidar_yaw_deg: float,
    lidar_pitch_deg: float,
    lidar_roll_deg: float,
) -> dict[str, np.ndarray]:
    """构建世界/相机/雷达之间的旋转和平移矩阵。"""
    r_world_to_cam = rotation_matrix_from_ypr(camera_yaw_deg, camera_pitch_deg, camera_roll_deg)
    t_world_to_cam = (-r_world_to_cam @ camera_world_pos_cm.reshape(3, 1)).astype(np.float32)

    r_lidar_extra = rotation_matrix_from_ypr(lidar_yaw_deg, lidar_pitch_deg, lidar_roll_deg)
    t_lidar_to_world = LIDAR_WORLD_POS_CM.reshape(3, 1).astype(np.float32)

    return {
        "r_world_to_cam": r_world_to_cam,
        "t_world_to_cam": t_world_to_cam,
        "r_lidar_extra": r_lidar_extra,
        "t_lidar_to_world": t_lidar_to_world,
    }


def world_to_cam(points_world: np.ndarray, r_world_to_cam: np.ndarray, t_world_to_cam: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return (r_world_to_cam @ pts.T).T + t_world_to_cam.reshape(1, 3)


def cam_to_pixels(points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """手写投影：先畸变，再乘内参。"""
    if len(points_cam) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)

    pts = np.asarray(points_cam, dtype=np.float32)
    front_mask = pts[:, 2] > 0.1
    pts = pts[front_mask]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32), front_mask

    x = pts[:, 0] / pts[:, 2]
    y = pts[:, 1] / pts[:, 2]
    r2 = x * x + y * y

    k1, k2, p1, p2, k3 = DIST_COEFFS.tolist()
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    fx = float(CAMERA_MATRIX[0, 0])
    fy = float(CAMERA_MATRIX[1, 1])
    cx = float(CAMERA_MATRIX[0, 2])
    cy = float(CAMERA_MATRIX[1, 2])
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    img_pts = np.stack([u, v], axis=1).astype(np.float32)
    return img_pts, front_mask


# =========================
# 点云读取
# =========================
def iter_batches(parquet_path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def get_first_frame_points(parquet_path: Path, frame_time_ns: int, time_window_ns: int, batch_size: int) -> pd.DataFrame:
    """只取第一帧对应时间窗口内的点。"""
    half = time_window_ns // 2
    start_ns = frame_time_ns - half
    end_ns = frame_time_ns + half
    parts: list[pd.DataFrame] = []

    for df in iter_batches(parquet_path, batch_size):
        require_columns(df, ["timestamp_ns", "x", "y", "z"])
        if len(df) == 0:
            continue
        if all(c in df.columns for c in ["intensity", "ring"]):
            df = df[["timestamp_ns", "x", "y", "z", "intensity", "ring"]].copy()
        else:
            df = df[["timestamp_ns", "x", "y", "z"]].copy()
        mask = (df["timestamp_ns"].to_numpy() >= start_ns) & (df["timestamp_ns"].to_numpy() <= end_ns)
        hit = df.loc[mask]
        if len(hit) > 0:
            parts.append(hit)

    if not parts:
        return pd.DataFrame(columns=["timestamp_ns", "x", "y", "z"])
    return pd.concat(parts, ignore_index=True)


# =========================
# 主流程
# =========================
def infer_video_start_ns_from_parquet(parquet_path: Path, batch_size: int = 1) -> int:
    """自动从 parquet 中提取视频起始时间戳（ns）。

    这里采用“读取 parquet 最前面的时间戳”的方式作为视频第一帧的对齐时间。
    这样就不需要手动输入 `video_start_ns`。
    """
    pq_file = pq.ParquetFile(parquet_path)
    first_batch = next(pq_file.iter_batches(batch_size=batch_size), None)
    if first_batch is None:
        raise ValueError(f"parquet 中没有数据: {parquet_path}")
    first_df = first_batch.to_pandas()
    require_columns(first_df, ["timestamp_ns"])
    return int(first_df["timestamp_ns"].iloc[0])


def run_first_frame_check(
    video_path: Path,
    parquet_path: Path,
    output_dir: Path,
    video_start_ns: Optional[int] = None,
    time_window_ms: float = TIME_WINDOW_MS,
    batch_size: int = BATCH_SIZE,
    draw_overlay: bool = True,
    save_debug_image: bool = True,
    camera_yaw_deg: float = CAMERA_YAW_DEG,
    camera_pitch_deg: float = CAMERA_PITCH_DEG,
    camera_roll_deg: float = CAMERA_ROLL_DEG,
    lidar_yaw_deg: float = LIDAR_EXTRA_YAW_DEG,
    lidar_pitch_deg: float = LIDAR_EXTRA_PITCH_DEG,
    lidar_roll_deg: float = LIDAR_EXTRA_ROLL_DEG,
) -> None:
    cap, fps, width, height, frame_count = load_video_meta(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")

    if video_start_ns is None:
        video_start_ns = infer_video_start_ns_from_parquet(parquet_path, batch_size=1)
        print(f"自动对齐得到 video_start_ns: {video_start_ns}")
    else:
        print(f"手动指定 video_start_ns: {video_start_ns}")

    first_frame_time_ns = frame_time_ns(0, fps, video_start_ns)
    time_window_ns = int(time_window_ms * 1_000_000)

    transforms = build_transforms(
        camera_world_pos_cm=CAMERA_WORLD_POS_CM,
        camera_yaw_deg=camera_yaw_deg,
        camera_pitch_deg=camera_pitch_deg,
        camera_roll_deg=camera_roll_deg,
        lidar_yaw_deg=lidar_yaw_deg,
        lidar_pitch_deg=lidar_pitch_deg,
        lidar_roll_deg=lidar_roll_deg,
    )

    fov_x, fov_y = calc_camera_fov(width, height, CAMERA_MATRIX)
    print("========== 第一帧空间对齐测试 ==========")
    print(f"视频: {video_path}")
    print(f"点云: {parquet_path}")
    print(f"视频尺寸: {width}x{height}, fps={fps:.3f}, frame_count={frame_count}")
    print(f"自动/输入得到的 video_start_ns: {video_start_ns}")
    print(f"第一帧时间戳: {first_frame_time_ns}")
    print(f"时间窗口: ±{time_window_ms / 2:.1f} ms")
    print(f"相机世界位置(cm): {CAMERA_WORLD_POS_CM.tolist()}")
    print(f"雷达世界位置(cm): {LIDAR_WORLD_POS_CM.tolist()}")
    print(f"相机旋转(ypr deg): ({camera_yaw_deg}, {camera_pitch_deg}, {camera_roll_deg})")
    print(f"雷达额外旋转(ypr deg): ({lidar_yaw_deg}, {lidar_pitch_deg}, {lidar_roll_deg})")
    print(f"相机FOV: horizontal={fov_x:.2f} deg, vertical={fov_y:.2f} deg")
    print(f"输出目录: {output_dir}")

    frame_df = get_first_frame_points(parquet_path, first_frame_time_ns, time_window_ns, batch_size)
    print(f"时间窗口内点数: {len(frame_df)}")

    if len(frame_df) == 0:
        print("该时间段没有点云数据，建议先检查 time_window_ms 或 video_start_ns")
        return

    xyz = frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)

    # 1) 世界坐标 -> 相机坐标
    pts_cam = world_to_cam(xyz, transforms["r_world_to_cam"], transforms["t_world_to_cam"])
    front_mask = pts_cam[:, 2] > 0.1
    pts_cam_front = pts_cam[front_mask]

    # 2) 相机坐标 -> 像素
    img_pts, front_mask2 = cam_to_pixels(pts_cam)
    # front_mask2 和 front_mask 一致，只是为了让逻辑清晰保留一次
    in_img_mask = (
        (img_pts[:, 0] >= 0)
        & (img_pts[:, 0] < width)
        & (img_pts[:, 1] >= 0)
        & (img_pts[:, 1] < height)
    ) if len(img_pts) else np.empty((0,), dtype=bool)
    img_pts_in = img_pts[in_img_mask]
    pts_cam_in = pts_cam_front[in_img_mask]

    # 3) 角度范围输出
    angle_range = calc_angle_range(pts_cam_front)
    bbox = bbox_int(img_pts_in)

    result = ProjectionResult(
        frame_idx=0,
        frame_time_ns=first_frame_time_ns,
        total_points=len(frame_df),
        front_points=len(pts_cam_front),
        in_image_points=len(img_pts_in),
        image_bbox=bbox,
        camera_angle_range=angle_range,
    )

    print("---------- 投影结果 ----------")
    print(f"总点数: {result.total_points}")
    print(f"相机前方点数(Z>0): {result.front_points}")
    print(f"落入图像点数: {result.in_image_points}")
    if result.image_bbox is not None:
        u0, v0, u1, v1 = result.image_bbox
        print(f"图像包围盒(u,v): ({u0}, {v0}) -> ({u1}, {v1})")
    else:
        print("图像包围盒(u,v): None")
    if result.camera_angle_range is not None:
        az0, az1, el0, el1 = result.camera_angle_range
        print(f"相机角度范围(度): azimuth[{az0:.2f}, {az1:.2f}], elevation[{el0:.2f}, {el1:.2f}]")

    # 4) 画点并保存
    vis = first_frame.copy()
    for (u, v) in img_pts_in.astype(np.int32, copy=False):
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(vis, (int(u), int(v)), DRAW_RADIUS, LIDAR_POINT_COLOR, -1)

    # 明确画出世界中的坐标轴：相机坐标系和雷达坐标系
    draw_axis_arrows(
        vis,
        origin_world_cm=CAMERA_WORLD_POS_CM,
        r_world_to_cam=transforms["r_world_to_cam"],
        t_world_to_cam=transforms["t_world_to_cam"],
        axis_length_cm=100.0,
        label="CAM axes",
        origin_color=CAM_AXIS_ORIGIN_COLOR,
        axis_colors={
            "X": (0, 0, 180),
            "Y": (0, 180, 0),
            "Z": (180, 0, 0),
        },
    )
    draw_axis_arrows(
        vis,
        origin_world_cm=LIDAR_WORLD_POS_CM,
        r_world_to_cam=transforms["r_world_to_cam"],
        t_world_to_cam=transforms["t_world_to_cam"],
        axis_length_cm=100.0,
        label="LIDAR axes",
        origin_color=LIDAR_AXIS_ORIGIN_COLOR,
        axis_colors={
            "X": (255, 120, 0),
            "Y": (255, 0, 255),
            "Z": (0, 200, 255),
        },
    )

    # 再额外画“正前方向”基准箭头，帮助区分“方向偏了多少”和“轴定义是否反了”
    # 相机正前方向：这里默认取相机坐标系 Z 轴正方向
    cam_forward = draw_reference_arrow(
        vis,
        origin_world_cm=CAMERA_WORLD_POS_CM,
        direction_world_cm=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        r_world_to_cam=transforms["r_world_to_cam"],
        t_world_to_cam=transforms["t_world_to_cam"],
        arrow_length_cm=150.0,
        label="CAM forward",
        color=CAM_FORWARD_COLOR,
    )
    # 雷达正前方向：用雷达自己的局部坐标前向作为基准
    lidar_forward = draw_reference_arrow(
        vis,
        origin_world_cm=LIDAR_WORLD_POS_CM,
        direction_world_cm=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        r_world_to_cam=transforms["r_world_to_cam"],
        t_world_to_cam=transforms["t_world_to_cam"],
        arrow_length_cm=150.0,
        label="LIDAR forward",
        color=LIDAR_FORWARD_COLOR,
    )

    if cam_forward is not None and lidar_forward is not None:
        cam_origin_uv, cam_end_uv = cam_forward
        lidar_origin_uv, lidar_end_uv = lidar_forward
        cam_vec = np.array([cam_end_uv[0] - cam_origin_uv[0], cam_end_uv[1] - cam_origin_uv[1]], dtype=np.float32)
        lidar_vec = np.array([lidar_end_uv[0] - lidar_origin_uv[0], lidar_end_uv[1] - lidar_origin_uv[1]], dtype=np.float32)
        diff_angle = angle_between_vectors_deg(cam_vec, lidar_vec)
        cv2.putText(
            vis,
            f"cam/lidar forward angle = {diff_angle:.2f} deg",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"frame0 points={len(img_pts_in)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"cam ypr=({camera_yaw_deg:.1f},{camera_pitch_deg:.1f},{camera_roll_deg:.1f})",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"lidar ypr=({lidar_yaw_deg:.1f},{lidar_pitch_deg:.1f},{lidar_roll_deg:.1f})",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"video_start_ns={video_start_ns}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if save_debug_image:
        img_path = output_dir / f"{video_path.stem}_first_frame_projection_{timestamp_tag}.png"
        cv2.imwrite(str(img_path), vis)
        print(f"调试图已保存: {img_path}")

    if draw_overlay:
        out_video = output_dir / f"{video_path.stem}_first_frame_projection_{timestamp_tag}.mp4"
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        writer.write(vis)
        writer.release()
        print(f"调试视频已保存: {out_video}")

    # 5) 给出参数提示
    print("---------- 调参建议 ----------")
    print("1. 如果点云整体偏左/偏右，优先调 camera_yaw_deg")
    print("2. 如果点云整体偏上/偏下，优先调 camera_pitch_deg")
    print("3. 如果点云旋转方向不对，优先调 camera_roll_deg")
    print("4. 如果雷达坐标与世界坐标并不重合，可调 lidar_extra_yaw/pitch/roll")
    print("5. 如果没有任何点落入图像，先检查 video_start_ns 和 TIME_WINDOW_MS")

    cap.release()
    print("========== 第一帧空间对齐测试结束 ==========")


# =========================
# 命令行入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只测试视频第一帧与点云时间段是否空间对齐")
    parser.add_argument("--video", required=True, type=Path, help="输入视频路径")
    parser.add_argument("--parquet", required=True, type=Path, help="输入点云 parquet 路径")
    parser.add_argument("--output-dir", default=Path("./vl_first_frame_out"), type=Path, help="输出目录")
    parser.add_argument("--video-start-ns", default=None, type=int, help="视频第一帧对应的时间戳(ns)，不传则自动从 parquet 首时间戳提取")
    parser.add_argument("--time-window-ms", default=TIME_WINDOW_MS, type=float, help="第一帧取点的时间窗口大小(ms)")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int, help="parquet 分块大小")
    parser.add_argument("--no-overlay", action="store_true", help="不保存叠加视频")
    parser.add_argument("--no-image", action="store_true", help="不保存调试图片")
    parser.add_argument("--camera-yaw", type=float, default=CAMERA_YAW_DEG, help="相机 yaw (deg)")
    parser.add_argument("--camera-pitch", type=float, default=CAMERA_PITCH_DEG, help="相机 pitch (deg)")
    parser.add_argument("--camera-roll", type=float, default=CAMERA_ROLL_DEG, help="相机 roll (deg)")
    parser.add_argument("--lidar-yaw", type=float, default=LIDAR_EXTRA_YAW_DEG, help="雷达额外 yaw (deg)")
    parser.add_argument("--lidar-pitch", type=float, default=LIDAR_EXTRA_PITCH_DEG, help="雷达额外 pitch (deg)")
    parser.add_argument("--lidar-roll", type=float, default=LIDAR_EXTRA_ROLL_DEG, help="雷达额外 roll (deg)")
    return parser


def main() -> None:
    # 硬编码模式：直接在这里修改参数，不依赖命令行。
    # 如果你后续要重新恢复命令行模式，只需要把下面的参数换回 args 即可。
    video_path = Path(r"G:\data\2025-01-12\video\20260112_162910_正前_000.mp4")
    parquet_path = Path(r"G:\data\2025-01-12\parquet_out\lidar_points_protocol_162920.parquet")
    output_dir = Path(r"G:\data\vl_first_frame_check_out")
    video_start_ns = None
    time_window_ms = 100.0
    batch_size = 500_000
    draw_overlay = True
    save_debug_image = True

    # 相机外参：默认固定，不走命令行
    camera_yaw_deg = 0.0
    camera_pitch_deg = 0.0
    camera_roll_deg = 0.0

    # 雷达外参：你可以在这里持续调参
    lidar_yaw_deg = -180.0
    lidar_pitch_deg = -180.0
    lidar_roll_deg = 180.0

    run_first_frame_check(
        video_path=video_path,
        parquet_path=parquet_path,
        output_dir=output_dir,
        video_start_ns=video_start_ns,
        time_window_ms=time_window_ms,
        batch_size=batch_size,
        draw_overlay=draw_overlay,
        save_debug_image=save_debug_image,
        camera_yaw_deg=camera_yaw_deg,
        camera_pitch_deg=camera_pitch_deg,
        camera_roll_deg=camera_roll_deg,
        lidar_yaw_deg=lidar_yaw_deg,
        lidar_pitch_deg=lidar_pitch_deg,
        lidar_roll_deg=lidar_roll_deg,
    )


if __name__ == "__main__":
    main()
