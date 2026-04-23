from __future__ import annotations

"""Clean LiDAR-to-video projection pipeline.

This module provides a callable projection API for external callers
(e.g. Java web backend) and a minimal CLI entry point for batch usage.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import argparse
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =========================
# 相机参数
# =========================
# 1) 相机内参矩阵
CAMERA_MATRIX = np.array(
    [
        [2041.23366196, 0.0, 997.68672528],
        [0.0, 2038.76906929, 544.68141109],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# 2) 相机畸变系数
DIST_COEFFS = np.array([-0.52939418, 0.37462897, 0.0, 0.0, 0.0], dtype=np.float32)


# =========================
# 设备/相机默认参数（固定值，供接口使用）
# =========================
LIDAR_WORLD_POS = np.array([1.5, -291.8, 187.8], dtype=np.float32)
FRONT_REF_WORLD_POS = np.array([-1.5, -203.8, 169.8], dtype=np.float32)
MMWAVE_WORLD_POS = np.array([0.0, 0.0, 58.5], dtype=np.float32)
CAMERA_WORLD_POS = FRONT_REF_WORLD_POS.copy()

# 默认外参：如后续由界面或上层服务传入，可覆盖这些固定值
DEFAULT_R_LIDAR_TO_CAM = np.array(
    [
        # 1、启动时用第二辆车校对
        # [-0.747909, 0.663802, 0.0],
        # [-0.082236, -0.092656, -0.992296],
        # [-0.658688, -0.742147, 0.123886],

        # 2、相对静止前车校对
        # [-0.744767, 0.667324, -0.046653],
        # [-0.082669, -0.092262, -0.992297],
        # [-0.662184, -0.739030, 0.123881],

        # 3、启动前用建筑弧度校对
        # [0.723261, 0.688997, -0.046653],
        # [-0.021385, -0.045179, 0.998750],
        # [-0.690244, 0.723355, -0.017941],

        # 4、修复后的parquet，以建筑弧度对齐
        # [-0.703847, 0.710352, 0.000000],
        # [-0.080507, -0.079770, -0.993557],
        # [-0.705775, -0.699312, 0.113334],

        # 5、修复后的parquet，运动时对齐
        [-0.629838, 0.776726, 0.000000],
        [0.004426, 0.003589, -0.999984],
        [-0.776714, -0.629828, -0.005698],

        # 6、修复水平后的parquet，静止时对齐
        # [-0.708285,  0.705927,  0.000000],
        # [-0.026790, -0.026880, -0.999280],
        # [-0.705418, -0.707775,  0.037951],
    ],
    dtype=np.float32,
)
# DEFAULT_T_WORLD_TO_CAM = np.zeros((3, 1), dtype=np.float32)
# m/数值
DEFAULT_T_WORLD_TO_CAM = np.array(
    [
        # 1、
        # [-3.36],
        # [0],
        # [38.67]
        
        # 2、
        # [-2.12],
        # [3.41],
        # [10.94],

        # 3、
        # [-3.43],
        # [0.48],
        # [56.09],

        # 4、
        # [-0.01],
        # [-1.68],
        # [4.79],

        # 5、
        [-0.50],
        [0.27],
        [13.91],

        # 6、
        # [-0.94],
        # [0.65],
        # [4.55],

    ],
    dtype=np.float32
)


# =========================
# 数据结构
# =========================
@dataclass
class ProjectionConfig:
    video_path: Path
    parquet_path: Path
    output_dir: Path | None = None
    output_video_path: Path | None = None
    batch_size: int = 500_000
    time_window_ms: float = 100.0
    video_start_ns: Optional[int] = None
    video_time_offset_sec: float = -10.0
    max_frames: Optional[int] = None
    draw_overlay: bool = True


@dataclass
class FrameProjectionInfo:
    frame_idx: int
    frame_time_ns: int
    point_count_in_window: int
    point_count_in_front: int
    projected_count: int
    image_bbox: Optional[tuple[int, int, int, int]]
    device_bbox_cm: Optional[tuple[float, float, float, float, float, float]]


# =========================
# 基础工具
# =========================
def _pick_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """识别 parquet 里的 xyz 字段名。"""
    candidates = [
        ("x", "y", "z"),
        ("X", "Y", "Z"),
        ("pos_x", "pos_y", "pos_z"),
        ("point_x", "point_y", "point_z"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    raise ValueError(f"无法识别 xyz 列，当前列：{list(df.columns)}")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"parquet 缺少必要字段: {missing}，当前列：{list(df.columns)}")


def _load_video_meta(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, float(fps), width, height, frame_count


def _frame_time_ns(frame_idx: int, fps: float, video_start_ns: int) -> int:
    """把帧序号换算为时间戳（ns）。"""

    # scale = (t_lidar_end - t_lidar_start) / (frame_count / fps * 1e9)
    # return video_start_ns + frame_idx * (1e9 / fps) * scale
    
    return int(video_start_ns + round(frame_idx * 1_000_000_000.0 / fps))


def _world_to_camera(points_world: np.ndarray) -> np.ndarray:
    """世界坐标系 → 相机坐标系。使用固定外参/镜像设置。"""
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if True:
        pts[:, 0] = -pts[:, 0]
    if False:
        pts[:, 1] = -pts[:, 1]
    if False:
        pts[:, 2] = -pts[:, 2]
    return (DEFAULT_R_LIDAR_TO_CAM @ pts.T).T + DEFAULT_T_WORLD_TO_CAM.reshape(1, 3)


def _project_camera_points(points_cam: np.ndarray, front_axis: str = "z") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把相机坐标系下的点投影到像素平面。"""
    if len(points_cam) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )

    pts = np.asarray(points_cam, dtype=np.float32)
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(front_axis)

    # front_mask = pts[:, axis_idx] > 0.1

    front_mask = pts[:, 2] > 0.1

    pts = pts[front_mask]
    if len(pts) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            front_mask,
        )

    # 使用 OpenCV 的投影函数：
    # 1) 旋转/平移到相机模型
    # 2) 畸变矫正
    # 3) 内参映射到像素坐标
    img_pts, _ = cv2.projectPoints(pts, np.zeros((3, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32), CAMERA_MATRIX, DIST_COEFFS)
    img_pts = img_pts.reshape(-1, 2)

    # depth_z = pts[:, 2].copy()
    depth = np.linalg.norm(pts, axis=1)

    # finite = np.isfinite(img_pts).all(axis=1)
    # return img_pts[finite], depth_z[finite], front_mask
    # ============修复：不移除 finite 点，保证长度和 front_mask 匹配=================
    return img_pts, depth, front_mask


def _bbox_int(points_uv: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    if len(points_uv) == 0:
        return None
    u_min = int(np.floor(np.min(points_uv[:, 0])))
    v_min = int(np.floor(np.min(points_uv[:, 1])))
    u_max = int(np.ceil(np.max(points_uv[:, 0])))
    v_max = int(np.ceil(np.max(points_uv[:, 1])))
    return u_min, v_min, u_max, v_max


def _bbox_cm(points_xyz: np.ndarray) -> Optional[tuple[float, float, float, float, float, float]]:
    if len(points_xyz) == 0:
        return None
    mins = np.min(points_xyz, axis=0)
    maxs = np.max(points_xyz, axis=0)
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


def _normalize_with_percentile(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    if len(values) == 0:
        return values
    v_min, v_max = np.percentile(values, [low, high])
    if np.isclose(v_min, v_max):
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - v_min) / (v_max - v_min), 0.0, 1.0).astype(np.float32)


# def _color_by_depth(depth: np.ndarray) -> np.ndarray:
#     """按相对离生成更接近 visualize_single.py 的亮色渐变 BGR。"""
#     if len(depth) == 0:
#         return np.empty((0, 3), dtype=np.uint8)

#     norm = _normalize_with_percentile(depth)
#     palette = np.array(
#         [
#             [255, 255, 0],   # 青蓝/黄系过渡
#             [128, 255, 0],   # 亮绿
#             [0, 255, 0],     # 绿
#             [0, 255, 128],   # 青绿
#             [0, 255, 255],   # 黄
#             [0, 165, 255],   # 橙
#             [0, 0, 255],     # 红
#         ],
#         dtype=np.float32,
#     )
#     scaled = norm * (len(palette) - 1)
#     left = np.floor(scaled).astype(np.int32)
#     right = np.clip(left + 1, 0, len(palette) - 1)
#     weight = (scaled - left).reshape(-1, 1)
#     colors = palette[left] * (1.0 - weight) + palette[right] * weight
#     colors = np.clip(colors * 1.05, 0, 255)
#     return colors.astype(np.uint8)

def _color_by_depth(depth: np.ndarray, max_dist: float = 80.0) -> np.ndarray:
    """
    按绝对距离上色（稳定，不随帧变化）
    depth: z or 欧氏距离（建议欧氏）
    """

    if len(depth) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # ✅ 关键：固定归一化
    norm = np.clip(depth / max_dist, 0.0, 1.0)

    # 渐变：红 → 黄 → 绿 → 青 → 蓝
    palette = np.array([
        [0, 0, 255],     # 红
        [0, 165, 255],   # 橙
        [0, 255, 255],   # 黄
        [0, 255, 0],     # 绿
        [255, 255, 0],   # 青
        [255, 0, 0],     # 蓝
    ], dtype=np.float32)

    scaled = norm * (len(palette) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(palette) - 1)
    weight = (scaled - left).reshape(-1, 1)

    colors = palette[left] * (1.0 - weight) + palette[right] * weight
    return colors.astype(np.uint8)


def _compute_image_mask(frame: np.ndarray) -> np.ndarray:
    """基于边缘的场景掩码"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    # 膨胀让轮廓更厚（避免点被误删）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)

    return mask > 0  # bool mask


def _color_consistency_filter(frame, pts_uv, threshold=40):
    """过滤颜色突变点"""
    h, w, _ = frame.shape
    valid = []

    for (u, v) in pts_uv.astype(np.int32):
        if not (1 <= u < w-1 and 1 <= v < h-1):
            valid.append(False)
            continue

        center = frame[v, u].astype(np.int32)
        patch = frame[v-1:v+2, u-1:u+2].astype(np.int32)

        diff = np.mean(np.abs(patch - center))
        valid.append(diff < threshold)

    return np.array(valid, dtype=bool)

# =========================
# 流式 parquet 读取与逐帧匹配
# =========================
def _load_point_cloud_arrays(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(parquet_path)
    parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        if len(df) == 0:
            continue
        _require_columns(df, ["timestamp_ns", "x", "y", "z"])
        parts.append(df[["timestamp_ns", "x", "y", "z"]].copy())
    if not parts:
        raise ValueError(f"parquet 中没有可用点云数据: {parquet_path}")
    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values("timestamp_ns", kind="mergesort").reset_index(drop=True)
    return all_df["timestamp_ns"].to_numpy(dtype=np.int64, copy=True), all_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)


def _stream_frames_points(
    timestamps_ns: np.ndarray,
    xyz: np.ndarray,
    fps: float,
    video_start_ns: int,
    time_window_ns: int,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """按每帧中心时间进行二分查找，获取对应点云窗口。"""
    current_frame_idx = 0
    half_window = time_window_ns // 2
    total = len(timestamps_ns)
    while True:
        current_frame_ns = _frame_time_ns(current_frame_idx, fps, video_start_ns)
        left = int(np.searchsorted(timestamps_ns, current_frame_ns - half_window, side="left"))
        right = int(np.searchsorted(timestamps_ns, current_frame_ns + half_window, side="right"))

        # 只取时间戳 == current_frame_ns 的点
        # left = int(np.searchsorted(timestamps_ns, current_frame_ns, side="left"))
        # right = left + 1

        if left >= total and current_frame_idx > 0:
            break
        if right <= left:
            frame_df = pd.DataFrame(columns=["timestamp_ns", "x", "y", "z"])
        else:
            frame_df = pd.DataFrame({
                "timestamp_ns": timestamps_ns[left:right],
                "x": xyz[left:right, 0],
                "y": xyz[left:right, 1],
                "z": xyz[left:right, 2],
            })
        yield current_frame_idx, current_frame_ns, frame_df
        current_frame_idx += 1


# =========================
# 每帧投影与调试输出
# =========================
def _project_frame_points(
    frame_df: pd.DataFrame,
    image_size: tuple[int, int],
    front_axis: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对一帧窗口内的点做空间变换与投影。"""
    if len(frame_df) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )

    xyz = frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
    pts_cam = _world_to_camera(xyz)
    img_pts, depth_z, front_mask = _project_camera_points(pts_cam, front_axis=front_axis)

    return img_pts, pts_cam[front_mask], depth_z, front_mask


def _clip_to_image(points_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(points_uv) == 0:
        return np.empty((0,), dtype=bool)
    return (
        np.isfinite(points_uv).all(axis=1)
        & (points_uv[:, 0] >= 0)
        & (points_uv[:, 0] < width)
        & (points_uv[:, 1] >= 0)
        & (points_uv[:, 1] < height)
    )


def run_projection(config: ProjectionConfig) -> Path | None:
    video_path = config.video_path
    parquet_path = config.parquet_path
    output_dir = config.output_dir or Path("./vl_projection_out")
    output_video_path = config.output_video_path
    batch_size = config.batch_size
    time_window_ms = max(config.time_window_ms, 100.0)
    video_start_ns = config.video_start_ns
    max_frames = config.max_frames
    draw_overlay = config.draw_overlay
    front_axis = "z"


    cap, fps, width, height, frame_count = _load_video_meta(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps_ns, xyz = _load_point_cloud_arrays(parquet_path)
    # ----repair-------
    t_lidar_start = timestamps_ns[0]
    t_lidar_end = timestamps_ns[-1]
    

    if video_start_ns is None:
        if len(timestamps_ns) == 0:
            raise ValueError(f"parquet 中没有数据: {parquet_path}")
        # video_start_ns = int(timestamps_ns[0])

        # --------repair---------
        video_start_ns = int(
        timestamps_ns[0] + config.video_time_offset_sec * 1_000_000_000
        )

    print("video_start_ns:", video_start_ns)
    print("first lidar ts:", timestamps_ns[0])
    print("last lidar ts:", timestamps_ns[-1])
    # else:
    #     video_start_ns = config.video_start_ns

    # 视频比雷达早 10 秒
    # video_start_ns = video_start_ns + 10 * 10**9

    time_window_ns = int(time_window_ms * 1_000_000)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if output_video_path is None:
        out_video_path = output_dir / f"{video_path.stem}_projection_{timestamp_tag}.mp4"
    else:
        out_video_path = Path(output_video_path)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if draw_overlay:
        writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_limit = frame_count if max_frames is None else min(frame_count, max_frames)
    frame_iter = _stream_frames_points(timestamps_ns=timestamps_ns, xyz=xyz, fps=fps, video_start_ns=video_start_ns, time_window_ns=time_window_ns)

    # for frame_idx, frame_time_ns, frame_df in frame_iter:
    #     if frame_idx >= frame_limit:
    #         break
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     if len(frame_df) == 0:
    #         if writer is not None:
    #             writer.write(frame)
    #         continue

    #     img_pts, pts_cam_front, depth_z, front_mask = _project_frame_points(frame_df, (width, height), front_axis)
    #     in_image_mask = _clip_to_image(img_pts, width, height)

    #     # ================= 颜色一致性过滤 =================
    #     color_mask = _color_consistency_filter(frame, img_pts_in)

    #     img_pts_in = img_pts_in[color_mask]
    #     depth_in = depth_in[color_mask]

    #     # img_pts_in = img_pts[in_image_mask]
    #     # depth_in = depth_z[in_image_mask]

    #     # ================= 图像掩码过滤 =================
    #     mask_img = _compute_image_mask(frame)

    #     valid_mask = []
    #     for (u, v) in img_pts_in.astype(np.int32):
    #         if 0 <= u < width and 0 <= v < height:
    #             valid_mask.append(mask_img[v, u])
    #         else:
    #             valid_mask.append(False)

    #     valid_mask = np.array(valid_mask, dtype=bool)

    #     img_pts_in = img_pts_in[valid_mask]
    #     depth_in = depth_in[valid_mask]

    #     if len(img_pts_in) > 0:
    #         draw_pts = img_pts_in.astype(np.int32, copy=False)
    #         colors = _color_by_depth(depth_in)
    #         for (u, v), c in zip(draw_pts, colors):
    #             if 0 <= u < width and 0 <= v < height:
    #                 cv2.circle(frame, (int(u), int(v)), 2, (int(c[0]), int(c[1]), int(c[2])), -1)

    #     if writer is not None:
    #         writer.write(frame)

    prev_depth_map = {}

    for frame_idx, frame_time_ns, frame_df in frame_iter:
        if frame_idx >= frame_limit:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if len(frame_df) == 0:
            if writer is not None:
                writer.write(frame)
            continue

        # ========= 投影 =========
        img_pts, pts_cam_front, depth_z, front_mask = _project_frame_points(
            frame_df, (width, height), front_axis
        )

        # ========= 图像范围过滤 =========
        in_image_mask = _clip_to_image(img_pts, width, height)
        img_pts_in = img_pts[in_image_mask]
        depth_in = depth_z[in_image_mask]

        if len(img_pts_in) == 0:
            if writer:
                writer.write(frame)
            continue

        # # ========= 图像掩码过滤 =========
        mask_img = _compute_image_mask(frame)

        valid_mask = []
        for (u, v) in img_pts_in.astype(np.int32):
            if 0 <= u < width and 0 <= v < height:
                valid_mask.append(mask_img[v, u])
            else:
                valid_mask.append(False)

        valid_mask = np.array(valid_mask, dtype=bool)

        img_pts_in = img_pts_in[valid_mask]
        depth_in = depth_in[valid_mask]

        if len(img_pts_in) == 0:
            if writer:
                writer.write(frame)
            continue

        # # ========= 颜色一致性过滤 =========
        color_mask = _color_consistency_filter(frame, img_pts_in)

        img_pts_in = img_pts_in[color_mask]
        depth_in = depth_in[color_mask]

        if len(img_pts_in) == 0:
            if writer:
                writer.write(frame)
            continue

        # ========= 深度时序过滤 =========
        new_map = {}
        temporal_mask = []

        for (u, v), d in zip(img_pts_in.astype(np.int32), depth_in):
            key = (u, v)

            if key in prev_depth_map:
                if abs(prev_depth_map[key] - d) > 0.5:
                    temporal_mask.append(False)
                    continue

            temporal_mask.append(True)
            new_map[key] = d

        temporal_mask = np.array(temporal_mask, dtype=bool)

        img_pts_in = img_pts_in[temporal_mask]
        depth_in = depth_in[temporal_mask]

        prev_depth_map = new_map

        if len(img_pts_in) == 0:
            if writer:
                writer.write(frame)
            continue

        # # ========= 绘制 =========
        # draw_pts = img_pts_in.astype(np.int32, copy=False)
        # colors = _color_by_depth(depth_in)

        # for (u, v), c in zip(draw_pts, colors):
        #     if 0 <= u < width and 0 <= v < height:
        #         cv2.circle(frame, (u, v), 2, (int(c[0]), int(c[1]), int(c[2])), -1)

        # # ========= 深度优先（Z-buffer） =========
        # z_buffer = {}

        # for (u, v), d in zip(img_pts_in.astype(np.int32), depth_in):
        #     key = (u, v)

        #     # 只保留更近的点（depth更小）
        #     if key not in z_buffer or d < z_buffer[key]:
        #         z_buffer[key] = d

        # if len(z_buffer) == 0:
        #     if writer:
        #         writer.write(frame)
        #     continue

        # # 转回数组
        # draw_pts = np.array(list(z_buffer.keys()), dtype=np.int32)
        # depth_draw = np.array(list(z_buffer.values()), dtype=np.float32)

        # colors = _color_by_depth(depth_draw)

        # # ========= 绘制 =========
        # for (u, v), c in zip(draw_pts, colors):
        #     if 0 <= u < width and 0 <= v < height:
        #         cv2.circle(frame, (u, v), 2, (int(c[0]), int(c[1]), int(c[2])), -1)


        # ========= 分层（近 / 远） =========
        z_cam = pts_cam_front[:, 2]

        # 同步mask
        z_cam = z_cam[in_image_mask]
        z_cam = z_cam[valid_mask]
        z_cam = z_cam[color_mask]
        z_cam = z_cam[temporal_mask]
        
        near_mask = pts_cam_front[:, 2][in_image_mask][valid_mask][color_mask][temporal_mask] > 15

        img_near = img_pts_in[near_mask]
        depth_near = depth_in[near_mask]

        img_far = img_pts_in[~near_mask]
        depth_far = depth_in[~near_mask]


        def zbuffer(points_uv, depth):
            zb = {}
            for (u, v), d in zip(points_uv.astype(np.int32), depth):
                key = (u, v)
                if key not in zb or d < zb[key]:
                    zb[key] = d
            if len(zb) == 0:
                return None, None
            pts = np.array(list(zb.keys()), dtype=np.int32)
            dep = np.array(list(zb.values()), dtype=np.float32)
            return pts, dep


        # ========= 远点先画 =========
        far_pts, far_depth = zbuffer(img_far, depth_far)

        if far_pts is not None:
            colors_far = _color_by_depth(far_depth)
            for (u, v), c in zip(far_pts, colors_far):
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame, (u, v), 2, (int(c[0]), int(c[1]), int(c[2])), -1)


        # ========= 近点后画（覆盖） =========
        near_pts, near_depth = zbuffer(img_near, depth_near)

        if near_pts is not None:
            colors_near = _color_by_depth(near_depth)
            for (u, v), c in zip(near_pts, colors_near):
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame, (u, v), 2, (int(c[0]), int(c[1]), int(c[2])), -1)
        
        
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
        return out_video_path
    return None


def run_projection_debug(
    video_path: Path,
    parquet_path: Path,
    output_dir: Path,
    batch_size: int = 500_000,
    time_window_ms: float = 150.0,
    video_start_ns: Optional[int] = None,
    max_frames: Optional[int] = None,
    draw_overlay: bool = True,
) -> None:
    out_video_path = run_projection(
        ProjectionConfig(
            video_path=video_path,
            parquet_path=parquet_path,
            output_dir=output_dir,
            batch_size=batch_size,
            time_window_ms=time_window_ms,
            video_start_ns=video_start_ns,
            max_frames=max_frames,
            draw_overlay=draw_overlay,
        )
    )
    if out_video_path is not None:
        print(f"叠加调试视频已输出: {out_video_path}")


# =========================
# 命令行入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiDAR parquet 与视频投影工具")
    parser.add_argument("--video", required=True, type=Path, help="输入视频路径")
    parser.add_argument("--parquet", required=True, type=Path, help="输入点云 parquet 路径")
    parser.add_argument("--output-dir", default=Path("G:/data/vl_projection_out"), type=Path, help="输出目录")
    parser.add_argument("--batch-size", default=500_000, type=int, help="parquet 分块大小")
    parser.add_argument("--time-window-ms", default=40.0, type=float, help="每帧时间窗口大小（毫秒）")
    parser.add_argument("--video-start-ns", default=None, type=int, help="视频第0帧对应的时间戳(ns)")
    parser.add_argument("--max-frames", default=None, type=int, help="最多处理多少帧")
    parser.add_argument("--draw-overlay", action="store_true", help="是否输出带投影点的视频")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_video_path = run_projection(
        ProjectionConfig(
            video_path=args.video,
            parquet_path=args.parquet,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            time_window_ms=args.time_window_ms,
            video_start_ns=args.video_start_ns,
            max_frames=args.max_frames,
            # draw_overlay=args.draw_overlay,
        )
    )
    if out_video_path is not None:
        print(f"叠加调试视频已输出: {out_video_path}")


if __name__ == "__main__":
    main()
