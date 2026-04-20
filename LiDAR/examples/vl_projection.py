from __future__ import annotations

"""Video-LiDAR projection debugger.

功能说明
--------
1. 读取视频和 parquet 点云。
2. 以 `timestamp_ns` 为基准，把点云按时间窗口对齐到每一帧视频。
3. 将雷达点从“世界坐标系”平移到“相机坐标系”（当前不考虑旋转）。
4. 先用畸变系数做投影，再用相机内参得到像素坐标。
5. 以流式 / 分块方式处理大 parquet，避免一次性把全量点云加载进内存。
6. 每帧输出详细调试信息，方便检查时间同步、空间变换和画面覆盖范围。

使用建议
--------
- 你的 parquet 最好按 `timestamp_ns` 升序保存；如果不是升序，本脚本仍可运行，但时间窗口的严格性会下降。
- 如果视频起始时间和点云起始时间不一致，可以通过 `--video-start-ns` 手动指定视频第一帧对应的时间戳。
- 当前外参默认只使用平移，不使用旋转。如果后续拿到真实旋转矩阵，替换 `R_WORLD_TO_CAM` 即可。
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
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
# 设备在世界坐标系中的位置（单位：cm）
# 说明：
# - 这里按你的要求，把这些值看作世界坐标系下的位置。
# - 脚本默认用“仅平移、无旋转”的方式做世界坐标 → 相机坐标变换。
# - 由于你提到“正前参考点”可看作相机所在位置，这里默认用它作为相机平移基准。
# =========================
LIDAR_WORLD_POS = np.array([1.5, -291.8, 187.8], dtype=np.float32)
FRONT_REF_WORLD_POS = np.array([-1.5, -203.8, 169.8], dtype=np.float32)
MMWAVE_WORLD_POS = np.array([0.0, 0.0, 58.5], dtype=np.float32)

# 当前默认把相机放在“正前参考点”
CAMERA_WORLD_POS = FRONT_REF_WORLD_POS.copy()


# R_WORLD_TO_CAM = np.eye(3, dtype=np.float32) # 无旋转
R_LIDAR_TO_CAM = np.array([
    # [ 0,  -1,  0],
    # [ 0,  0, -1],
    # [ 1,  0,  0]
    [ 0,  0,  1],
    [ 1,  0,  0],
    [ 0,  1,  0]
], dtype=np.float32)
R_WORLD_TO_CAM = R_LIDAR_TO_CAM

# T_WORLD_TO_CAM = (-R_WORLD_TO_CAM @ CAMERA_WORLD_POS.reshape(3, 1)).astype(np.float32)
T_WORLD_TO_CAM = np.zeros((3, 1), dtype=np.float32) # 因为有世界坐标，所以无平移

R_VEC, _ = cv2.Rodrigues(R_WORLD_TO_CAM)
R_VEC = R_VEC.astype(np.float32)
T_VEC = T_WORLD_TO_CAM.astype(np.float32)


# =========================
# 数据结构
# =========================
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
    return int(video_start_ns + round(frame_idx * 1_000_000_000.0 / fps))


def _world_to_camera(points_world: np.ndarray) -> np.ndarray:
    """世界坐标系 → 相机坐标系（当前仅平移，不含旋转）。"""
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    
    # 修复：打印信息
    res = (R_WORLD_TO_CAM @ pts.T).T + T_WORLD_TO_CAM.reshape(1, 3)
    print("相机坐标系 Z 值范围：", res[:,2].min(), res[:,2].max())

    # 修复：Z 轴取反，让点出现在相机前方
    # res[:, 2] = -res[:, 2]

    # 修复：全局坐标反向，视角彻底镜像
    # res = -res
    return res


def _project_camera_points(points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把相机坐标系下的点投影到像素平面。

    返回值：
    - img_pts: 像素坐标 (u, v)
    - depth_z: 相机坐标系下的 Z 深度
    - front_mask: Z>0 的点掩码结果
    """
    if len(points_cam) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )

    pts = np.asarray(points_cam, dtype=np.float32)
    # front_mask = pts[:, 2] > 0.1
    # front_mask = pts[:, 2] < -0.1
    front_mask = pts[:, 1] > 0.1 #正前方
    # front_mask = pts[:, 1] < -0.1
    # front_mask = pts[:, 0] > 0.1
    # front_mask = pts[:, 0] < -0.1
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
    img_pts, _ = cv2.projectPoints(pts, R_VEC, T_VEC, CAMERA_MATRIX, DIST_COEFFS)
    img_pts = img_pts.reshape(-1, 2)
    depth_z = pts[:, 2].copy()

    # finite = np.isfinite(img_pts).all(axis=1)
    # return img_pts[finite], depth_z[finite], front_mask
    # ============修复：不移除 finite 点，保证长度和 front_mask 匹配=================
    return img_pts, depth_z, front_mask


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


# =========================
# 流式 parquet 读取与逐帧匹配
# =========================
def _iter_point_batches(parquet_path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """按批次读取 parquet，避免一次性加载大文件。"""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        yield df


def _stream_frames_points(
    parquet_path: Path,
    fps: float,
    video_start_ns: int,
    time_window_ns: int,
    batch_size: int,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """流式地产生“某帧对应的点云窗口”。

    这里采用“时间有序”的处理方式：
    - 先按批读取 parquet
    - 每批内部按 timestamp_ns 排序
    - 将点按时间推进到当前帧窗口中

    适合 parquet 大文件，内存占用较小。
    如果你的 parquet 全局不是按 timestamp_ns 升序，建议先预排序。
    """
    current_frame_idx = 0
    current_frame_ns = _frame_time_ns(current_frame_idx, fps, video_start_ns)
    half_window = time_window_ns // 2
    window_start = current_frame_ns - half_window
    window_end = current_frame_ns + half_window

    buffer_parts: list[pd.DataFrame] = []
    buffer_min_ts = None

    for df in _iter_point_batches(parquet_path, batch_size=batch_size):
        _require_columns(df, ["timestamp_ns", "x", "y", "z"])
        if len(df) == 0:
            continue

        # 只保留当前需要的字段，降低后续开销
        df = df[["timestamp_ns", "x", "y", "z", "intensity", "ring"]].copy() if all(c in df.columns for c in ["intensity", "ring"]) else df[["timestamp_ns", "x", "y", "z"]].copy()
        df = df.sort_values("timestamp_ns", kind="mergesort")

        if buffer_min_ts is None and len(df) > 0:
            buffer_min_ts = int(df["timestamp_ns"].iloc[0])

        buffer_parts.append(df)
        buffer = pd.concat(buffer_parts, ignore_index=True)

        # 按时间窗口不断吐出可处理的帧
        while len(buffer) > 0:
            # 当前帧的窗口中，取出落在时间范围内的点
            mask = (buffer["timestamp_ns"].to_numpy() >= window_start) & (buffer["timestamp_ns"].to_numpy() <= window_end)
            frame_df = buffer.loc[mask]

            # 处理完当前窗口之前的点：这些点不会再用于未来帧
            # 由于窗口是连续向前移动的，因此小于当前窗口起点的点可以丢弃
            keep_mask = buffer["timestamp_ns"].to_numpy() >= window_start
            buffer = buffer.loc[keep_mask].reset_index(drop=True)
            buffer_parts = [buffer] if len(buffer) else []

            yield current_frame_idx, current_frame_ns, frame_df

            current_frame_idx += 1
            current_frame_ns = _frame_time_ns(current_frame_idx, fps, video_start_ns)
            window_start = current_frame_ns - half_window
            window_end = current_frame_ns + half_window

            # 如果缓存里已经没有更早的点，而新的批次还没到下一帧时间，暂停吐出
            if len(buffer) == 0:
                break


# =========================
# 每帧投影与调试输出
# =========================
def _project_frame_points(
    frame_df: pd.DataFrame,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一帧窗口内的点做空间变换与投影。"""
    if len(frame_df) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )

    xyz = frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
    pts_cam = _world_to_camera(xyz)
    img_pts, depth_z, front_mask = _project_camera_points(pts_cam)

    # front_mask 是对原始点做的前方筛选，img_pts/depth_z 是筛选后的结果
    return img_pts, pts_cam[front_mask], front_mask


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


def run_projection_debug(
    video_path: Path,
    parquet_path: Path,
    output_dir: Path,
    batch_size: int = 500_000,
    time_window_ms: float = 40.0,
    video_start_ns: Optional[int] = None,
    max_frames: Optional[int] = None,
    draw_overlay: bool = False,
) -> None:
    cap, fps, width, height, frame_count = _load_video_meta(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if video_start_ns is None:
        # 默认策略：把视频第 0 帧时间映射到点云第 0 个时间戳。
        # 如果你的相机和雷达有更准确的同步基准，建议用命令行参数显式指定。
        pq_file = pq.ParquetFile(parquet_path)
        first_batch = next(pq_file.iter_batches(batch_size=1), None)
        if first_batch is None:
            raise ValueError(f"parquet 中没有数据: {parquet_path}")
        first_df = first_batch.to_pandas()
        _require_columns(first_df, ["timestamp_ns"])
        video_start_ns = int(first_df["timestamp_ns"].iloc[0])

    time_window_ns = int(time_window_ms * 1_000_000)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_video_path = output_dir / f"{video_path.stem}_projection_{timestamp_tag}}.mp4"
    writer = None
    if draw_overlay:
        writer = cv2.VideoWriter(
            str(out_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    print("========== LiDAR / Video 投影调试开始 ==========")
    print(f"视频: {video_path}")
    print(f"点云: {parquet_path}")
    print(f"视频尺寸: {width}x{height}, fps={fps:.3f}, frame_count={frame_count}")
    print(f"时间窗口: ±{time_window_ms/2:.1f} ms")
    print(f"相机世界坐标位置(cm): {CAMERA_WORLD_POS.tolist()}")
    print(f"激光雷达世界坐标位置(cm): {LIDAR_WORLD_POS.tolist()}")
    print(f"毫米波雷达世界坐标位置(cm): {MMWAVE_WORLD_POS.tolist()}")
    print(f"视频起始时间戳(video_start_ns): {video_start_ns}")

    frame_limit = frame_count if max_frames is None else min(frame_count, max_frames)

    # 为了满足“分块流式处理”，这里用 parquet 批量推进时间窗口。
    # 你可以把 batch_size 设大一点提高吞吐，也可以设小一点减少内存。
    frame_iter = _stream_frames_points(
        parquet_path=parquet_path,
        fps=fps,
        video_start_ns=video_start_ns,
        time_window_ns=time_window_ns,
        batch_size=batch_size,
    )

    for frame_idx, frame_time_ns, frame_df in frame_iter:
        if frame_idx >= frame_limit:
            break

        # 从视频中读出当前帧
        ret, frame = cap.read()
        if not ret:
            print(f"视频已结束，停止于 frame_idx={frame_idx}")
            break

        if len(frame_df) == 0:
            print(
                f"[frame {frame_idx:05d}] t={frame_time_ns} ns | window_points=0 | projected=0 | no lidar points in window"
            )
            if writer is not None:
                writer.write(frame)
            continue

        xyz = frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
        pts_cam = _world_to_camera(xyz)

        # front_mask = pts_cam[:, 2] > 0.1
        # pts_cam_front = pts_cam[front_mask]
        # 投影
        # img_pts, _depth_z, _ = _project_camera_points(pts_cam)
        # in_image_mask = _clip_to_image(img_pts, width, height)
        # img_pts_in = img_pts[in_image_mask]
        # pts_cam_in = pts_cam_front[in_image_mask]

        # ====================== 修复 ======================
        # 1. 先获取前方点掩码 + 投影点
        img_pts, depth_z, front_mask = _project_camera_points(pts_cam)
        
        # 2. 计算画面内掩码（和 img_pts 长度完全一致）
        in_image_mask = _clip_to_image(img_pts, width, height)
        
        # 3. 维度严格匹配：img_pts → in_image_mask → pts_cam_front
        pts_cam_front = pts_cam[front_mask]  # 前方点
        img_pts_in = img_pts[in_image_mask]   # 画面内投影点
        pts_cam_in = pts_cam_front[in_image_mask]  # 画面内相机坐标点
        
        # 统计调试信息
        image_bbox = _bbox_int(img_pts_in)
        device_bbox = _bbox_cm(xyz)

        info = FrameProjectionInfo(
            frame_idx=frame_idx,
            frame_time_ns=frame_time_ns,
            point_count_in_window=len(frame_df),
            point_count_in_front=len(pts_cam_front),
            projected_count=len(img_pts_in),
            image_bbox=image_bbox,
            device_bbox_cm=device_bbox,
        )

        # 输出每一帧的调试信息，便于排查：
        # 1) 时间窗口内是否有点
        # 2) 点是否在相机前方
        # 3) 投影后是否落在图像范围内
        # 4) 投影结果是否出现异常坐标
        print(
            f"[frame {info.frame_idx:05d}] t={info.frame_time_ns} ns | "
            f"window_points={info.point_count_in_window} | front_points={info.point_count_in_front} | "
            f"projected_in_image={info.projected_count}"
        )
        if info.image_bbox is not None:
            u0, v0, u1, v1 = info.image_bbox
            print(f"  image_bbox(u,v)=({u0}, {v0}) -> ({u1}, {v1})")
        else:
            print("  image_bbox(u,v)=None")

        if info.device_bbox_cm is not None:
            x0, x1, y0, y1, z0, z1 = info.device_bbox_cm
            print(
                f"  world_bbox(cm): x[{x0:.2f}, {x1:.2f}], y[{y0:.2f}, {y1:.2f}], z[{z0:.2f}, {z1:.2f}]"
            )

        # 把投影点画到当前帧上，方便人工检查。
        # 颜色采用红色小圆点，若你想看深度分布，可自行改成伪彩色。
        if len(img_pts_in) > 0:
            draw_pts = img_pts_in.astype(np.int32, copy=False)
            for (u, v) in draw_pts:
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame, (int(u), int(v)), 2, (0, 0, 255), -1)

        # 如果要额外观察相机画面与投影，可以输出到视频
        if writer is not None:
            cv2.putText(
                frame,
                f"frame={frame_idx} points={len(img_pts_in)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"叠加调试视频已输出: {out_video_path}")

    print("========== LiDAR / Video 投影调试结束 ==========")


# =========================
# 命令行入口
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiDAR parquet 与视频逐帧投影调试脚本")
    parser.add_argument("--video", required=True, type=Path, help="输入视频路径")
    parser.add_argument("--parquet", required=True, type=Path, help="输入点云 parquet 路径")
    parser.add_argument("--output-dir", default=Path("./vl_projection_out"), type=Path, help="输出目录")
    parser.add_argument("--batch-size", default=500_000, type=int, help="parquet 分块大小")
    parser.add_argument("--time-window-ms", default=40.0, type=float, help="每帧时间窗口大小（毫秒）")
    parser.add_argument("--video-start-ns", default=None, type=int, help="视频第0帧对应的时间戳(ns)，不填则自动用 parquet 首点时间")
    parser.add_argument("--max-frames", default=None, type=int, help="最多处理多少帧，用于调试")
    parser.add_argument("--draw-overlay", action="store_true", help="是否输出带投影点的调试视频")
    return parser


def main() -> None:
    video_path = Path(r"G:\data\2025-01-12\video\20260112_162910_正前_000.mp4")
    parquet_path = Path(r"G:\data\2025-01-12\parquet_out\lidar_points_protocol_162920.parquet")
    output_dir = Path(r"G:\data\vl_projection_out")
    batch_size = 500_000
    time_window_ms = 100.0
    video_start_ns = None
    max_frames = 50    #None
    draw_overlay = True
    # args = build_arg_parser().parse_args()
    run_projection_debug(
        # video_path=args.video,
        # parquet_path=args.parquet,
        # output_dir=args.output_dir,
        # batch_size=args.batch_size,
        # time_window_ms=args.time_window_ms,
        # video_start_ns=args.video_start_ns,
        # max_frames=args.max_frames,
        # draw_overlay=args.draw_overlay,
        video_path=video_path,
        parquet_path=parquet_path,
        output_dir=output_dir,
        batch_size=batch_size,
        time_window_ms=time_window_ms,
        video_start_ns=video_start_ns,
        max_frames=max_frames,
        draw_overlay=draw_overlay,
    )


if __name__ == "__main__":
    main()
