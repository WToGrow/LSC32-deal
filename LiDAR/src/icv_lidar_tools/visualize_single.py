from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pyarrow.parquet as pq

try:
    from icv_lidar_tools.lidar.pcap_parser_protocol import ProtocolPcapParser
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from icv_lidar_tools.lidar.pcap_parser_protocol import ProtocolPcapParser


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


def _detect_time_column(column_names: list[str]) -> str | None:
    """Detect a reasonable timestamp column from parquet field names."""
    candidates = (
        # "timestamp_ns",
        "timestamp",
        "time_stamp",
        "time",
        "ts",
        "stamp",
        "sec",
        "nsec",
    )
    lowered = {name.lower(): name for name in column_names}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    return None


def _apply_color_mode(pcd: o3d.geometry.PointCloud, xyz: np.ndarray, intensity: np.ndarray, color_mode: str) -> None:
    """Assign colors based on intensity / height / distance."""
    if len(xyz) == 0:
        pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
        return

    mode = color_mode.lower().strip()
    colors = np.zeros((len(xyz), 3), dtype=np.float64)

    if mode == "height":
        feature = _normalize_with_percentile(xyz[:, 2])
    elif mode == "distance":
        feature = _normalize_with_percentile(np.linalg.norm(xyz[:, :3], axis=1))
    else:
        feature = _normalize_with_percentile(intensity)

    colors[:] = _bright_colormap(feature)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def _smooth_scene_points(
    xyz: np.ndarray,
    intensity: np.ndarray,
    scene_scale: float,
    adaptive_distance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Light denoise for scene visualization."""
    if len(xyz) == 0:
        return xyz, intensity

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    voxel_size = max(scene_scale * 0.01, 0.05)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(pcd.points)
    if len(xyz) == 0:
        return xyz, intensity[:0]

    if len(xyz) > 32:
        if adaptive_distance:
            distances = np.linalg.norm(xyz[:, :3], axis=1)
            dist_norm = _normalize_with_percentile(distances, low=5.0, high=95.0)
            nb_neighbors = np.clip(np.round(18 - 4 * dist_norm).astype(int), 10, 18)
            std_ratio = np.clip(1.9 + 0.5 * dist_norm, 1.9, 2.4)
            keep_mask = np.zeros(len(xyz), dtype=bool)
            for neighbor_count in np.unique(nb_neighbors):
                idx = np.flatnonzero(nb_neighbors == neighbor_count)
                sub_pcd = pcd.select_by_index(idx)
                if len(sub_pcd.points) <= neighbor_count + 1:
                    keep_mask[idx] = True
                    continue
                filtered, sub_indices = sub_pcd.remove_statistical_outlier(
                    nb_neighbors=int(neighbor_count),
                    std_ratio=float(np.median(std_ratio[idx])),
                )
                keep_mask[idx[sub_indices]] = True
            if np.any(keep_mask):
                xyz = xyz[keep_mask]
                intensity = intensity[: len(xyz)]
        else:
            filtered, indices = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.1)
            xyz = np.asarray(filtered.points)
            intensity = intensity[indices] if len(intensity) >= len(indices) else intensity[: len(xyz)]

    return xyz, intensity


def _complete_vehicle_by_symmetry(xyz: np.ndarray, intensity: np.ndarray, scene_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Complete vehicle-like clusters by mirroring their left-right symmetry."""
    if len(xyz) == 0:
        return xyz, intensity

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    labels = np.array(pcd.cluster_dbscan(eps=max(scene_scale * 0.035, 0.25), min_points=18, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return xyz, intensity

    completed_xyz = [xyz]
    completed_intensity = [intensity]

    for label in range(labels.max() + 1):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 40:
            continue
        cluster = xyz[idx]
        cluster_extent = cluster.ptp(axis=0)
        if cluster_extent[0] < scene_scale * 0.06 or cluster_extent[1] > cluster_extent[0] * 1.1:
            continue

        y_center = float(np.median(cluster[:, 1]))
        mirrored = cluster.copy()
        mirrored[:, 1] = 2.0 * y_center - mirrored[:, 1]

        merged = np.vstack((cluster, mirrored))
        center = merged.mean(axis=0)
        merged[:, 0] = 0.75 * merged[:, 0] + 0.25 * center[0]
        merged[:, 2] = np.maximum(merged[:, 2], np.percentile(cluster[:, 2], 20.0))
        completed_xyz.append(merged)
        completed_intensity.append(np.full(len(merged), int(np.median(intensity[idx])), dtype=intensity.dtype))

    xyz_out = np.concatenate(completed_xyz, axis=0)
    intensity_out = np.concatenate(completed_intensity, axis=0)
    return xyz_out, intensity_out


def _reconstruct_surface_fill(xyz: np.ndarray, intensity: np.ndarray, scene_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a light surface-inspired fill using voxel centers and local interpolation."""
    if len(xyz) < 50:
        return xyz, intensity

    voxel_size = max(scene_scale * 0.02, 0.08)
    grid = np.floor(xyz / voxel_size)
    unique_grid, inverse = np.unique(grid, axis=0, return_inverse=True)
    centers = np.zeros_like(unique_grid, dtype=np.float64)
    voxel_intensity = np.zeros(len(unique_grid), dtype=np.float64)
    for i in range(len(unique_grid)):
        mask = inverse == i
        centers[i] = xyz[mask].mean(axis=0)
        voxel_intensity[i] = intensity[mask].mean()

    return np.vstack((xyz, centers)), np.concatenate((intensity, voxel_intensity.astype(intensity.dtype)))


def _build_scene_geometries(xyz: np.ndarray, intensity: np.ndarray, color_mode: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _apply_color_mode(pcd, xyz, intensity, color_mode)

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    scene_scale = float(np.max(extent)) if np.all(np.isfinite(extent)) else 1.0
    frame_size = max(scene_scale * 0.06, 0.6)
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    coord_origin = np.array(
        [
            bbox_max[0] + frame_size * 0.28,
            bbox_min[1] - frame_size * 0.28,
            bbox_min[2] + frame_size * 0.06,
        ],
        dtype=np.float64,
    )
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=coord_origin)
    return pcd, coord, bbox, scene_scale


def visualize_points(
    parquet_path: str,
    color_mode: str = "intensity",
    adaptive_distance: bool = False,
    scene_mode: str = "first_scene",
    apply_correction: bool = True,
    do_completion: bool = True,
):
    print("加载点云场景...")
    print(f"当前着色模式：{color_mode}（可选：intensity / height / distance）")
    print(f"清理模式：{'按距离自适应' if adaptive_distance else '统一轻量离群点清理'}")
    print(f"点云修复：{'开启' if apply_correction else '关闭'}")
    print(f"补齐增强：{'开启' if do_completion else '关闭'}")
    print(
        f"场景模式：{scene_mode}（"
        f"first_scene=首个时间戳连续场景，"
        f"all_t0=全文件汇总T0场景并清理，"
        f"all_t0_raw=全文件汇总T0场景不清理）"
    )

    pq_file = pq.ParquetFile(parquet_path)
    schema_names = list(pq_file.schema_arrow.names)
    time_column = _detect_time_column(schema_names)
    if time_column:
        print(f"检测到时间列：{time_column}，将仅显示其首个时间片场景")
    else:
        print("未检测到显式时间列，将仅显示第一批点云")

    chunk_size = 500_000
    accumulated_points = 0
    xyz_list: list[np.ndarray] = []
    intensity_list: list[np.ndarray] = []
    bbox_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bbox_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    first_time_value = None
    collecting_first_scene = True
    mode = scene_mode.lower().strip()
    raw_t0_mode = mode == "all_t0_raw"

    for batch in pq_file.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        remaining = 100_000_000 - accumulated_points
        if remaining <= 0:
            break
        df_chunk = df_chunk.iloc[:remaining]

        if time_column and time_column in df_chunk.columns:
            time_values = df_chunk[time_column].to_numpy()
            if len(time_values) == 0:
                continue

            if first_time_value is None:
                first_time_value = time_values[0]
                print(f"首个时间戳 T0：{first_time_value}")

            if mode in ("all_t0", "all_t0_raw"):
                # 读取整个文件，汇总所有 T0 的点，方便和首帧连续读取模式做对比
                mask = time_values == first_time_value
                if np.any(mask):
                    df_scene = df_chunk.loc[mask].copy()
                    xyz_chunk = df_scene[["x", "y", "z"]].to_numpy()
                    intensity_chunk = df_scene["intensity"].to_numpy()
                    xyz_list.append(xyz_chunk)
                    intensity_list.append(intensity_chunk)
                    bbox_min = np.minimum(bbox_min, xyz_chunk.min(axis=0))
                    bbox_max = np.maximum(bbox_max, xyz_chunk.max(axis=0))
                    accumulated_points += len(df_scene)
                    print(f"已累计全文件T0场景点数：{accumulated_points}")
                continue

            # first_scene：只收集首个时间戳 T0 的点；一旦后续批次不再包含 T0，就停止
            mask = time_values == first_time_value
            if not np.any(mask):
                collecting_first_scene = False
                break

            df_scene = df_chunk.loc[mask].copy()
            xyz_chunk = df_scene[["x", "y", "z"]].to_numpy()
            intensity_chunk = df_scene["intensity"].to_numpy()
            xyz_list.append(xyz_chunk)
            intensity_list.append(intensity_chunk)
            bbox_min = np.minimum(bbox_min, xyz_chunk.min(axis=0))
            bbox_max = np.maximum(bbox_max, xyz_chunk.max(axis=0))
            accumulated_points += len(df_scene)
            print(f"已加载首个时间戳场景点数：{accumulated_points}")

            if np.any(time_values != first_time_value):
                collecting_first_scene = False
                break
        else:
            if accumulated_points > 0:
                break
            if len(df_chunk) > 0:
                xyz_chunk = df_chunk[["x", "y", "z"]].to_numpy()
                intensity_chunk = df_chunk["intensity"].to_numpy()
                xyz_list.append(xyz_chunk)
                intensity_list.append(intensity_chunk)
                bbox_min = np.minimum(bbox_min, xyz_chunk.min(axis=0))
                bbox_max = np.maximum(bbox_max, xyz_chunk.max(axis=0))
                accumulated_points += len(df_chunk)
                print(f"已加载首个场景点数：{accumulated_points}")
                collecting_first_scene = False
                break

    if not xyz_list:
        raise ValueError(f"文件中未读取到有效点云数据: {parquet_path}")

    xyz_all = np.concatenate(xyz_list, axis=0)
    intensity_all = np.concatenate(intensity_list, axis=0)
    print(f"场景点数：{len(xyz_all)}")

    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(xyz_all)
    if np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max)):
        raw_bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    else:
        raw_bbox = raw_pcd.get_axis_aligned_bounding_box()

    raw_extent = raw_bbox.get_extent()
    raw_scene_scale = float(np.max(raw_extent)) if np.all(np.isfinite(raw_extent)) else 1.0

    if apply_correction:
        xyz_all = ProtocolPcapParser.correct_point_cloud_offset_distortion(xyz_all)
        print("已应用相机/雷达偏移与畸变修正")

    if do_completion:
        xyz_all, intensity_all = _complete_vehicle_by_symmetry(xyz_all, intensity_all, raw_scene_scale)
        xyz_all, intensity_all = _reconstruct_surface_fill(xyz_all, intensity_all, raw_scene_scale)
        print("已应用对称补齐与表面增强")

    if raw_t0_mode:
        print("原始T0模式：不做离群点过滤")
    else:
        xyz_all, intensity_all = _smooth_scene_points(
            xyz_all,
            intensity_all,
            raw_scene_scale,
            adaptive_distance=adaptive_distance,
        )
        print(f"清理后点数：{len(xyz_all)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_all)
    _apply_color_mode(pcd, xyz_all, intensity_all, color_mode)

    bbox = pcd.get_axis_aligned_bounding_box()

    extent = bbox.get_extent()
    scene_scale = float(np.max(extent)) if np.all(np.isfinite(extent)) else 1.0
    frame_size = max(scene_scale * 0.12, 1.0)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=bbox.get_center())

    vis = o3d.visualization.Visualizer()
    vis.create_window("LiDAR-GNSS 首个场景", width=1600, height=900)
    vis.add_geometry(pcd)
    vis.add_geometry(coord)

    opt = vis.get_render_option()
    opt.point_size = 1.2
    opt.background_color = np.array([0.03, 0.03, 0.04])

    ctr = vis.get_view_control()
    ctr.set_lookat(bbox.get_center())
    ctr.set_front([-1.0, -1.0, 0.55])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.55)

    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    visualize_points(
        parquet_path="G:/data/parquet/lidar_points_protocol_162920.parquet",
        color_mode="intensity",
        adaptive_distance=False,
        scene_mode="all_t0_raw",
        apply_correction=False,
        do_completion=False,
    )
