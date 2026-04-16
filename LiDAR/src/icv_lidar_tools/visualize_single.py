import numpy as np
import open3d as o3d
import pyarrow.parquet as pq


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


def visualize_points(parquet_path: str, color_mode: str = "intensity"):
    print("加载点云场景...")
    print(f"当前着色模式：{color_mode}（可选：intensity / height / distance）")
    print("已固定为 first_scene 模式：仅显示首个时间戳连续场景")

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

            mask = time_values == first_time_value
            if not np.any(mask):
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
                break

    if not xyz_list:
        raise ValueError(f"文件中未读取到有效点云数据: {parquet_path}")

    xyz_all = np.concatenate(xyz_list, axis=0)
    intensity_all = np.concatenate(intensity_list, axis=0)
    print(f"场景点数：{len(xyz_all)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_all)
    _apply_color_mode(pcd, xyz_all, intensity_all, color_mode)

    bbox = pcd.get_axis_aligned_bounding_box()
    scene_scale = float(np.max(bbox.get_extent())) if np.all(np.isfinite(bbox.get_extent())) else 1.0
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
    )
