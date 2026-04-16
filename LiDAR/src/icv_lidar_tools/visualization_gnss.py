from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_gnss_track(df: pd.DataFrame, out_path: str | Path | None = None) -> None:
    if not {"longitude", "latitude"}.issubset(df.columns):
        raise ValueError("DataFrame must contain longitude and latitude columns")

    plt.figure(figsize=(8, 6))
    plt.plot(df["longitude"], df["latitude"], linewidth=1.2)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("GNSS Track")
    plt.grid(True, linestyle="--", alpha=0.5)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lidar_xy(points_df: pd.DataFrame, out_path: str | Path | None = None) -> None:
    if not {"x", "y"}.issubset(points_df.columns):
        raise ValueError("DataFrame must contain x and y columns")

    plt.figure(figsize=(8, 6))
    plt.scatter(points_df["x"], points_df["y"], s=1)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("LiDAR XY Projection")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
