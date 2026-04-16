from __future__ import annotations

import pandas as pd


class TimeAligner:
    """Timestamp-based nearest-neighbor aligner."""

    @staticmethod
    def align_nearest(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_ts_col: str = "timestamp",
        right_ts_col: str = "timestamp",
        tolerance_ms: int = 50,
    ) -> pd.DataFrame:
        left = left_df.copy()
        right = right_df.copy()

        left[left_ts_col] = pd.to_datetime(left[left_ts_col], utc=True, errors="coerce")
        right[right_ts_col] = pd.to_datetime(right[right_ts_col], utc=True, errors="coerce")

        left = left.dropna(subset=[left_ts_col]).sort_values(left_ts_col)
        right = right.dropna(subset=[right_ts_col]).sort_values(right_ts_col)

        aligned = pd.merge_asof(
            left,
            right,
            left_on=left_ts_col,
            right_on=right_ts_col,
            direction="nearest",
            tolerance=pd.Timedelta(milliseconds=tolerance_ms),
            suffixes=("_left", "_right"),
        )
        return aligned
