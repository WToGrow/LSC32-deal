from __future__ import annotations

from pathlib import Path

import pandas as pd


class Exporter:
    @staticmethod
    def to_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index, encoding="utf-8-sig")

    @staticmethod
    def to_excel(df: pd.DataFrame, path: str | Path, index: bool = False, sheet_name: str = "data") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=index, sheet_name=sheet_name)
