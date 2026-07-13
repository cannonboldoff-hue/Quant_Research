"""Parquet/CSV helpers with a consistent schema. Prefer parquet for speed."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def write_ohlcv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path) if path.suffix == ".parquet" else df.to_csv(path, index=False)
