"""Shared table IO helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    df.columns = [str(col).upper() for col in df.columns]
    return df


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")


def parse_report_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    yyyymmdd_mask = text.str.fullmatch(r"\d{8}").fillna(False)

    if yyyymmdd_mask.any():
        out.loc[yyyymmdd_mask] = pd.to_datetime(
            text.loc[yyyymmdd_mask],
            format="%Y%m%d",
            errors="coerce",
        )
    if (~yyyymmdd_mask).any():
        out.loc[~yyyymmdd_mask] = pd.to_datetime(series.loc[~yyyymmdd_mask], errors="coerce")

    return out
