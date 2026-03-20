"""Download selected FRED series to a wide daily table.

This is optional support for richer market-rate history. The deposit competition
builder works without this file and can use Treasury-only inputs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from bankfragility.tables import save_table

API_URL = "https://api.stlouisfed.org/fred/series/observations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series-id",
        action="append",
        required=True,
        help="Repeat for each FRED series, e.g. --series-id IORB",
    )
    parser.add_argument("--start", default="", help="Optional observation_start YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Optional observation_end YYYY-MM-DD")
    parser.add_argument("--api-key-env", default="FRED_API_KEY")
    parser.add_argument("--out", type=Path, required=True, help="Output .csv or .parquet")
    return parser.parse_args()


def observations_to_frame(series_id: str, observations: Sequence[dict[str, Any]]) -> pd.DataFrame:
    if not observations:
        return pd.DataFrame(
            {
                "DATE": pd.Series(dtype="datetime64[ns]"),
                series_id.upper(): pd.Series(dtype="float64"),
            }
        )

    raw = pd.DataFrame(observations)
    out = pd.DataFrame(
        {
            "DATE": pd.to_datetime(raw["date"], errors="coerce"),
            series_id.upper(): pd.to_numeric(raw["value"].replace(".", pd.NA), errors="coerce"),
        }
    )
    out = out.dropna(subset=["DATE"]).sort_values("DATE")
    out = out.drop_duplicates(subset=["DATE"], keep="last")
    return out.reset_index(drop=True)


def merge_series_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame({"DATE": pd.Series(dtype="datetime64[ns]")})

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="DATE", how="outer")
    out = out.sort_values("DATE").drop_duplicates(subset=["DATE"], keep="last")
    return out.reset_index(drop=True)


def fetch_series(
    series_id: str,
    api_key: str,
    start: str = "",
    end: str = "",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    session = session or requests.Session()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end

    response = session.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations", [])
    return observations_to_frame(series_id=series_id, observations=observations)


def download_series(
    series_ids: Sequence[str],
    api_key: str,
    start: str = "",
    end: str = "",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    session = session or requests.Session()
    frames = [
        fetch_series(series_id=series_id, api_key=api_key, start=start, end=end, session=session)
        for series_id in series_ids
    ]
    return merge_series_frames(frames)


def run_download(args: argparse.Namespace) -> pd.DataFrame:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{args.api_key_env} is not set. "
            "Set a FRED API key in the environment before using this optional downloader."
        )

    df = download_series(
        series_ids=args.series_id,
        api_key=api_key,
        start=args.start,
        end=args.end,
    )
    save_table(df, args.out)
    print(f"Saved {len(df):,} rows to {args.out}", file=sys.stderr)
    return df


def main() -> None:
    args = parse_args()
    run_download(args)


if __name__ == "__main__":
    main()
