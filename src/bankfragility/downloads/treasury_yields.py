"""Download daily Treasury yield curve data from Treasury.gov.

Tries the direct CSV endpoint first, then falls back to XML parsing.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import requests

CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
    "?type={series}&field_tdr_date_value={year}&page&_format=csv"
)
XML_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xmlview?data={series}&field_tdr_date_value={year}"
)

# Standard tenors in the yield curve CSV
TENOR_COLUMNS = [
    "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo",
    "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="daily_treasury_yield_curve")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output .csv or .parquet")
    return parser.parse_args()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to uppercase with underscore separators."""
    rename: dict[str, str] = {}
    for col in df.columns:
        clean = str(col).strip()
        if clean.lower() == "date":
            rename[col] = "DATE"
        else:
            # "1 Mo" → "YC_1MO", "10 Yr" → "YC_10YR"
            upper = clean.upper().replace(" ", "")
            rename[col] = f"YC_{upper}"
    return df.rename(columns=rename)


def fetch_csv(
    year: int,
    series: str = "daily_treasury_yield_curve",
    session: requests.Session | None = None,
) -> pd.DataFrame | None:
    """Try the direct CSV endpoint."""
    session = session or requests.Session()
    url = CSV_URL.format(year=year, series=series)
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
    except Exception:
        return None
    text = r.text.strip()
    if not text:
        return None
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return None
    return _normalize_columns(df)


def _parse_xml_entries(xml_bytes: bytes) -> pd.DataFrame:
    """Parse Treasury XML Atom feed into a DataFrame."""
    root = ElementTree.fromstring(xml_bytes)
    ns_data = "http://schemas.microsoft.com/ado/2007/08/dataservices"
    ns_meta = "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
    ns_atom = "http://www.w3.org/2005/Atom"

    rows: list[dict[str, Any]] = []
    for entry in root.iter(f"{{{ns_atom}}}entry"):
        props = entry.find(f".//{{{ns_meta}}}properties")
        if props is None:
            continue
        row: dict[str, Any] = {}
        for child in props:
            tag = re.sub(r"\{.*\}", "", child.tag)
            text = child.text
            row[tag] = text
        rows.append(row)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Rename known XML columns
    rename: dict[str, str] = {}
    for col in df.columns:
        upper = col.upper()
        if upper == "NEW_DATE":
            rename[col] = "DATE"
        elif upper.startswith("BC_"):
            # BC_1MONTH → YC_1MO, BC_2YEAR → YC_2YR, etc.
            tenor = upper[3:].replace("MONTH", "MO").replace("YEAR", "YR")
            rename[col] = f"YC_{tenor}"
        else:
            rename[col] = upper
    df = df.rename(columns=rename)
    # Convert rate columns to float
    for col in df.columns:
        if col.startswith("YC_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_xml(
    year: int,
    series: str = "daily_treasury_yield_curve",
    session: requests.Session | None = None,
) -> pd.DataFrame | None:
    """Fall back to the XML Atom feed."""
    session = session or requests.Session()
    url = XML_URL.format(year=year, series=series)
    try:
        r = session.get(url, timeout=120)
        r.raise_for_status()
    except Exception:
        return None
    df = _parse_xml_entries(r.content)
    return df if not df.empty else None


def download_yields(
    year: int,
    series: str = "daily_treasury_yield_curve",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download Treasury yield curve for a given year.  CSV first, then XML."""
    session = session or requests.Session()
    df = fetch_csv(year, series, session)
    if df is not None and not df.empty:
        return df
    df = fetch_xml(year, series, session)
    if df is not None and not df.empty:
        return df
    raise RuntimeError(f"Could not download Treasury yields for {year}.")


def run_download(args: argparse.Namespace) -> pd.DataFrame:
    df = download_yields(year=args.year, series=args.series)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".parquet":
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}", file=sys.stderr)
    return df


def main() -> None:
    args = parse_args()
    run_download(args)


if __name__ == "__main__":
    main()
