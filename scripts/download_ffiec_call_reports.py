#!/usr/bin/env python3
"""Download FFIEC Call Report bulk/XBRL data using ffiec-data-collector.

This is a seed script. The upstream package may change method signatures over time,
so keep it pinned and tested.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from ffiec_data_collector import FFIECDownloader
except Exception as exc:  # pragma: no cover - dependency/runtime guard
    raise SystemExit(
        "Could not import ffiec_data_collector. Install requirements first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quarters",
        nargs="*",
        default=[],
        help="Explicit quarter-end dates, e.g. 20240331 20240630",
    )
    parser.add_argument("--start-year", type=int, help="Optional start year")
    parser.add_argument("--end-year", type=int, help="Optional end year")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/ffiec_call_reports"),
        help="Directory to store downloaded files",
    )
    return parser.parse_args()


def quarter_range(start_year: int, end_year: int) -> list[str]:
    quarters: list[str] = []
    for year in range(start_year, end_year + 1):
        for mmdd in ("0331", "0630", "0930", "1231"):
            quarters.append(f"{year}{mmdd}")
    return quarters


def normalize_download_output(result: object) -> list[Path]:
    if result is None:
        return []
    if isinstance(result, (str, Path)):
        return [Path(result)]
    if isinstance(result, (list, tuple, set)):
        paths: list[Path] = []
        for item in result:
            if isinstance(item, (str, Path)):
                paths.append(Path(item))
        return paths
    return []


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    quarters = list(args.quarters)
    if args.start_year and args.end_year:
        quarters.extend(quarter_range(args.start_year, args.end_year))
    quarters = sorted(set(quarters))
    if not quarters:
        raise SystemExit("Supply --quarters or both --start-year and --end-year")

    downloader = FFIECDownloader()

    for quarter in quarters:
        print(f"Downloading FFIEC Call Report data for {quarter}...", file=sys.stderr)
        result = downloader.download_cdr_single_period(quarter)
        paths = normalize_download_output(result)

        if not paths:
            print(
                f"WARNING: download method returned no file paths for {quarter}. "
                "Check the installed ffiec-data-collector version.",
                file=sys.stderr,
            )
            continue

        quarter_dir = args.out_dir / quarter
        quarter_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            if not src.exists():
                print(f"WARNING: reported path does not exist: {src}", file=sys.stderr)
                continue
            dest = quarter_dir / src.name
            if src.resolve() != dest.resolve():
                shutil.move(str(src), dest)
            print(f"Saved {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
