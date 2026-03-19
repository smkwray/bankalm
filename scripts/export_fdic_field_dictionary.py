#!/usr/bin/env python3
"""Export useful sheets from FDIC's Common Financial Reports workbook to CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reference/All_Financial_Reports.xlsx"),
        help="Path to the FDIC workbook",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reference"),
        help="Output directory for CSV exports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    workbook = pd.ExcelFile(args.input)
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name)
        safe_name = (
            sheet_name.lower()
            .replace(" ", "_")
            .replace("&", "and")
            .replace("-", "_")
        )
        out_path = args.out_dir / f"{safe_name}.csv"
        df.to_csv(out_path, index=False)

    print(f"Exported {len(workbook.sheet_names)} sheets to {args.out_dir}")


if __name__ == "__main__":
    main()
