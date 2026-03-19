#!/usr/bin/env python3
"""Scan downloaded FFIEC XBRL/XML files for maturity/repricing-related concept labels.

This is intentionally simple. It is a seed discovery tool to help identify
candidate XBRL concepts to map into standard repricing buckets.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_KEYWORDS = [
    "maturity",
    "repricing",
    "reprice",
    "remaining",
    "three months",
    "twelve months",
    "one year",
    "three years",
    "five years",
    "fifteen years",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Folder of XBRL/XML files")
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_KEYWORDS,
        help="Keywords to search for",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".xml", ".xsd", ".xbrl", ".htm", ".html"],
        help="File extensions to scan",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = [re.compile(re.escape(k), re.IGNORECASE) for k in args.keywords]

    rows: list[dict[str, object]] = []
    for path in args.root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {ext.lower() for ext in args.extensions}:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(p.search(line) for p in patterns):
                rows.append(
                    {
                        "file": str(path),
                        "line_number": lineno,
                        "snippet": line.strip()[:1000],
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "line_number", "snippet"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} candidate snippets to {args.out}")


if __name__ == "__main__":
    main()
