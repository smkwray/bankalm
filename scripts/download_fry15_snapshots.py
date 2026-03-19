#!/usr/bin/env python3
"""Scrape and download FR Y-15 snapshot CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PAGE_URL = "https://www.ffiec.gov/npw/FinancialReport/FRY15Reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-url", default=PAGE_URL)
    parser.add_argument("--match", nargs="*", default=[], help="Optional substrings to filter the CSV links")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def discover_csv_links(page_url: str) -> list[str]:
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        full_url = urljoin(page_url, href)
        if full_url.lower().endswith(".csv") and full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)
    return urls


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def main() -> None:
    args = parse_args()
    urls = discover_csv_links(args.page_url)
    if args.match:
        urls = [u for u in urls if all(m.lower() in u.lower() for m in args.match)]

    if not urls:
        raise SystemExit("No CSV links discovered.")

    for url in urls:
        print(url)
        if args.list_only:
            continue
        download_file(url, args.out_dir / Path(url).name)


if __name__ == "__main__":
    main()
