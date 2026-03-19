#!/usr/bin/env python3
"""Scrape and download NIC structure bulk files from the FFIEC page."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PAGE_URL = "https://www.ffiec.gov/npw/FinancialReport/DataDownload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-url", default=PAGE_URL)
    parser.add_argument("--match", nargs="*", default=[], help="Optional substrings to filter links")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def discover_links(page_url: str) -> list[tuple[str, str]]:
    headers = {"User-Agent": "bankalm-research/1.0 (public-data pipeline)"}
    response = requests.get(page_url, headers=headers, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        full_url = urljoin(page_url, href)
        text = " ".join(anchor.stripped_strings) or Path(href).name
        if any(full_url.lower().endswith(ext) for ext in (".zip", ".csv", ".xml", ".txt")):
            if full_url not in seen:
                seen.add(full_url)
                links.append((text, full_url))
    return links


def matches(url: str, label: str, filters: list[str]) -> bool:
    if not filters:
        return True
    haystack = f"{label} {url}".lower()
    return all(f.lower() in haystack for f in filters)


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
    links = [
        (label, url)
        for label, url in discover_links(args.page_url)
        if matches(url, label, args.match)
    ]

    if not links:
        raise SystemExit("No matching download links discovered.")

    for label, url in links:
        filename = Path(url).name
        print(f"{label}: {url}")
        if args.list_only:
            continue
        download_file(url, args.out_dir / filename)


if __name__ == "__main__":
    main()
