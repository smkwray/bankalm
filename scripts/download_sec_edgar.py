#!/usr/bin/env python3
"""Download SEC EDGAR issuer-level data and recent public filings.

Examples
--------
python scripts/download_sec_edgar.py --tickers BAC JPM USB --download submissions companyfacts recent-filings --out-dir data/raw/sec

python scripts/download_sec_edgar.py --bulk submissions companyfacts --out-dir data/raw/sec
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
BULK_COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
BULK_SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/submissions/submissions.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=[], help="Ticker symbols")
    parser.add_argument("--tickers-file", type=Path, help="CSV file with a ticker column")
    parser.add_argument("--ciks", nargs="*", default=[], help="Explicit CIKs")
    parser.add_argument(
        "--download",
        nargs="*",
        default=["submissions", "companyfacts"],
        choices=["submissions", "companyfacts", "recent-filings"],
        help="Issuer-level payloads to download",
    )
    parser.add_argument(
        "--bulk",
        nargs="*",
        default=[],
        choices=["submissions", "companyfacts"],
        help="Optional bulk archives to download",
    )
    parser.add_argument(
        "--forms",
        nargs="*",
        default=["10-K", "10-Q"],
        help="When downloading recent filings, restrict to these forms",
    )
    parser.add_argument("--max-filings", type=int, default=6, help="Recent filings per issuer")
    parser.add_argument("--sleep", type=float, default=0.25, help="Pause between SEC requests")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def get_headers() -> dict[str, str]:
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        print(
            "WARNING: SEC_USER_AGENT is not set. Set it to a descriptive value like "
            "'Your Name your_email@example.com'.",
            file=sys.stderr,
        )
        user_agent = "public-data-bank-project research@example.com"
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def save_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_json(session: requests.Session, url: str) -> object:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_bytes(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = [t.strip().upper() for t in args.tickers if t.strip()]
    if args.tickers_file:
        df = pd.read_csv(args.tickers_file)
        if "ticker" in [c.lower() for c in df.columns]:
            ticker_col = next(c for c in df.columns if c.lower() == "ticker")
            tickers.extend([str(x).strip().upper() for x in df[ticker_col].dropna()])
        else:
            tickers.extend([str(x).strip().upper() for x in df.iloc[:, 0].dropna()])
    return sorted(set(tickers))


def parse_ticker_map(payload: object) -> dict[str, str]:
    # SEC's file is usually {"fields":[...], "data":[...]}.
    if isinstance(payload, dict) and "fields" in payload and "data" in payload:
        fields = payload["fields"]
        rows = payload["data"]
        try:
            ticker_ix = fields.index("ticker")
            cik_ix = fields.index("cik")
        except ValueError:
            return {}
        mapping = {}
        for row in rows:
            if ticker_ix < len(row) and cik_ix < len(row):
                mapping[str(row[ticker_ix]).upper()] = str(row[cik_ix]).zfill(10)
        return mapping

    # Fallback if the structure changes.
    mapping: dict[str, str] = {}
    if isinstance(payload, dict):
        for _, row in payload.items():
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).upper().strip()
            cik = str(row.get("cik_str", row.get("cik", ""))).strip()
            if ticker and cik:
                mapping[ticker] = cik.zfill(10)
    return mapping


def recent_filing_rows(submissions_payload: object) -> list[dict[str, object]]:
    if not isinstance(submissions_payload, dict):
        return []
    recent = (
        submissions_payload.get("filings", {}) or {}
    ).get("recent", {}) or {}
    if not isinstance(recent, dict):
        return []

    keys = list(recent.keys())
    if not keys:
        return []

    row_count = max(len(recent.get(k, [])) for k in keys)
    rows: list[dict[str, object]] = []
    for i in range(row_count):
        row = {}
        for key in keys:
            values = recent.get(key, [])
            row[key] = values[i] if i < len(values) else None
        rows.append(row)
    return rows


def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_document}"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(get_headers())

    if args.bulk:
        if "companyfacts" in args.bulk:
            save_bytes(fetch_bytes(session, BULK_COMPANYFACTS_URL), args.out_dir / "companyfacts.zip")
        if "submissions" in args.bulk:
            save_bytes(fetch_bytes(session, BULK_SUBMISSIONS_URL), args.out_dir / "submissions.zip")

    tickers = load_tickers(args)
    ciks = {str(c).zfill(10) for c in args.ciks if str(c).strip()}

    ticker_map = {}
    if tickers and not ciks:
        ticker_map_payload = fetch_json(session, TICKER_MAP_URL)
        save_json(ticker_map_payload, args.out_dir / "ticker_map.json")
        ticker_map = parse_ticker_map(ticker_map_payload)
        missing = [t for t in tickers if t not in ticker_map]
        if missing:
            print(f"WARNING: could not resolve tickers: {missing}", file=sys.stderr)
        ciks.update({ticker_map[t] for t in tickers if t in ticker_map})

    for cik in sorted(ciks):
        issuer_dir = args.out_dir / cik
        issuer_dir.mkdir(parents=True, exist_ok=True)

        submissions_payload = None
        if "submissions" in args.download or "recent-filings" in args.download:
            submissions_payload = fetch_json(session, SUBMISSIONS_URL.format(cik=cik))
            save_json(submissions_payload, issuer_dir / "submissions.json")
            time.sleep(args.sleep)

        if "companyfacts" in args.download:
            companyfacts_payload = fetch_json(session, COMPANYFACTS_URL.format(cik=cik))
            save_json(companyfacts_payload, issuer_dir / "companyfacts.json")
            time.sleep(args.sleep)

        if "recent-filings" in args.download:
            if submissions_payload is None:
                submissions_payload = fetch_json(session, SUBMISSIONS_URL.format(cik=cik))
                time.sleep(args.sleep)
            rows = recent_filing_rows(submissions_payload)
            allowed_forms = {f.upper() for f in args.forms}
            selected = [
                row for row in rows
                if str(row.get("form", "")).upper() in allowed_forms
                and row.get("accessionNumber")
                and row.get("primaryDocument")
            ][: args.max_filings]

            filings_dir = issuer_dir / "filings"
            filings_dir.mkdir(parents=True, exist_ok=True)

            for row in selected:
                accession = str(row["accessionNumber"])
                primary_doc = str(row["primaryDocument"])
                form = str(row.get("form", "UNK")).replace("/", "_")
                filing_date = str(row.get("filingDate", "unknown"))
                url = filing_document_url(cik, accession, primary_doc)
                content = fetch_bytes(session, url)
                filename = f"{filing_date}_{form}_{primary_doc}"
                save_bytes(content, filings_dir / filename)
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
