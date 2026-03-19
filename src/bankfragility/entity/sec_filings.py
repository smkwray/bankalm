"""Download and parse SEC 10-Q/10-K filings for uninsured-deposit disclosures.

Fetches filing indexes from EDGAR, downloads HTML filings, and extracts
Item 1406 (Regulation S-K) uninsured deposit disclosures and basic
securities-footnote data where available.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from bankfragility.tables import save_table

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{doc}"
SEC_USER_AGENT = "bankalm-research/1.0 research@example.com"
SEC_RATE_LIMIT = 0.12  # seconds between requests (SEC limit: 10/sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True, help="SEC mapping parquet with CERT, SEC_CIK")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/sec/filings"), help="Cache directory for HTML filings")
    parser.add_argument("--max-filings", type=int, default=4, help="Max filings per CIK to download")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def fetch_filing_index(
    cik: int,
    session: requests.Session,
) -> list[dict[str, str]]:
    """Fetch recent 10-Q/10-K filing metadata from EDGAR."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    r = session.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    filings = []
    for i, form in enumerate(forms):
        if form in ("10-Q", "10-K"):
            acc_clean = accessions[i].replace("-", "")
            filings.append({
                "cik": cik,
                "form": form,
                "filing_date": dates[i],
                "report_date": report_dates[i] if i < len(report_dates) else "",
                "accession": accessions[i],
                "url": SEC_ARCHIVES_URL.format(
                    cik=cik, accession_clean=acc_clean, doc=docs[i],
                ),
                "doc": docs[i],
            })
    return filings


def download_filing_html(
    url: str,
    cache_path: Path,
    session: requests.Session,
) -> str:
    """Download filing HTML, using cache if available."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    time.sleep(SEC_RATE_LIMIT)
    r = session.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=120)
    r.raise_for_status()
    html = r.text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


def parse_uninsured_deposits(html: str) -> dict[str, Any]:
    """Extract uninsured deposit amounts from filing HTML.

    Searches for Item 1406 (Regulation S-K) disclosure patterns and
    dollar amounts near "uninsured" deposit keywords.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    result: dict[str, Any] = {
        "sec_uninsured_found": False,
        "sec_uninsured_amount": None,
        "sec_uninsured_context": None,
    }

    # Search for uninsured deposit disclosure patterns
    patterns = [
        r"uninsured\s+(?:domestic\s+)?deposits?\s+(?:were|was|of|totaled|totaling|approximat\w+)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand)?",
        r"(?:estimated|approximate)\s+(?:amount\s+of\s+)?uninsured\s+deposits?\s+\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand)?",
        r"deposits?\s+(?:that\s+)?exceed(?:ed|ing)?\s+(?:the\s+)?(?:FDIC\s+)?(?:insurance\s+)?(?:limit|coverage)\s+(?:were|was|totaled)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            try:
                amount = float(amount_str)
            except ValueError:
                continue

            # Determine scale from context
            context_window = text[max(0, match.start() - 50):match.end() + 100]
            context_lower = context_window.lower()
            if "billion" in context_lower:
                amount *= 1_000_000  # convert to thousands
            elif "million" in context_lower:
                amount *= 1_000  # convert to thousands
            elif amount > 1_000_000:
                pass  # likely already in thousands

            result["sec_uninsured_found"] = True
            result["sec_uninsured_amount"] = amount
            result["sec_uninsured_context"] = context_window[:200].strip()
            return result

    # Fallback: try inline XBRL tag (us-gaap:DepositLiabilityUninsured)
    xbrl_match = re.search(
        r'name="us-gaap:DepositLiabilityUninsured"[^>]*?scale="(\d+)"[^>]*>([\d,]+(?:\.\d+)?)</ix:nonFraction>',
        html, re.IGNORECASE | re.DOTALL,
    )
    if not xbrl_match:
        # Try alternate attribute order
        xbrl_match = re.search(
            r'name="us-gaap:DepositLiabilityUninsured"[^>]*>([\d,]+(?:\.\d+)?)</ix:nonFraction>',
            html, re.IGNORECASE | re.DOTALL,
        )
    if xbrl_match:
        groups = xbrl_match.groups()
        if len(groups) == 2:
            scale = int(groups[0])
            raw_val = float(groups[1].replace(",", ""))
            amount = raw_val * (10 ** scale) / 1000  # normalize to thousands
        else:
            raw_val = float(groups[0].replace(",", ""))
            amount = raw_val  # assume already in thousands
        result["sec_uninsured_found"] = True
        result["sec_uninsured_amount"] = amount
        result["sec_uninsured_context"] = f"XBRL: us-gaap:DepositLiabilityUninsured = {amount:,.0f} (thousands)"

    return result


def parse_securities_footnote(html: str) -> dict[str, Any]:
    """Extract basic securities composition from filing HTML."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    result: dict[str, Any] = {
        "sec_afs_found": False,
        "sec_htm_found": False,
    }

    # Check for AFS/HTM mentions
    if re.search(r"available[\-\s]for[\-\s]sale", text, re.IGNORECASE):
        result["sec_afs_found"] = True
    if re.search(r"held[\-\s]to[\-\s]maturity", text, re.IGNORECASE):
        result["sec_htm_found"] = True

    return result


def process_filings(
    mapping: pd.DataFrame,
    cache_dir: Path,
    max_filings: int = 4,
) -> pd.DataFrame:
    """Download and parse filings for all mapped banks."""
    session = requests.Session()
    rows: list[dict[str, Any]] = []

    mapped = mapping[mapping["SEC_CIK"].notna()].copy()
    if mapped.empty:
        return pd.DataFrame()

    for _, bank in mapped.iterrows():
        cik = int(bank["SEC_CIK"])
        cert = str(bank["CERT"])
        ticker = bank.get("SEC_TICKER", "")

        print(f"Processing CIK {cik} ({ticker})...", file=sys.stderr)

        try:
            filings = fetch_filing_index(cik, session)
        except Exception as e:
            print(f"  Failed to fetch index: {e}", file=sys.stderr)
            continue

        time.sleep(SEC_RATE_LIMIT)

        for filing in filings[:max_filings]:
            cache_path = cache_dir / f"{cik}" / f"{filing['accession'].replace('-', '_')}_{filing['doc']}"

            try:
                html = download_filing_html(filing["url"], cache_path, session)
            except Exception as e:
                print(f"  Failed to download {filing['doc']}: {e}", file=sys.stderr)
                continue

            uninsured = parse_uninsured_deposits(html)
            securities = parse_securities_footnote(html)

            row = {
                "CERT": cert,
                "SEC_CIK": cik,
                "SEC_TICKER": ticker,
                "FORM": filing["form"],
                "FILING_DATE": filing["filing_date"],
                "REPORT_DATE": filing["report_date"],
                "FILING_URL": filing["url"],
                **uninsured,
                **securities,
            }
            rows.append(row)

            status = "FOUND" if uninsured["sec_uninsured_found"] else "not found"
            print(f"  {filing['filing_date']} {filing['form']}: uninsured {status}", file=sys.stderr)

    return pd.DataFrame(rows)


def run_filing_parser(args: argparse.Namespace) -> pd.DataFrame:
    mapping = pd.read_parquet(args.mapping)
    result = process_filings(mapping, args.cache_dir, args.max_filings)
    if result.empty:
        print("No filings processed.", file=sys.stderr)
        return result
    save_table(result, args.out)
    found = result["sec_uninsured_found"].sum()
    print(f"Processed {len(result)} filings, found uninsured disclosure in {found}. Saved to {args.out}", file=sys.stderr)
    return result


def main() -> None:
    args = parse_args()
    run_filing_parser(args)


if __name__ == "__main__":
    main()
