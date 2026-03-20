#!/usr/bin/env python3
"""Check pipeline data freshness, coverage, and integrity."""
from __future__ import annotations

import argparse
import calendar
import glob
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SITE = ROOT / "site" / "data"

SOURCE_STARTS = {
    "financials": date(2007, 3, 31),
    "ffiec": date(2020, 3, 31),
}
SOD_START_YEAR = 2010
SOD_MAX_YEAR = 2024
TREASURY_START_YEAR = 2022


def _quarter_end_for(year: int, quarter: int) -> date:
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = ends[quarter]
    return date(year, month, day)


def latest_completed_quarter(today: date | None = None) -> date:
    today = today or date.today()
    quarter = (today.month - 1) // 3 + 1
    end = _quarter_end_for(today.year, quarter)
    if today < end:
        quarter -= 1
        if quarter == 0:
            return _quarter_end_for(today.year - 1, 4)
        end = _quarter_end_for(today.year, quarter)
    return end


def quarter_range(start: date, end: date) -> list[str]:
    out: list[str] = []
    current = start
    while current <= end:
        out.append(current.strftime("%Y%m%d"))
        month = current.month + 3
        year = current.year
        if month > 12:
            month -= 12
            year += 1
        last_day = calendar.monthrange(year, month)[1]
        current = date(year, month, last_day)
    return out


def latest_completed_year(today: date | None = None) -> int:
    return (today or date.today()).year - 1


def source_window(kind: str, today: date | None = None) -> list[str] | str:
    today = today or date.today()
    latest_quarter = latest_completed_quarter(today)
    latest_year = latest_completed_year(today)
    if kind == "financials":
        return quarter_range(SOURCE_STARTS["financials"], latest_quarter)
    if kind == "ffiec":
        return quarter_range(SOURCE_STARTS["ffiec"], latest_quarter)
    if kind == "report-quarter":
        return latest_quarter.strftime("%Y%m%d")
    if kind == "sod-years":
        return [str(year) for year in range(SOD_START_YEAR, min(latest_year, SOD_MAX_YEAR) + 1)]
    if kind == "treasury-years":
        return [str(year) for year in range(TREASURY_START_YEAR, latest_year + 1)]
    raise ValueError(f"Unknown source window kind: {kind}")


def _format_source_window(kind: str, today: date | None = None) -> str:
    value = source_window(kind, today=today)
    if isinstance(value, list):
        return " ".join(value)
    return value


def _parse_date_from_filename(path: str) -> date | None:
    name = Path(path).name
    for pattern, fmt in ((r"(20\d{6})", "%Y%m%d"), (r"((?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])20\d{2})", "%m%d%Y")):
        match = re.search(pattern, name)
        if not match:
            continue
        try:
            return datetime.strptime(match.group(1), fmt).date()
        except ValueError:
            continue
    return None


def _parse_year_from_filename(path: str) -> int | None:
    match = re.search(r"(20\d{2})", Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def _max_present(values):
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _lag_quarters(actual: date | None, target: date) -> int | None:
    if actual is None:
        return None
    return (target.year - actual.year) * 4 + ((target.month - 1) // 3) - ((actual.month - 1) // 3)


def _lag_years(actual: int | None, target: int) -> int | None:
    if actual is None:
        return None
    return target - actual


def _source_status(label: str, actual: str | int | None, target: str | int, lag: int | None, unit: str) -> None:
    if actual is None:
        print(f"  [MISS] {label}: no files found (target {target})")
        return
    lag_text = f"{lag}{unit}" if lag is not None else "n/a"
    print(f"  [OK]   {label}: latest {actual} / target {target} / lag {lag_text}")


def check_files(label: str, pattern: str) -> int:
    files = sorted(glob.glob(str(ROOT / pattern)))
    count = len(files)
    total_size = sum(Path(f).stat().st_size for f in files)
    if count > 0:
        print(f"  [OK]   {label}: {count} files ({total_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  [MISS] {label}: no files found")
    return count


def check_parquet(label: str, path: str):
    import pandas as pd

    p = ROOT / path
    if not p.exists():
        print(f"  [MISS] {label}: {path}")
        return None
    df = pd.read_parquet(p)
    size_mb = p.stat().st_size / 1024 / 1024
    quarters = df["REPDTE"].nunique() if "REPDTE" in df.columns else "N/A"
    banks = df["CERT"].nunique() if "CERT" in df.columns else len(df)
    print(f"  [OK]   {label}: {len(df):,} rows, {len(df.columns)} cols, {banks:,} banks, {quarters} quarters ({size_mb:.1f} MB)")
    return df


def report_source_freshness(today: date | None = None) -> None:
    today = today or date.today()
    latest_q = latest_completed_quarter(today)
    latest_y = latest_completed_year(today)

    financials_files = sorted(glob.glob(str(ROOT / "data/raw/fdic/universe_financials_*.parquet")))
    financials_latest = _max_present(_parse_date_from_filename(f) for f in financials_files)
    _source_status(
        "FDIC financials",
        financials_latest.strftime("%Y-%m-%d") if financials_latest else None,
        latest_q.strftime("%Y-%m-%d"),
        _lag_quarters(financials_latest, latest_q),
        "Q",
    )

    ffiec_files = sorted(glob.glob(str(ROOT / "data/raw/ffiec/FFIEC CDR Call Bulk All Schedules *.zip")))
    ffiec_latest = _max_present(_parse_date_from_filename(f) for f in ffiec_files)
    _source_status(
        "FFIEC CDR zips",
        ffiec_latest.strftime("%Y-%m-%d") if ffiec_latest else None,
        latest_q.strftime("%Y-%m-%d"),
        _lag_quarters(ffiec_latest, latest_q),
        "Q",
    )

    sod_files = sorted(glob.glob(str(ROOT / "data/raw/fdic/universe_sod_*.parquet")))
    sod_latest = _max_present(_parse_year_from_filename(f) for f in sod_files)
    sod_target = min(latest_y, SOD_MAX_YEAR)
    _source_status(
        "FDIC SOD",
        str(sod_latest) if sod_latest is not None else None,
        str(sod_target),
        _lag_years(sod_latest, sod_target),
        "Y",
    )

    treasury_files = sorted(glob.glob(str(ROOT / "data/raw/treasury/yields_*.parquet")))
    treasury_latest = _max_present(_parse_year_from_filename(f) for f in treasury_files)
    _source_status(
        "Treasury yields",
        str(treasury_latest) if treasury_latest is not None else None,
        str(latest_y),
        _lag_years(treasury_latest, latest_y),
        "Y",
    )


def print_source_window(kind: str) -> None:
    print(_format_source_window(kind))


def _load_site_freshness() -> dict[str, object] | None:
    path = SITE / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pipeline data freshness, coverage, and integrity.")
    parser.add_argument(
        "--source-window",
        choices=["financials", "ffiec", "report-quarter", "sod-years", "treasury-years"],
        help="Print the canonical source window for Makefile consumption and exit.",
    )
    args, _ = parser.parse_known_args()

    if args.source_window:
        print_source_window(args.source_window)
        return

    import pandas as pd

    print("=" * 70)
    print("PIPELINE STATUS CHECK")
    print("=" * 70)

    print("\n--- Freshness ---")
    report_source_freshness()
    site_manifest = _load_site_freshness()
    if site_manifest and isinstance(site_manifest.get("freshness"), dict):
        freshness = site_manifest["freshness"]
        snapshot = freshness.get("site_snapshot_as_of", "n/a")
        stale = freshness.get("stale", False)
        warnings = freshness.get("coverage_warnings") or []
        status = "STALE" if stale else "OK"
        print(f"  [{status}] Site manifest snapshot: as of {snapshot}")
        for warning in warnings:
            print(f"         - {warning}")

    print("\n--- Raw Data ---")
    check_files("FDIC financials", "data/raw/fdic/universe_financials_*.parquet")
    check_files("FDIC institutions", "data/raw/fdic/universe_institutions.parquet")
    check_files("FDIC SOD", "data/raw/fdic/universe_sod_*.parquet")
    check_files("FDIC failures", "data/raw/fdic/failures.parquet")
    check_files("FFIEC CDR zips", "data/raw/ffiec/*.zip")
    check_files("Treasury yields", "data/raw/treasury/yields_*.parquet")
    check_files("SEC tickers cache", "data/raw/sec/company_tickers.json")
    check_files("SEC filing cache", "data/raw/sec/filings/*/*.htm*")

    print("\n--- Processed Outputs ---")
    check_parquet("Bank panel", "data/processed/universe_bank_panel.parquet")
    check_parquet("Stickiness", "data/processed/universe_deposit_stickiness.parquet")
    check_parquet("ALM mismatch", "data/processed/universe_alm_features.parquet")
    check_parquet("Treasury ext", "data/processed/universe_treasury_features.parquet")
    df = check_parquet("Indices", "data/processed/universe_bank_indices.parquet")
    check_parquet("Supervised", "data/processed/universe_supervised_stickiness.parquet")
    check_parquet("Publishable mart", "data/processed/universe_publishable_mart.parquet")
    check_parquet("Core panel", "data/processed/universe_core_panel.parquet")
    check_parquet("Enriched panel", "data/processed/universe_enriched_panel.parquet")
    check_parquet("Failure backtest", "data/processed/universe_failure_backtest.parquet")
    check_parquet("SEC mapping", "data/processed/universe_sec_mapping.parquet")
    check_parquet("SEC filings", "data/processed/smoke_sec_filings.parquet")

    print("\n--- FFIEC Repricing Extractions ---")
    check_files("Repricing parquets", "data/processed/ffiec_repricing_*.parquet")

    print("\n--- Reports ---")
    check_files("Universe reports", "data/reports/universe/*.parquet")
    check_files("2023 validation reports", "data/reports/validation_2023/*.parquet")
    check_files("2008 validation reports", "data/reports/validation_2008/*.parquet")

    if df is not None:
        print("\n--- Data Layer Coverage ---")
        print(f"{'Quarter':>12s}  {'Banks':>6s}  {'SOD':>5s}  {'FFIEC':>5s}  {'TSY':>5s}  {'Version':>16s}")
        for q in sorted(df["REPDTE"].unique()):
            qdata = df[df["REPDTE"] == q]
            sod = qdata["SOD_TOTAL_DEPOSITS"].notna().sum() if "SOD_TOTAL_DEPOSITS" in qdata.columns else 0
            ffiec = qdata["DURATION_GAP_LITE"].notna().sum() if "DURATION_GAP_LITE" in qdata.columns else 0
            tsy = qdata["HAS_TREASURY_YIELD_HISTORY"].fillna(0).astype(int).sum() if "HAS_TREASURY_YIELD_HISTORY" in qdata.columns else 0
            ver = qdata["INDEX_VERSION"].iloc[0] if "INDEX_VERSION" in qdata.columns else "?"
            print(f"{pd.Timestamp(q).strftime('%Y-%m-%d'):>12s}  {len(qdata):>6,}  {sod:>5,}  {ffiec:>5,}  {tsy:>5,}  {ver:>16s}")

    print("\n--- Integrity ---")
    from bankfragility.validation.consistency import validate_panel
    if df is not None:
        report = validate_panel(df)
        impossible = report[report["CHECK"].isin(["COREDEP_GT_DEPDOM", "SCPLEDGE_GT_SC", "NEGATIVE_DEPOSIT"])]
        if impossible.empty:
            print("  [OK]   No impossible-ratio violations")
        else:
            print(f"  [WARN] {len(impossible)} impossible-ratio violations found")

        dupes = df.duplicated(subset=["CERT", "REPDTE"], keep=False)
        if not dupes.any():
            print("  [OK]   No duplicate CERT+REPDTE rows")
        else:
            print(f"  [WARN] {dupes.sum()} duplicate CERT+REPDTE rows")

    print("\n" + "=" * 70)
    print("STATUS CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
