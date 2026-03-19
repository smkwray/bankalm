#!/usr/bin/env python3
"""Check pipeline data freshness, coverage, and integrity."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def check_files(label: str, pattern: str) -> int:
    files = sorted(glob.glob(str(ROOT / pattern)))
    count = len(files)
    total_size = sum(Path(f).stat().st_size for f in files)
    if count > 0:
        print(f"  [OK]   {label}: {count} files ({total_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  [MISS] {label}: no files found")
    return count


def check_parquet(label: str, path: str) -> pd.DataFrame | None:
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


def main() -> None:
    print("=" * 70)
    print("PIPELINE STATUS CHECK")
    print("=" * 70)

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
    check_parquet("SEC mapping", "data/processed/smoke_sec_mapping.parquet")
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

    # Validation checks
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
