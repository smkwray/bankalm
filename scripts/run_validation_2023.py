#!/usr/bin/env python3
"""Run the 2023 regional bank stress validation and produce a report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bankfragility.tables import read_table


def main() -> None:
    indices_path = ROOT / "data" / "processed" / "validation_bank_indices.parquet"
    if not indices_path.exists():
        raise SystemExit(
            "Run the full validation pipeline first. See handoff.md for commands."
        )

    df = read_table(indices_path)

    # Q4-2022: last full quarter before SVB failure
    q4 = df[df["REPDTE"] == pd.Timestamp("2022-12-31")].copy()
    svb = q4[q4["CERT"].astype(str) == "24735"]

    if svb.empty:
        raise SystemExit("SVB (CERT 24735) not found in validation data.")

    svb_row = svb.iloc[0]

    print("=" * 70)
    print("2023 REGIONAL BANK STRESS VALIDATION")
    print(f"Quarter: Q4-2022 | Banks: {len(q4)}")
    print("=" * 70)

    # Cross-peer raw feature ranking
    print("\nSVB CROSS-PEER RANKING (Q4-2022):")
    for col, label, ascending in [
        ("UNINSURED_SHARE", "Uninsured share", False),
        ("TREASURY_TO_UNINSURED", "Treasury / uninsured", True),
        ("VOLATILE_TO_LIQUID_LOWER", "Volatile / liquid", False),
    ]:
        if col not in q4.columns:
            continue
        valid = q4[col].dropna()
        rank = (valid > svb_row[col]).sum() + 1 if not ascending else (valid < svb_row[col]).sum() + 1
        print(f"  {label:>30s}: {svb_row[col]:8.4f}  (rank {rank}/{len(valid)})")

    # SVB time series
    svb_ts = df[df["CERT"].astype(str) == "24735"].sort_values("REPDTE")
    if len(svb_ts) > 1:
        print("\nSVB DETERIORATION (time series):")
        first = svb_ts.iloc[0]
        last = svb_ts.iloc[-1]
        for col, label in [
            ("UNINSURED_SHARE", "Uninsured share"),
            ("VOLATILE_TO_LIQUID_LOWER", "Volatile/liquid"),
            ("TREASURY_TO_UNINSURED", "Treasury coverage"),
        ]:
            if col in svb_ts.columns:
                v0, v1 = first[col], last[col]
                delta = v1 - v0
                print(f"  {label:>25s}: {v0:.4f} → {v1:.4f}  (Δ{delta:+.4f})")

    # Verdict
    print("\nVERDICT:")
    checks_passed = 0
    total_checks = 0

    # Check 1: SVB has highest uninsured share
    total_checks += 1
    if svb_row["UNINSURED_SHARE"] == q4["UNINSURED_SHARE"].max():
        checks_passed += 1
        print("  [PASS] SVB has the highest uninsured share in the sample")
    else:
        print("  [FAIL] SVB does not have the highest uninsured share")

    # Check 2: SVB is in worst quartile on VOLATILE_TO_LIQUID_LOWER
    total_checks += 1
    vtl_pct = (q4["VOLATILE_TO_LIQUID_LOWER"] <= svb_row["VOLATILE_TO_LIQUID_LOWER"]).mean()
    if vtl_pct >= 0.5:
        checks_passed += 1
        print(f"  [PASS] SVB volatile/liquid ratio is worse than {vtl_pct:.0%} of sample")
    else:
        print(f"  [FAIL] SVB volatile/liquid ratio is only worse than {vtl_pct:.0%}")

    # Check 3: SVB Treasury coverage is below median
    total_checks += 1
    treas_valid = q4["TREASURY_TO_UNINSURED"].dropna()
    if len(treas_valid) > 0:
        treas_median = treas_valid.median()
        if svb_row["TREASURY_TO_UNINSURED"] < treas_median:
            checks_passed += 1
            print(f"  [PASS] SVB Treasury coverage ({svb_row['TREASURY_TO_UNINSURED']:.1%}) below median ({treas_median:.1%})")
        else:
            print(f"  [FAIL] SVB Treasury coverage above median")

    print(f"\n  Result: {checks_passed}/{total_checks} validation checks passed")


if __name__ == "__main__":
    main()
