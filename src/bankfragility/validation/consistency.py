"""Internal consistency checks for the bank-quarter panel and feature outputs.

Each check returns a DataFrame of flagged rows with a CHECK and DETAIL column.
``validate_panel`` runs all checks and returns a combined report.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _flag(df: pd.DataFrame, mask: pd.Series, check: str, detail: str) -> pd.DataFrame:
    """Build a flagged-row slice with CHECK and DETAIL columns."""
    flagged = df.loc[mask, ["CERT", "REPDTE"]].copy()
    flagged["CHECK"] = check
    flagged["DETAIL"] = detail
    return flagged


# ---------------------------------------------------------------------------
# Impossible ratio checks
# ---------------------------------------------------------------------------

def check_coredep_exceeds_depdom(df: pd.DataFrame) -> pd.DataFrame:
    """COREDEP should never exceed DEPDOM."""
    if "COREDEP" not in df.columns or "DEPDOM" not in df.columns:
        return pd.DataFrame()
    mask = df["COREDEP"] > df["DEPDOM"]
    return _flag(df, mask, "COREDEP_GT_DEPDOM", "COREDEP > DEPDOM")


def check_pledged_exceeds_securities(df: pd.DataFrame) -> pd.DataFrame:
    """Pledged securities should not exceed the total securities book."""
    if "SCPLEDGE" not in df.columns or "SC" not in df.columns:
        return pd.DataFrame()
    mask = df["SCPLEDGE"] > df["SC"]
    return _flag(df, mask, "SCPLEDGE_GT_SC", "SCPLEDGE > SC")


def check_negative_deposits(df: pd.DataFrame) -> pd.DataFrame:
    """Core deposit and domestic deposit columns should not be negative."""
    flags: list[pd.DataFrame] = []
    for col in ["DEPDOM", "COREDEP", "DEP"]:
        if col in df.columns:
            mask = df[col] < 0
            if mask.any():
                flags.append(_flag(df, mask, "NEGATIVE_DEPOSIT", f"{col} < 0"))
    return pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()


def check_shares_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Deposit-composition shares should be in [0, 1]."""
    share_cols = [
        "UNINSURED_SHARE", "BROKERED_SHARE", "LIST_SERVICE_SHARE",
        "LARGE_ACCOUNT_SHARE", "CORE_DEPOSIT_SHARE", "NONINTEREST_SHARE",
        "TIME_DEPOSIT_SHARE", "SHORT_FHLB_SHARE",
    ]
    flags: list[pd.DataFrame] = []
    for col in share_cols:
        if col not in df.columns:
            continue
        mask_neg = df[col] < 0
        mask_high = df[col] > 1.0
        if mask_neg.any():
            flags.append(_flag(df, mask_neg, "SHARE_NEGATIVE", f"{col} < 0"))
        if mask_high.any():
            flags.append(_flag(df, mask_high, "SHARE_GT_1", f"{col} > 1.0"))
    return pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()


def check_negative_coverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage and liquidity ratios should not be negative."""
    coverage_cols = [
        "TREASURY_TO_UNINSURED", "TREASURY_AGENCY_TO_RUNNABLE",
        "HQLA_NARROW_LOWER_TO_RUNNABLE", "VOLATILE_TO_LIQUID_LOWER",
    ]
    flags: list[pd.DataFrame] = []
    for col in coverage_cols:
        if col in df.columns:
            mask = df[col] < 0
            if mask.any():
                flags.append(_flag(df, mask, "NEGATIVE_COVERAGE", f"{col} < 0"))
    return pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()


# ---------------------------------------------------------------------------
# Balance / bucket identity checks
# ---------------------------------------------------------------------------

def check_securities_bucket_sum(df: pd.DataFrame, tolerance: float = 0.20) -> pd.DataFrame:
    """Securities maturity buckets should not exceed the debt-securities total.

    SCNM* buckets cover non-mortgage securities only, so sum < total is expected.
    Only flag when buckets exceed the total (impossible) or when total is positive
    but all buckets are zero (suspicious missing data).
    """
    bucket_cols = ["SCNM3LES", "SCNM3T12", "SCNM1T3", "SCNM3T5", "SCNM5T15", "SCNMOV15"]
    total_col = "SCRDEBT" if "SCRDEBT" in df.columns else "SC"
    if total_col not in df.columns:
        return pd.DataFrame()
    present = [c for c in bucket_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    bucket_sum = df[present].sum(axis=1)
    total = pd.to_numeric(df[total_col], errors="coerce")
    ratio = np.where(total > 0, bucket_sum / total, np.nan)
    series = pd.Series(ratio, index=df.index)
    # Flag if buckets exceed total (impossible direction)
    mask = series.notna() & (series > 1.0 + tolerance)
    return _flag(df, mask, "SEC_BUCKET_SUM_EXCEEDS_TOTAL", f"sum(SCNM*) exceeds {total_col} by >{tolerance:.0%}")


def check_fhlb_bucket_sum(df: pd.DataFrame, tolerance: float = 0.20) -> pd.DataFrame:
    """FHLB term buckets should roughly sum to total FHLB borrowings."""
    bucket_cols = ["OTBFH1L", "OTBFH1T3", "OTBFH3T5", "OTBFHOV5"]
    total_col = "OTHBFHLB"
    if total_col not in df.columns:
        return pd.DataFrame()
    present = [c for c in bucket_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    bucket_sum = df[present].sum(axis=1)
    total = pd.to_numeric(df[total_col], errors="coerce")
    ratio = np.where(total > 0, bucket_sum / total, np.nan)
    mask = pd.Series(ratio, index=df.index)
    mask = mask.notna() & ((mask < 1.0 - tolerance) | (mask > 1.0 + tolerance))
    return _flag(df, mask, "FHLB_BUCKET_SUM_MISMATCH", f"sum(OTBFH*) deviates from OTHBFHLB by >{tolerance:.0%}")


# ---------------------------------------------------------------------------
# Temporal and entity stability checks
# ---------------------------------------------------------------------------

def check_repdte_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """Report dates should be monotonically increasing within each CERT."""
    if "CERT" not in df.columns or "REPDTE" not in df.columns:
        return pd.DataFrame()
    sorted_df = df.sort_values(["CERT", "REPDTE"])
    prev = sorted_df.groupby("CERT")["REPDTE"].shift(1)
    mask = sorted_df["REPDTE"] <= prev
    mask = mask.fillna(False)
    return _flag(sorted_df, mask, "NON_MONOTONIC_REPDTE", "REPDTE <= previous REPDTE for same CERT")


def check_entity_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """Flag banks where RSSDHCR changes between consecutive quarters (potential merger/restructure)."""
    if "CERT" not in df.columns or "RSSDHCR" not in df.columns:
        return pd.DataFrame()
    sorted_df = df.sort_values(["CERT", "REPDTE"])
    prev_hcr = sorted_df.groupby("CERT")["RSSDHCR"].shift(1)
    mask = (
        sorted_df["RSSDHCR"].notna()
        & prev_hcr.notna()
        & (sorted_df["RSSDHCR"] != prev_hcr)
    )
    return _flag(sorted_df, mask, "RSSDHCR_CHANGE", "RSSDHCR changed between consecutive quarters")


# ---------------------------------------------------------------------------
# SOD checks
# ---------------------------------------------------------------------------

def check_sod_positive_deposits(df: pd.DataFrame) -> pd.DataFrame:
    """SOD total deposits should be positive when present."""
    if "SOD_TOTAL_DEPOSITS" not in df.columns:
        return pd.DataFrame()
    mask = df["SOD_TOTAL_DEPOSITS"].notna() & (df["SOD_TOTAL_DEPOSITS"] <= 0)
    return _flag(df, mask, "SOD_NONPOSITIVE_DEPOSITS", "SOD_TOTAL_DEPOSITS <= 0")


def check_sod_hhi_range(df: pd.DataFrame) -> pd.DataFrame:
    """HHI values should be in (0, 1]."""
    flags: list[pd.DataFrame] = []
    for col in ["SOD_DEPOSIT_HHI_STATE", "SOD_DEPOSIT_HHI_COUNTY"]:
        if col not in df.columns:
            continue
        valid = df[col].notna()
        mask = valid & ((df[col] < 0) | (df[col] > 1.0))
        if mask.any():
            flags.append(_flag(df, mask, "SOD_HHI_OUT_OF_RANGE", f"{col} outside [0, 1]"))
    return pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_coredep_exceeds_depdom,
    check_pledged_exceeds_securities,
    check_negative_deposits,
    check_shares_out_of_range,
    check_negative_coverage_ratios,
    check_securities_bucket_sum,
    check_fhlb_bucket_sum,
    check_repdte_monotonicity,
    check_entity_jumps,
    check_sod_positive_deposits,
    check_sod_hhi_range,
]


def validate_panel(df: pd.DataFrame, checks: list[Any] | None = None) -> pd.DataFrame:
    """Run all consistency checks and return a combined report of flagged rows.

    Returns an empty DataFrame if no issues are found.
    """
    checks = checks or ALL_CHECKS
    flags: list[pd.DataFrame] = []
    for check_fn in checks:
        result = check_fn(df)
        if not result.empty:
            flags.append(result)
    if not flags:
        return pd.DataFrame(columns=["CERT", "REPDTE", "CHECK", "DETAIL"])
    return pd.concat(flags, ignore_index=True)
