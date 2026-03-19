"""Extract repricing and maturity bucket data from FFIEC Call Report bulk TSV files.

Reads schedule-level TSV files from a Call Report zip, extracts MDRM-coded
maturity bucket fields, maps IDRSSD to CERT, and produces a bank-quarter
DataFrame with standardized repricing columns.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bankfragility.tables import save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="Path to FFIEC CDR Call Bulk zip")
    parser.add_argument("--map", type=Path, required=True, help="YAML repricing map config")
    parser.add_argument(
        "--institutions",
        type=Path,
        help="Parquet with CERT and FED_RSSD columns for IDRSSD→CERT mapping",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_schedule_from_zip(
    zf: zipfile.ZipFile,
    schedule_pattern: str,
) -> pd.DataFrame:
    """Read one or more parts of a schedule from the zip, concatenate columns."""
    matching = [
        name
        for name in zf.namelist()
        if schedule_pattern in name and name.endswith(".txt")
    ]
    if not matching:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for name in sorted(matching):
        with zf.open(name) as f:
            raw = pd.read_csv(f, sep="\t", dtype=str)
        # Row 0 is labels, row 1+ is data
        data = raw.iloc[1:].copy()
        data.columns = raw.columns
        frames.append(data)

    if not frames:
        return pd.DataFrame()

    # Multi-part schedules share IDRSSD — merge on it
    result = frames[0]
    for extra in frames[1:]:
        new_cols = [c for c in extra.columns if c not in result.columns]
        if new_cols:
            result = result.merge(
                extra[["IDRSSD"] + new_cols],
                on="IDRSSD",
                how="outer",
            )
    return result


def get_schedule_labels(zf: zipfile.ZipFile, schedule_pattern: str) -> dict[str, str]:
    """Extract MDRM → label mapping from row 0 of a schedule."""
    matching = [
        name
        for name in zf.namelist()
        if schedule_pattern in name and name.endswith(".txt")
    ]
    labels: dict[str, str] = {}
    for name in sorted(matching):
        with zf.open(name) as f:
            raw = pd.read_csv(f, sep="\t", nrows=1, dtype=str)
        if raw.empty:
            continue
        row = raw.iloc[0]
        for col in raw.columns[1:]:
            val = row[col]
            if pd.notna(val):
                labels[col] = str(val).strip()
    return labels


def collect_mdrm_codes(repricing_cfg: dict[str, Any]) -> set[str]:
    """Recursively collect all MDRM codes from the repricing map."""
    codes: set[str] = set()
    for value in repricing_cfg.values():
        if isinstance(value, str) and len(value) >= 7:
            codes.add(value)
        elif isinstance(value, dict):
            codes.update(collect_mdrm_codes(value))
    return codes


def extract_repricing_data(
    zf: zipfile.ZipFile,
    repricing_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Extract all mapped MDRM fields from the relevant schedules."""
    target_codes = collect_mdrm_codes(repricing_cfg)

    # Determine which schedules to read
    schedule_patterns = [
        "Schedule RCCI",
        "Schedule RCE ",
        "Schedule RCEI",
        "Schedule RCM ",
        "Schedule RCB ",
    ]

    all_data: pd.DataFrame | None = None
    for pattern in schedule_patterns:
        df = read_schedule_from_zip(zf, pattern)
        if df.empty:
            continue
        keep = ["IDRSSD"] + [c for c in df.columns if c in target_codes]
        if len(keep) <= 1:
            continue
        subset = df[keep].copy()
        if all_data is None:
            all_data = subset
        else:
            new_cols = [c for c in subset.columns if c not in all_data.columns]
            if new_cols:
                all_data = all_data.merge(
                    subset[["IDRSSD"] + new_cols],
                    on="IDRSSD",
                    how="outer",
                )

    if all_data is None:
        return pd.DataFrame()

    # Convert numeric
    for col in all_data.columns:
        if col != "IDRSSD":
            all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data["IDRSSD"] = pd.to_numeric(all_data["IDRSSD"], errors="coerce").astype("Int64")
    return all_data


def map_idrssd_to_cert(
    data: pd.DataFrame,
    institutions: pd.DataFrame,
) -> pd.DataFrame:
    """Join IDRSSD to CERT using the institutions table."""
    inst = institutions.copy()
    rssd_col = "FED_RSSD" if "FED_RSSD" in inst.columns else "RSSDID"
    if rssd_col not in inst.columns:
        return data
    inst[rssd_col] = pd.to_numeric(inst[rssd_col], errors="coerce").astype("Int64")
    inst["CERT"] = inst["CERT"].astype(str).str.strip()
    mapping = inst.drop_duplicates(subset=[rssd_col])[[rssd_col, "CERT"]]
    return data.merge(mapping, left_on="IDRSSD", right_on=rssd_col, how="left")


def build_repricing_features(
    data: pd.DataFrame,
    repricing_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute standardized repricing bucket features from raw MDRM values."""
    out = data.copy()
    midpoints = repricing_cfg.get("horizon_midpoints", {})

    # --- Loan repricing buckets (prefer domestic RCON, fall back to consolidated RCFD) ---
    loan_cfg = repricing_cfg.get("loan_maturity", {})
    other_loans = loan_cfg.get("other_loans", {})
    other_loans_cons = loan_cfg.get("other_loans_consolidated", {})

    for bucket, label in [
        ("0_3m", "LOAN_0_3M"),
        ("3_12m", "LOAN_3_12M"),
        ("1_3y", "LOAN_1_3Y"),
        ("3_5y", "LOAN_3_5Y"),
        ("5_15y", "LOAN_5_15Y"),
        ("15y_plus", "LOAN_15Y_PLUS"),
    ]:
        rcon = other_loans.get(bucket, "")
        rcfd = other_loans_cons.get(bucket, "")
        rcon_present = rcon in out.columns
        rcfd_present = rcfd in out.columns
        if rcon_present and rcfd_present:
            # Per-row fallback: prefer RCON when non-null, else RCFD
            out[label] = out[rcon].fillna(out[rcfd])
        elif rcon_present:
            out[label] = out[rcon]
        elif rcfd_present:
            out[label] = out[rcfd]
        else:
            out[label] = np.nan

    # --- RE-secured loan maturity buckets (add to total loan picture) ---
    re_first_lien = loan_cfg.get("re_first_lien", {})
    for bucket, label in [
        ("0_3m", "RE_LOAN_0_3M"),
        ("3_12m", "RE_LOAN_3_12M"),
        ("1_3y", "RE_LOAN_1_3Y"),
        ("3_5y", "RE_LOAN_3_5Y"),
        ("5_15y", "RE_LOAN_5_15Y"),
        ("15y_plus", "RE_LOAN_15Y_PLUS"),
    ]:
        mdrm = re_first_lien.get(bucket, "")
        out[label] = out[mdrm] if mdrm in out.columns else np.nan

    # --- Time deposit repricing buckets ---
    td_cfg = repricing_cfg.get("time_deposit_maturity", {})
    for size_label, size_key in [("TD_SMALL", "small"), ("TD_LARGE", "large")]:
        buckets = td_cfg.get(size_key, {})
        for bucket, col_label in [
            ("0_3m", f"{size_label}_0_3M"),
            ("3_12m", f"{size_label}_3_12M"),
            ("1_3y", f"{size_label}_1_3Y"),
            ("3y_plus", f"{size_label}_3Y_PLUS"),
        ]:
            mdrm = buckets.get(bucket, "")
            out[col_label] = out[mdrm] if mdrm in out.columns else np.nan

    # --- Borrowings ---
    borr_cfg = repricing_cfg.get("borrowings_maturity", {})
    for label, key, fallback_key in [
        ("FHLB_LE_1Y", "fhlb_le_1y", "fhlb_le_1y_consolidated"),
        ("OTHER_BORR_LE_1Y", "other_borrowings_le_1y", "other_borrowings_le_1y_consolidated"),
    ]:
        mdrm = borr_cfg.get(key, "")
        fallback = borr_cfg.get(fallback_key, "")
        if mdrm in out.columns:
            out[label] = out[mdrm]
        elif fallback in out.columns:
            out[label] = out[fallback]
        else:
            out[label] = np.nan

    # --- Derived features ---
    loan_buckets = ["LOAN_0_3M", "LOAN_3_12M", "LOAN_1_3Y", "LOAN_3_5Y", "LOAN_5_15Y", "LOAN_15Y_PLUS"]
    re_buckets = ["RE_LOAN_0_3M", "RE_LOAN_3_12M", "RE_LOAN_1_3Y", "RE_LOAN_3_5Y", "RE_LOAN_5_15Y", "RE_LOAN_15Y_PLUS"]
    td_buckets = ["TD_SMALL_0_3M", "TD_SMALL_3_12M", "TD_SMALL_1_3Y", "TD_SMALL_3Y_PLUS",
                  "TD_LARGE_0_3M", "TD_LARGE_3_12M", "TD_LARGE_1_3Y", "TD_LARGE_3Y_PLUS"]

    def _safe_sum(cols: list[str], default_zero: bool = False) -> pd.Series:
        present = [c for c in cols if c in out.columns and out[c].notna().any()]
        if present:
            return out[present].sum(axis=1)
        return pd.Series(0.0 if default_zero else np.nan, index=out.index)

    # Totals
    out["LOAN_MATURITY_TOTAL"] = _safe_sum(loan_buckets)
    out["RE_LOAN_MATURITY_TOTAL"] = _safe_sum(re_buckets)
    out["ALL_LOAN_MATURITY_TOTAL"] = out["LOAN_MATURITY_TOTAL"].fillna(0) + out["RE_LOAN_MATURITY_TOTAL"].fillna(0)
    out["TD_MATURITY_TOTAL"] = _safe_sum(td_buckets)

    # --- Weighted average maturity (duration proxy) ---
    def _wam(bucket_cols: list[str], total_col: str) -> pd.Series:
        horizons = ["0_3m", "3_12m", "1_3y", "3_5y", "5_15y", "15y_plus"]
        weighted = pd.Series(0.0, index=out.index)
        for col, hz in zip(bucket_cols, horizons):
            mid = midpoints.get(hz, 0)
            if col in out.columns:
                weighted += out[col].fillna(0) * mid
        total = out[total_col]
        return np.where(total > 0, weighted / total, np.nan)

    out["LOAN_WAM_PROXY"] = _wam(loan_buckets, "LOAN_MATURITY_TOTAL")
    out["RE_LOAN_WAM_PROXY"] = _wam(re_buckets, "RE_LOAN_MATURITY_TOTAL")

    # --- Repricing gap by horizon ---
    # Asset side: loans repricing/maturing in each bucket
    # Liability side: time deposits maturing in each bucket
    # Gap = assets - liabilities (positive = more assets repricing than liabilities)
    asset_map = {
        "0_3m": ["LOAN_0_3M", "RE_LOAN_0_3M"],
        "3_12m": ["LOAN_3_12M", "RE_LOAN_3_12M"],
        "1_3y": ["LOAN_1_3Y", "RE_LOAN_1_3Y"],
        "3_5y": ["LOAN_3_5Y", "RE_LOAN_3_5Y"],
        "5y_plus": ["LOAN_5_15Y", "LOAN_15Y_PLUS", "RE_LOAN_5_15Y", "RE_LOAN_15Y_PLUS"],
    }
    liability_map = {
        "0_3m": ["TD_SMALL_0_3M", "TD_LARGE_0_3M"],
        "3_12m": ["TD_SMALL_3_12M", "TD_LARGE_3_12M"],
        "1_3y": ["TD_SMALL_1_3Y", "TD_LARGE_1_3Y"],
        "3_5y": ["TD_SMALL_3Y_PLUS", "TD_LARGE_3Y_PLUS"],
        "5y_plus": [],  # No standard TD bucket beyond 3y+ in Call Report
    }

    cumulative_gap = pd.Series(0.0, index=out.index)
    for hz in ["0_3m", "3_12m", "1_3y", "3_5y", "5y_plus"]:
        asset_total = _safe_sum(asset_map.get(hz, []), default_zero=True)
        liab_total = _safe_sum(liability_map.get(hz, []), default_zero=True)
        gap = asset_total - liab_total
        out[f"REPRICING_GAP_{hz.upper()}"] = gap
        cumulative_gap = cumulative_gap + gap
        out[f"CUMULATIVE_GAP_{hz.upper()}"] = cumulative_gap

    # --- Duration-gap-lite ---
    # Simple proxy: (asset WAM - liability WAM) where liability WAM uses TD maturity
    td_bucket_midpoints = [
        ("TD_SMALL_0_3M", midpoints.get("0_3m", 0.125)),
        ("TD_SMALL_3_12M", midpoints.get("3_12m", 0.625)),
        ("TD_SMALL_1_3Y", midpoints.get("1_3y", 2.0)),
        ("TD_SMALL_3Y_PLUS", midpoints.get("3y_plus", 5.0)),
        ("TD_LARGE_0_3M", midpoints.get("0_3m", 0.125)),
        ("TD_LARGE_3_12M", midpoints.get("3_12m", 0.625)),
        ("TD_LARGE_1_3Y", midpoints.get("1_3y", 2.0)),
        ("TD_LARGE_3Y_PLUS", midpoints.get("3y_plus", 5.0)),
    ]
    td_weighted = sum(
        out[col].fillna(0) * mid for col, mid in td_bucket_midpoints if col in out.columns
    )
    td_total = out["TD_MATURITY_TOTAL"]
    out["TD_WAM_PROXY"] = np.where(td_total > 0, td_weighted / td_total, np.nan)

    # Duration gap lite = asset WAM - liability WAM
    out["DURATION_GAP_LITE"] = pd.to_numeric(out["LOAN_WAM_PROXY"], errors="coerce") - pd.to_numeric(out["TD_WAM_PROXY"], errors="coerce")

    return out


def infer_repdte_from_zip(zip_path: Path) -> str | None:
    """Extract report date from zip filename and normalize to YYYYMMDD."""
    import re
    match = re.search(r"(\d{8})", zip_path.name)
    if not match:
        return None
    raw = match.group(1)
    # FFIEC zips use MMDDYYYY; convert to YYYYMMDD
    if len(raw) == 8 and int(raw[:2]) <= 12:
        return raw[4:8] + raw[:4]  # MMDDYYYY → YYYYMMDD
    return raw


def run_extraction(args: argparse.Namespace) -> pd.DataFrame:
    with open(args.map, "r", encoding="utf-8") as f:
        repricing_cfg = yaml.safe_load(f)

    zf = zipfile.ZipFile(args.zip)
    raw = extract_repricing_data(zf, repricing_cfg)
    if raw.empty:
        raise SystemExit("No repricing data found in the zip.")

    if args.institutions:
        inst = pd.read_parquet(args.institutions)
        raw = map_idrssd_to_cert(raw, inst)

    out = build_repricing_features(raw, repricing_cfg)

    # Add REPDTE from filename
    repdte = infer_repdte_from_zip(args.zip)
    if repdte:
        out["REPDTE"] = repdte

    save_table(out, args.out)
    print(f"Saved {len(out)} rows to {args.out}", file=sys.stderr)
    return out


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
