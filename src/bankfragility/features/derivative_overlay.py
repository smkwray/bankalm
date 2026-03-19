"""Derivative overlay for ALM mismatch adjustment.

Extracts interest rate derivative positions from FFIEC Schedule RC-L and
computes hedge indicators.  Banks with no IR derivatives bear full
balance-sheet rate risk (unhedged).  Banks with derivatives may have
partially or fully hedged their repricing exposure.

This overlay does NOT attempt to reconstruct net hedge effectiveness —
that requires trade-level data.  Instead it provides:
- HAS_IR_DERIVATIVES: binary flag
- IR_DERIV_NOTIONAL_TO_ASSETS: notional / total assets (informational)
- UNHEDGED_DURATION_FLAG: 1 if no IR derivatives AND positive duration gap
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bankfragility.tables import save_table

# Key MDRM codes for IR derivatives from Schedule RC-L
IR_DERIV_CODES = {
    "ir_swaps": ("RCFD3450", "RCON3450"),          # IR swap notional
    "ir_futures": ("RCFD8693", "RCON8693"),         # IR futures notional
    "ir_forwards": ("RCFD8697", "RCON8697"),        # IR forward notional
    "ir_written_options": ("RCFDA126", "RCONA126"), # Written IR options
    "ir_purchased_options": ("RCFDA127", "RCONA127"), # Purchased IR options
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="FFIEC CDR zip")
    parser.add_argument("--institutions", type=Path, help="For IDRSSD→CERT mapping")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def extract_derivative_data(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Extract IR derivative fields from RC-L schedule."""
    target_codes: set[str] = set()
    for rcfd, rcon in IR_DERIV_CODES.values():
        target_codes.add(rcfd)
        target_codes.add(rcon)

    matching = [n for n in zf.namelist() if "Schedule RCL" in n and n.endswith(".txt")]
    if not matching:
        return pd.DataFrame()

    all_data: pd.DataFrame | None = None
    for name in sorted(matching):
        with zf.open(name) as f:
            raw = pd.read_csv(f, sep="\t", dtype=str)
        data = raw.iloc[1:]  # skip label row
        data.columns = raw.columns
        keep = ["IDRSSD"] + [c for c in data.columns if c in target_codes]
        if len(keep) <= 1:
            continue
        subset = data[keep].copy()
        if all_data is None:
            all_data = subset
        else:
            new_cols = [c for c in subset.columns if c not in all_data.columns]
            if new_cols:
                all_data = all_data.merge(subset[["IDRSSD"] + new_cols], on="IDRSSD", how="outer")

    if all_data is None:
        return pd.DataFrame()

    for col in all_data.columns:
        if col != "IDRSSD":
            all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data["IDRSSD"] = pd.to_numeric(all_data["IDRSSD"], errors="coerce").astype("Int64")
    return all_data


def build_derivative_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute derivative overlay features from raw RC-L data."""
    out = data.copy()

    # For each IR derivative type, prefer RCFD (consolidated), fill with RCON (domestic)
    for label, (rcfd, rcon) in IR_DERIV_CODES.items():
        col_name = label.upper()
        if rcfd in out.columns and rcon in out.columns:
            out[col_name] = out[rcfd].fillna(out[rcon])
        elif rcfd in out.columns:
            out[col_name] = out[rcfd]
        elif rcon in out.columns:
            out[col_name] = out[rcon]
        else:
            out[col_name] = np.nan

    # Total IR derivative notional
    ir_cols = [c.upper() for c in IR_DERIV_CODES.keys() if c.upper() in out.columns]
    out["IR_DERIV_TOTAL_NOTIONAL"] = out[ir_cols].sum(axis=1) if ir_cols else np.nan

    # Binary flag: does this bank use IR derivatives?
    out["HAS_IR_DERIVATIVES"] = (out["IR_DERIV_TOTAL_NOTIONAL"] > 0).astype(int)

    # Notional-to-assets ratio (informational — not comparable for dealer vs end-user banks)
    if "ASSET" in out.columns:
        asset = pd.to_numeric(out["ASSET"], errors="coerce")
        out["IR_DERIV_NOTIONAL_TO_ASSETS"] = np.where(
            asset > 0, out["IR_DERIV_TOTAL_NOTIONAL"] / asset, np.nan,
        )

    return out


def map_idrssd_to_cert(data: pd.DataFrame, institutions: pd.DataFrame) -> pd.DataFrame:
    inst = institutions.copy()
    rssd_col = "FED_RSSD" if "FED_RSSD" in inst.columns else "RSSDID"
    if rssd_col not in inst.columns:
        return data
    inst[rssd_col] = pd.to_numeric(inst[rssd_col], errors="coerce").astype("Int64")
    inst["CERT"] = inst["CERT"].astype(str).str.strip()
    mapping = inst.drop_duplicates(subset=[rssd_col])[[rssd_col, "CERT"]]
    return data.merge(mapping, left_on="IDRSSD", right_on=rssd_col, how="left")


def infer_repdte_from_zip(zip_path: Path) -> str | None:
    import re
    match = re.search(r"(\d{8})", zip_path.name)
    if not match:
        return None
    raw = match.group(1)
    if len(raw) == 8 and int(raw[:2]) <= 12:
        return raw[4:8] + raw[:4]
    return raw


def run_derivative_extraction(args: argparse.Namespace) -> pd.DataFrame:
    zf = zipfile.ZipFile(args.zip)
    raw = extract_derivative_data(zf)
    if raw.empty:
        raise SystemExit("No derivative data found.")

    if args.institutions:
        inst = pd.read_parquet(args.institutions)
        raw = map_idrssd_to_cert(raw, inst)

    out = build_derivative_features(raw)

    repdte = infer_repdte_from_zip(args.zip)
    if repdte:
        out["REPDTE"] = repdte

    save_table(out, args.out)
    has_derivs = out["HAS_IR_DERIVATIVES"].sum()
    print(f"Derivatives: {has_derivs}/{len(out)} banks with IR derivatives. Saved to {args.out}", file=sys.stderr)
    return out


def main() -> None:
    args = parse_args()
    run_derivative_extraction(args)


if __name__ == "__main__":
    main()
