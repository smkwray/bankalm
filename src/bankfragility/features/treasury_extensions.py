"""Build Treasury-focused liquidity and stress-overlay features."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bankfragility.tables import parse_report_date, read_table, save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--treasury-glob", default="", help="Optional glob for downloaded Treasury yield history")
    parser.add_argument(
        "--treasury-duration-years",
        type=float,
        default=4.5,
        help="Fallback assumed Treasury duration when maturity-by-type detail is unavailable",
    )
    parser.add_argument(
        "--shock-bps",
        nargs="*",
        type=float,
        default=[100, 200, 300],
        help="Parallel upward yield shocks in basis points",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(np.where((b.notna()) & (b != 0), a / b, np.nan), index=a.index)


def load_treasury_history(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = [read_table(path) for path in paths]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out = out.dropna(subset=["DATE"]).sort_values("DATE").drop_duplicates(subset=["DATE"], keep="last")
    return out.reset_index(drop=True)


def build_treasury_rate_history_features(
    repdte: pd.Series,
    treasury_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join latest available Treasury yields on or before REPDTE and derive regime features."""
    out = pd.DataFrame(index=repdte.index)
    out["REPDTE"] = parse_report_date(repdte).astype("datetime64[ns]")

    if treasury_history is None or treasury_history.empty:
        out["HAS_TREASURY_YIELD_HISTORY"] = 0
        return out.drop(columns=["REPDTE"])

    history = treasury_history.copy()
    history.columns = [str(col).upper() for col in history.columns]
    history["DATE"] = pd.to_datetime(history["DATE"], errors="coerce").astype("datetime64[ns]")
    history = history.dropna(subset=["DATE"]).sort_values("DATE")

    yield_cols = [col for col in ["YC_3MO", "YC_2YR", "YC_10YR", "YC_30YR"] if col in history.columns]
    if not yield_cols:
        out["HAS_TREASURY_YIELD_HISTORY"] = 0
        return out.drop(columns=["REPDTE"])

    unique_repdte = (
        out[["REPDTE"]]
        .dropna()
        .drop_duplicates()
        .sort_values("REPDTE")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        unique_repdte,
        history[["DATE"] + yield_cols].sort_values("DATE"),
        left_on="REPDTE",
        right_on="DATE",
        direction="backward",
    )
    merged = merged.rename(columns={"DATE": "TREASURY_YIELD_DATE"})
    merged["HAS_TREASURY_YIELD_HISTORY"] = merged["TREASURY_YIELD_DATE"].notna().astype(int)

    if {"YC_10YR", "YC_3MO"} <= set(merged.columns):
        merged["YC_10Y_3M_SLOPE_BP"] = (merged["YC_10YR"] - merged["YC_3MO"]) * 100.0
    if {"YC_10YR", "YC_2YR"} <= set(merged.columns):
        merged["YC_10Y_2Y_SLOPE_BP"] = (merged["YC_10YR"] - merged["YC_2YR"]) * 100.0

    for base_col in [col for col in ["YC_2YR", "YC_10YR", "YC_10Y_3M_SLOPE_BP", "YC_10Y_2Y_SLOPE_BP"] if col in merged.columns]:
        scale = 1.0 if base_col.endswith("_SLOPE_BP") else 100.0
        merged[f"{base_col}_QOQ_CHANGE_BP"] = (merged[base_col] - merged[base_col].shift(1)) * scale
        merged[f"{base_col}_YOY_CHANGE_BP"] = (merged[base_col] - merged[base_col].shift(4)) * scale

    result = out.merge(merged, on="REPDTE", how="left").drop(columns=["REPDTE"])
    if "HAS_TREASURY_YIELD_HISTORY" not in result.columns:
        result["HAS_TREASURY_YIELD_HISTORY"] = 0
    return result


def build_treasury_extensions(
    df: pd.DataFrame,
    treasury_history: pd.DataFrame | None = None,
    treasury_duration_years: float = 4.5,
    shock_bps: list[float] | tuple[float, ...] = (100, 200, 300),
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).upper() for col in out.columns]
    if "REPDTE" in out.columns:
        out["REPDTE"] = parse_report_date(out["REPDTE"]).astype("datetime64[ns]")

    scust = num(out, "SCUST").fillna(0)
    scage = num(out, "SCAGE").fillna(0)
    spledge = num(out, "SCPLEDGE").fillna(0)
    chbal = num(out, "CHBAL").fillna(0)
    frepo = num(out, "FREPO").fillna(0)

    depuna = num(out, "DEPUNA")
    runnable = num(out, "RUNNABLE_FUNDING_PROXY")
    if runnable.isna().all():
        runnable = num(out, "VOLIAB")

    out["TREASURY_TO_UNINSURED"] = safe_div(scust, depuna)
    out["TREASURY_AGENCY_TO_RUNNABLE"] = safe_div(scust + 0.85 * scage, runnable)

    out["HQLA_NARROW_UPPER"] = chbal + frepo + scust + 0.85 * scage
    out["HQLA_NARROW_LOWER"] = (out["HQLA_NARROW_UPPER"] - spledge).clip(lower=0)
    out["HQLA_NARROW_LOWER_TO_RUNNABLE"] = safe_div(out["HQLA_NARROW_LOWER"], runnable)

    sec_dur = num(out, "SECURITY_DURATION_PROXY")
    treasury_dur = sec_dur.clip(lower=2.0, upper=6.0).fillna(treasury_duration_years)
    out["TREASURY_DURATION_ASSUMPTION"] = treasury_dur

    for shock_bp in shock_bps:
        shock = float(shock_bp) / 10_000.0
        label = str(int(shock_bp))
        treas_loss = scust * treasury_dur * shock
        agency_loss = scage * np.minimum(treasury_dur + 0.5, 7.0) * shock
        out[f"TREASURY_LOSS_{label}BP"] = treas_loss
        out[f"TREASURY_AFTER_{label}BP"] = (scust - treas_loss).clip(lower=0)
        out[f"TREASURY_TO_UNINSURED_AFTER_{label}BP"] = safe_div(out[f"TREASURY_AFTER_{label}BP"], depuna)

        hqla_after = chbal + frepo + (scust - treas_loss).clip(lower=0) + 0.85 * (scage - agency_loss).clip(lower=0)
        hqla_after_lower = (hqla_after - spledge).clip(lower=0)
        out[f"HQLA_NARROW_LOWER_TO_RUNNABLE_AFTER_{label}BP"] = safe_div(hqla_after_lower, runnable)

    if "REPDTE" in out.columns:
        rate_features = build_treasury_rate_history_features(out["REPDTE"], treasury_history)
        out = pd.concat([out, rate_features], axis=1)
    else:
        out["HAS_TREASURY_YIELD_HISTORY"] = 0

    return out


def main() -> None:
    args = parse_args()
    df = read_table(args.input)
    treasury_history = load_treasury_history(args.treasury_glob) if args.treasury_glob else None
    out = build_treasury_extensions(
        df=df,
        treasury_history=treasury_history,
        treasury_duration_years=args.treasury_duration_years,
        shock_bps=args.shock_bps,
    )
    save_table(out, args.out)
    print(f"Saved {len(out):,} rows to {args.out}")


if __name__ == "__main__":
    main()
