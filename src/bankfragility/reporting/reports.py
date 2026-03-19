"""Generate reporting outputs from the bank-quarter index pipeline.

Produces:
- Bank-level drill-down: one bank across time
- Quarter league table: all banks ranked for one quarter
- Peer-group summary: aggregate stats per peer group per quarter
- Scenario comparison: side-by-side deposit life estimates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bankfragility.tables import read_table, save_table
from bankfragility.reporting.site_exports import (
    build_publishable_mart,
    split_publishable_panels,
    write_site_exports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, required=True, help="Bank indices parquet")
    parser.add_argument("--supervised", type=Path, help="Optional supervised overlay parquet")
    parser.add_argument("--validation-metrics", type=Path, help="Optional cohort backtest metrics csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--mart-out", type=Path, help="Optional publishable mart parquet path")
    parser.add_argument("--core-panel-out", type=Path, help="Optional full-history core panel parquet path")
    parser.add_argument("--enriched-panel-out", type=Path, help="Optional recent-history enriched panel parquet path")
    parser.add_argument("--site-dir", type=Path, help="Optional site/data output directory")
    parser.add_argument("--cert", type=str, help="Optional: single CERT for drill-down report")
    parser.add_argument("--quarter", type=str, help="Optional: YYYYMMDD for league table")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Index columns used across reports
# ---------------------------------------------------------------------------

INDEX_COLS = [
    "RUN_RISK_INDEX", "DEPOSIT_STICKINESS_INDEX", "ALM_MISMATCH_INDEX",
    "TREASURY_BUFFER_INDEX", "FUNDING_FRAGILITY_INDEX",
]

FEATURE_COLS = [
    "UNINSURED_SHARE", "BROKERED_SHARE", "CORE_DEPOSIT_SHARE",
    "NONINTEREST_SHARE", "TIME_DEPOSIT_SHARE",
    "VOLATILE_TO_LIQUID_LOWER", "LONG_TERM_ASSETS_TO_STABLE_FUNDING_BASELINE",
    "LOANS_TO_CORE_DEPOSITS", "TREASURY_TO_UNINSURED",
    "TREASURY_AGENCY_TO_RUNNABLE", "HQLA_NARROW_LOWER_TO_RUNNABLE",
    "HAS_TREASURY_YIELD_HISTORY", "YC_2YR", "YC_10YR", "YC_10Y_2Y_SLOPE_BP",
    "SECURITY_DURATION_PROXY", "SHORT_FHLB_SHARE", "SUPERVISED_OUTFLOW_SCORE",
    "RUN_RISK_SCORE", "STICKINESS_SCORE",
]

SCENARIO_COLS = [
    "DEPOSIT_WAL_BASELINE", "DEPOSIT_WAL_ADVERSE", "DEPOSIT_WAL_SEVERE",
    "DEPOSIT_STABLE_EQUIV_BASELINE", "DEPOSIT_STABLE_EQUIV_ADVERSE",
    "DEPOSIT_STABLE_EQUIV_SEVERE",
]

FFIEC_COLS = [
    "LOAN_WAM_PROXY", "TD_WAM_PROXY", "DURATION_GAP_LITE",
    "REPRICING_GAP_0_3M", "CUMULATIVE_GAP_3_5Y",
]

SOD_COLS = [
    "SOD_BRANCH_COUNT", "SOD_STATE_COUNT", "SOD_DEPOSIT_HHI_STATE",
    "SOD_COUNTY_COUNT",
]

TREASURY_REGIME_COLS = [
    "TREASURY_YIELD_DATE", "HAS_TREASURY_YIELD_HISTORY",
    "YC_3MO", "YC_2YR", "YC_10YR", "YC_30YR",
    "YC_10Y_3M_SLOPE_BP", "YC_10Y_2Y_SLOPE_BP",
    "YC_2YR_QOQ_CHANGE_BP", "YC_10YR_QOQ_CHANGE_BP",
    "YC_10Y_3M_SLOPE_BP_QOQ_CHANGE_BP", "YC_10Y_2Y_SLOPE_BP_QOQ_CHANGE_BP",
]


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Bank drill-down
# ---------------------------------------------------------------------------

def bank_drill_down(df: pd.DataFrame, cert: str) -> pd.DataFrame:
    """One bank across all quarters — key features and indices."""
    bank = df[df["CERT"].astype(str).str.strip() == str(cert).strip()].copy()
    if bank.empty:
        return pd.DataFrame()
    bank = bank.sort_values("REPDTE")

    id_cols = ["CERT", "REPDTE", "NAMEFULL", "PEER_GROUP", "ASSET"]
    cols = _available(bank, id_cols + INDEX_COLS + FEATURE_COLS + SCENARIO_COLS + FFIEC_COLS + SOD_COLS)
    return bank[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quarter league table
# ---------------------------------------------------------------------------

def quarter_league_table(df: pd.DataFrame, quarter: str | pd.Timestamp) -> pd.DataFrame:
    """All banks ranked by FUNDING_FRAGILITY_INDEX for one quarter."""
    if isinstance(quarter, str):
        quarter = pd.Timestamp(quarter)
    q = df[df["REPDTE"] == quarter].copy()
    if q.empty:
        return pd.DataFrame()

    id_cols = ["CERT", "NAMEFULL", "PEER_GROUP", "ASSET"]
    rank_cols = INDEX_COLS + ["RUN_RISK_SCORE"]
    feature_subset = ["UNINSURED_SHARE", "VOLATILE_TO_LIQUID_LOWER",
                      "TREASURY_TO_UNINSURED", "LOANS_TO_CORE_DEPOSITS"]
    cols = _available(q, id_cols + rank_cols + feature_subset)
    out = q[cols].sort_values("FUNDING_FRAGILITY_INDEX", ascending=False)

    # Add cross-peer rank
    if "FUNDING_FRAGILITY_INDEX" in out.columns:
        out["FRAGILITY_RANK"] = out["FUNDING_FRAGILITY_INDEX"].rank(ascending=False, method="min").astype(int)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Peer-group summary
# ---------------------------------------------------------------------------

def peer_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats (median, p25, p75, count) per peer group per quarter."""
    if "PEER_GROUP" not in df.columns or "REPDTE" not in df.columns:
        return pd.DataFrame()

    group_cols = ["REPDTE", "PEER_GROUP"]
    agg_targets = _available(df, INDEX_COLS + ["RUN_RISK_SCORE", "UNINSURED_SHARE",
                                                "VOLATILE_TO_LIQUID_LOWER", "TREASURY_TO_UNINSURED"])
    if not agg_targets:
        return pd.DataFrame()

    agg_funcs: dict[str, list[str]] = {col: ["median", "mean", "count"] for col in agg_targets}
    summary = df.groupby(group_cols, dropna=False).agg(agg_funcs)
    summary.columns = ["_".join(col).upper() for col in summary.columns]
    return summary.reset_index()


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------

def scenario_comparison(df: pd.DataFrame, quarter: str | pd.Timestamp) -> pd.DataFrame:
    """Side-by-side deposit life estimates across scenarios for one quarter."""
    if isinstance(quarter, str):
        quarter = pd.Timestamp(quarter)
    q = df[df["REPDTE"] == quarter].copy()
    if q.empty:
        return pd.DataFrame()

    id_cols = ["CERT", "NAMEFULL", "PEER_GROUP", "RUN_RISK_SCORE"]
    cols = _available(q, id_cols + SCENARIO_COLS)
    return q[cols].sort_values("RUN_RISK_SCORE", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Treasury regime summary
# ---------------------------------------------------------------------------

def treasury_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter summarizing the Treasury-rate backdrop when available."""
    if "REPDTE" not in df.columns:
        return pd.DataFrame()

    cols = _available(df, ["REPDTE"] + TREASURY_REGIME_COLS)
    if len(cols) <= 1:
        return pd.DataFrame()

    summary = (
        df[cols]
        .sort_values("REPDTE")
        .groupby("REPDTE", as_index=False)
        .first()
    )
    if "YC_10Y_3M_SLOPE_BP" in summary.columns:
        summary["CURVE_REGIME"] = np.where(
            summary["YC_10Y_3M_SLOPE_BP"] < 0,
            "inverted",
            "upward_sloping",
        )
    return summary


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run_reports(args: argparse.Namespace) -> None:
    indices = read_table(args.indices)
    supervised = read_table(args.supervised) if args.supervised else None
    metrics = pd.read_csv(args.validation_metrics) if args.validation_metrics and args.validation_metrics.exists() else None

    df = build_publishable_mart(indices, supervised)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    core_panel, enriched_panel = split_publishable_panels(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports_generated = 0

    if args.mart_out:
        save_table(df, args.mart_out)
        print(f"Publishable mart: {len(df)} rows → {args.mart_out}", file=sys.stderr)
        reports_generated += 1
    if args.core_panel_out:
        save_table(core_panel, args.core_panel_out)
        print(f"Full-history core panel: {len(core_panel)} rows → {args.core_panel_out}", file=sys.stderr)
        reports_generated += 1
    if args.enriched_panel_out:
        save_table(enriched_panel, args.enriched_panel_out)
        print(f"Recent-history enriched panel: {len(enriched_panel)} rows → {args.enriched_panel_out}", file=sys.stderr)
        reports_generated += 1

    # Bank drill-down
    if args.cert:
        drill = bank_drill_down(df, args.cert)
        if not drill.empty:
            path = args.out_dir / f"drill_down_cert_{args.cert}.parquet"
            save_table(drill, path)
            print(f"Bank drill-down: {len(drill)} rows → {path}", file=sys.stderr)
            reports_generated += 1

    # Quarter league table
    if args.quarter:
        league = quarter_league_table(df, args.quarter)
        if not league.empty:
            path = args.out_dir / f"league_table_{args.quarter}.parquet"
            save_table(league, path)
            print(f"League table: {len(league)} banks → {path}", file=sys.stderr)
            reports_generated += 1

        scenario = scenario_comparison(df, args.quarter)
        if not scenario.empty:
            path = args.out_dir / f"scenarios_{args.quarter}.parquet"
            save_table(scenario, path)
            print(f"Scenario comparison: {len(scenario)} banks → {path}", file=sys.stderr)
            reports_generated += 1

    # Peer-group summary (always generated)
    summary = peer_group_summary(df)
    if not summary.empty:
        path = args.out_dir / "peer_group_summary.parquet"
        save_table(summary, path)
        print(f"Peer-group summary: {len(summary)} rows → {path}", file=sys.stderr)
        reports_generated += 1

    treasury_summary = treasury_regime_summary(df)
    if not treasury_summary.empty:
        path = args.out_dir / "treasury_regime_summary.parquet"
        save_table(treasury_summary, path)
        print(f"Treasury regime summary: {len(treasury_summary)} rows → {path}", file=sys.stderr)
        reports_generated += 1

    if reports_generated == 0:
        print("No reports generated. Use --cert and/or --quarter.", file=sys.stderr)
    else:
        print(f"Generated {reports_generated} reports in {args.out_dir}", file=sys.stderr)

    if args.site_dir:
        write_site_exports(
            df,
            args.site_dir,
            full_history_core=core_panel,
            recent_history_enriched=enriched_panel,
            validation_metrics=metrics,
        )
        print(f"Site exports written to {args.site_dir}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    run_reports(args)


if __name__ == "__main__":
    main()
