"""Quarter-aligned cohort backtest for FDIC bank failures.

Builds a bank-quarter panel where each row is labeled for whether the bank
fails within fixed forward horizons. This avoids comparing failed banks at
their last pre-failure quarter against survivors many years later.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bankfragility.tables import read_table, save_table

DEFAULT_HORIZONS = (1, 2, 4)
SCORE_COLUMNS = [
    ("RUN_RISK_INDEX", "Run risk index"),
    ("FUNDING_FRAGILITY_INDEX", "Composite fragility"),
    ("ALM_MISMATCH_INDEX", "ALM mismatch index"),
]
EPISODE_WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "full_sample": (None, None),
    "gfc_2008": ("2007-01-01", "2010-12-31"),
    "covid_2020": ("2019-01-01", "2021-12-31"),
    "banking_2023": ("2022-01-01", "2024-12-31"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--min-year", type=int, default=2007)
    parser.add_argument("--max-year", type=int, default=2024)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _quarter_distance(repdte: pd.Series, faildate: pd.Series) -> pd.Series:
    """Compute integer quarters between report date and failure date."""
    rep_year = repdte.dt.year
    rep_q = (repdte.dt.month - 1) // 3
    fail_year = faildate.dt.year
    fail_q = (faildate.dt.month - 1) // 3
    return (fail_year - rep_year) * 4 + (fail_q - rep_q)


def _forward_observed(repdte: pd.Series, horizon_quarters: int, evaluation_end: pd.Timestamp) -> pd.Series:
    end_dates = repdte + pd.offsets.QuarterEnd(horizon_quarters)
    return end_dates <= evaluation_end


def build_failure_dataset(
    indices: pd.DataFrame,
    failures: pd.DataFrame,
    min_year: int = 2007,
    max_year: int = 2024,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build one row per bank-quarter with forward failure horizon labels."""
    idx = indices.copy()
    idx["REPDTE"] = pd.to_datetime(idx["REPDTE"])
    idx["CERT_NUM"] = pd.to_numeric(idx["CERT"], errors="coerce").astype("Int64")
    idx = idx.dropna(subset=["REPDTE", "CERT_NUM"])
    idx = idx[
        (idx["REPDTE"].dt.year >= min_year)
        & (idx["REPDTE"].dt.year <= max_year)
    ].copy()

    fail = failures.copy()
    fail["FAILDATE"] = pd.to_datetime(fail["FAILDATE"])
    fail["CERT_NUM"] = pd.to_numeric(fail["CERT"], errors="coerce").astype("Int64")
    fail = fail.dropna(subset=["CERT_NUM", "FAILDATE"])
    fail = fail[
        (fail["FAILDATE"].dt.year >= min_year)
        & (fail["FAILDATE"].dt.year <= max_year)
    ]
    fail = fail.sort_values(["CERT_NUM", "FAILDATE"]).drop_duplicates(subset=["CERT_NUM"], keep="last")

    out = idx.merge(
        fail[["CERT_NUM", "FAILDATE"]],
        on="CERT_NUM",
        how="left",
    )
    out["FAIL_DATE"] = out["FAILDATE"]
    out["FAILED_BANK"] = out["FAILDATE"].notna().astype(int)

    quarters_to_failure = _quarter_distance(
        out["REPDTE"],
        pd.to_datetime(out["FAILDATE"]),
    )
    future_failure_mask = out["FAILDATE"].notna() & (out["FAILDATE"] > out["REPDTE"])
    out["QUARTERS_TO_FAILURE"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[future_failure_mask, "QUARTERS_TO_FAILURE"] = quarters_to_failure.loc[future_failure_mask].astype("Int64")

    evaluation_end = pd.Timestamp(f"{max_year}-12-31")
    for horizon in horizons:
        observed_col = f"FORWARD_OBSERVED_{horizon}Q"
        label_col = f"FAIL_WITHIN_{horizon}Q"
        out[observed_col] = _forward_observed(out["REPDTE"], horizon, evaluation_end)
        out[label_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        observed_mask = out[observed_col].fillna(False)
        fail_mask = (
            out["QUARTERS_TO_FAILURE"].notna()
            & (pd.to_numeric(out["QUARTERS_TO_FAILURE"], errors="coerce") > 0)
            & (pd.to_numeric(out["QUARTERS_TO_FAILURE"], errors="coerce") <= horizon)
        )
        out.loc[observed_mask, label_col] = fail_mask.loc[observed_mask].astype("Int64")

    return out


def compute_backtest_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    observed_col: str | None = None,
) -> dict[str, float]:
    if observed_col is not None:
        observed_mask = pd.to_numeric(df[observed_col], errors="coerce").fillna(0).astype(int).astype(bool)
        df = df.loc[observed_mask]
    valid = df[[score_col, label_col]].dropna()
    if valid.empty or valid[label_col].nunique() < 2:
        return {"auc": np.nan, "n_failures": 0, "n_total": len(valid)}

    y_true = valid[label_col].astype(int).values
    y_score = valid[score_col].astype(float).values
    n_failures = int(y_true.sum())
    n_total = len(valid)

    auc = roc_auc_score(y_true, y_score)
    metrics: dict[str, float] = {"auc": auc, "n_failures": n_failures, "n_total": n_total}

    for pct in [5, 10, 20]:
        threshold = np.percentile(y_score, 100 - pct)
        flagged = y_score >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f"precision_at_{pct}pct"] = precision
        metrics[f"recall_at_{pct}pct"] = recall

    return metrics


def build_metrics_table(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    score_cols: list[tuple[str, str]] = SCORE_COLUMNS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repdte = pd.to_datetime(df["REPDTE"])

    for slice_name, (start, end) in EPISODE_WINDOWS.items():
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= repdte >= pd.Timestamp(start)
        if end is not None:
            mask &= repdte <= pd.Timestamp(end)
        slice_df = df[mask]

        for horizon in horizons:
            label_col = f"FAIL_WITHIN_{horizon}Q"
            observed_col = f"FORWARD_OBSERVED_{horizon}Q"
            for score_col, score_label in score_cols:
                if score_col not in slice_df.columns:
                    continue
                metrics = compute_backtest_metrics(slice_df, score_col, label_col, observed_col=observed_col)
                rows.append(
                    {
                        "SLICE": slice_name,
                        "HORIZON_QUARTERS": horizon,
                        "SCORE_COL": score_col,
                        "SCORE_LABEL": score_label,
                        "AUC": metrics["auc"],
                        "N_FAILURES": metrics["n_failures"],
                        "N_TOTAL": metrics["n_total"],
                        "PRECISION_AT_5PCT": metrics.get("precision_at_5pct", np.nan),
                        "RECALL_AT_5PCT": metrics.get("recall_at_5pct", np.nan),
                        "PRECISION_AT_10PCT": metrics.get("precision_at_10pct", np.nan),
                        "RECALL_AT_10PCT": metrics.get("recall_at_10pct", np.nan),
                        "PRECISION_AT_20PCT": metrics.get("precision_at_20pct", np.nan),
                        "RECALL_AT_20PCT": metrics.get("recall_at_20pct", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def run_backtest(args: argparse.Namespace) -> pd.DataFrame:
    indices = read_table(args.indices)
    failures = read_table(args.failures)

    labeled = build_failure_dataset(indices, failures, args.min_year, args.max_year)
    if labeled.empty:
        raise SystemExit("No bank-quarter rows matched the evaluation window.")

    metrics = build_metrics_table(labeled)
    metrics_out = args.out.with_name(f"{args.out.stem}_metrics.csv")

    print("=" * 72, file=sys.stderr)
    print("FAILURE COHORT BACKTEST (quarter-aligned bank-quarter evaluation)", file=sys.stderr)
    print(f"Rows: {len(labeled):,}", file=sys.stderr)
    print(f"Window: {args.min_year}-{args.max_year}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    for horizon in DEFAULT_HORIZONS:
        print(f"\nForward horizon: {horizon}Q", file=sys.stderr)
        horizon_metrics = metrics[
            (metrics["SLICE"] == "full_sample")
            & (metrics["HORIZON_QUARTERS"] == horizon)
        ]
        for _, row in horizon_metrics.iterrows():
            auc = row["AUC"]
            auc_text = "nan" if pd.isna(auc) else f"{auc:.4f}"
            print(
                f"  {row['SCORE_LABEL']}: AUC={auc_text} "
                f"({int(row['N_FAILURES']):,} failures / {int(row['N_TOTAL']):,} bank-quarters)",
                file=sys.stderr,
            )
            print(
                f"    Top 20%: precision={row['PRECISION_AT_20PCT']:.3f}, "
                f"recall={row['RECALL_AT_20PCT']:.3f}",
                file=sys.stderr,
            )

    save_table(labeled, args.out)
    metrics.to_csv(metrics_out, index=False)
    print(f"\nSaved {len(labeled):,} labeled rows to {args.out}", file=sys.stderr)
    print(f"Saved {len(metrics):,} metric rows to {metrics_out}", file=sys.stderr)
    return labeled


def main() -> None:
    args = parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
