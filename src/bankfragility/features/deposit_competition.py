"""Build deposit competition / outside-option pressure features.

This module asks a simple question:
how exposed is a bank when safe outside options pay more than the bank pays
on deposits?

It is designed to fit the existing bankALM flow:
deposit stickiness -> ALM -> Treasury extensions -> deposit competition.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bankfragility.tables import parse_report_date, read_table, save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--market-rates",
        type=Path,
        default=None,
        help="Optional wide daily market-rate history table with DATE plus series columns",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def clip01(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.clip(out, 0.0, 1.0), index=series.index, dtype="float64")


def add_pct_rank(
    df: pd.DataFrame,
    value_col: str,
    by_col: str,
    out_col: str,
    higher_is_better: bool,
    fill_value: float,
) -> None:
    def ranker(series: pd.Series) -> pd.Series:
        ranked = series.rank(pct=True, method="average", ascending=True)
        if higher_is_better:
            ranked = 1 - ranked
        return ranked.fillna(fill_value)

    df[out_col] = df.groupby(by_col, dropna=False)[value_col].transform(ranker)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config must deserialize to a mapping.")
    return cfg


def load_market_rate_history(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    out = read_table(path)
    out.columns = [str(col).upper() for col in out.columns]
    if "DATE" not in out.columns:
        raise ValueError("Market-rate history must contain DATE.")
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=["DATE"]).sort_values("DATE")
    out = out.drop_duplicates(subset=["DATE"], keep="last")
    return out.reset_index(drop=True)


def build_market_rate_history_features(
    repdte: pd.Series,
    market_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join latest available market-rate observations on or before REPDTE."""
    out = pd.DataFrame(index=repdte.index)
    out["REPDTE"] = parse_report_date(repdte).astype("datetime64[ns]")

    if market_history is None or market_history.empty:
        out["HAS_MARKET_RATE_HISTORY"] = 0
        return out.drop(columns=["REPDTE"])

    history = market_history.copy()
    history.columns = [str(col).upper() for col in history.columns]
    history["DATE"] = pd.to_datetime(history["DATE"], errors="coerce").astype("datetime64[ns]")
    history = history.dropna(subset=["DATE"]).sort_values("DATE")

    value_cols = [col for col in history.columns if col != "DATE"]
    if not value_cols:
        out["HAS_MARKET_RATE_HISTORY"] = 0
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
        history[["DATE"] + value_cols].sort_values("DATE"),
        left_on="REPDTE",
        right_on="DATE",
        direction="backward",
    )
    merged = merged.rename(columns={"DATE": "MARKET_RATE_DATE"})
    merged["HAS_MARKET_RATE_HISTORY"] = merged["MARKET_RATE_DATE"].notna().astype(int)

    result = out.merge(merged, on="REPDTE", how="left").drop(columns=["REPDTE"])
    if "HAS_MARKET_RATE_HISTORY" not in result.columns:
        result["HAS_MARKET_RATE_HISTORY"] = 0
    return result


def _convert_raw_rate_columns_to_bp(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    mapping = cfg.get("raw_rate_columns_to_bp", {})
    if not isinstance(mapping, dict):
        return

    for raw_col, out_col in mapping.items():
        raw_col = str(raw_col).upper()
        out_col = str(out_col).upper()
        if out_col in df.columns or raw_col not in df.columns:
            continue
        # Market-rate inputs are stored in percent units, e.g. 5.25.
        df[out_col] = safe_num(df, raw_col) * 100.0


def _build_outside_option_rate(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    selection = cfg.get("benchmark_selection", {})
    method = str(selection.get("method", "max_available")).lower()
    benchmark_cols = [str(col).upper() for col in selection.get("benchmark_priority", [])]
    benchmark_cols = [col for col in benchmark_cols if col in df.columns]

    if "YC_3MO_BP" not in df.columns and "YC_3MO" in df.columns:
        df["YC_3MO_BP"] = safe_num(df, "YC_3MO") * 100.0
    if not benchmark_cols and "YC_3MO_BP" in df.columns:
        benchmark_cols = ["YC_3MO_BP"]

    if not benchmark_cols:
        df["OUTSIDE_OPTION_RATE_BP"] = np.nan
        df["OUTSIDE_OPTION_SOURCE"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["OUTSIDE_OPTION_RATE_AVAILABLE_COUNT"] = 0
        return

    rate_frame = df[benchmark_cols].apply(pd.to_numeric, errors="coerce")
    available_count = rate_frame.notna().sum(axis=1)
    df["OUTSIDE_OPTION_RATE_AVAILABLE_COUNT"] = available_count

    if method == "first_available":
        out_rate = pd.Series(np.nan, index=df.index, dtype="float64")
        out_source = pd.Series(pd.NA, index=df.index, dtype="string")
        for col in benchmark_cols:
            use_mask = out_rate.isna() & rate_frame[col].notna()
            out_rate.loc[use_mask] = rate_frame.loc[use_mask, col]
            out_source.loc[use_mask] = col
        df["OUTSIDE_OPTION_RATE_BP"] = out_rate
        df["OUTSIDE_OPTION_SOURCE"] = out_source
        return

    out_rate = rate_frame.max(axis=1, skipna=True)
    out_source = pd.Series(pd.NA, index=df.index, dtype="string")
    has_any = available_count > 0
    if has_any.any():
        out_source.loc[has_any] = rate_frame.loc[has_any].idxmax(axis=1).astype("string")
    df["OUTSIDE_OPTION_RATE_BP"] = out_rate
    df["OUTSIDE_OPTION_SOURCE"] = out_source


def _build_rate_sensitive_exposure(df: pd.DataFrame, exposure_cfg: dict[str, Any]) -> pd.Series:
    exposure_value = pd.Series(0.0, index=df.index, dtype="float64")
    exposure_weight = pd.Series(0.0, index=df.index, dtype="float64")

    for col, weight in exposure_cfg.items():
        weight = float(weight)
        if weight <= 0:
            continue
        series = clip01(safe_num(df, str(col).upper()))
        available = series.notna()
        if not available.any():
            continue
        exposure_value = exposure_value.add(series.where(available, 0.0) * weight, fill_value=0.0)
        exposure_weight = exposure_weight.add(
            pd.Series(np.where(available, weight, 0.0), index=df.index, dtype="float64"),
            fill_value=0.0,
        )

    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = exposure_weight > 0
    out.loc[valid] = exposure_value.loc[valid] / exposure_weight.loc[valid]
    return out.clip(lower=0.0, upper=1.0)


def build_deposit_competition_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    market_rate_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).upper() for col in out.columns]

    if "CERT" not in out.columns or "REPDTE" not in out.columns:
        raise ValueError("Input must contain CERT and REPDTE.")

    out["CERT"] = out["CERT"].astype(str).str.strip()
    out["REPDTE"] = parse_report_date(out["REPDTE"]).astype("datetime64[ns]")
    out = out.dropna(subset=["CERT", "REPDTE"]).sort_values(["CERT", "REPDTE"]).reset_index(drop=True)

    if market_rate_history is not None and not market_rate_history.empty:
        market_features = build_market_rate_history_features(out["REPDTE"], market_rate_history)
        out = pd.concat([out, market_features], axis=1)
    else:
        out["HAS_MARKET_RATE_HISTORY"] = 0

    _convert_raw_rate_columns_to_bp(out, cfg)
    _build_outside_option_rate(out, cfg)

    # Deposit cost in bankALM is a decimal ratio, so convert to basis points.
    out["DOMESTIC_DEPOSIT_COST_BP"] = safe_num(out, "DOMESTIC_DEPOSIT_COST") * 10_000.0

    premium_floor_bp = float(cfg.get("premium_floor_bp", 0.0))
    out["OUTSIDE_OPTION_PREMIUM_BP"] = out["OUTSIDE_OPTION_RATE_BP"] - out["DOMESTIC_DEPOSIT_COST_BP"]
    out["OUTSIDE_OPTION_PREMIUM_POS_BP"] = out["OUTSIDE_OPTION_PREMIUM_BP"].clip(lower=premium_floor_bp)

    out["OUTSIDE_OPTION_RATE_BP_LAG1"] = out.groupby("CERT", dropna=False)["OUTSIDE_OPTION_RATE_BP"].shift(1)
    out["DOMESTIC_DEPOSIT_COST_BP_LAG1"] = out.groupby("CERT", dropna=False)["DOMESTIC_DEPOSIT_COST_BP"].shift(1)
    out["OUTSIDE_OPTION_RATE_QOQ_CHANGE_BP"] = out["OUTSIDE_OPTION_RATE_BP"] - out["OUTSIDE_OPTION_RATE_BP_LAG1"]
    out["DOMESTIC_DEPOSIT_COST_QOQ_CHANGE_BP"] = (
        out["DOMESTIC_DEPOSIT_COST_BP"] - out["DOMESTIC_DEPOSIT_COST_BP_LAG1"]
    )
    out["PASS_THROUGH_GAP_BP"] = (
        out["OUTSIDE_OPTION_RATE_QOQ_CHANGE_BP"] - out["DOMESTIC_DEPOSIT_COST_QOQ_CHANGE_BP"]
    ).clip(lower=0)

    exposure_cfg = cfg.get("rate_sensitive_exposure_weights", {})
    if not isinstance(exposure_cfg, dict) or not exposure_cfg:
        raise ValueError("Config must include rate_sensitive_exposure_weights.")

    out["RATE_SENSITIVE_DEPOSIT_EXPOSURE"] = _build_rate_sensitive_exposure(out, exposure_cfg)

    out["PREMIUM_X_RATE_SENSITIVE_EXPOSURE"] = (
        out["OUTSIDE_OPTION_PREMIUM_POS_BP"] * out["RATE_SENSITIVE_DEPOSIT_EXPOSURE"]
    )

    for share_col in [
        "UNINSURED_SHARE",
        "LARGE_ACCOUNT_SHARE",
        "BROKERED_SHARE",
        "LIST_SERVICE_SHARE",
        "TIME_DEPOSIT_SHARE",
    ]:
        out[f"PREMIUM_X_{share_col}"] = out["OUTSIDE_OPTION_PREMIUM_POS_BP"] * clip01(safe_num(out, share_col))

    out["PREMIUM_X_DEP_DRAWDOWN_4Q"] = (
        out["OUTSIDE_OPTION_PREMIUM_POS_BP"] * clip01(safe_num(out, "DEP_DRAWDOWN_4Q"))
    )
    out["PREMIUM_X_SHORT_FHLB_SHARE"] = (
        out["OUTSIDE_OPTION_PREMIUM_POS_BP"] * clip01(safe_num(out, "SHORT_FHLB_SHARE"))
    )

    score_cfg = cfg.get("transparent_pressure", {})
    components = score_cfg.get("components", {})
    fill_value = float(score_cfg.get("missing_rank_fill", 0.50))

    component_cols: list[tuple[str, float]] = []
    for _, spec in components.items():
        if not isinstance(spec, dict):
            continue
        raw_col = str(spec.get("column", "")).upper()
        if not raw_col:
            continue
        if raw_col not in out.columns:
            out[raw_col] = np.nan
        rank_col = f"RANK_{raw_col}"
        orientation = str(spec.get("orientation", "higher_is_worse")).lower()
        higher_is_better = orientation == "higher_is_better"
        add_pct_rank(
            df=out,
            value_col=raw_col,
            by_col="REPDTE",
            out_col=rank_col,
            higher_is_better=higher_is_better,
            fill_value=fill_value,
        )
        component_cols.append((rank_col, float(spec.get("weight", 0.0))))

    total_weight = sum(weight for _, weight in component_cols if weight > 0)
    if total_weight <= 0:
        raise ValueError("Transparent pressure score must have positive total weight.")

    out["DEPOSIT_COMPETITION_PRESSURE_SCORE"] = 0.0
    for rank_col, weight in component_cols:
        if weight <= 0:
            continue
        out["DEPOSIT_COMPETITION_PRESSURE_SCORE"] += weight * out[rank_col].fillna(fill_value)

    out["DEPOSIT_COMPETITION_PRESSURE_SCORE"] = (
        100.0 * out["DEPOSIT_COMPETITION_PRESSURE_SCORE"] / total_weight
    )
    out["DEPOSIT_COMPETITION_RESILIENCE_SCORE"] = 100.0 - out["DEPOSIT_COMPETITION_PRESSURE_SCORE"]

    return out


def run_builder(
    input_path: Path,
    config_path: Path,
    out_path: Path,
    market_rates_path: Path | None = None,
) -> pd.DataFrame:
    df = read_table(input_path)
    cfg = load_config(config_path)
    market_rates = load_market_rate_history(market_rates_path)
    out = build_deposit_competition_features(
        df=df,
        cfg=cfg,
        market_rate_history=market_rates,
    )
    save_table(out, out_path)
    return out


def main() -> None:
    args = parse_args()
    out = run_builder(
        input_path=args.input,
        config_path=args.config,
        out_path=args.out,
        market_rates_path=args.market_rates,
    )
    print(f"Saved {len(out):,} rows to {args.out}")


if __name__ == "__main__":
    main()
