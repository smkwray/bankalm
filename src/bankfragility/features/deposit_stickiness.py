"""Build relative deposit stickiness / run-risk features and deposit-life scenarios."""
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
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = np.where((den.notna()) & (den != 0), num / den, np.nan)
    return pd.Series(out, index=num.index, dtype="float64")


def rowwise_first(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    series_list: list[pd.Series] = []
    for col in candidates:
        if col in df.columns:
            series_list.append(pd.to_numeric(df[col], errors="coerce"))
    if not series_list:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    out = series_list[0].copy()
    for series in series_list[1:]:
        out = out.fillna(series)
    return out


def add_pct_rank(
    df: pd.DataFrame,
    value_col: str,
    by_col: str,
    out_col: str,
    higher_is_better: bool,
    fill_value: float,
) -> None:
    # Output a risk percentile where higher means worse / less sticky.
    def ranker(series: pd.Series) -> pd.Series:
        ranked = series.rank(pct=True, method="average", ascending=True)
        if higher_is_better:
            ranked = 1 - ranked
        return ranked.fillna(fill_value)

    df[out_col] = df.groupby(by_col, dropna=False)[value_col].transform(ranker)


def clip01(series: pd.Series) -> pd.Series:
    return pd.Series(np.clip(pd.to_numeric(series, errors="coerce"), 0.0, 1.0), index=series.index)


def load_scenario_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Scenario config must deserialize to a mapping.")
    return config


def build_deposit_stickiness_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).upper() for col in out.columns]

    if "CERT" not in out.columns or "REPDTE" not in out.columns:
        raise ValueError("Input must contain CERT and REPDTE.")

    out["CERT"] = out["CERT"].astype(str).str.strip()
    out["REPDTE"] = parse_report_date(out["REPDTE"])
    out = out.dropna(subset=["CERT", "REPDTE"]).sort_values(["CERT", "REPDTE"]).reset_index(drop=True)

    depdom = rowwise_first(out, ["DEPDOM", "DEP"])
    out["DEPDOM_EFFECTIVE"] = depdom
    # DEPUNA=0 for small banks often means "not reported" rather than "zero uninsured."
    # Treat DEPUNA=0 as unknown (NaN) when DEPDOM > 0, so it doesn't falsely lower run risk.
    depuna = safe_num(out, "DEPUNA")
    depuna_censored = depuna.where((depuna > 0) | depuna.isna() | (depdom <= 0))
    out["DEPUNA_MISSING"] = (depuna == 0) & (depdom > 0)
    out["UNINSURED_SHARE"] = safe_div(depuna_censored, depdom)
    out["BROKERED_SHARE"] = safe_div(safe_num(out, "BRO"), depdom)
    out["LIST_SERVICE_SHARE"] = safe_div(safe_num(out, "DEPLSNB"), depdom)
    out["LARGE_ACCOUNT_SHARE"] = safe_div(
        safe_num(out, "IDDEPLAM"),
        safe_num(out, "IDDEPLAM") + safe_num(out, "IDDEPSAM"),
    )
    out["CORE_DEPOSIT_SHARE"] = safe_div(safe_num(out, "COREDEP"), depdom)
    out["NONINTEREST_SHARE"] = safe_div(safe_num(out, "DEPNIDOM"), depdom)
    out["TRN_EX_DDT"] = (safe_num(out, "TRN") - safe_num(out, "DDT")).clip(lower=0)
    out["NMD_TOTAL"] = (
        safe_num(out, "DDT").fillna(0)
        + out["TRN_EX_DDT"].fillna(0)
        + safe_num(out, "NTRSMMDA").fillna(0)
        + safe_num(out, "NTRSOTH").fillna(0)
    )

    # Time deposit buckets: modern $250k threshold first, then legacy $100k threshold.
    out["TD_LARGE_0_3"] = rowwise_first(out, ["IDCD3LES", "CD3LES"])
    out["TD_LARGE_3_12"] = rowwise_first(out, ["IDCD3T12", "CD3T12"])
    out["TD_LARGE_1_3"] = rowwise_first(out, ["IDCD1T3", "CD1T3"])
    out["TD_LARGE_3PLUS"] = rowwise_first(out, ["IDCDOV3", "CDOV3"])

    out["TD_SMALL_0_3"] = rowwise_first(out, ["IDCD3LESS", "CD3LESS"])
    out["TD_SMALL_3_12"] = rowwise_first(out, ["IDCD3T12S", "CD3T12S"])
    out["TD_SMALL_1_3"] = rowwise_first(out, ["IDCD1T3S", "CD1T3S"])
    out["TD_SMALL_3PLUS"] = rowwise_first(out, ["IDCDOV3S", "CDOV3S"])

    out["TD_BUCKET_SUM"] = (
        out["TD_LARGE_0_3"].fillna(0)
        + out["TD_LARGE_3_12"].fillna(0)
        + out["TD_LARGE_1_3"].fillna(0)
        + out["TD_LARGE_3PLUS"].fillna(0)
        + out["TD_SMALL_0_3"].fillna(0)
        + out["TD_SMALL_3_12"].fillna(0)
        + out["TD_SMALL_1_3"].fillna(0)
        + out["TD_SMALL_3PLUS"].fillna(0)
    )
    ntrtime = safe_num(out, "NTRTIME")
    out["TIME_DEPOSITS_EFFECTIVE"] = np.where(out["TD_BUCKET_SUM"] > 0, out["TD_BUCKET_SUM"], ntrtime)
    out["TIME_DEPOSIT_SHARE"] = safe_div(out["TIME_DEPOSITS_EFFECTIVE"], depdom)

    # Deposit pricing proxy.
    out["DEPDOM_LAG1"] = out.groupby("CERT", dropna=False)["DEPDOM_EFFECTIVE"].shift(1)
    out["DEPDOM_LAG4"] = out.groupby("CERT", dropna=False)["DEPDOM_EFFECTIVE"].shift(4)
    out["AVG_DEPDOM_QTR"] = (out["DEPDOM_EFFECTIVE"] + out["DEPDOM_LAG1"]) / 2.0
    out["DOMESTIC_DEPOSIT_COST"] = safe_div(safe_num(out, "EDEPDOMQ") * 4.0, out["AVG_DEPDOM_QTR"])

    # Growth / drawdown.
    out["DEP_GROWTH_QOQ"] = safe_div(
        out["DEPDOM_EFFECTIVE"] - out["DEPDOM_LAG1"],
        out["DEPDOM_LAG1"],
    )
    out["DEP_GROWTH_YOY"] = safe_div(
        out["DEPDOM_EFFECTIVE"] - out["DEPDOM_LAG4"],
        out["DEPDOM_LAG4"],
    )
    out["DEP_GROWTH_VOL_4Q"] = (
        out.groupby("CERT", dropna=False)["DEP_GROWTH_QOQ"]
        .rolling(window=4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    rolling_peak = (
        out.groupby("CERT", dropna=False)["DEPDOM_EFFECTIVE"]
        .rolling(window=4, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    out["DEP_DRAWDOWN_4Q"] = safe_div(rolling_peak - out["DEPDOM_EFFECTIVE"], rolling_peak)

    # Cross-sectional transparent score ranks by quarter.
    score_cfg = cfg["transparent_run_risk"]
    fill_value = float(score_cfg.get("missing_rank_fill", 0.50))
    component_cols: list[tuple[str, float]] = []

    feature_map = {
        "uninsured_share": "UNINSURED_SHARE",
        "brokered_share": "BROKERED_SHARE",
        "list_service_share": "LIST_SERVICE_SHARE",
        "large_account_share": "LARGE_ACCOUNT_SHARE",
        "core_deposit_share": "CORE_DEPOSIT_SHARE",
        "noninterest_share": "NONINTEREST_SHARE",
        "time_deposit_share": "TIME_DEPOSIT_SHARE",
        "domestic_deposit_cost": "DOMESTIC_DEPOSIT_COST",
        "dep_growth_vol_4q": "DEP_GROWTH_VOL_4Q",
        "dep_drawdown_4q": "DEP_DRAWDOWN_4Q",
    }

    for feature_name, spec in score_cfg["components"].items():
        raw_col = feature_map[feature_name]
        rank_col = f"RANK_{feature_name.upper()}"
        orientation = str(spec["orientation"]).lower()
        higher_is_better = orientation == "higher_is_better"
        add_pct_rank(
            df=out,
            value_col=raw_col,
            by_col="REPDTE",
            out_col=rank_col,
            higher_is_better=higher_is_better,
            fill_value=fill_value,
        )
        component_cols.append((rank_col, float(spec["weight"])))

    total_weight = sum(weight for _, weight in component_cols)
    if total_weight <= 0:
        raise ValueError("Total transparent run-risk weight must be positive.")

    out["RUN_RISK_SCORE"] = 0.0
    for rank_col, weight in component_cols:
        out["RUN_RISK_SCORE"] += weight * out[rank_col].fillna(fill_value)
    out["RUN_RISK_SCORE"] = 100.0 * out["RUN_RISK_SCORE"] / total_weight
    out["STICKINESS_SCORE"] = 100.0 - out["RUN_RISK_SCORE"]
    risk_scalar = clip01(out["RUN_RISK_SCORE"] / 100.0)

    # Scenario-based non-maturity lives.
    scenario_cfg = cfg["life_scenarios"]
    stable_horizon = float(cfg.get("stable_equivalent_horizon_years", 2.0))
    td_mid = cfg["time_deposit_bucket_midpoints_years"]

    td_wal_num = (
        out["TD_SMALL_0_3"].fillna(0) * float(td_mid["small_0_3m"])
        + out["TD_SMALL_3_12"].fillna(0) * float(td_mid["small_3_12m"])
        + out["TD_SMALL_1_3"].fillna(0) * float(td_mid["small_1_3y"])
        + out["TD_SMALL_3PLUS"].fillna(0) * float(td_mid["small_3plus_y"])
        + out["TD_LARGE_0_3"].fillna(0) * float(td_mid["large_0_3m"])
        + out["TD_LARGE_3_12"].fillna(0) * float(td_mid["large_3_12m"])
        + out["TD_LARGE_1_3"].fillna(0) * float(td_mid["large_1_3y"])
        + out["TD_LARGE_3PLUS"].fillna(0) * float(td_mid["large_3plus_y"])
    )
    out["TIME_DEPOSIT_WAL"] = safe_div(td_wal_num, out["TIME_DEPOSITS_EFFECTIVE"])

    td_stable_equiv = (
        out["TD_SMALL_0_3"].fillna(0) * min(float(td_mid["small_0_3m"]) / stable_horizon, 1.0)
        + out["TD_SMALL_3_12"].fillna(0) * min(float(td_mid["small_3_12m"]) / stable_horizon, 1.0)
        + out["TD_SMALL_1_3"].fillna(0) * min(float(td_mid["small_1_3y"]) / stable_horizon, 1.0)
        + out["TD_SMALL_3PLUS"].fillna(0) * min(float(td_mid["small_3plus_y"]) / stable_horizon, 1.0)
        + out["TD_LARGE_0_3"].fillna(0) * min(float(td_mid["large_0_3m"]) / stable_horizon, 1.0)
        + out["TD_LARGE_3_12"].fillna(0) * min(float(td_mid["large_3_12m"]) / stable_horizon, 1.0)
        + out["TD_LARGE_1_3"].fillna(0) * min(float(td_mid["large_1_3y"]) / stable_horizon, 1.0)
        + out["TD_LARGE_3PLUS"].fillna(0) * min(float(td_mid["large_3plus_y"]) / stable_horizon, 1.0)
    )

    # Non-maturity deposit categories.
    cat_amounts = {
        "DDT": safe_num(out, "DDT").fillna(0),
        "TRN_EX_DDT": out["TRN_EX_DDT"].fillna(0),
        "MMDA": safe_num(out, "NTRSMMDA").fillna(0),
        "OTHER_SAVINGS": safe_num(out, "NTRSOTH").fillna(0),
    }

    for scenario_name, spec in scenario_cfg.items():
        max_adj = float(spec["max_downward_adjustment_fraction"])
        cat_lives = spec["category_lives_years"]

        ddt_life = np.maximum(
            float(cat_lives["ddt"]["floor"]),
            float(cat_lives["ddt"]["base"]) * (1.0 - max_adj * risk_scalar),
        )
        trn_ex_ddt_life = np.maximum(
            float(cat_lives["trn_ex_ddt"]["floor"]),
            float(cat_lives["trn_ex_ddt"]["base"]) * (1.0 - max_adj * risk_scalar),
        )
        mmda_life = np.maximum(
            float(cat_lives["mmda"]["floor"]),
            float(cat_lives["mmda"]["base"]) * (1.0 - max_adj * risk_scalar),
        )
        other_savings_life = np.maximum(
            float(cat_lives["other_savings"]["floor"]),
            float(cat_lives["other_savings"]["base"]) * (1.0 - max_adj * risk_scalar),
        )

        out[f"DDT_LIFE_{scenario_name.upper()}"] = ddt_life
        out[f"TRN_EX_DDT_LIFE_{scenario_name.upper()}"] = trn_ex_ddt_life
        out[f"MMDA_LIFE_{scenario_name.upper()}"] = mmda_life
        out[f"OTHER_SAVINGS_LIFE_{scenario_name.upper()}"] = other_savings_life

        nmd_wal_num = (
            cat_amounts["DDT"] * ddt_life
            + cat_amounts["TRN_EX_DDT"] * trn_ex_ddt_life
            + cat_amounts["MMDA"] * mmda_life
            + cat_amounts["OTHER_SAVINGS"] * other_savings_life
        )
        out[f"NMD_WAL_{scenario_name.upper()}"] = safe_div(nmd_wal_num, out["NMD_TOTAL"])

        deposit_wal_num = nmd_wal_num + out["TIME_DEPOSIT_WAL"].fillna(0) * out["TIME_DEPOSITS_EFFECTIVE"].fillna(0)
        out[f"DEPOSIT_WAL_{scenario_name.upper()}"] = safe_div(
            deposit_wal_num,
            out["NMD_TOTAL"].fillna(0) + out["TIME_DEPOSITS_EFFECTIVE"].fillna(0),
        )

        nmd_stable_equiv = (
            cat_amounts["DDT"] * np.minimum(ddt_life / stable_horizon, 1.0)
            + cat_amounts["TRN_EX_DDT"] * np.minimum(trn_ex_ddt_life / stable_horizon, 1.0)
            + cat_amounts["MMDA"] * np.minimum(mmda_life / stable_horizon, 1.0)
            + cat_amounts["OTHER_SAVINGS"] * np.minimum(other_savings_life / stable_horizon, 1.0)
        )
        out[f"DEPOSIT_STABLE_EQUIV_{scenario_name.upper()}"] = nmd_stable_equiv + td_stable_equiv

    return out


def run_builder(input_path: Path, scenario_config_path: Path, out_path: Path) -> pd.DataFrame:
    data = read_table(input_path)
    config = load_scenario_config(scenario_config_path)
    result = build_deposit_stickiness_features(data, config)
    save_table(result, out_path)
    return result


def main() -> None:
    args = parse_args()
    result = run_builder(
        input_path=args.input,
        scenario_config_path=args.scenario_config,
        out_path=args.out,
    )
    print(f"Saved {len(result):,} rows to {args.out}")


if __name__ == "__main__":
    main()
