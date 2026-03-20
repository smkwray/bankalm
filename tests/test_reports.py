from __future__ import annotations

import pandas as pd

from bankfragility.reporting.reports import (
    bank_drill_down,
    peer_group_summary,
    quarter_league_table,
    scenario_comparison,
    treasury_regime_summary,
)


def _sample_indices() -> pd.DataFrame:
    return pd.DataFrame({
        "CERT": ["100", "100", "200", "200"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"]),
        "NAMEFULL": ["Bank A", "Bank A", "Bank B", "Bank B"],
        "PEER_GROUP": ["community", "community", "community", "community"],
        "ASSET": [500.0, 510.0, 800.0, 820.0],
        "RUN_RISK_INDEX": [80.0, 75.0, 40.0, 45.0],
        "DEPOSIT_STICKINESS_INDEX": [20.0, 25.0, 60.0, 55.0],
        "ALM_MISMATCH_INDEX": [60.0, 65.0, 30.0, 35.0],
        "TREASURY_BUFFER_INDEX": [40.0, 35.0, 70.0, 65.0],
        "DEPOSIT_COMPETITION_PRESSURE_INDEX": [68.0, 70.0, 35.0, 38.0],
        "DEPOSIT_COMPETITION_RESILIENCE_INDEX": [32.0, 30.0, 65.0, 62.0],
        "FUNDING_FRAGILITY_INDEX": [72.0, 76.0, 38.0, 42.0],
        "RUN_RISK_SCORE": [80.0, 75.0, 40.0, 45.0],
        "DEPOSIT_COMPETITION_PRESSURE_SCORE": [66.0, 69.0, 33.0, 36.0],
        "UNINSURED_SHARE": [0.7, 0.65, 0.2, 0.25],
        "VOLATILE_TO_LIQUID_LOWER": [2.5, 2.8, 0.5, 0.6],
        "TREASURY_TO_UNINSURED": [0.1, 0.08, 0.5, 0.45],
        "OUTSIDE_OPTION_PREMIUM_BP": [410.0, 430.0, 120.0, 140.0],
        "PASS_THROUGH_GAP_BP": [85.0, 90.0, 15.0, 20.0],
        "RATE_SENSITIVE_DEPOSIT_EXPOSURE": [0.45, 0.47, 0.18, 0.20],
        "DEPOSIT_WAL_BASELINE": [1.5, 1.4, 3.0, 2.8],
        "TREASURY_YIELD_DATE": pd.to_datetime(["2024-03-29", "2024-06-28", "2024-03-29", "2024-06-28"]),
        "HAS_TREASURY_YIELD_HISTORY": [1, 1, 1, 1],
        "YC_2YR": [4.5, 4.7, 4.5, 4.7],
        "YC_10YR": [4.2, 4.4, 4.2, 4.4],
        "YC_10Y_3M_SLOPE_BP": [-80.0, -40.0, -80.0, -40.0],
    })


def test_bank_drill_down_returns_single_bank_time_series() -> None:
    df = _sample_indices()
    result = bank_drill_down(df, "100")
    assert len(result) == 2
    assert list(result["REPDTE"]) == [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]
    assert "FUNDING_FRAGILITY_INDEX" in result.columns
    assert "DEPOSIT_COMPETITION_PRESSURE_INDEX" in result.columns


def test_bank_drill_down_returns_empty_for_missing_cert() -> None:
    df = _sample_indices()
    assert bank_drill_down(df, "999").empty


def test_quarter_league_table_ranks_by_fragility() -> None:
    df = _sample_indices()
    result = quarter_league_table(df, "2024-03-31")
    assert len(result) == 2
    assert result.iloc[0]["CERT"] == "100"  # higher fragility first
    assert result.iloc[0]["FRAGILITY_RANK"] == 1
    assert "DEPOSIT_COMPETITION_PRESSURE_SCORE" in result.columns


def test_peer_group_summary_aggregates_correctly() -> None:
    df = _sample_indices()
    result = peer_group_summary(df)
    assert len(result) >= 2  # at least 2 quarters
    assert "FUNDING_FRAGILITY_INDEX_MEDIAN" in result.columns
    assert "DEPOSIT_COMPETITION_PRESSURE_INDEX_MEDIAN" in result.columns


def test_scenario_comparison_orders_by_risk() -> None:
    df = _sample_indices()
    result = scenario_comparison(df, "2024-03-31")
    assert len(result) == 2
    assert result.iloc[0]["RUN_RISK_SCORE"] > result.iloc[1]["RUN_RISK_SCORE"]
    assert "DEPOSIT_WAL_BASELINE" in result.columns


def test_treasury_regime_summary_returns_one_row_per_quarter() -> None:
    df = _sample_indices()
    result = treasury_regime_summary(df)
    assert len(result) == 2
    assert "YC_10YR" in result.columns
    assert "CURVE_REGIME" in result.columns
