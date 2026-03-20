from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bankfragility.features.deposit_competition import (
    build_deposit_competition_features,
    build_market_rate_history_features,
    load_market_rate_history,
    run_builder,
)


def _make_cfg() -> dict:
    return {
        "benchmark_selection": {
            "method": "max_available",
            "benchmark_priority": ["IORB_BP", "RRPONTSYAWARD_BP", "YC_3MO_BP"],
        },
        "raw_rate_columns_to_bp": {
            "IORB": "IORB_BP",
            "RRPONTSYAWARD": "RRPONTSYAWARD_BP",
            "YC_3MO": "YC_3MO_BP",
        },
        "premium_floor_bp": 0.0,
        "rate_sensitive_exposure_weights": {
            "UNINSURED_SHARE": 0.45,
            "BROKERED_SHARE": 0.15,
            "LIST_SERVICE_SHARE": 0.10,
            "TIME_DEPOSIT_SHARE": 0.05,
        },
        "transparent_pressure": {
            "missing_rank_fill": 0.5,
            "components": {
                "outside_option_premium_pos_bp": {
                    "column": "OUTSIDE_OPTION_PREMIUM_POS_BP",
                    "weight": 1.0,
                    "orientation": "higher_is_worse",
                }
            },
        },
    }


def test_missing_market_rate_path_is_treated_as_optional(tmp_path: Path) -> None:
    history = load_market_rate_history(tmp_path / "does-not-exist.csv")
    assert history.empty


def test_build_market_rate_history_features_uses_latest_observation_on_or_before_repdte() -> None:
    repdte = pd.Series(pd.to_datetime(["2024-03-31", "2024-06-30"]))
    history = pd.DataFrame(
        {
            "DATE": ["2024-03-28", "2024-06-28"],
            "IORB": [5.40, 5.40],
            "RRPONTSYAWARD": [5.30, 5.30],
        }
    )

    out = build_market_rate_history_features(repdte, history)

    assert out.loc[0, "HAS_MARKET_RATE_HISTORY"] == 1
    assert out.loc[0, "MARKET_RATE_DATE"] == pd.Timestamp("2024-03-28")
    assert out.loc[1, "MARKET_RATE_DATE"] == pd.Timestamp("2024-06-28")


def test_premium_scaling_uses_basis_points() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "YC_3MO": [5.00],
            "DOMESTIC_DEPOSIT_COST": [0.01],
            "UNINSURED_SHARE": [0.50],
            "BROKERED_SHARE": [0.00],
            "LIST_SERVICE_SHARE": [0.00],
            "TIME_DEPOSIT_SHARE": [0.10],
            "DEP_DRAWDOWN_4Q": [0.10],
            "SHORT_FHLB_SHARE": [0.20],
        }
    )

    out = build_deposit_competition_features(df, _make_cfg())

    assert out.loc[0, "YC_3MO_BP"] == pytest.approx(500.0)
    assert out.loc[0, "DOMESTIC_DEPOSIT_COST_BP"] == pytest.approx(100.0)
    assert out.loc[0, "OUTSIDE_OPTION_PREMIUM_BP"] == pytest.approx(400.0)


def test_builder_prefers_higher_optional_market_rate_when_available() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "YC_3MO": [5.00],
            "DOMESTIC_DEPOSIT_COST": [0.01],
            "UNINSURED_SHARE": [0.50],
            "BROKERED_SHARE": [0.00],
            "LIST_SERVICE_SHARE": [0.00],
            "TIME_DEPOSIT_SHARE": [0.10],
            "DEP_DRAWDOWN_4Q": [0.10],
            "SHORT_FHLB_SHARE": [0.20],
        }
    )
    market_history = pd.DataFrame(
        {
            "DATE": ["2024-03-28"],
            "IORB": [5.40],
            "RRPONTSYAWARD": [5.30],
        }
    )

    out = build_deposit_competition_features(df, _make_cfg(), market_rate_history=market_history)

    assert out.loc[0, "OUTSIDE_OPTION_RATE_BP"] == pytest.approx(540.0)
    assert out.loc[0, "OUTSIDE_OPTION_SOURCE"] == "IORB_BP"
    assert out.loc[0, "OUTSIDE_OPTION_PREMIUM_BP"] == pytest.approx(440.0)


def test_pressure_score_orders_higher_premium_as_worse() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "YC_3MO": [5.00, 5.00],
            "DOMESTIC_DEPOSIT_COST": [0.01, 0.03],
            "UNINSURED_SHARE": [0.50, 0.50],
            "BROKERED_SHARE": [0.00, 0.00],
            "LIST_SERVICE_SHARE": [0.00, 0.00],
            "TIME_DEPOSIT_SHARE": [0.10, 0.10],
            "DEP_DRAWDOWN_4Q": [0.10, 0.10],
            "SHORT_FHLB_SHARE": [0.20, 0.20],
        }
    )

    out = build_deposit_competition_features(df, _make_cfg()).sort_values("CERT").reset_index(drop=True)

    assert out.loc[0, "OUTSIDE_OPTION_PREMIUM_BP"] > out.loc[1, "OUTSIDE_OPTION_PREMIUM_BP"]
    assert out.loc[0, "DEPOSIT_COMPETITION_PRESSURE_SCORE"] > out.loc[1, "DEPOSIT_COMPETITION_PRESSURE_SCORE"]
    assert out.loc[0, "DEPOSIT_COMPETITION_RESILIENCE_SCORE"] < out.loc[1, "DEPOSIT_COMPETITION_RESILIENCE_SCORE"]


def test_rate_sensitive_exposure_renormalizes_available_shares() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "YC_3MO": [5.00, 5.00],
            "DOMESTIC_DEPOSIT_COST": [0.01, 0.01],
            "UNINSURED_SHARE": [1.0, pd.NA],
            "BROKERED_SHARE": [0.0, pd.NA],
            "LIST_SERVICE_SHARE": [pd.NA, pd.NA],
            "TIME_DEPOSIT_SHARE": [pd.NA, pd.NA],
            "DEP_DRAWDOWN_4Q": [0.10, 0.10],
            "SHORT_FHLB_SHARE": [0.20, 0.20],
        }
    )

    out = build_deposit_competition_features(df, _make_cfg()).sort_values("CERT").reset_index(drop=True)

    expected = (0.45 * 1.0 + 0.15 * 0.0) / (0.45 + 0.15)
    assert out.loc[0, "RATE_SENSITIVE_DEPOSIT_EXPOSURE"] == pytest.approx(expected)
    assert pd.isna(out.loc[1, "RATE_SENSITIVE_DEPOSIT_EXPOSURE"])


def test_run_builder_without_market_rates_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    config_path = tmp_path / "config.yaml"
    out_path = tmp_path / "out.csv"

    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "YC_3MO": [5.00],
            "DOMESTIC_DEPOSIT_COST": [0.01],
            "UNINSURED_SHARE": [0.50],
            "BROKERED_SHARE": [0.00],
            "LIST_SERVICE_SHARE": [0.00],
            "TIME_DEPOSIT_SHARE": [0.10],
            "DEP_DRAWDOWN_4Q": [0.10],
            "SHORT_FHLB_SHARE": [0.20],
        }
    )
    df.to_csv(input_path, index=False)

    config_path.write_text(
        """
benchmark_selection:
  method: max_available
  benchmark_priority: [IORB_BP, RRPONTSYAWARD_BP, YC_3MO_BP]
raw_rate_columns_to_bp:
  YC_3MO: YC_3MO_BP
premium_floor_bp: 0.0
rate_sensitive_exposure_weights:
  UNINSURED_SHARE: 0.45
  BROKERED_SHARE: 0.15
  LIST_SERVICE_SHARE: 0.10
  TIME_DEPOSIT_SHARE: 0.05
transparent_pressure:
  missing_rank_fill: 0.50
  components:
    outside_option_premium_pos_bp:
      column: OUTSIDE_OPTION_PREMIUM_POS_BP
      weight: 1.0
      orientation: higher_is_worse
""".strip(),
        encoding="utf-8",
    )

    out = run_builder(input_path=input_path, config_path=config_path, out_path=out_path, market_rates_path=None)

    assert out_path.exists()
    assert out.loc[0, "HAS_MARKET_RATE_HISTORY"] == 0
