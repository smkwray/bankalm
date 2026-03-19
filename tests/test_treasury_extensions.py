from __future__ import annotations

import pandas as pd
import pytest

from bankfragility.features.treasury_extensions import (
    build_treasury_extensions,
    build_treasury_rate_history_features,
)


def test_build_treasury_extensions_adds_shock_outputs() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "SCUST": [100.0],
            "SCAGE": [40.0],
            "SCPLEDGE": [10.0],
            "CHBAL": [50.0],
            "FREPO": [20.0],
            "DEPUNA": [200.0],
            "RUNNABLE_FUNDING_PROXY": [100.0],
            "SECURITY_DURATION_PROXY": [4.0],
        }
    )

    out = build_treasury_extensions(df, shock_bps=[100])

    assert out.loc[0, "HQLA_NARROW_LOWER"] == pytest.approx(194.0)
    assert out.loc[0, "TREASURY_LOSS_100BP"] == pytest.approx(4.0)
    assert out.loc[0, "TREASURY_TO_UNINSURED_AFTER_100BP"] == pytest.approx(0.48)


def test_build_treasury_extensions_falls_back_to_voliab() -> None:
    df = pd.DataFrame(
        {
            "SCUST": [20.0],
            "SCAGE": [0.0],
            "CHBAL": [10.0],
            "FREPO": [0.0],
            "SCPLEDGE": [0.0],
            "DEPUNA": [40.0],
            "VOLIAB": [50.0],
        }
    )

    out = build_treasury_extensions(df, shock_bps=[100])

    assert out.loc[0, "HQLA_NARROW_LOWER_TO_RUNNABLE"] == pytest.approx(0.6)


def test_build_treasury_rate_history_features_uses_latest_observation_on_or_before_repdte() -> None:
    repdte = pd.Series(pd.to_datetime(["2024-03-31", "2024-06-30"]))
    history = pd.DataFrame(
        {
            "DATE": ["2024-03-28", "2024-06-28"],
            "YC_3MO": [5.2, 5.0],
            "YC_2YR": [4.6, 4.7],
            "YC_10YR": [4.2, 4.4],
            "YC_30YR": [4.4, 4.6],
        }
    )

    out = build_treasury_rate_history_features(repdte, history)

    assert out.loc[0, "HAS_TREASURY_YIELD_HISTORY"] == 1
    assert out.loc[0, "TREASURY_YIELD_DATE"] == pd.Timestamp("2024-03-28")
    assert out.loc[0, "YC_10Y_2Y_SLOPE_BP"] == pytest.approx(-40.0)
    assert out.loc[1, "YC_10YR_QOQ_CHANGE_BP"] == pytest.approx(20.0)


def test_build_treasury_extensions_adds_rate_history_columns_when_history_available() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "SCUST": [100.0],
            "SCAGE": [40.0],
            "SCPLEDGE": [10.0],
            "CHBAL": [50.0],
            "FREPO": [20.0],
            "DEPUNA": [200.0],
            "RUNNABLE_FUNDING_PROXY": [100.0],
            "SECURITY_DURATION_PROXY": [4.0],
        }
    )
    history = pd.DataFrame(
        {
            "DATE": ["2024-03-28"],
            "YC_3MO": [5.2],
            "YC_2YR": [4.6],
            "YC_10YR": [4.2],
            "YC_30YR": [4.4],
        }
    )

    out = build_treasury_extensions(df, treasury_history=history, shock_bps=[100])

    assert out.loc[0, "HAS_TREASURY_YIELD_HISTORY"] == 1
    assert out.loc[0, "YC_2YR"] == pytest.approx(4.6)
    assert out.loc[0, "YC_10Y_3M_SLOPE_BP"] == pytest.approx(-100.0)
