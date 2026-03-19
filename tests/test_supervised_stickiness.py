from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankfragility.models.supervised_stickiness import (
    build_supervised_overlay,
    compute_deposit_growth,
    compute_next_deposit_growth,
    compute_next_quarter_observed,
    label_absolute_outflow,
    label_relative_outflow,
    select_features,
)


def _sample_data(n_banks: int = 4, n_quarters: int = 6) -> pd.DataFrame:
    """Build a small multi-bank multi-quarter dataset for testing."""
    rows = []
    rng = np.random.RandomState(42)
    for cert in range(100, 100 + n_banks):
        base_dep = rng.uniform(500, 5000)
        for q in range(n_quarters):
            quarter = f"202{2 + q // 4}-{['03','06','09','12'][q % 4]}-{['31','30','30','31'][q % 4]}"
            growth = rng.normal(0.02, 0.05)
            base_dep *= (1 + growth)
            rows.append({
                "CERT": str(cert),
                "REPDTE": quarter,
                "DEPDOM": base_dep,
                "UNINSURED_SHARE": rng.uniform(0, 0.8),
                "BROKERED_SHARE": rng.uniform(0, 0.3),
                "CORE_DEPOSIT_SHARE": rng.uniform(0.5, 1.0),
                "NONINTEREST_SHARE": rng.uniform(0, 0.5),
                "VOLATILE_TO_LIQUID_LOWER": rng.uniform(0.2, 5.0),
                "LOANS_TO_CORE_DEPOSITS": rng.uniform(0.5, 2.0),
            })
    return pd.DataFrame(rows)


def test_label_is_t_plus_1_not_contemporaneous() -> None:
    """The severe outflow label should reflect NEXT quarter's outcome, not current quarter."""
    df = pd.DataFrame({
        "CERT": ["A", "A", "A", "A"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]),
        "DEPDOM": [100.0, 110.0, 80.0, 85.0],  # big drop in Q3
    })
    next_growth = compute_next_deposit_growth(df)
    next_obs = compute_next_quarter_observed(df)
    labels = label_relative_outflow(next_growth, df["REPDTE"], next_obs, percentile=50.0)
    # Q2 should get the label for Q3's drop (T+1), not its own growth
    # Q3 should get the label for Q4's outcome
    # The label at Q2 reflects the Q2→Q3 growth which is negative
    q2_label = labels.iloc[1]
    assert q2_label == 1, f"Q2 should be labeled 1 (next quarter has severe drop), got {q2_label}"
    # Last quarter has no observed next quarter → censored
    q4_label = labels.iloc[3]
    assert pd.isna(q4_label), f"Q4 should be censored (no next quarter), got {q4_label}"


def test_compute_deposit_growth_returns_qoq_changes() -> None:
    df = pd.DataFrame({
        "CERT": ["A", "A", "A"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"]),
        "DEPDOM": [100.0, 110.0, 99.0],
    })
    growth = compute_deposit_growth(df)
    assert pd.isna(growth.iloc[0])  # first quarter has no prior
    assert growth.iloc[1] == pytest.approx(0.10)
    assert growth.iloc[2] == pytest.approx(-0.10, abs=0.001)


def test_relative_outflow_label_marks_bottom_percentile() -> None:
    df = _sample_data(n_banks=10, n_quarters=4)
    next_growth = compute_next_deposit_growth(df)
    next_obs = compute_next_quarter_observed(df)
    labels = label_relative_outflow(next_growth, pd.to_datetime(df["REPDTE"]), next_obs, percentile=20.0)
    # With 10 banks × 4 quarters, some should be labeled
    observed = labels.dropna()
    assert observed.sum() > 0
    assert observed.sum() < len(observed)


def test_absolute_outflow_requires_actual_negative_growth() -> None:
    df = pd.DataFrame({
        "CERT": ["A", "A", "B", "B"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"]),
        "DEPDOM": [100.0, 110.0, 100.0, 80.0],
    })
    next_growth = compute_next_deposit_growth(df)
    next_obs = compute_next_quarter_observed(df)
    labels = label_absolute_outflow(next_growth, df["REPDTE"], next_obs, percentile=50.0)
    assert labels.iloc[0] == 0
    assert labels.iloc[2] == 1


def test_select_features_excludes_zero_variance() -> None:
    df = pd.DataFrame({
        "UNINSURED_SHARE": [0.5] * 20,  # zero variance
        "BROKERED_SHARE": np.random.rand(20),  # has variance
        "CORE_DEPOSIT_SHARE": np.random.rand(20),
    })
    features = select_features(df)
    assert "UNINSURED_SHARE" not in features
    assert "BROKERED_SHARE" in features


def test_build_supervised_overlay_produces_probability_column() -> None:
    df = _sample_data(n_banks=8, n_quarters=8)
    result = build_supervised_overlay(df, outflow_percentile=10.0)
    assert "SUPERVISED_OUTFLOW_SCORE" in result.columns
    assert "SUPERVISED_RISK_SCORE" in result.columns
    assert "SEVERE_RELATIVE_OUTFLOW" in result.columns
    assert "SEVERE_ABSOLUTE_OUTFLOW" in result.columns
    assert "NEXT_DEP_GROWTH_QOQ" in result.columns
    assert "NEXT_Q_OBSERVED" in result.columns
    # Scores should be in [0, 1]
    probs = result["SUPERVISED_OUTFLOW_SCORE"].dropna()
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_build_supervised_overlay_censors_terminal_rows() -> None:
    df = pd.DataFrame({
        "CERT": ["A", "A", "B", "B"],
        "REPDTE": ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
        "DEPDOM": [100.0, 90.0, 200.0, 180.0],
        "UNINSURED_SHARE": [0.5, 0.5, 0.4, 0.4],
        "BROKERED_SHARE": [0.1, 0.1, 0.1, 0.1],
        "CORE_DEPOSIT_SHARE": [0.8, 0.8, 0.8, 0.8],
        "NONINTEREST_SHARE": [0.1, 0.1, 0.1, 0.1],
        "VOLATILE_TO_LIQUID_LOWER": [1.0, 1.0, 1.0, 1.0],
        "LOANS_TO_CORE_DEPOSITS": [1.0, 1.0, 1.0, 1.0],
    })
    result = build_supervised_overlay(df, outflow_percentile=50.0)
    terminal = result.groupby("CERT").tail(1)
    assert terminal["NEXT_Q_OBSERVED"].eq(False).all()
    assert terminal["SEVERE_RELATIVE_OUTFLOW"].isna().all()
    assert terminal["SEVERE_ABSOLUTE_OUTFLOW"].isna().all()


def test_build_supervised_overlay_with_insufficient_features() -> None:
    """If too few features have variance, model should gracefully produce NaN."""
    df = pd.DataFrame({
        "CERT": ["A", "A"],
        "REPDTE": ["2024-03-31", "2024-06-30"],
        "DEPDOM": [100.0, 90.0],
    })
    result = build_supervised_overlay(df)
    assert pd.isna(result["SUPERVISED_RISK_SCORE"].iloc[0])
