from __future__ import annotations

import pandas as pd

from bankfragility.validation.failure_backtest import (
    _quarter_distance,
    build_failure_dataset,
    build_metrics_table,
    compute_backtest_metrics,
)


def test_quarter_distance_same_next_quarter() -> None:
    rep = pd.Series(pd.to_datetime(["2008-12-31"]))
    fail = pd.Series(pd.to_datetime(["2009-01-25"]))
    assert _quarter_distance(rep, fail).iloc[0] == 1


def test_build_failure_dataset_adds_forward_horizon_labels() -> None:
    indices = pd.DataFrame({
        "CERT": ["100", "100", "200", "200"],
        "REPDTE": pd.to_datetime(["2008-03-31", "2008-06-30", "2008-03-31", "2008-06-30"]),
        "RUN_RISK_INDEX": [80.0, 85.0, 30.0, 35.0],
    })
    failures = pd.DataFrame({"CERT": [100.0], "FAILDATE": ["2008-12-01"]})
    result = build_failure_dataset(indices, failures, min_year=2007, max_year=2008)

    q2_failed = result[(result["CERT"] == "100") & (result["REPDTE"] == pd.Timestamp("2008-06-30"))].iloc[0]
    assert q2_failed["FORWARD_OBSERVED_2Q"] == 1
    assert q2_failed["FAIL_WITHIN_2Q"] == 1
    assert q2_failed["FAIL_WITHIN_1Q"] == 0

    q2_survivor = result[(result["CERT"] == "200") & (result["REPDTE"] == pd.Timestamp("2008-06-30"))].iloc[0]
    assert q2_survivor["FAIL_WITHIN_2Q"] == 0


def test_build_failure_dataset_censors_unobservable_horizons() -> None:
    indices = pd.DataFrame({
        "CERT": ["100", "200"],
        "REPDTE": pd.to_datetime(["2024-09-30", "2024-09-30"]),
        "RUN_RISK_INDEX": [80.0, 20.0],
    })
    failures = pd.DataFrame({"CERT": [100.0], "FAILDATE": ["2025-01-20"]})
    result = build_failure_dataset(indices, failures, min_year=2024, max_year=2024)

    row = result[result["CERT"] == "100"].iloc[0]
    assert row["FORWARD_OBSERVED_2Q"] == 0
    assert pd.isna(row["FAIL_WITHIN_2Q"])
    assert row["FORWARD_OBSERVED_1Q"] == 1


def test_compute_backtest_metrics_filters_on_observed_horizon() -> None:
    df = pd.DataFrame({
        "score": [90, 85, 20, 10],
        "FAIL_WITHIN_1Q": [1, 1, 0, pd.NA],
        "FORWARD_OBSERVED_1Q": [1, 1, 1, 0],
    })
    m = compute_backtest_metrics(df, "score", "FAIL_WITHIN_1Q", observed_col="FORWARD_OBSERVED_1Q")
    assert m["auc"] > 0.9
    assert m["n_failures"] == 2
    assert m["n_total"] == 3


def test_build_metrics_table_includes_episode_and_horizon_slices() -> None:
    df = pd.DataFrame({
        "CERT": ["100", "200", "100", "200"],
        "REPDTE": pd.to_datetime(["2023-03-31", "2023-03-31", "2023-06-30", "2023-06-30"]),
        "RUN_RISK_INDEX": [90.0, 10.0, 85.0, 20.0],
        "FUNDING_FRAGILITY_INDEX": [88.0, 12.0, 84.0, 22.0],
        "ALM_MISMATCH_INDEX": [70.0, 30.0, 65.0, 35.0],
        "DEPOSIT_COMPETITION_PRESSURE_INDEX": [75.0, 25.0, 72.0, 28.0],
        "FAIL_WITHIN_1Q": [1, 0, 1, 0],
        "FAIL_WITHIN_2Q": [1, 0, 1, 0],
        "FAIL_WITHIN_4Q": [1, 0, 1, 0],
        "FORWARD_OBSERVED_1Q": [1, 1, 1, 1],
        "FORWARD_OBSERVED_2Q": [1, 1, 1, 1],
        "FORWARD_OBSERVED_4Q": [1, 1, 1, 1],
    })
    metrics = build_metrics_table(df)
    full_sample = metrics[(metrics["SLICE"] == "full_sample") & (metrics["HORIZON_QUARTERS"] == 1)]
    assert not full_sample.empty
    assert set(full_sample["SCORE_COL"]) >= {
        "RUN_RISK_INDEX",
        "FUNDING_FRAGILITY_INDEX",
        "ALM_MISMATCH_INDEX",
        "DEPOSIT_COMPETITION_PRESSURE_INDEX",
    }
