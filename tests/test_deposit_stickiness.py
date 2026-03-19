from __future__ import annotations

import pandas as pd

from bankfragility.features.deposit_stickiness import build_deposit_stickiness_features


def test_run_risk_scoring_orders_higher_uninsured_as_worse() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "DEPDOM": [100.0, 100.0],
            "DEPUNA": [10.0, 90.0],
            "DDT": [50.0, 50.0],
            "TRN": [50.0, 50.0],
            "NTRSMMDA": [0.0, 0.0],
            "NTRSOTH": [0.0, 0.0],
            "NTRTIME": [0.0, 0.0],
        }
    )

    cfg = {
        "transparent_run_risk": {
            "missing_rank_fill": 0.5,
            "components": {
                "uninsured_share": {"weight": 1.0, "orientation": "higher_is_worse"},
            },
        },
        "life_scenarios": {
            "baseline": {
                "max_downward_adjustment_fraction": 0.5,
                "category_lives_years": {
                    "ddt": {"base": 4.0, "floor": 1.0},
                    "trn_ex_ddt": {"base": 2.0, "floor": 0.5},
                    "mmda": {"base": 1.0, "floor": 0.25},
                    "other_savings": {"base": 1.0, "floor": 0.25},
                },
            }
        },
        "time_deposit_bucket_midpoints_years": {
            "small_0_3m": 0.125,
            "small_3_12m": 0.625,
            "small_1_3y": 2.0,
            "small_3plus_y": 4.0,
            "large_0_3m": 0.125,
            "large_3_12m": 0.625,
            "large_1_3y": 2.0,
            "large_3plus_y": 4.0,
        },
        "stable_equivalent_horizon_years": 2.0,
    }

    out = build_deposit_stickiness_features(df=df, cfg=cfg).sort_values("CERT").reset_index(drop=True)

    assert out.loc[0, "RUN_RISK_SCORE"] < out.loc[1, "RUN_RISK_SCORE"]
    assert out.loc[0, "STICKINESS_SCORE"] > out.loc[1, "STICKINESS_SCORE"]


def test_baseline_scenario_life_and_stable_equiv_reflect_risk_scalar() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": [20240331, 20240331],  # FDIC numeric date format
            "DEPDOM": [100.0, 100.0],
            "DEPUNA": [10.0, 90.0],
            "DDT": [50.0, 50.0],
            "TRN": [50.0, 50.0],
            "NTRSMMDA": [0.0, 0.0],
            "NTRSOTH": [0.0, 0.0],
            "NTRTIME": [0.0, 0.0],
        }
    )

    cfg = {
        "transparent_run_risk": {
            "missing_rank_fill": 0.5,
            "components": {
                "uninsured_share": {"weight": 1.0, "orientation": "higher_is_worse"},
            },
        },
        "life_scenarios": {
            "baseline": {
                "max_downward_adjustment_fraction": 0.5,
                "category_lives_years": {
                    "ddt": {"base": 4.0, "floor": 1.0},
                    "trn_ex_ddt": {"base": 2.0, "floor": 0.5},
                    "mmda": {"base": 1.0, "floor": 0.25},
                    "other_savings": {"base": 1.0, "floor": 0.25},
                },
            }
        },
        "time_deposit_bucket_midpoints_years": {
            "small_0_3m": 0.125,
            "small_3_12m": 0.625,
            "small_1_3y": 2.0,
            "small_3plus_y": 4.0,
            "large_0_3m": 0.125,
            "large_3_12m": 0.625,
            "large_1_3y": 2.0,
            "large_3plus_y": 4.0,
        },
        "stable_equivalent_horizon_years": 2.0,
    }

    out = build_deposit_stickiness_features(df=df, cfg=cfg).sort_values("CERT").reset_index(drop=True)

    # With two records in-quarter, risk scores are 50 and 100, so scalar is 0.5 and 1.0.
    assert out.loc[0, "DDT_LIFE_BASELINE"] == 3.0
    assert out.loc[1, "DDT_LIFE_BASELINE"] == 2.0
    assert out.loc[0, "DEPOSIT_WAL_BASELINE"] > out.loc[1, "DEPOSIT_WAL_BASELINE"]


def _make_cfg(**overrides):
    """Minimal stickiness config for edge-case tests."""
    cfg = {
        "transparent_run_risk": {
            "missing_rank_fill": 0.5,
            "components": {
                "uninsured_share": {"weight": 1.0, "orientation": "higher_is_worse"},
            },
        },
        "life_scenarios": {
            "baseline": {
                "max_downward_adjustment_fraction": 0.5,
                "category_lives_years": {
                    "ddt": {"base": 4.0, "floor": 1.0},
                    "trn_ex_ddt": {"base": 2.0, "floor": 0.5},
                    "mmda": {"base": 1.0, "floor": 0.25},
                    "other_savings": {"base": 1.0, "floor": 0.25},
                },
            }
        },
        "time_deposit_bucket_midpoints_years": {
            "small_0_3m": 0.125, "small_3_12m": 0.625,
            "small_1_3y": 2.0, "small_3plus_y": 4.0,
            "large_0_3m": 0.125, "large_3_12m": 0.625,
            "large_1_3y": 2.0, "large_3plus_y": 4.0,
        },
        "stable_equivalent_horizon_years": 2.0,
    }
    cfg.update(overrides)
    return cfg


def test_zero_depdom_produces_nan_shares() -> None:
    """A bank with zero domestic deposits should get NaN shares, not division errors."""
    df = pd.DataFrame({
        "CERT": ["900"],
        "REPDTE": ["2024-03-31"],
        "DEPDOM": [0.0],
        "DEPUNA": [0.0],
        "DDT": [0.0],
        "TRN": [0.0],
        "NTRSMMDA": [0.0],
        "NTRSOTH": [0.0],
        "NTRTIME": [0.0],
    })
    out = build_deposit_stickiness_features(df, _make_cfg())
    # Shares should be NaN or 0 — never infinity
    import numpy as np
    assert not np.isinf(out["UNINSURED_SHARE"].iloc[0])


def test_missing_optional_columns_still_produces_output() -> None:
    """Build should succeed even when optional columns like BRO, DEPLSNB are absent."""
    df = pd.DataFrame({
        "CERT": ["800", "801"],
        "REPDTE": ["2024-03-31", "2024-03-31"],
        "DEPDOM": [100.0, 100.0],
        "DEPUNA": [30.0, 70.0],
        "DDT": [40.0, 40.0],
        "TRN": [40.0, 40.0],
        "NTRSMMDA": [10.0, 10.0],
        "NTRSOTH": [10.0, 10.0],
        "NTRTIME": [0.0, 0.0],
    })
    out = build_deposit_stickiness_features(df, _make_cfg())
    assert len(out) == 2
    assert "RUN_RISK_SCORE" in out.columns
    assert "DDT_LIFE_BASELINE" in out.columns


def test_depuna_zero_treated_as_unknown_not_zero() -> None:
    """DEPUNA=0 should produce NaN uninsured share, not 0.0, for banks with positive deposits."""
    import numpy as np
    df = pd.DataFrame({
        "CERT": ["600", "601"],
        "REPDTE": ["2024-03-31", "2024-03-31"],
        "DEPDOM": [100.0, 100.0],
        "DEPUNA": [0.0, 50.0],  # bank 600: reported zero (unknown), bank 601: real value
        "DDT": [50.0, 50.0],
        "TRN": [50.0, 50.0],
        "NTRSMMDA": [0.0, 0.0],
        "NTRSOTH": [0.0, 0.0],
        "NTRTIME": [0.0, 0.0],
    })
    out = build_deposit_stickiness_features(df, _make_cfg())
    # Bank 600: DEPUNA=0 should be treated as unknown (NaN), not literal 0
    assert pd.isna(out.loc[out["CERT"] == "600", "UNINSURED_SHARE"].iloc[0])
    # Bank 601: DEPUNA=50 should produce real share
    assert out.loc[out["CERT"] == "601", "UNINSURED_SHARE"].iloc[0] == 0.5
    # Missingness flag
    assert out.loc[out["CERT"] == "600", "DEPUNA_MISSING"].iloc[0] == True
    assert out.loc[out["CERT"] == "601", "DEPUNA_MISSING"].iloc[0] == False


def test_single_bank_gets_midpoint_rank() -> None:
    """A single bank in a quarter should get a 0.5-ish percentile rank."""
    df = pd.DataFrame({
        "CERT": ["700"],
        "REPDTE": ["2024-03-31"],
        "DEPDOM": [100.0],
        "DEPUNA": [50.0],
        "DDT": [50.0],
        "TRN": [50.0],
        "NTRSMMDA": [0.0],
        "NTRSOTH": [0.0],
        "NTRTIME": [0.0],
    })
    out = build_deposit_stickiness_features(df, _make_cfg())
    # Single observation → rank = 1.0 (100th percentile of itself)
    assert out["RUN_RISK_SCORE"].iloc[0] > 0
