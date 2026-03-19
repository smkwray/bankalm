from __future__ import annotations

import pandas as pd
import pytest

from bankfragility.staging.ffiec_repricing import (
    build_repricing_features,
    collect_mdrm_codes,
    infer_repdte_from_zip,
)


def test_collect_mdrm_codes_extracts_nested_values() -> None:
    cfg = {
        "loan_maturity": {
            "other_loans": {"0_3m": "RCONA570", "3_12m": "RCONA571"},
        },
        "time_deposit_maturity": {
            "small": {"0_3m": "RCONHK07"},
        },
        "horizon_midpoints": {"0_3m": 0.125},  # not a code
    }
    codes = collect_mdrm_codes(cfg)
    assert "RCONA570" in codes
    assert "RCONA571" in codes
    assert "RCONHK07" in codes
    assert "0.125" not in codes


def test_infer_repdte_from_zip() -> None:
    from pathlib import Path

    assert infer_repdte_from_zip(Path("FFIEC CDR Call Bulk All Schedules 06302023.zip")) == "20230630"
    assert infer_repdte_from_zip(Path("no_date.zip")) is None


def test_build_repricing_features_computes_loan_wam() -> None:
    data = pd.DataFrame({
        "IDRSSD": [1],
        "RCONA570": [100.0],   # 0-3m
        "RCONA571": [0.0],     # 3-12m
        "RCONA572": [0.0],     # 1-3y
        "RCONA573": [0.0],     # 3-5y
        "RCONA574": [0.0],     # 5-15y
        "RCONA575": [100.0],   # 15y+
    })
    cfg = {
        "loan_maturity": {
            "other_loans": {
                "0_3m": "RCONA570", "3_12m": "RCONA571",
                "1_3y": "RCONA572", "3_5y": "RCONA573",
                "5_15y": "RCONA574", "15y_plus": "RCONA575",
            },
            "other_loans_consolidated": {},
        },
        "time_deposit_maturity": {},
        "borrowings_maturity": {},
        "horizon_midpoints": {
            "0_3m": 0.125, "3_12m": 0.625, "1_3y": 2.0,
            "3_5y": 4.0, "5_15y": 10.0, "15y_plus": 20.0,
        },
    }
    result = build_repricing_features(data, cfg)
    # WAM = (100*0.125 + 100*20) / 200 = 10.0625
    assert result.loc[0, "LOAN_WAM_PROXY"] == pytest.approx(10.0625, rel=0.01)
    assert result.loc[0, "LOAN_MATURITY_TOTAL"] == 200.0


def test_build_repricing_features_computes_repricing_gap_and_duration_gap() -> None:
    data = pd.DataFrame({
        "IDRSSD": [1],
        "RCONA570": [500.0],   # LOAN 0-3m
        "RCONA571": [200.0],   # LOAN 3-12m
        "RCONA572": [100.0],   # LOAN 1-3y
        "RCONA573": [50.0],    # LOAN 3-5y
        "RCONA574": [0.0],
        "RCONA575": [0.0],
        "RCONHK07": [300.0],   # TD small 0-3m
        "RCONHK08": [200.0],   # TD small 3-12m
        "RCONHK09": [50.0],    # TD small 1-3y
        "RCONHK10": [10.0],    # TD small 3y+
        "RCONHK12": [100.0],   # TD large 0-3m
        "RCONHK13": [50.0],    # TD large 3-12m
        "RCONHK14": [0.0],
        "RCONHK15": [0.0],
    })
    cfg = {
        "loan_maturity": {
            "other_loans": {
                "0_3m": "RCONA570", "3_12m": "RCONA571",
                "1_3y": "RCONA572", "3_5y": "RCONA573",
                "5_15y": "RCONA574", "15y_plus": "RCONA575",
            },
            "other_loans_consolidated": {},
            "re_first_lien": {},
        },
        "time_deposit_maturity": {
            "small": {"0_3m": "RCONHK07", "3_12m": "RCONHK08", "1_3y": "RCONHK09", "3y_plus": "RCONHK10"},
            "large": {"0_3m": "RCONHK12", "3_12m": "RCONHK13", "1_3y": "RCONHK14", "3y_plus": "RCONHK15"},
        },
        "borrowings_maturity": {},
        "horizon_midpoints": {
            "0_3m": 0.125, "3_12m": 0.625, "1_3y": 2.0,
            "3_5y": 4.0, "5_15y": 10.0, "15y_plus": 20.0, "3y_plus": 5.0,
        },
    }
    result = build_repricing_features(data, cfg)

    # Repricing gap 0-3m: asset (500) - liability (300+100) = 100
    assert result.loc[0, "REPRICING_GAP_0_3M"] == pytest.approx(100.0)
    # Cumulative gap should accumulate
    assert result.loc[0, "CUMULATIVE_GAP_0_3M"] == pytest.approx(100.0)
    # Duration gap = loan WAM - TD WAM
    assert pd.notna(result.loc[0, "DURATION_GAP_LITE"])
    assert result.loc[0, "DURATION_GAP_LITE"] > 0  # loans longer than TDs


def test_build_repricing_features_produces_5y_plus_gap() -> None:
    data = pd.DataFrame({
        "IDRSSD": [1],
        "RCONA570": [100.0],   # LOAN 0-3m
        "RCONA574": [200.0],   # LOAN 5-15y
        "RCONA575": [50.0],    # LOAN 15y+
    })
    cfg = {
        "loan_maturity": {
            "other_loans": {
                "0_3m": "RCONA570", "3_12m": "", "1_3y": "", "3_5y": "",
                "5_15y": "RCONA574", "15y_plus": "RCONA575",
            },
            "other_loans_consolidated": {},
            "re_first_lien": {},
        },
        "time_deposit_maturity": {},
        "borrowings_maturity": {},
        "horizon_midpoints": {
            "0_3m": 0.125, "3_12m": 0.625, "1_3y": 2.0,
            "3_5y": 4.0, "5_15y": 10.0, "15y_plus": 20.0, "3y_plus": 5.0,
        },
    }
    result = build_repricing_features(data, cfg)
    assert "REPRICING_GAP_5Y_PLUS" in result.columns
    assert "CUMULATIVE_GAP_5Y_PLUS" in result.columns
    # 5Y+ asset = 200+50=250, 5Y+ liability = 0 → gap = 250
    assert result.loc[0, "REPRICING_GAP_5Y_PLUS"] == pytest.approx(250.0)


def test_build_repricing_features_rcfd_fallback_per_row() -> None:
    """RCFD values should fill in where RCON is NaN."""
    data = pd.DataFrame({
        "IDRSSD": [1, 2],
        "RCONA570": [100.0, float("nan")],  # bank 1 has RCON, bank 2 doesn't
        "RCFDA570": [200.0, 500.0],           # RCFD available for both
    })
    cfg = {
        "loan_maturity": {
            "other_loans": {"0_3m": "RCONA570"},
            "other_loans_consolidated": {"0_3m": "RCFDA570"},
            "re_first_lien": {},
        },
        "time_deposit_maturity": {},
        "borrowings_maturity": {},
        "horizon_midpoints": {"0_3m": 0.125},
    }
    result = build_repricing_features(data, cfg)
    # Bank 1: RCON preferred (100), bank 2: RCFD fallback (500)
    assert result.loc[0, "LOAN_0_3M"] == 100.0
    assert result.loc[1, "LOAN_0_3M"] == 500.0


def test_build_repricing_features_handles_missing_columns() -> None:
    """Build should succeed even when no MDRM codes are present."""
    data = pd.DataFrame({"IDRSSD": [1]})
    cfg = {
        "loan_maturity": {"other_loans": {"0_3m": "RCONA570"}, "other_loans_consolidated": {}},
        "time_deposit_maturity": {},
        "borrowings_maturity": {},
        "horizon_midpoints": {"0_3m": 0.125},
    }
    result = build_repricing_features(data, cfg)
    assert "LOAN_0_3M" in result.columns
    assert pd.isna(result.loc[0, "LOAN_0_3M"])
