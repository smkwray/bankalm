from __future__ import annotations

import pandas as pd

from bankfragility.features.alm_mismatch import build_alm_mismatch_features


def test_uninsured_fallback_drives_runnable_proxy_when_depuna_missing() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "DEPUNA": [None],
            "IDDEPLAM": [200.0],
            "BRO": [10.0],
            "DEPLSNB": [5.0],
            "VOLIAB": [80.0],
        }
    )

    out = build_alm_mismatch_features(df, small_bank_uninsured_fallback_factor=0.50)

    # 0.5*200 + 10 + 5 = 115, which should exceed VOLIAB=80.
    assert out.loc[0, "RUNNABLE_FUNDING_PROXY"] == 115.0


def test_long_term_assets_to_stable_funding_baseline_uses_expected_components() -> None:
    df = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "EQ": [20.0],
            "DEPOSIT_STABLE_EQUIV_BASELINE": [60.0],
            "OTBFH1T3": [10.0],
            "OTBFH3T5": [5.0],
            "OTBFHOV5": [5.0],
            "ASSTLT": [180.0],
        }
    )

    out = build_alm_mismatch_features(df)

    # Stable funding = 20 + 60 + 10 + 5 + 5 = 100, ratio = 180/100 = 1.8
    assert out.loc[0, "STABLE_FUNDING_BASELINE"] == 100.0
    assert out.loc[0, "LONG_TERM_ASSETS_TO_STABLE_FUNDING_BASELINE"] == 1.8


def test_depuna_zero_treated_as_unknown_in_alm() -> None:
    """DEPUNA=0 should trigger the fallback proxy, not be treated as zero uninsured."""
    df = pd.DataFrame({
        "CERT": ["1001"],
        "REPDTE": ["2024-03-31"],
        "DEPUNA": [0.0],       # reported zero = unknown for small banks
        "IDDEPLAM": [200.0],
        "BRO": [10.0],
        "DEPLSNB": [5.0],
        "VOLIAB": [80.0],
    })
    out = build_alm_mismatch_features(df, small_bank_uninsured_fallback_factor=0.50)
    # DEPUNA=0 → treated as NaN → fallback to 0.5*IDDEPLAM = 100
    # Runnable = max(VOLIAB=80, 100+10+5=115) = 115
    assert out.loc[0, "RUNNABLE_FUNDING_PROXY"] == 115.0
