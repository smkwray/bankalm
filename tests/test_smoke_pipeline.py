"""Fixture-based end-to-end smoke test: synthetic data → indices → site exports."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _make_panel(n_banks: int = 4, n_quarters: int = 2) -> pd.DataFrame:
    """Create a minimal synthetic bank-quarter panel with required columns."""
    rng = np.random.default_rng(42)
    certs = [str(1000 + i) for i in range(n_banks)]
    quarters = pd.date_range("2023-06-30", periods=n_quarters, freq="QE")
    rows = []
    for cert in certs:
        for q in quarters:
            rows.append({
                "CERT": cert,
                "REPDTE": q.strftime("%Y-%m-%d"),
                "NAMEFULL": f"Test Bank {cert}",
                "PEER_GROUP": "community",
                "ASSET": rng.uniform(100_000, 5_000_000),
                "DEPDOM": rng.uniform(50_000, 3_000_000),
                "DEPUNA": rng.uniform(5_000, 500_000),
                "BROKERED_SHARE": rng.uniform(0, 0.3),
                "LIST_SERVICE_SHARE": rng.uniform(0, 0.1),
                "LARGE_ACCOUNT_SHARE": rng.uniform(0, 0.2),
                "TIME_DEPOSIT_SHARE": rng.uniform(0.1, 0.5),
                "NONINTEREST_SHARE": rng.uniform(0.1, 0.5),
                "CORE_DEPOSIT_SHARE": rng.uniform(0.4, 0.9),
                "SC": rng.uniform(10_000, 500_000),
                "SCUST": rng.uniform(5_000, 200_000),
                "SCAGE": rng.uniform(2_000, 100_000),
                "LNLSNET": rng.uniform(30_000, 2_000_000),
                "EINTEXP": rng.uniform(100, 10_000),
                "INTINC": rng.uniform(200, 20_000),
                "NITEFHLB": rng.uniform(0, 100_000),
                "STALP": "TX",
                "BKCLASS": "NM",
            })
    return pd.DataFrame(rows)


def test_smoke_stickiness_to_indices():
    """Stickiness → ALM → Treasury → Indices produces valid output."""
    from bankfragility.features.deposit_stickiness import build_deposit_stickiness_features
    from bankfragility.features.alm_mismatch import build_alm_mismatch_features
    from bankfragility.features.treasury_extensions import build_treasury_extensions
    from bankfragility.models.indices import build_indices_frame

    panel = _make_panel()
    cfg = {
        "transparent_run_risk": {
            "missing_rank_fill": 0.50,
            "components": {
                "uninsured_share": {"weight": 0.20, "orientation": "higher_is_worse"},
                "brokered_share": {"weight": 0.17, "orientation": "higher_is_worse"},
                "time_deposit_share": {"weight": 0.05, "orientation": "higher_is_worse"},
                "domestic_deposit_cost": {"weight": 0.08, "orientation": "higher_is_worse"},
                "dep_growth_vol_4q": {"weight": 0.05, "orientation": "higher_is_worse"},
                "dep_drawdown_4q": {"weight": 0.05, "orientation": "higher_is_worse"},
            },
        },
        "life_scenarios": {
            "baseline": {
                "max_downward_adjustment_fraction": 0.60,
                "category_lives_years": {
                    "ddt": {"base": 4.0, "floor": 1.0},
                    "trn_ex_ddt": {"base": 2.5, "floor": 0.50},
                    "mmda": {"base": 2.0, "floor": 0.25},
                    "other_savings": {"base": 3.0, "floor": 0.50},
                },
            },
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
    stickiness = build_deposit_stickiness_features(panel, cfg)
    assert "RUN_RISK_SCORE" in stickiness.columns
    assert len(stickiness) == len(panel)

    alm = build_alm_mismatch_features(stickiness)
    assert "VOLATILE_TO_LIQUID_LOWER" in alm.columns

    treasury = build_treasury_extensions(alm)
    assert "TREASURY_TO_UNINSURED_AFTER_100BP" in treasury.columns

    peer_cfg = [{"name": "community", "min_assets": 0, "max_assets": None}]
    weight_cfg = {
        "alm_mismatch_components": {
            "long_term_assets_to_stable_funding_baseline": 0.30,
            "volatile_to_liquid_lower": 0.30,
            "security_vs_deposit_gap_baseline": 0.20,
            "loans_to_core_deposits": 0.10,
            "short_fhlb_share": 0.10,
        },
        "treasury_buffer_components": {
            "treasury_to_uninsured_after_100bp": 0.35,
            "treasury_agency_to_runnable": 0.35,
            "hqla_narrow_lower_to_runnable": 0.30,
        },
        "composite_fragility_weights": {
            "run_risk_index": 0.40,
            "alm_mismatch_index": 0.40,
            "inverse_treasury_buffer_index": 0.20,
        },
    }
    indices = build_indices_frame(stickiness, alm, treasury, peer_cfg, weight_cfg)
    assert "RUN_RISK_INDEX" in indices.columns
    assert "FUNDING_FRAGILITY_INDEX" in indices.columns
    assert indices["RUN_RISK_INDEX"].between(0, 100).all()


def test_smoke_indices_to_site_exports(tmp_path):
    """Indices → site exports produces valid manifest and bank JSONs."""
    from bankfragility.reporting.site_exports import write_site_exports

    mart = pd.DataFrame({
        "CERT": ["1000", "1001"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-03-31"]),
        "NAMEFULL": ["Test Bank A", "Test Bank B"],
        "PEER_GROUP": ["community", "community"],
        "ASSET": [500_000.0, 800_000.0],
        "DEPDOM": [300_000.0, 450_000.0],
        "RUN_RISK_INDEX": [72.0, 38.0],
        "ALM_MISMATCH_INDEX": [60.0, 30.0],
        "TREASURY_BUFFER_INDEX": [20.0, 70.0],
        "FUNDING_FRAGILITY_INDEX": [68.0, 35.0],
    })
    failures = pd.DataFrame({
        "CERT": [1000.0],
        "FAILDATE": ["3/10/2023"],
        "NAME": ["Test Bank A"],
    })

    write_site_exports(mart, tmp_path, failures=failures)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["pipeline"]["unique_banks"] == 2

    bank_a = json.loads((tmp_path / "banks" / "1000.json").read_text())
    bank_b = json.loads((tmp_path / "banks" / "1001.json").read_text())
    assert bank_a["failed"] is True
    assert bank_a["fail_date"] == "2023-03-10"
    assert bank_b["failed"] is False
    assert bank_b["fail_date"] is None
