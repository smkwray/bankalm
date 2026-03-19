from __future__ import annotations

import pandas as pd

from bankfragility.models.indices import assign_peer_group, build_indices_frame


def test_assign_peer_group_uses_asset_buckets() -> None:
    assets = pd.Series([5e8, 5e9, 5e10, 5e11])
    peer_cfg = [
        {"name": "community", "min_assets": 0, "max_assets": 1e9},
        {"name": "regional", "min_assets": 1e9, "max_assets": 1e10},
        {"name": "large_regional", "min_assets": 1e10, "max_assets": 1e11},
        {"name": "very_large", "min_assets": 1e11, "max_assets": None},
    ]

    out = assign_peer_group(assets, peer_cfg)

    assert out.tolist() == ["community", "regional", "large_regional", "very_large"]


def test_build_indices_frame_ranks_worse_bank_higher_fragility() -> None:
    stickiness = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "ASSET": [5e8, 7e8],
            "RUN_RISK_SCORE": [80.0, 20.0],
        }
    )
    alm = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "LONG_TERM_ASSETS_TO_STABLE_FUNDING_BASELINE": [2.0, 0.5],
            "VOLATILE_TO_LIQUID_LOWER": [3.0, 0.5],
            "SECURITY_VS_DEPOSIT_GAP_BASELINE": [5.0, 1.0],
            "LOANS_TO_CORE_DEPOSITS": [1.3, 0.8],
            "SHORT_FHLB_SHARE": [0.8, 0.1],
        }
    )
    treasury = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "TREASURY_TO_UNINSURED_AFTER_100BP": [0.2, 0.9],
            "TREASURY_AGENCY_TO_RUNNABLE": [0.3, 1.0],
            "HQLA_NARROW_LOWER_TO_RUNNABLE": [0.4, 1.2],
        }
    )
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

    out = build_indices_frame(stickiness, alm, treasury, peer_cfg, weight_cfg).sort_values("CERT").reset_index(drop=True)

    assert out.loc[0, "FUNDING_FRAGILITY_INDEX"] > out.loc[1, "FUNDING_FRAGILITY_INDEX"]
    assert out.loc[0, "PEER_GROUP"] == "community"


def test_build_indices_frame_ignores_overlapping_carried_columns() -> None:
    stickiness = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "ASSET": [5e8],
            "RUN_RISK_SCORE": [60.0],
        }
    )
    alm = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "ASSET": [5e8],
            "RUN_RISK_SCORE": [60.0],
            "VOLATILE_TO_LIQUID_LOWER": [1.5],
        }
    )
    treasury = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "ASSET": [5e8],
            "RUN_RISK_SCORE": [60.0],
            "VOLATILE_TO_LIQUID_LOWER": [1.5],
            "TREASURY_AGENCY_TO_RUNNABLE": [0.8],
        }
    )
    peer_cfg = [{"name": "community", "min_assets": 0, "max_assets": None}]
    weight_cfg = {
        "alm_mismatch_components": {"volatile_to_liquid_lower": 1.0},
        "treasury_buffer_components": {"treasury_agency_to_runnable": 1.0},
        "composite_fragility_weights": {
            "run_risk_index": 0.4,
            "alm_mismatch_index": 0.4,
            "inverse_treasury_buffer_index": 0.2,
        },
    }

    out = build_indices_frame(stickiness, alm, treasury, peer_cfg, weight_cfg)

    assert len(out) == 1
    assert "TREASURY_BUFFER_INDEX" in out.columns


def test_build_indices_frame_tags_version_per_row() -> None:
    stickiness = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-06-30"],
            "ASSET": [5e8, 5e8],
            "RUN_RISK_SCORE": [60.0, 55.0],
            "HAS_FFIEC_FEATURES": [0, 1],
            "HAS_SOD_FEATURES": [1, 1],
        }
    )
    alm = pd.DataFrame({"CERT": ["1001", "1002"], "REPDTE": ["2024-03-31", "2024-06-30"]})
    treasury = pd.DataFrame({"CERT": ["1001", "1002"], "REPDTE": ["2024-03-31", "2024-06-30"]})
    peer_cfg = [{"name": "community", "min_assets": 0, "max_assets": None}]
    weight_cfg = {
        "alm_mismatch_components": {},
        "treasury_buffer_components": {},
        "composite_fragility_weights": {
            "run_risk_index": 0.5,
            "alm_mismatch_index": 0.3,
            "inverse_treasury_buffer_index": 0.2,
        },
    }

    out = build_indices_frame(stickiness, alm, treasury, peer_cfg, weight_cfg, version="v0_fdic_only")
    assert out.loc[out["CERT"] == "1001", "INDEX_VERSION"].iloc[0] == "v0_fdic_only"
    assert out.loc[out["CERT"] == "1002", "INDEX_VERSION"].iloc[0] == "v1_ffiec_hybrid"

    out2 = build_indices_frame(stickiness, alm, treasury, peer_cfg, weight_cfg, version="v1_custom")
    assert (out2["INDEX_VERSION"] == "v1_custom").all()


def test_build_indices_frame_adds_coverage_flags_when_missing() -> None:
    stickiness = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": ["2024-03-31"],
            "ASSET": [5e8],
            "RUN_RISK_SCORE": [60.0],
            "SOD_TOTAL_DEPOSITS": [100.0],
        }
    )
    alm = pd.DataFrame({"CERT": ["1001"], "REPDTE": ["2024-03-31"], "DURATION_GAP_LITE": [2.0]})
    treasury = pd.DataFrame({"CERT": ["1001"], "REPDTE": ["2024-03-31"]})
    peer_cfg = [{"name": "community", "min_assets": 0, "max_assets": None}]
    weight_cfg = {
        "alm_mismatch_components": {},
        "treasury_buffer_components": {},
        "composite_fragility_weights": {
            "run_risk_index": 0.5,
            "alm_mismatch_index": 0.3,
            "inverse_treasury_buffer_index": 0.2,
        },
    }

    out = build_indices_frame(stickiness, alm, treasury, peer_cfg, weight_cfg)
    assert out.loc[0, "HAS_SOD_FEATURES"] == 1
    assert out.loc[0, "HAS_FFIEC_FEATURES"] == 1
    assert out.loc[0, "HAS_HISTORICAL_ENTITY_MAP"] == 0
