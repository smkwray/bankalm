from __future__ import annotations

import json

import pandas as pd

from bankfragility.reporting.site_exports import (
    build_publishable_mart,
    split_publishable_panels,
    write_site_exports,
)


def test_build_publishable_mart_merges_supervised_overlay() -> None:
    indices = pd.DataFrame({
        "CERT": ["100"],
        "REPDTE": ["2024-03-31"],
        "RUN_RISK_INDEX": [80.0],
        "FUNDING_FRAGILITY_INDEX": [75.0],
    })
    supervised = pd.DataFrame({
        "CERT": ["100"],
        "REPDTE": ["2024-03-31"],
        "SUPERVISED_OUTFLOW_SCORE": [0.42],
        "NEXT_Q_OBSERVED": [True],
    })

    mart = build_publishable_mart(indices, supervised)
    assert mart.loc[0, "SUPERVISED_OUTFLOW_SCORE"] == 0.42
    assert mart.loc[0, "SUPERVISED_EXPERIMENTAL"] == 1


def test_write_site_exports_generates_manifest_and_league_json(tmp_path) -> None:
    mart = pd.DataFrame({
        "CERT": ["100", "200"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-03-31"]),
        "NAMEFULL": ["Bank A", "Bank B"],
        "PEER_GROUP": ["community", "regional"],
        "ASSET": [500.0, 800.0],
        "DEPDOM": [300.0, 450.0],
        "UNINSURED_SHARE": [0.7, 0.2],
        "STALP": ["NY", "CA"],
        "BKCLASS": ["NM", "SM"],
        "RUN_RISK_SCORE": [72.0, 38.0],
        "STICKINESS_SCORE": [28.0, 62.0],
        "RANK_UNINSURED_SHARE": [0.9, 0.2],
        "RANK_BROKERED_SHARE": [0.8, 0.4],
        "RUN_RISK_INDEX": [80.0, 40.0],
        "ALM_MISMATCH_INDEX": [60.0, 30.0],
        "TREASURY_BUFFER_INDEX": [20.0, 70.0],
        "FUNDING_FRAGILITY_INDEX": [75.0, 35.0],
        "ALM_MISMATCH_INDEX_CONTRIB_LONG_TERM_ASSETS_TO_STABLE_FUNDING_BASELINE": [20.0, 8.0],
        "ALM_MISMATCH_INDEX_CONTRIB_VOLATILE_TO_LIQUID_LOWER": [18.0, 9.0],
        "TREASURY_BUFFER_INDEX_CONTRIB_TREASURY_TO_UNINSURED_AFTER_100BP": [8.0, 24.0],
        "TREASURY_YIELD_DATE": pd.to_datetime(["2024-03-29", "2024-03-29"]),
        "HAS_TREASURY_YIELD_HISTORY": [1, 1],
        "YC_2YR": [4.6, 4.6],
        "YC_10YR": [4.2, 4.2],
        "YC_10Y_3M_SLOPE_BP": [-80.0, -80.0],
        "YC_10Y_2Y_SLOPE_BP": [-40.0, -40.0],
        "YC_10YR_QOQ_CHANGE_BP": [25.0, 25.0],
        "HAS_FFIEC_FEATURES": [1, 0],
        "HAS_SOD_FEATURES": [1, 1],
        "INDEX_VERSION": ["v1_ffiec_hybrid", "v0_fdic_only"],
    })
    metrics = pd.DataFrame({
        "SLICE": ["full_sample", "full_sample", "full_sample"],
        "HORIZON_QUARTERS": [4, 4, 4],
        "SCORE_COL": ["RUN_RISK_INDEX", "ALM_MISMATCH_INDEX", "FUNDING_FRAGILITY_INDEX"],
        "AUC": [0.75, 0.55, 0.70],
        "N_FAILURES": [12, 12, 12],
        "N_TOTAL": [100, 100, 100],
        "RECALL_AT_20PCT": [0.50, 0.20, 0.40],
    })

    write_site_exports(mart, tmp_path, validation_metrics=metrics)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    league = json.loads((tmp_path / "league.json").read_text(encoding="utf-8"))

    assert manifest["pipeline"]["bank_quarters"] == 2
    assert manifest["pipeline"]["failures_tested"] == 12
    assert manifest["indices"][0]["headline_horizon_quarters"] == 4
    assert manifest["site_panel"] == "recent_history_enriched"
    assert manifest["published_panels"]["full_history_core"]["bank_quarters"] == 2
    assert manifest["published_panels"]["recent_history_enriched"]["bank_quarters"] == 2
    assert manifest["treasury_regime"]["yield_date"] == "2024-03-29"
    assert manifest["treasury_regime"]["y10"] == 4.2
    assert "run_risk" in manifest["index_methodology"]
    assert manifest["index_methodology"]["funding_fragility"]["components"][0]["label"] == "Run Risk Index"
    assert league[0]["name"] == "Bank A"
    assert league[0]["funding_fragility"] == 75.0
    assert league[0]["peer_group_bank_count"] == 1
    assert league[0]["treasury_yield_date"] == "2024-03-29"
    assert league[0]["yc_10yr"] == 4.2
    assert league[0]["run_risk_components"][0]["label"] == "Uninsured deposits"
    assert league[0]["alm_components"][0]["label"] == "Long-term assets / stable funding"
    assert league[0]["composite_components"][0]["label"] == "Run Risk Index"


def test_split_publishable_panels_creates_full_history_core_and_recent_enriched() -> None:
    mart = pd.DataFrame({
        "CERT": ["100", "100", "200"],
        "REPDTE": pd.to_datetime(["2019-12-31", "2020-03-31", "2020-03-31"]),
        "NAMEFULL": ["Bank A", "Bank A", "Bank B"],
        "RUN_RISK_INDEX": [50.0, 55.0, 60.0],
        "FUNDING_FRAGILITY_INDEX": [45.0, 50.0, 65.0],
        "HAS_FFIEC_FEATURES": [0, 1, 1],
        "HAS_DERIVATIVE_FEATURES": [0, 0, 1],
        "HAS_SOD_FEATURES": [1, 1, 1],
        "INDEX_VERSION": ["v0_fdic_only", "v1_ffiec_hybrid", "v1_ffiec_hybrid"],
        "SUPERVISED_OUTFLOW_SCORE": [None, 0.2, 0.3],
        "DURATION_GAP_LITE": [None, 1.2, 1.5],
    })

    core, enriched = split_publishable_panels(mart)

    assert len(core) == 3
    assert len(enriched) == 2
    assert (core["PANEL_VARIANT"] == "full_history_core").all()
    assert (enriched["PANEL_VARIANT"] == "recent_history_enriched").all()
    assert "SUPERVISED_OUTFLOW_SCORE" not in core.columns
    assert "DURATION_GAP_LITE" in enriched.columns
