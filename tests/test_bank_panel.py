from __future__ import annotations

import pandas as pd
import pytest

from bankfragility.staging.bank_panel import (
    _normalize_institution_columns,
    aggregate_sod_features,
    build_bank_panel,
)


def test_sod_join_uses_prior_year_for_first_half() -> None:
    financials = pd.DataFrame(
        {
            "CERT": ["1001", "1001"],
            "REPDTE": ["2024-03-31", "2024-09-30"],
            "DEPDOM": [100.0, 120.0],
        }
    )
    sod = pd.DataFrame(
        {
            "CERT": ["1001", "1001"],
            "YEAR": [2023, 2024],
            "DEPSUMBR": [80.0, 95.0],
            "UNINUMBR": [1, 1],
            "STALP": ["NY", "NY"],
            "CNTYNAME": ["Albany", "Albany"],
        }
    )

    panel = build_bank_panel(financials=financials, sod=sod)

    assert panel["SOD_TOTAL_DEPOSITS"].tolist() == [80.0, 95.0]
    assert panel["SOD_JOIN_YEAR"].tolist() == [2023, 2024]


def test_build_bank_panel_parses_fdic_numeric_report_dates() -> None:
    financials = pd.DataFrame(
        {
            "CERT": ["1001"],
            "REPDTE": [19840331],
            "DEPDOM": [100.0],
        }
    )

    panel = build_bank_panel(financials=financials)

    assert panel.loc[0, "REPDTE"] == pd.Timestamp("1984-03-31")


def test_build_bank_panel_rejects_duplicate_cert_repdte() -> None:
    financials = pd.DataFrame(
        {
            "CERT": ["1001", "1001"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "DEPDOM": [100.0, 120.0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate CERT\\+REPDTE rows detected"):
        build_bank_panel(financials=financials)


def test_normalize_institution_columns_maps_api_names() -> None:
    inst = pd.DataFrame(
        {
            "CERT": [1],
            "FED_RSSD": [123456],
            "NAME": ["First Bank"],
            "RSSDHCR": [999],
        }
    )
    result = _normalize_institution_columns(inst)
    assert "RSSDID" in result.columns
    assert "NAMEFULL" in result.columns
    assert result.loc[0, "RSSDID"] == 123456
    assert result.loc[0, "NAMEFULL"] == "First Bank"


def test_normalize_institution_columns_preserves_canonical_names() -> None:
    inst = pd.DataFrame(
        {
            "CERT": [1],
            "RSSDID": [123456],
            "NAMEFULL": ["First Bank"],
        }
    )
    result = _normalize_institution_columns(inst)
    assert result.loc[0, "RSSDID"] == 123456
    assert result.loc[0, "NAMEFULL"] == "First Bank"


def test_join_institutions_fills_metadata_from_api_fields() -> None:
    financials = pd.DataFrame(
        {
            "CERT": ["100"],
            "REPDTE": ["2024-03-31"],
            "DEPDOM": [500.0],
        }
    )
    institutions = pd.DataFrame(
        {
            "CERT": [100],
            "FED_RSSD": [42],
            "NAME": ["Community Bank"],
            "RSSDHCR": [99],
            "BKCLASS": ["NM"],
            "STALP": ["CA"],
            "CITY": ["Oakland"],
        }
    )
    panel = build_bank_panel(financials=financials, institutions=institutions)
    assert panel.loc[0, "RSSDID"] == 42
    assert panel.loc[0, "NAMEFULL"] == "Community Bank"
    assert panel.loc[0, "RSSDHCR"] == 99


def test_aggregate_sod_features_uses_branch_level_columns() -> None:
    sod = pd.DataFrame(
        {
            "CERT": ["200", "200", "200"],
            "YEAR": [2023, 2023, 2023],
            "DEPSUMBR": [100.0, 200.0, 300.0],
            "UNINUMBR": [1, 2, 3],
            "STALPBR": ["NY", "NY", "NJ"],
            "CNTYNAMB": ["Kings", "Queens", "Bergen"],
            "BRNUM": [0, 1, 2],
        }
    )
    result = aggregate_sod_features(sod)
    assert len(result) == 1
    assert result.loc[0, "SOD_TOTAL_DEPOSITS"] == 600.0
    assert result.loc[0, "SOD_BRANCH_COUNT"] == 3
    assert result.loc[0, "SOD_STATE_COUNT"] == 2
    assert result.loc[0, "SOD_COUNTY_COUNT"] == 3
    assert result.loc[0, "SOD_TOP_STATE_SHARE"] == pytest.approx(0.5)


def test_full_three_way_join_produces_expected_columns() -> None:
    financials = pd.DataFrame(
        {
            "CERT": ["300"],
            "REPDTE": ["2024-09-30"],
            "DEPDOM": [1000.0],
        }
    )
    institutions = pd.DataFrame(
        {
            "CERT": [300],
            "FED_RSSD": [77],
            "NAME": ["Regional Bancshares"],
            "STALP": ["TX"],
        }
    )
    sod = pd.DataFrame(
        {
            "CERT": ["300"],
            "YEAR": [2024],
            "DEPSUMBR": [500.0],
            "UNINUMBR": [1],
            "STALPBR": ["TX"],
            "CNTYNAMB": ["Harris"],
        }
    )
    panel = build_bank_panel(financials=financials, institutions=institutions, sod=sod)
    assert panel.loc[0, "NAMEFULL"] == "Regional Bancshares"
    assert panel.loc[0, "RSSDID"] == 77
    assert panel.loc[0, "SOD_TOTAL_DEPOSITS"] == 500.0
    assert panel.loc[0, "SOD_JOIN_YEAR"] == 2024
    assert panel.loc[0, "HAS_SOD_FEATURES"] == 1


def test_aggregate_sod_features_handles_zero_deposit_branches() -> None:
    """Branches with zero deposits should be counted but not distort HHI."""
    sod = pd.DataFrame({
        "CERT": ["400", "400", "400"],
        "YEAR": [2023, 2023, 2023],
        "DEPSUMBR": [100.0, 0.0, 200.0],
        "UNINUMBR": [1, 2, 3],
        "STALP": ["NY", "NY", "NJ"],
        "CNTYNAME": ["Kings", "Kings", "Bergen"],
    })
    result = aggregate_sod_features(sod)
    assert result.loc[0, "SOD_TOTAL_DEPOSITS"] == 300.0
    assert result.loc[0, "SOD_BRANCH_COUNT"] == 3  # zero-dep branch still counted
    assert result.loc[0, "SOD_STATE_COUNT"] == 2


def test_join_ffiec_repricing_merges_on_cert_repdte() -> None:
    financials = pd.DataFrame({
        "CERT": ["100", "100"],
        "REPDTE": ["2023-06-30", "2023-09-30"],
        "DEPDOM": [500.0, 600.0],
    })
    ffiec = pd.DataFrame({
        "CERT": ["100"],
        "REPDTE": ["20230630"],
        "DURATION_GAP_LITE": [2.5],
        "LOAN_WAM_PROXY": [3.0],
    })
    panel = build_bank_panel(financials=financials, ffiec_repricing=ffiec)
    # Q2-2023 should have FFIEC data, Q3-2023 should be NaN
    q2 = panel[panel["REPDTE"] == pd.Timestamp("2023-06-30")].iloc[0]
    q3 = panel[panel["REPDTE"] == pd.Timestamp("2023-09-30")].iloc[0]
    assert q2["DURATION_GAP_LITE"] == 2.5
    assert pd.isna(q3["DURATION_GAP_LITE"])
    assert q2["HAS_FFIEC_FEATURES"] == 1
    assert q3["HAS_FFIEC_FEATURES"] == 0


def test_aggregate_sod_features_single_branch() -> None:
    """A bank with one branch should have HHI = 1.0 and state count = 1."""
    sod = pd.DataFrame({
        "CERT": ["500"],
        "YEAR": [2023],
        "DEPSUMBR": [1000.0],
        "UNINUMBR": [1],
        "STALP": ["TX"],
        "CNTYNAME": ["Harris"],
    })
    result = aggregate_sod_features(sod)
    assert result.loc[0, "SOD_DEPOSIT_HHI_STATE"] == 1.0
    assert result.loc[0, "SOD_DEPOSIT_HHI_COUNTY"] == 1.0
    assert result.loc[0, "SOD_STATE_COUNT"] == 1
    assert result.loc[0, "SOD_COUNTY_COUNT"] == 1
