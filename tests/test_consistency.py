from __future__ import annotations

import pandas as pd

from bankfragility.validation.consistency import (
    check_coredep_exceeds_depdom,
    check_entity_jumps,
    check_fhlb_bucket_sum,
    check_negative_coverage_ratios,
    check_negative_deposits,
    check_pledged_exceeds_securities,
    check_repdte_monotonicity,
    check_securities_bucket_sum,
    check_shares_out_of_range,
    check_sod_hhi_range,
    check_sod_positive_deposits,
    validate_panel,
)


def _base(extras: dict | None = None) -> pd.DataFrame:
    row = {"CERT": "100", "REPDTE": pd.Timestamp("2024-03-31")}
    if extras:
        row.update(extras)
    return pd.DataFrame([row])


# ---- Impossible ratio checks ----


def test_coredep_exceeds_depdom_flags_violation() -> None:
    df = _base({"COREDEP": 200.0, "DEPDOM": 100.0})
    assert len(check_coredep_exceeds_depdom(df)) == 1


def test_coredep_within_depdom_clean() -> None:
    df = _base({"COREDEP": 80.0, "DEPDOM": 100.0})
    assert check_coredep_exceeds_depdom(df).empty


def test_pledged_exceeds_securities_flags() -> None:
    df = _base({"SCPLEDGE": 50.0, "SC": 30.0})
    assert len(check_pledged_exceeds_securities(df)) == 1


def test_pledged_within_securities_clean() -> None:
    df = _base({"SCPLEDGE": 20.0, "SC": 30.0})
    assert check_pledged_exceeds_securities(df).empty


def test_negative_deposits_flagged() -> None:
    df = _base({"DEPDOM": -5.0})
    result = check_negative_deposits(df)
    assert len(result) == 1
    assert result.loc[0, "CHECK"] == "NEGATIVE_DEPOSIT"


def test_shares_out_of_range_flags_negative_and_above_one() -> None:
    df = _base({"UNINSURED_SHARE": -0.1, "BROKERED_SHARE": 1.2})
    result = check_shares_out_of_range(df)
    assert len(result) == 2
    checks = set(result["CHECK"])
    assert "SHARE_NEGATIVE" in checks
    assert "SHARE_GT_1" in checks


def test_shares_in_range_clean() -> None:
    df = _base({"UNINSURED_SHARE": 0.5, "BROKERED_SHARE": 0.0, "CORE_DEPOSIT_SHARE": 1.0})
    assert check_shares_out_of_range(df).empty


def test_negative_coverage_ratio_flags() -> None:
    df = _base({"TREASURY_TO_UNINSURED": -0.3})
    result = check_negative_coverage_ratios(df)
    assert len(result) == 1


# ---- Balance / bucket identity checks ----


def test_securities_bucket_sum_flags_when_exceeds_total() -> None:
    df = _base({
        "SC": 100.0,
        "SCNM3LES": 50.0, "SCNM3T12": 50.0, "SCNM1T3": 50.0,
        "SCNM3T5": 0.0, "SCNM5T15": 0.0, "SCNMOV15": 0.0,
    })
    # Sum = 150, SC = 100, ratio = 1.5 → flags (exceeds total)
    result = check_securities_bucket_sum(df)
    assert len(result) == 1


def test_securities_bucket_sum_clean_when_subset() -> None:
    df = _base({
        "SC": 100.0,
        "SCNM3LES": 5.0, "SCNM3T12": 5.0, "SCNM1T3": 5.0,
        "SCNM3T5": 5.0, "SCNM5T15": 5.0, "SCNMOV15": 5.0,
    })
    # Sum = 30, SC = 100, ratio = 0.3 → clean (SCNM = non-mortgage subset)
    assert check_securities_bucket_sum(df).empty


def test_fhlb_bucket_sum_flags_deviation() -> None:
    df = _base({
        "OTHBFHLB": 1000.0,
        "OTBFH1L": 100.0, "OTBFH1T3": 100.0, "OTBFH3T5": 100.0, "OTBFHOV5": 100.0,
    })
    # Sum = 400, total = 1000, ratio = 0.4 → flags
    result = check_fhlb_bucket_sum(df)
    assert len(result) == 1


# ---- Temporal and entity stability ----


def test_repdte_monotonicity_flags_duplicates() -> None:
    df = pd.DataFrame({
        "CERT": ["100", "100", "100"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31"]),
    })
    result = check_repdte_monotonicity(df)
    assert len(result) == 1


def test_repdte_monotonicity_clean_for_ordered_dates() -> None:
    df = pd.DataFrame({
        "CERT": ["100", "100"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30"]),
    })
    assert check_repdte_monotonicity(df).empty


def test_entity_jumps_flags_rssdhcr_change() -> None:
    df = pd.DataFrame({
        "CERT": ["100", "100"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "RSSDHCR": [999.0, 888.0],
    })
    result = check_entity_jumps(df)
    assert len(result) == 1
    assert result.iloc[0]["CHECK"] == "RSSDHCR_CHANGE"


def test_entity_jumps_clean_for_stable_hcr() -> None:
    df = pd.DataFrame({
        "CERT": ["100", "100"],
        "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "RSSDHCR": [999.0, 999.0],
    })
    assert check_entity_jumps(df).empty


# ---- SOD checks ----


def test_sod_nonpositive_deposits_flagged() -> None:
    df = _base({"SOD_TOTAL_DEPOSITS": 0.0})
    assert len(check_sod_positive_deposits(df)) == 1


def test_sod_positive_deposits_clean() -> None:
    df = _base({"SOD_TOTAL_DEPOSITS": 1000.0})
    assert check_sod_positive_deposits(df).empty


def test_sod_hhi_out_of_range_flags() -> None:
    df = _base({"SOD_DEPOSIT_HHI_STATE": 1.5})
    result = check_sod_hhi_range(df)
    assert len(result) == 1


def test_sod_hhi_in_range_clean() -> None:
    df = _base({"SOD_DEPOSIT_HHI_STATE": 0.25, "SOD_DEPOSIT_HHI_COUNTY": 0.10})
    assert check_sod_hhi_range(df).empty


# ---- Combined runner ----


def test_validate_panel_returns_empty_for_clean_data() -> None:
    df = _base({
        "COREDEP": 80.0, "DEPDOM": 100.0, "SC": 50.0, "SCPLEDGE": 10.0,
        "SOD_TOTAL_DEPOSITS": 500.0, "SOD_DEPOSIT_HHI_STATE": 0.3,
    })
    report = validate_panel(df)
    assert report.empty


def test_validate_panel_catches_multiple_violations() -> None:
    df = _base({
        "COREDEP": 200.0, "DEPDOM": 100.0,  # violation
        "SCPLEDGE": 50.0, "SC": 30.0,       # violation
        "UNINSURED_SHARE": -0.1,             # violation
    })
    report = validate_panel(df)
    checks = set(report["CHECK"])
    assert "COREDEP_GT_DEPDOM" in checks
    assert "SCPLEDGE_GT_SC" in checks
    assert "SHARE_NEGATIVE" in checks


def test_validate_panel_on_real_smoke_data() -> None:
    """Run validation on the actual smoke output — should be clean or close to it."""
    import os
    path = os.path.join(
        os.path.dirname(__file__), "..",
        "data", "processed", "smoke_bank_indices.parquet",
    )
    if not os.path.exists(path):
        return  # skip if smoke data not present
    df = pd.read_parquet(path)
    report = validate_panel(df)
    # Real data may have some flags but should have no impossible-ratio violations
    impossible = report[report["CHECK"].isin(["COREDEP_GT_DEPDOM", "SCPLEDGE_GT_SC", "NEGATIVE_DEPOSIT"])]
    assert impossible.empty, f"Found impossible violations:\n{impossible}"
