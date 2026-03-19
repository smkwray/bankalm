"""Validation tests for 2023 regional bank stress episode.

These tests verify that the pipeline correctly identifies known risk
factors in Silicon Valley Bank (CERT 24735) before its March 2023 failure.
Tests skip gracefully if the validation data hasn't been generated.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

VALIDATION_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "processed", "validation_bank_indices.parquet",
)


@pytest.fixture
def validation_data() -> pd.DataFrame:
    if not os.path.exists(VALIDATION_PATH):
        pytest.skip("Validation data not present — run the validation pipeline first")
    return pd.read_parquet(VALIDATION_PATH)


@pytest.fixture
def svb_q4_2022(validation_data: pd.DataFrame) -> pd.Series:
    q4 = validation_data[validation_data["REPDTE"] == pd.Timestamp("2022-12-31")]
    svb = q4[q4["CERT"].astype(str) == "24735"]
    if svb.empty:
        pytest.skip("SVB (CERT 24735) not in validation data")
    return svb.iloc[0]


def test_svb_has_highest_uninsured_share(validation_data: pd.DataFrame, svb_q4_2022: pd.Series) -> None:
    """SVB should rank #1 on uninsured deposit share in Q4-2022."""
    q4 = validation_data[validation_data["REPDTE"] == pd.Timestamp("2022-12-31")]
    assert svb_q4_2022["UNINSURED_SHARE"] == q4["UNINSURED_SHARE"].max()


def test_svb_uninsured_share_exceeds_90_percent(svb_q4_2022: pd.Series) -> None:
    """SVB's uninsured share should be extreme (>90%)."""
    assert svb_q4_2022["UNINSURED_SHARE"] > 0.90


def test_svb_treasury_coverage_below_15_percent(svb_q4_2022: pd.Series) -> None:
    """SVB's Treasury coverage of uninsured should be critically low."""
    assert svb_q4_2022["TREASURY_TO_UNINSURED"] < 0.15


def test_svb_volatile_to_liquid_worsened_through_2022(validation_data: pd.DataFrame) -> None:
    """SVB's volatile/liquid ratio should deteriorate between Q1 and Q4 2022."""
    svb = validation_data[validation_data["CERT"].astype(str) == "24735"].sort_values("REPDTE")
    if len(svb) < 2:
        pytest.skip("Need multiple SVB quarters")
    first_vtl = svb.iloc[0]["VOLATILE_TO_LIQUID_LOWER"]
    last_vtl = svb.iloc[-1]["VOLATILE_TO_LIQUID_LOWER"]
    assert last_vtl > first_vtl, f"Expected deterioration: {first_vtl} → {last_vtl}"
