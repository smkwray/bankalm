from __future__ import annotations

import pandas as pd

import pandas as pd

from bankfragility.entity.sec_mapping import (
    _normalize_name,
    apply_overrides,
    build_sec_lookup,
    match_institutions_to_sec,
)


def test_normalize_name_strips_suffixes_and_punctuation() -> None:
    assert _normalize_name("JPMorgan Chase & Co.") == "jpmorgan chase"
    assert _normalize_name("Bank of America, National Association") == "bank of america"
    assert _normalize_name("BOKF, National Association") == "bokf"
    assert _normalize_name("Frost  Bank") == "frost bank"


def test_build_sec_lookup_creates_normalized_names() -> None:
    tickers = [
        {"cik_str": 123, "ticker": "ABC", "title": "ABC Holdings Inc."},
    ]
    df = build_sec_lookup(tickers)
    assert "SEC_NAME_NORM" in df.columns
    assert df.loc[0, "SEC_NAME_NORM"] == "abc"


def test_match_institutions_exact_match() -> None:
    inst = pd.DataFrame({"CERT": ["1"], "NAME": ["Acme Bancorp Inc."]})
    sec = build_sec_lookup([{"cik_str": 999, "ticker": "ACM", "title": "Acme Bancorp Inc."}])
    result = match_institutions_to_sec(inst, sec)
    assert result.loc[0, "SEC_CIK"] == 999
    assert result.loc[0, "SEC_MATCH_TYPE"] == "exact"


def test_match_institutions_no_false_positive_on_common_words() -> None:
    """Banks with only common-word names should NOT fuzzy-match random SEC entities."""
    inst = pd.DataFrame({"CERT": ["1"], "NAME": ["Bank of Holly Springs"]})
    sec = build_sec_lookup([
        {"cik_str": 70858, "ticker": "BAC", "title": "Bank of America Corp /DE/"},
    ])
    result = match_institutions_to_sec(inst, sec)
    # Should NOT match Bank of America
    assert pd.isna(result.loc[0, "SEC_CIK"])


def test_match_uses_nic_top_holder_name_when_available() -> None:
    """When NIC structure provides a top-holder name, SEC matching should use it."""
    inst = pd.DataFrame({
        "CERT": ["1"],
        "FED_RSSD": [100],
        "NAME": ["Subsidiary Bank NA"],  # bank-level name won't match
    })
    sec = build_sec_lookup([
        {"cik_str": 42, "ticker": "BFC", "title": "BOK Financial Corporation"},
    ])
    nic = pd.DataFrame({
        "ID_RSSD": [100],
        "TOP_HOLDER_NAME": ["BOK FINANCIAL CORPORATION"],  # HC name will match
    })
    result = match_institutions_to_sec(inst, sec, nic_structure=nic)
    assert result.loc[0, "SEC_CIK"] == 42
    assert result.loc[0, "SEC_MATCH_TYPE"] == "exact"


def test_apply_overrides_replaces_values() -> None:
    mapping = pd.DataFrame({
        "CERT": ["100", "200"],
        "SEC_CIK": [None, None],
        "SEC_TICKER": [None, None],
        "SEC_MATCH_TYPE": [None, None],
    })
    overrides = {"100": {"cik": 42, "ticker": "XYZ"}}
    result = apply_overrides(mapping, overrides)
    assert result.loc[0, "SEC_CIK"] == 42
    assert result.loc[0, "SEC_TICKER"] == "XYZ"
    assert result.loc[0, "SEC_MATCH_TYPE"] == "override"
    assert pd.isna(result.loc[1, "SEC_CIK"])
