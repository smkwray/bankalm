from __future__ import annotations

from bankfragility.entity.sec_filings import (
    parse_securities_footnote,
    parse_uninsured_deposits,
)


def test_parse_uninsured_deposits_finds_basic_pattern() -> None:
    html = "<html><body>Estimated uninsured deposits totaled $21.2 billion at December 31, 2025.</body></html>"
    result = parse_uninsured_deposits(html)
    assert result["sec_uninsured_found"] is True
    assert result["sec_uninsured_amount"] == 21_200_000.0  # billions → thousands


def test_parse_uninsured_deposits_finds_were_pattern() -> None:
    html = "<html><body>Firmwide estimated uninsured deposits were $1,558.6 billion at the end of the period.</body></html>"
    result = parse_uninsured_deposits(html)
    assert result["sec_uninsured_found"] is True
    assert result["sec_uninsured_amount"] == 1_558_600_000.0


def test_parse_uninsured_deposits_returns_false_when_not_found() -> None:
    html = "<html><body>This filing has no deposit disclosures.</body></html>"
    result = parse_uninsured_deposits(html)
    assert result["sec_uninsured_found"] is False
    assert result["sec_uninsured_amount"] is None


def test_parse_securities_footnote_detects_afs_htm() -> None:
    html = "<html><body>Available-for-sale securities... Held-to-maturity investments...</body></html>"
    result = parse_securities_footnote(html)
    assert result["sec_afs_found"] is True
    assert result["sec_htm_found"] is True


def test_parse_securities_footnote_no_match() -> None:
    html = "<html><body>No securities discussed here.</body></html>"
    result = parse_securities_footnote(html)
    assert result["sec_afs_found"] is False
    assert result["sec_htm_found"] is False
