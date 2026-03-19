from __future__ import annotations

import pandas as pd

from bankfragility.downloads.treasury_yields import _normalize_columns, _parse_xml_entries


def test_normalize_columns_renames_tenors() -> None:
    df = pd.DataFrame(
        {
            "Date": ["01/02/2023"],
            "1 Mo": [4.17],
            "10 Yr": [3.79],
            "30 Yr": [3.88],
        }
    )
    result = _normalize_columns(df)
    assert "DATE" in result.columns
    assert "YC_1MO" in result.columns
    assert "YC_10YR" in result.columns
    assert "YC_30YR" in result.columns


def test_parse_xml_entries_extracts_rates() -> None:
    xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
    <content type="application/xml">
    <m:properties xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
    <d:NEW_DATE xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices" m:type="Edm.DateTime">2023-01-03T00:00:00</d:NEW_DATE>
    <d:BC_1MONTH xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices" m:type="Edm.Double">4.17</d:BC_1MONTH>
    <d:BC_10YEAR xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices" m:type="Edm.Double">3.79</d:BC_10YEAR>
    </m:properties>
    </content>
    </entry>
    </feed>"""
    result = _parse_xml_entries(xml)
    assert len(result) == 1
    assert "DATE" in result.columns
    assert "YC_1MO" in result.columns
    assert "YC_10YR" in result.columns
    assert result.loc[0, "YC_1MO"] == 4.17
    assert result.loc[0, "YC_10YR"] == 3.79
