from __future__ import annotations

import pandas as pd

from bankfragility.downloads.fred_series import merge_series_frames, observations_to_frame


def test_observations_to_frame_converts_dates_and_values() -> None:
    observations = [
        {"date": "2024-03-28", "value": "5.40"},
        {"date": "2024-03-29", "value": "."},
    ]
    out = observations_to_frame("IORB", observations)

    assert list(out.columns) == ["DATE", "IORB"]
    assert out.loc[0, "DATE"] == pd.Timestamp("2024-03-28")
    assert out.loc[0, "IORB"] == 5.40
    assert pd.isna(out.loc[1, "IORB"])


def test_merge_series_frames_builds_wide_table() -> None:
    a = pd.DataFrame({"DATE": pd.to_datetime(["2024-03-28"]), "IORB": [5.40]})
    b = pd.DataFrame({"DATE": pd.to_datetime(["2024-03-28"]), "RRPONTSYAWARD": [5.30]})

    out = merge_series_frames([a, b])

    assert list(out.columns) == ["DATE", "IORB", "RRPONTSYAWARD"]
    assert out.loc[0, "IORB"] == 5.40
    assert out.loc[0, "RRPONTSYAWARD"] == 5.30
