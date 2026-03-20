from __future__ import annotations

import pandas as pd

from bankfragility.models.indices import build_indices_frame


def test_build_indices_frame_adds_deposit_competition_index_when_score_is_present() -> None:
    stickiness = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "ASSET": [5e8, 7e8],
            "RUN_RISK_SCORE": [60.0, 40.0],
        }
    )
    alm = pd.DataFrame({"CERT": ["1001", "1002"], "REPDTE": ["2024-03-31", "2024-03-31"]})
    treasury = pd.DataFrame({"CERT": ["1001", "1002"], "REPDTE": ["2024-03-31", "2024-03-31"]})
    deposit_competition = pd.DataFrame(
        {
            "CERT": ["1001", "1002"],
            "REPDTE": ["2024-03-31", "2024-03-31"],
            "DEPOSIT_COMPETITION_PRESSURE_SCORE": [80.0, 20.0],
        }
    )
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

    out = build_indices_frame(
        stickiness=stickiness,
        alm=alm,
        treasury=treasury,
        deposit_competition=deposit_competition,
        peer_cfg=peer_cfg,
        weight_cfg=weight_cfg,
    ).sort_values("CERT").reset_index(drop=True)

    assert out.loc[0, "DEPOSIT_COMPETITION_PRESSURE_INDEX"] > out.loc[1, "DEPOSIT_COMPETITION_PRESSURE_INDEX"]
    assert out.loc[0, "DEPOSIT_COMPETITION_RESILIENCE_INDEX"] < out.loc[1, "DEPOSIT_COMPETITION_RESILIENCE_INDEX"]
