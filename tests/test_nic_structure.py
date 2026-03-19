from __future__ import annotations

import pandas as pd

from bankfragility.entity.nic_structure import build_top_holder_map


def test_build_top_holder_map_walks_chain() -> None:
    """A → B → C should give A's top holder as C."""
    rels = pd.DataFrame({
        "ID_RSSD_PARENT": [200, 300],
        "ID_RSSD_OFFSPRING": [100, 200],
    })
    result = build_top_holder_map(rels)
    row_a = result[result["ID_RSSD"] == 100].iloc[0]
    assert row_a["TOP_HOLDER_RSSD"] == 300


def test_build_top_holder_map_handles_top_entity() -> None:
    """An entity with no parent should be its own top holder."""
    rels = pd.DataFrame({
        "ID_RSSD_PARENT": [200],
        "ID_RSSD_OFFSPRING": [100],
    })
    result = build_top_holder_map(rels)
    row_top = result[result["ID_RSSD"] == 200].iloc[0]
    assert row_top["TOP_HOLDER_RSSD"] == 200
