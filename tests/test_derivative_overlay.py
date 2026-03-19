from __future__ import annotations

import numpy as np
import pandas as pd

from bankfragility.features.derivative_overlay import build_derivative_features


def test_build_derivative_features_flags_banks_with_swaps() -> None:
    data = pd.DataFrame({
        "IDRSSD": [1, 2],
        "RCFD3450": [1000.0, 0.0],  # IR swaps: bank 1 has, bank 2 doesn't
    })
    result = build_derivative_features(data)
    assert result.loc[0, "HAS_IR_DERIVATIVES"] == 1
    assert result.loc[1, "HAS_IR_DERIVATIVES"] == 0
    assert result.loc[0, "IR_DERIV_TOTAL_NOTIONAL"] > 0


def test_build_derivative_features_handles_no_derivative_columns() -> None:
    data = pd.DataFrame({"IDRSSD": [1]})
    result = build_derivative_features(data)
    assert "HAS_IR_DERIVATIVES" in result.columns
