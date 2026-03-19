"""Build coarse public-data asset-liability mismatch features.

**Pre-hedge proxy disclaimer:**
This module uses FDIC public financial data only.  It does not incorporate
derivative overlay information (interest-rate swaps, caps, floors) that
banks use to hedge their balance-sheet repricing exposure.  As a result,
the mismatch metrics produced here represent a *pre-hedge structural proxy*,
not a post-hedge effective position.

When FFIEC Schedule RC-L derivative data is parsed and integrated, these
metrics can be refined into a post-hedge view.  Until then, interpret the
outputs as "how exposed does the raw balance sheet look?" rather than
"how exposed is the bank after hedging?"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bankfragility.tables import read_table, save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--small-bank-uninsured-fallback-factor",
        type=float,
        default=0.50,
        help="Fallback multiplier on IDDEPLAM when DEPUNA is missing",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def safe_div(num_s: pd.Series, den_s: pd.Series) -> pd.Series:
    out = np.where((den_s.notna()) & (den_s != 0), num_s / den_s, np.nan)
    return pd.Series(out, index=num_s.index, dtype="float64")


def rowwise_max(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(np.maximum(a.fillna(-np.inf), b.fillna(-np.inf)), index=a.index)


def build_alm_mismatch_features(
    df: pd.DataFrame,
    small_bank_uninsured_fallback_factor: float = 0.50,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).upper() for col in out.columns]

    depuna = num(out, "DEPUNA")
    iddeplam = num(out, "IDDEPLAM")
    # Treat DEPUNA=0 as unknown (same as NaN) — small banks often report 0 instead of actual uninsured
    depuna_clean = depuna.where(depuna > 0)
    uninsured_proxy = depuna_clean.where(depuna_clean.notna(), small_bank_uninsured_fallback_factor * iddeplam)

    bro = num(out, "BRO").fillna(0)
    list_service = num(out, "DEPLSNB").fillna(0)
    voliab = num(out, "VOLIAB")
    out["RUNNABLE_FUNDING_PROXY"] = rowwise_max(voliab, uninsured_proxy.fillna(0) + bro + list_service)

    chbal = num(out, "CHBAL").fillna(0)
    frepo = num(out, "FREPO").fillna(0)
    scust = num(out, "SCUST").fillna(0)
    scage = num(out, "SCAGE").fillna(0)
    scaf = num(out, "SCAF").fillna(0)
    spledge = num(out, "SCPLEDGE").fillna(0)

    out["LIQUID_ASSETS_NARROW"] = chbal + frepo + scust + 0.85 * scage
    out["LIQUID_ASSETS_BROAD"] = out["LIQUID_ASSETS_NARROW"] + 0.50 * (scaf - scust - scage).clip(lower=0)

    out["UNPLEDGED_LIQUID_NARROW_LOWER"] = (out["LIQUID_ASSETS_NARROW"] - spledge).clip(lower=0)
    out["UNPLEDGED_LIQUID_BROAD_LOWER"] = (out["LIQUID_ASSETS_BROAD"] - spledge).clip(lower=0)

    out["VOLATILE_TO_LIQUID_LOWER"] = safe_div(out["RUNNABLE_FUNDING_PROXY"], out["UNPLEDGED_LIQUID_NARROW_LOWER"])
    out["VOLATILE_TO_LIQUID_BROAD_LOWER"] = safe_div(out["RUNNABLE_FUNDING_PROXY"], out["UNPLEDGED_LIQUID_BROAD_LOWER"])

    eq = num(out, "EQ").fillna(0)
    dep_stable_base = num(out, "DEPOSIT_STABLE_EQUIV_BASELINE")
    if dep_stable_base.isna().all():
        dep_stable_base = num(out, "COREDEP").fillna(0)

    long_fhlb = num(out, "OTBFH1T3").fillna(0) + num(out, "OTBFH3T5").fillna(0) + num(out, "OTBFHOV5").fillna(0)
    asstlt = num(out, "ASSTLT")

    out["STABLE_FUNDING_BASELINE"] = eq + dep_stable_base.fillna(0) + long_fhlb
    out["LONG_TERM_ASSETS_TO_STABLE_FUNDING_BASELINE"] = safe_div(asstlt, out["STABLE_FUNDING_BASELINE"])
    out["STRUCTURAL_TERM_GAP_BASELINE"] = asstlt - out["STABLE_FUNDING_BASELINE"]

    b_0_3 = num(out, "SCNM3LES").fillna(0)
    b_3_12 = num(out, "SCNM3T12").fillna(0)
    b_1_3 = num(out, "SCNM1T3").fillna(0)
    b_3_5 = num(out, "SCNM3T5").fillna(0)
    b_5_15 = num(out, "SCNM5T15").fillna(0)
    b_15p = num(out, "SCNMOV15").fillna(0)

    sec_total = num(out, "SCRDEBT")
    sec_total = sec_total.where(sec_total.notna() & (sec_total != 0), b_0_3 + b_3_12 + b_1_3 + b_3_5 + b_5_15 + b_15p)

    sec_dur_num = (
        b_0_3 * 0.125
        + b_3_12 * 0.625
        + b_1_3 * 2.0
        + b_3_5 * 4.0
        + b_5_15 * 10.0
        + b_15p * 20.0
    )
    out["SECURITY_DURATION_PROXY"] = safe_div(sec_dur_num, sec_total)

    deposit_wal_base = num(out, "DEPOSIT_WAL_BASELINE")
    out["SECURITY_VS_DEPOSIT_GAP_BASELINE"] = out["SECURITY_DURATION_PROXY"] - deposit_wal_base

    lnlsnet = num(out, "LNLSNET")
    coredep = num(out, "COREDEP")
    idlncorr = num(out, "IDLNCORR")
    out["LOANS_TO_CORE_DEPOSITS"] = safe_div(lnlsnet, coredep)
    out["LOANS_TO_CORE_DEPOSITS"] = out["LOANS_TO_CORE_DEPOSITS"].fillna(idlncorr / 100.0)

    othbfhlb = num(out, "OTHBFHLB")
    short_fhlb = num(out, "OTBFH1L")
    out["SHORT_FHLB_SHARE"] = safe_div(short_fhlb, othbfhlb)

    bucket_names = ["0_3M", "3_12M", "1_3Y", "3_5Y", "5Y_PLUS"]
    if all(f"ASSET_BKT_{bucket}" in out.columns for bucket in bucket_names) and all(
        f"LIAB_BKT_{bucket}" in out.columns for bucket in bucket_names
    ):
        for bucket in bucket_names:
            out[f"REPRICING_GAP_{bucket}"] = num(out, f"ASSET_BKT_{bucket}") - num(out, f"LIAB_BKT_{bucket}")
        out["CUMULATIVE_GAP_1Y"] = out["REPRICING_GAP_0_3M"].fillna(0) + out["REPRICING_GAP_3_12M"].fillna(0)

    return out


def run(
    input_path: Path,
    out_path: Path,
    small_bank_uninsured_fallback_factor: float = 0.50,
) -> pd.DataFrame:
    df = read_table(input_path)
    out = build_alm_mismatch_features(
        df=df,
        small_bank_uninsured_fallback_factor=small_bank_uninsured_fallback_factor,
    )
    save_table(out, out_path)
    return out


def main() -> None:
    args = parse_args()
    out = run(
        input_path=args.input,
        out_path=args.out,
        small_bank_uninsured_fallback_factor=args.small_bank_uninsured_fallback_factor,
    )
    print(f"Saved {len(out):,} rows to {args.out}")


if __name__ == "__main__":
    main()
