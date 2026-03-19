"""Build a bank-quarter panel from FDIC financials, institutions, and SOD."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bankfragility.tables import parse_report_date, read_table, save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financials-glob", required=True, help="Glob for financial files (.csv or .parquet)")
    parser.add_argument("--institutions-glob", default="", help="Optional glob for institution files")
    parser.add_argument("--sod-glob", default="", help="Optional glob for SOD branch files")
    parser.add_argument("--ffiec-repricing-glob", default="", help="Optional glob for FFIEC repricing feature files")
    parser.add_argument("--derivatives-glob", default="", help="Optional glob for derivative overlay files")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_glob(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {pattern}")
    frames = [read_table(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def safe_to_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def aggregate_sod_features(sod: pd.DataFrame) -> pd.DataFrame:
    sod = sod.copy()
    cert_col = first_present(list(sod.columns), ["CERT", "FDICCERT"])
    year_col = first_present(list(sod.columns), ["YEAR", "SOD_YEAR"])
    dep_col = first_present(list(sod.columns), ["DEPSUMBR", "DEPSUM", "SUMDEP", "DEPDOM", "DEP"])
    state_col = first_present(list(sod.columns), ["STALPBR", "STALP", "STNAMEBR", "STNAME", "STATE"])
    county_col = first_present(list(sod.columns), ["CNTYNAMB", "CNTYNUMB", "CNTYNAME", "COUNTY", "CNTYNUM", "COUNTYNAME"])
    branch_col = first_present(list(sod.columns), ["UNINUMBR", "BRNUM", "OFFNUM", "OFFICENUM", "BRANCHID"])

    if cert_col is None or dep_col is None:
        raise ValueError("Could not infer required SOD columns.")

    if year_col is None:
        repdte_col = first_present(list(sod.columns), ["REPDTE", "REPORT_DATE", "DATE"])
        if repdte_col is None:
            raise ValueError("Could not infer a year column for SOD.")
        sod[repdte_col] = parse_report_date(sod[repdte_col])
        sod["YEAR"] = sod[repdte_col].dt.year
        year_col = "YEAR"

    sod[cert_col] = sod[cert_col].astype(str).str.strip()
    sod[year_col] = pd.to_numeric(sod[year_col], errors="coerce").astype("Int64")
    sod[dep_col] = pd.to_numeric(sod[dep_col], errors="coerce").fillna(0.0)

    grouped = sod.groupby([cert_col, year_col], dropna=False)
    out = grouped[dep_col].sum().rename("SOD_TOTAL_DEPOSITS").reset_index()
    out = out.rename(columns={cert_col: "CERT", year_col: "SOD_YEAR"})

    if branch_col:
        branch_counts = (
            sod.groupby([cert_col, year_col], dropna=False)[branch_col]
            .nunique()
            .rename("SOD_BRANCH_COUNT")
            .reset_index()
            .rename(columns={cert_col: "CERT", year_col: "SOD_YEAR"})
        )
        out = out.merge(branch_counts, on=["CERT", "SOD_YEAR"], how="left")

    if state_col:
        state = sod.groupby([cert_col, year_col, state_col], dropna=False)[dep_col].sum().reset_index()
        state = state.rename(columns={cert_col: "CERT", year_col: "SOD_YEAR", dep_col: "DEP_AMT"})
        state_total = state.groupby(["CERT", "SOD_YEAR"], dropna=False)["DEP_AMT"].transform("sum")
        state["DEP_SHARE"] = np.where(state_total > 0, state["DEP_AMT"] / state_total, np.nan)
        state_agg = (
            state.groupby(["CERT", "SOD_YEAR"], dropna=False)
            .agg(
                SOD_STATE_COUNT=(state_col, "nunique"),
                SOD_TOP_STATE_SHARE=("DEP_SHARE", "max"),
                SOD_DEPOSIT_HHI_STATE=("DEP_SHARE", lambda s: float(np.square(s.fillna(0)).sum())),
            )
            .reset_index()
        )
        out = out.merge(state_agg, on=["CERT", "SOD_YEAR"], how="left")

    if county_col:
        county = sod.groupby([cert_col, year_col, county_col], dropna=False)[dep_col].sum().reset_index()
        county = county.rename(columns={cert_col: "CERT", year_col: "SOD_YEAR", dep_col: "DEP_AMT"})
        county_total = county.groupby(["CERT", "SOD_YEAR"], dropna=False)["DEP_AMT"].transform("sum")
        county["DEP_SHARE"] = np.where(county_total > 0, county["DEP_AMT"] / county_total, np.nan)
        county_agg = (
            county.groupby(["CERT", "SOD_YEAR"], dropna=False)
            .agg(
                SOD_COUNTY_COUNT=(county_col, "nunique"),
                SOD_DEPOSIT_HHI_COUNTY=("DEP_SHARE", lambda s: float(np.square(s.fillna(0)).sum())),
            )
            .reset_index()
        )
        out = out.merge(county_agg, on=["CERT", "SOD_YEAR"], how="left")

    return out


def _normalize_institution_columns(institutions: pd.DataFrame) -> pd.DataFrame:
    """Rename API-native field names to canonical panel names."""
    rename_map: dict[str, str] = {}
    cols = set(institutions.columns)
    if "FED_RSSD" in cols and "RSSDID" not in cols:
        rename_map["FED_RSSD"] = "RSSDID"
    if "ID" in cols and "RSSDID" not in cols and "FED_RSSD" not in cols:
        rename_map["ID"] = "RSSDID"
    if "NAME" in cols and "NAMEFULL" not in cols:
        rename_map["NAME"] = "NAMEFULL"
    if rename_map:
        institutions = institutions.rename(columns=rename_map)
    return institutions


def join_institutions(financials: pd.DataFrame, institutions: pd.DataFrame) -> pd.DataFrame:
    institutions = _normalize_institution_columns(institutions)
    institutions["CERT"] = institutions["CERT"].astype(str).str.strip()
    institutions = institutions.drop_duplicates(subset=["CERT"], keep="last")
    institution_keep = [
        col for col in ["CERT", "RSSDID", "RSSDHCR", "NAMEFULL", "BKCLASS", "STALP", "CITY"]
        if col in institutions.columns
    ]
    if not institution_keep:
        return financials

    out = financials.merge(
        institutions[institution_keep],
        on="CERT",
        how="left",
        suffixes=("", "_INST"),
    )
    for col in ["RSSDID", "RSSDHCR", "NAMEFULL", "BKCLASS", "STALP", "CITY"]:
        inst_col = f"{col}_INST"
        if inst_col in out.columns:
            if col not in out.columns:
                out[col] = out[inst_col]
            else:
                out[col] = out[col].fillna(out[inst_col])
            out = out.drop(columns=[inst_col])
    return out


def join_sod(financials: pd.DataFrame, sod: pd.DataFrame) -> pd.DataFrame:
    sod_features = aggregate_sod_features(sod)
    out = financials.copy()
    out["SOD_JOIN_YEAR"] = np.where(
        out["REPDTE"].dt.month <= 6,
        out["REPDTE"].dt.year - 1,
        out["REPDTE"].dt.year,
    )
    return out.merge(
        sod_features,
        left_on=["CERT", "SOD_JOIN_YEAR"],
        right_on=["CERT", "SOD_YEAR"],
        how="left",
    )


def join_ffiec_repricing(financials: pd.DataFrame, ffiec: pd.DataFrame) -> pd.DataFrame:
    """Left-join FFIEC repricing features on CERT + REPDTE."""
    ffiec = ffiec.copy()
    ffiec["CERT"] = ffiec["CERT"].astype(str).str.strip()
    ffiec["REPDTE"] = parse_report_date(ffiec["REPDTE"])

    # Only keep feature columns (drop raw MDRM codes and join helpers)
    feature_cols = [
        c for c in ffiec.columns
        if c.startswith(("LOAN_", "RE_LOAN_", "TD_SMALL_", "TD_LARGE_", "TD_WAM",
                         "TD_MATURITY", "FHLB_", "OTHER_BORR_", "REPRICING_",
                         "CUMULATIVE_", "DURATION_", "ALL_LOAN_"))
    ]
    keep = ["CERT", "REPDTE"] + feature_cols
    keep = [c for c in keep if c in ffiec.columns]
    ffiec_subset = ffiec[keep].drop_duplicates(subset=["CERT", "REPDTE"], keep="last")

    # Avoid column collisions
    existing = set(financials.columns)
    new_cols = [c for c in feature_cols if c not in existing]
    if not new_cols:
        return financials

    merge_cols = ["CERT", "REPDTE"] + new_cols
    return financials.merge(
        ffiec_subset[[c for c in merge_cols if c in ffiec_subset.columns]],
        on=["CERT", "REPDTE"],
        how="left",
    )


def join_derivatives(financials: pd.DataFrame, derivatives: pd.DataFrame) -> pd.DataFrame:
    """Left-join derivative overlay features on CERT + REPDTE."""
    derivatives = derivatives.copy()
    derivatives["CERT"] = derivatives["CERT"].astype(str).str.strip()
    derivatives["REPDTE"] = parse_report_date(derivatives["REPDTE"])

    feature_cols = [
        c for c in derivatives.columns
        if c.startswith(("IR_", "HAS_IR_"))
    ]
    keep = ["CERT", "REPDTE"] + feature_cols
    keep = [c for c in keep if c in derivatives.columns]
    deriv_subset = derivatives[keep].drop_duplicates(subset=["CERT", "REPDTE"], keep="last")

    existing = set(financials.columns)
    new_cols = [c for c in feature_cols if c not in existing]
    if not new_cols:
        return financials

    merge_cols = ["CERT", "REPDTE"] + new_cols
    return financials.merge(
        deriv_subset[[c for c in merge_cols if c in deriv_subset.columns]],
        on=["CERT", "REPDTE"],
        how="left",
    )


def validate_unique_keys(financials: pd.DataFrame) -> None:
    dupes = financials.duplicated(subset=["CERT", "REPDTE"], keep=False)
    if not dupes.any():
        return
    duplicate_count = int(dupes.sum())
    raise ValueError(f"Duplicate CERT+REPDTE rows detected: {duplicate_count}")


def _row_has_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [col for col in cols if col in df.columns]
    if not available:
        return pd.Series(False, index=df.index)
    return df[available].notna().any(axis=1)


def add_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HAS_SOD_FEATURES"] = _row_has_any(
        out,
        [
            "SOD_TOTAL_DEPOSITS",
            "SOD_BRANCH_COUNT",
            "SOD_STATE_COUNT",
            "SOD_DEPOSIT_HHI_STATE",
            "SOD_COUNTY_COUNT",
        ],
    ).astype(int)
    out["HAS_FFIEC_FEATURES"] = _row_has_any(
        out,
        [
            "DURATION_GAP_LITE",
            "LOAN_WAM_PROXY",
            "TD_WAM_PROXY",
            "REPRICING_GAP_0_3M",
            "CUMULATIVE_GAP_3_5Y",
        ],
    ).astype(int)
    out["HAS_DERIVATIVE_FEATURES"] = _row_has_any(
        out,
        [
            "HAS_IR_DERIVATIVES",
            "IR_SWAP_PAY_FIXED_NOTIONAL",
            "IR_SWAP_RECEIVE_FIXED_NOTIONAL",
        ],
    ).astype(int)
    # Current institutions joins are not historical as-of mappings.
    out["HAS_HISTORICAL_ENTITY_MAP"] = 0
    return out


def build_bank_panel(
    financials: pd.DataFrame,
    institutions: pd.DataFrame | None = None,
    sod: pd.DataFrame | None = None,
    ffiec_repricing: pd.DataFrame | None = None,
    derivatives: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = safe_to_numeric(financials, exclude={"CERT", "NAMEFULL", "NAME", "STALP", "CITY", "BKCLASS"})
    if "CERT" not in out.columns or "REPDTE" not in out.columns:
        raise ValueError("Financials input must include CERT and REPDTE.")

    out["CERT"] = out["CERT"].astype(str).str.strip()
    out["REPDTE"] = parse_report_date(out["REPDTE"])
    out = out.dropna(subset=["CERT", "REPDTE"]).sort_values(["CERT", "REPDTE"]).reset_index(drop=True)

    if institutions is not None and not institutions.empty:
        out = join_institutions(out, institutions)
    if sod is not None and not sod.empty:
        out = join_sod(out, sod)
    if ffiec_repricing is not None and not ffiec_repricing.empty:
        out = join_ffiec_repricing(out, ffiec_repricing)
    if derivatives is not None and not derivatives.empty:
        out = join_derivatives(out, derivatives)

    out = add_coverage_flags(out)
    validate_unique_keys(out)
    return out


def main() -> None:
    args = parse_args()
    financials = load_glob(args.financials_glob)
    institutions = load_glob(args.institutions_glob) if args.institutions_glob else None
    sod = load_glob(args.sod_glob) if args.sod_glob else None
    ffiec_repricing = load_glob(args.ffiec_repricing_glob) if args.ffiec_repricing_glob else None
    derivatives = load_glob(args.derivatives_glob) if args.derivatives_glob else None

    panel = build_bank_panel(
        financials=financials, institutions=institutions,
        sod=sod, ffiec_repricing=ffiec_repricing, derivatives=derivatives,
    )
    save_table(panel, args.out)
    print(f"Saved {len(panel):,} rows to {args.out}")


if __name__ == "__main__":
    main()
