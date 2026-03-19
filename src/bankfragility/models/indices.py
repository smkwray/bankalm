"""Build peer-normalized stickiness, run-risk, ALM mismatch, Treasury buffer, and composite indices."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bankfragility.tables import parse_report_date, read_table, save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stickiness", type=Path, required=True)
    parser.add_argument("--alm", type=Path, required=True)
    parser.add_argument("--treasury", type=Path, required=True)
    parser.add_argument("--peer-groups", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default=DEFAULT_INDEX_VERSION, help="Index methodology version tag")
    return parser.parse_args()


def merge_inputs(stickiness: pd.DataFrame, alm: pd.DataFrame, treasury: pd.DataFrame) -> pd.DataFrame:
    keys = ["CERT", "REPDTE"]
    base = stickiness.copy()
    base["REPDTE"] = parse_report_date(base["REPDTE"])
    for other in (alm, treasury):
        other = other.copy()
        other["REPDTE"] = parse_report_date(other["REPDTE"])
        extra_cols = [col for col in other.columns if col not in keys and col not in base.columns]
        if not extra_cols:
            continue
        base = base.merge(other[keys + extra_cols], on=keys, how="left")
    return base


def assign_peer_group(assets: pd.Series, peer_cfg: list[dict[str, object]]) -> pd.Series:
    out = pd.Series("unassigned", index=assets.index, dtype="object")
    numeric_assets = pd.to_numeric(assets, errors="coerce")
    for spec in peer_cfg:
        name = str(spec["name"])
        min_assets = float(spec.get("min_assets") or 0)
        max_assets = spec.get("max_assets")
        if max_assets is None:
            mask = numeric_assets >= min_assets
        else:
            mask = (numeric_assets >= min_assets) & (numeric_assets < float(max_assets))
        out = out.where(~mask, name)
    return out


def percentile_by_group(df: pd.DataFrame, value_col: str, group_cols: list[str], higher_is_better: bool) -> pd.Series:
    def ranker(s: pd.Series) -> pd.Series:
        r = s.rank(pct=True, method="average", ascending=True)
        if not higher_is_better:
            r = 1 - r
        return 100.0 * r.fillna(0.5)

    return df.groupby(group_cols, dropna=False)[value_col].transform(ranker)


def weighted_index(
    df: pd.DataFrame,
    components: dict[str, float],
    group_cols: list[str],
    higher_is_better: bool,
    out_col: str,
) -> None:
    valid = [(col.upper(), float(weight)) for col, weight in components.items() if col.upper() in df.columns and float(weight) > 0]
    if not valid:
        df[out_col] = np.nan
        return

    total_weight = sum(weight for _, weight in valid)
    df[out_col] = 0.0
    for col, weight in valid:
        pct = percentile_by_group(df, col, group_cols, higher_is_better=higher_is_better)
        contrib_col = f"{out_col}_CONTRIB_{col}"
        df[contrib_col] = pct * weight / total_weight
        df[out_col] += df[contrib_col]
    df[out_col] = df[out_col].clip(lower=0, upper=100)


DEFAULT_INDEX_VERSION = "v0_fdic_only"
FFIEC_VERSION = "v1_ffiec_hybrid"


def _build_index_version(df: pd.DataFrame, requested_version: str) -> pd.Series:
    if requested_version != DEFAULT_INDEX_VERSION:
        return pd.Series(requested_version, index=df.index, dtype="object")

    ffiec_source = df["HAS_FFIEC_FEATURES"] if "HAS_FFIEC_FEATURES" in df.columns else pd.Series(0, index=df.index)
    ffiec_flag = pd.to_numeric(ffiec_source, errors="coerce").fillna(0).astype(int) > 0
    version = pd.Series(DEFAULT_INDEX_VERSION, index=df.index, dtype="object")
    version.loc[ffiec_flag] = FFIEC_VERSION
    return version


def build_indices_frame(
    stickiness: pd.DataFrame,
    alm: pd.DataFrame,
    treasury: pd.DataFrame,
    peer_cfg: list[dict[str, object]],
    weight_cfg: dict[str, dict[str, float]],
    version: str = DEFAULT_INDEX_VERSION,
) -> pd.DataFrame:
    df = merge_inputs(stickiness, alm, treasury)
    df["REPDTE"] = parse_report_date(df["REPDTE"])
    df["ASSET"] = pd.to_numeric(df["ASSET"], errors="coerce")
    df["PEER_GROUP"] = assign_peer_group(df["ASSET"], peer_cfg)

    group_cols = ["REPDTE", "PEER_GROUP"]

    if "RUN_RISK_SCORE" in df.columns:
        df["RUN_RISK_INDEX"] = percentile_by_group(df, "RUN_RISK_SCORE", group_cols, higher_is_better=True)
    else:
        df["RUN_RISK_INDEX"] = np.nan
    df["DEPOSIT_STICKINESS_INDEX"] = 100.0 - df["RUN_RISK_INDEX"]

    weighted_index(
        df=df,
        components=weight_cfg["alm_mismatch_components"],
        group_cols=group_cols,
        higher_is_better=True,
        out_col="ALM_MISMATCH_INDEX",
    )
    weighted_index(
        df=df,
        components=weight_cfg["treasury_buffer_components"],
        group_cols=group_cols,
        higher_is_better=True,
        out_col="TREASURY_BUFFER_INDEX",
    )

    comp = weight_cfg["composite_fragility_weights"]
    inv_treas = 100.0 - df["TREASURY_BUFFER_INDEX"]
    df["FUNDING_FRAGILITY_INDEX"] = (
        float(comp["run_risk_index"]) * df["RUN_RISK_INDEX"]
        + float(comp["alm_mismatch_index"]) * df["ALM_MISMATCH_INDEX"]
        + float(comp["inverse_treasury_buffer_index"]) * inv_treas
    )
    df["FUNDING_FRAGILITY_INDEX"] = df["FUNDING_FRAGILITY_INDEX"].clip(lower=0, upper=100)

    if "HAS_SOD_FEATURES" not in df.columns:
        sod_cols = [c for c in df.columns if c.startswith("SOD_")]
        df["HAS_SOD_FEATURES"] = df[sod_cols].notna().any(axis=1).astype(int) if sod_cols else 0
    if "HAS_FFIEC_FEATURES" not in df.columns:
        ffiec_cols = [c for c in df.columns if c.startswith(("DURATION_GAP", "REPRICING_GAP", "LOAN_WAM", "TD_WAM"))]
        df["HAS_FFIEC_FEATURES"] = df[ffiec_cols].notna().any(axis=1).astype(int) if ffiec_cols else 0
    if "HAS_HISTORICAL_ENTITY_MAP" not in df.columns:
        df["HAS_HISTORICAL_ENTITY_MAP"] = 0

    df["INDEX_VERSION"] = _build_index_version(df, version)

    # Flag banks with no IR derivative hedging and positive duration gap
    if "HAS_IR_DERIVATIVES" in df.columns and "DURATION_GAP_LITE" in df.columns:
        df["UNHEDGED_DURATION_FLAG"] = (
            (df["HAS_IR_DERIVATIVES"].fillna(0) == 0)
            & (df["DURATION_GAP_LITE"].fillna(0) > 0)
        ).astype(int)
    elif "HAS_IR_DERIVATIVES" in df.columns:
        df["UNHEDGED_DURATION_FLAG"] = (df["HAS_IR_DERIVATIVES"].fillna(0) == 0).astype(int)

    return df


def main() -> None:
    args = parse_args()
    stickiness = read_table(args.stickiness)
    alm = read_table(args.alm)
    treasury = read_table(args.treasury)

    with args.peer_groups.open("r", encoding="utf-8") as handle:
        peer_cfg = yaml.safe_load(handle)["peer_groups"]
    with args.weights.open("r", encoding="utf-8") as handle:
        weight_cfg = yaml.safe_load(handle)

    out = build_indices_frame(
        stickiness=stickiness,
        alm=alm,
        treasury=treasury,
        peer_cfg=peer_cfg,
        weight_cfg=weight_cfg,
        version=args.version,
    )
    save_table(out, args.out)
    print(f"Saved {len(out):,} rows to {args.out}")


if __name__ == "__main__":
    main()
