"""Build publishable mart, split publishable panels, and site export artifacts."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUN_RISK_COMPONENTS = [
    {"key": "uninsured_share", "label": "Uninsured deposits", "weight": 0.20, "rank_col": "RANK_UNINSURED_SHARE"},
    {"key": "brokered_share", "label": "Brokered deposits", "weight": 0.17, "rank_col": "RANK_BROKERED_SHARE"},
    {"key": "list_service_share", "label": "List-service deposits", "weight": 0.08, "rank_col": "RANK_LIST_SERVICE_SHARE"},
    {"key": "large_account_share", "label": "Large-account concentration", "weight": 0.10, "rank_col": "RANK_LARGE_ACCOUNT_SHARE"},
    {"key": "time_deposit_share", "label": "Time-deposit share", "weight": 0.05, "rank_col": "RANK_TIME_DEPOSIT_SHARE"},
    {"key": "domestic_deposit_cost", "label": "Domestic deposit cost", "weight": 0.08, "rank_col": "RANK_DOMESTIC_DEPOSIT_COST"},
    {"key": "dep_growth_vol_4q", "label": "Deposit-growth volatility", "weight": 0.05, "rank_col": "RANK_DEP_GROWTH_VOL_4Q"},
    {"key": "dep_drawdown_4q", "label": "Deposit drawdown", "weight": 0.05, "rank_col": "RANK_DEP_DRAWDOWN_4Q"},
]

ALM_COMPONENTS = [
    {
        "key": "long_term_assets_to_stable_funding_baseline",
        "label": "Long-term assets / stable funding",
        "weight": 0.30,
        "contrib_col": "ALM_MISMATCH_INDEX_CONTRIB_LONG_TERM_ASSETS_TO_STABLE_FUNDING_BASELINE",
    },
    {
        "key": "volatile_to_liquid_lower",
        "label": "Volatile funding / liquid assets",
        "weight": 0.30,
        "contrib_col": "ALM_MISMATCH_INDEX_CONTRIB_VOLATILE_TO_LIQUID_LOWER",
    },
    {
        "key": "security_vs_deposit_gap_baseline",
        "label": "Securities / deposit gap",
        "weight": 0.20,
        "contrib_col": "ALM_MISMATCH_INDEX_CONTRIB_SECURITY_VS_DEPOSIT_GAP_BASELINE",
    },
    {
        "key": "loans_to_core_deposits",
        "label": "Loans / core deposits",
        "weight": 0.10,
        "contrib_col": "ALM_MISMATCH_INDEX_CONTRIB_LOANS_TO_CORE_DEPOSITS",
    },
    {
        "key": "short_fhlb_share",
        "label": "Short FHLB funding share",
        "weight": 0.10,
        "contrib_col": "ALM_MISMATCH_INDEX_CONTRIB_SHORT_FHLB_SHARE",
    },
]

TREASURY_BUFFER_COMPONENTS = [
    {
        "key": "treasury_to_uninsured_after_100bp",
        "label": "Treasuries / uninsured after 100 bp shock",
        "weight": 0.35,
        "contrib_col": "TREASURY_BUFFER_INDEX_CONTRIB_TREASURY_TO_UNINSURED_AFTER_100BP",
    },
    {
        "key": "treasury_agency_to_runnable",
        "label": "Treasury-agency / runnable funding",
        "weight": 0.35,
        "contrib_col": "TREASURY_BUFFER_INDEX_CONTRIB_TREASURY_AGENCY_TO_RUNNABLE",
    },
    {
        "key": "hqla_narrow_lower_to_runnable",
        "label": "Narrow HQLA / runnable funding",
        "weight": 0.30,
        "contrib_col": "TREASURY_BUFFER_INDEX_CONTRIB_HQLA_NARROW_LOWER_TO_RUNNABLE",
    },
]

DEPOSIT_COMPETITION_COMPONENTS = [
    {
        "key": "outside_option_premium_pos_bp",
        "label": "Outside-option premium",
        "weight": 0.35,
        "rank_col": "RANK_OUTSIDE_OPTION_PREMIUM_POS_BP",
    },
    {
        "key": "pass_through_gap_bp",
        "label": "Pass-through gap",
        "weight": 0.20,
        "rank_col": "RANK_PASS_THROUGH_GAP_BP",
    },
    {
        "key": "rate_sensitive_deposit_exposure",
        "label": "Rate-sensitive deposit exposure",
        "weight": 0.10,
        "rank_col": "RANK_RATE_SENSITIVE_DEPOSIT_EXPOSURE",
    },
    {
        "key": "premium_x_rate_sensitive_exposure",
        "label": "Premium × rate-sensitive exposure",
        "weight": 0.20,
        "rank_col": "RANK_PREMIUM_X_RATE_SENSITIVE_EXPOSURE",
    },
    {
        "key": "premium_x_dep_drawdown_4q",
        "label": "Premium × deposit drawdown",
        "weight": 0.10,
        "rank_col": "RANK_PREMIUM_X_DEP_DRAWDOWN_4Q",
    },
    {
        "key": "premium_x_short_fhlb_share",
        "label": "Premium × short FHLB share",
        "weight": 0.05,
        "rank_col": "RANK_PREMIUM_X_SHORT_FHLB_SHARE",
    },
]

COMPOSITE_COMPONENTS = [
    {"key": "run_risk_index", "label": "Run Risk Index", "weight": 0.40},
    {"key": "alm_mismatch_index", "label": "ALM Mismatch Index", "weight": 0.40},
    {"key": "inverse_treasury_buffer_index", "label": "Inverse Treasury Buffer", "weight": 0.20},
]

INDEX_METHODOLOGY = {
    "run_risk": {
        "title": "Run Risk Index",
        "summary": "Percentile rank of the transparent run-risk score within the same quarter and asset-size peer group.",
        "formula": "RUN_RISK_INDEX = percentile rank of RUN_RISK_SCORE within REPDTE x PEER_GROUP",
        "scale_note": "Higher means more fragile funding within the peer group.",
        "components": RUN_RISK_COMPONENTS,
    },
    "alm_mismatch": {
        "title": "ALM Mismatch Index",
        "summary": "Weighted percentile composite of structural public-data ALM mismatch proxies within the same quarter and peer group.",
        "formula": "ALM_MISMATCH_INDEX = weighted percentile composite within REPDTE x PEER_GROUP",
        "scale_note": "Higher means larger structural mismatch by public-data proxy.",
        "components": ALM_COMPONENTS,
    },
    "deposit_competition": {
        "title": "Deposit Competition Pressure Index",
        "summary": "Percentile rank of the transparent deposit-competition pressure score within the same quarter and peer group.",
        "formula": "DEPOSIT_COMPETITION_PRESSURE_INDEX = percentile rank of DEPOSIT_COMPETITION_PRESSURE_SCORE within REPDTE x PEER_GROUP",
        "scale_note": "Higher means greater pressure from safe outside options and rate-sensitive funding exposure within the peer group.",
        "components": DEPOSIT_COMPETITION_COMPONENTS,
    },
    "funding_fragility": {
        "title": "Composite Fragility Index",
        "summary": "Weighted mix of run risk, ALM mismatch, and the inverse of Treasury buffer strength.",
        "formula": "0.40 x RUN_RISK_INDEX + 0.40 x ALM_MISMATCH_INDEX + 0.20 x (100 - TREASURY_BUFFER_INDEX)",
        "scale_note": "Higher means more overall fragility on the transparent public-data screen.",
        "components": COMPOSITE_COMPONENTS,
    },
    "treasury_buffer": {
        "title": "Treasury Buffer Index",
        "summary": "Weighted percentile composite of public liquidity buffer ratios within the same quarter and peer group.",
        "formula": "TREASURY_BUFFER_INDEX = weighted percentile composite within REPDTE x PEER_GROUP",
        "scale_note": "Higher means stronger public Treasury/HQLA buffer, which lowers composite fragility.",
        "components": TREASURY_BUFFER_COMPONENTS,
    },
}

SITE_INDEX_META = {
    "run_risk": {
        "score_col": "RUN_RISK_INDEX",
        "title": "Run Risk Index",
        "accent": "crimson",
        "description": (
            "Transparent screening score for funding fragility using uninsured deposits, "
            "volatile funding mix, liquidity coverage, and recent deposit weakness."
        ),
    },
    "alm_mismatch": {
        "score_col": "ALM_MISMATCH_INDEX",
        "title": "ALM Mismatch Index",
        "accent": "amber",
        "description": (
            "Structural balance-sheet mismatch proxy built from public call-report ratios. "
            "This is a heuristic scenario lens, not a full internal ALM model."
        ),
    },
    "deposit_competition": {
        "score_col": "DEPOSIT_COMPETITION_PRESSURE_INDEX",
        "title": "Deposit Competition Index",
        "accent": "emerald",
        "description": (
            "Peer-normalized pressure from safe outside options, deposit pass-through gaps, "
            "and rate-sensitive funding exposure."
        ),
    },
    "funding_fragility": {
        "score_col": "FUNDING_FRAGILITY_INDEX",
        "title": "Composite Fragility Index",
        "accent": "indigo",
        "description": (
            "Peer-normalized composite of the transparent run-risk, ALM-mismatch, and "
            "Treasury-buffer indices for exploratory public-data screening."
        ),
    },
}

CORE_PANEL_COLUMNS = [
    "CERT", "REPDTE", "NAMEFULL", "PEER_GROUP", "ASSET", "BKCLASS", "STALP", "CITY",
    "RSSDID", "RSSDHCR",
    "DEPDOM", "DEPDOM_EFFECTIVE", "UNINSURED_SHARE", "BROKERED_SHARE", "CORE_DEPOSIT_SHARE",
    "NONINTEREST_SHARE", "TIME_DEPOSIT_SHARE", "DOMESTIC_DEPOSIT_COST",
    "DEP_GROWTH_QOQ", "DEP_GROWTH_VOL_4Q", "DEP_DRAWDOWN_4Q",
    "TREASURY_TO_UNINSURED", "TREASURY_AGENCY_TO_RUNNABLE", "HQLA_NARROW_LOWER_TO_RUNNABLE",
    "HAS_TREASURY_YIELD_HISTORY",
    "RUN_RISK_SCORE", "STICKINESS_SCORE",
    "RUN_RISK_INDEX", "DEPOSIT_STICKINESS_INDEX", "ALM_MISMATCH_INDEX",
    "TREASURY_BUFFER_INDEX", "FUNDING_FRAGILITY_INDEX",
    "DEPOSIT_WAL_BASELINE", "DEPOSIT_WAL_ADVERSE", "DEPOSIT_WAL_SEVERE",
    "DEPOSIT_STABLE_EQUIV_BASELINE", "DEPOSIT_STABLE_EQUIV_ADVERSE", "DEPOSIT_STABLE_EQUIV_SEVERE",
    "INDEX_VERSION", "HAS_SOD_FEATURES", "HAS_FFIEC_FEATURES", "HAS_DERIVATIVE_FEATURES",
    "HAS_HISTORICAL_ENTITY_MAP",
]


def _dedupe_merge(base: pd.DataFrame, other: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    other = other.copy()
    extra_cols = [col for col in other.columns if col not in keys and col not in base.columns]
    if not extra_cols:
        return base
    return base.merge(other[keys + extra_cols], on=keys, how="left")


def build_publishable_mart(indices: pd.DataFrame, supervised: pd.DataFrame | None = None) -> pd.DataFrame:
    keys = ["CERT", "REPDTE"]
    mart = indices.copy()
    mart["REPDTE"] = pd.to_datetime(mart["REPDTE"])

    if supervised is not None and not supervised.empty:
        sup = supervised.copy()
        sup["REPDTE"] = pd.to_datetime(sup["REPDTE"])
        mart = _dedupe_merge(mart, sup, keys)

    if "HAS_HISTORICAL_ENTITY_MAP" not in mart.columns:
        mart["HAS_HISTORICAL_ENTITY_MAP"] = 0
    mart["SUPERVISED_EXPERIMENTAL"] = mart["SUPERVISED_OUTFLOW_SCORE"].notna().astype(int) if "SUPERVISED_OUTFLOW_SCORE" in mart.columns else 0
    return mart


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = [col for col in columns if col in df.columns]
    return df[keep].copy()


def infer_enriched_start_repdte(mart: pd.DataFrame) -> pd.Timestamp | None:
    flags = pd.Series(False, index=mart.index)
    for col in ["HAS_FFIEC_FEATURES", "HAS_DERIVATIVE_FEATURES"]:
        if col in mart.columns:
            flags |= pd.to_numeric(mart[col], errors="coerce").fillna(0).astype(int) > 0
    if "HAS_TREASURY_YIELD_HISTORY" in mart.columns:
        flags |= pd.to_numeric(mart["HAS_TREASURY_YIELD_HISTORY"], errors="coerce").fillna(0).astype(int) > 0
    if not flags.any():
        return None
    repdte = pd.to_datetime(mart.loc[flags, "REPDTE"])
    return repdte.min()


def split_publishable_panels(mart: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the integrated mart into a historically consistent core panel and a recent enriched panel."""
    out = mart.copy()
    out["REPDTE"] = pd.to_datetime(out["REPDTE"])

    core = _select_columns(out, CORE_PANEL_COLUMNS)
    core["PANEL_VARIANT"] = "full_history_core"

    enriched_start = infer_enriched_start_repdte(out)
    if enriched_start is None:
        enriched = out.copy()
    else:
        enriched = out[out["REPDTE"] >= enriched_start].copy()
    enriched["PANEL_VARIANT"] = "recent_history_enriched"
    if enriched_start is not None:
        core["ENRICHED_WINDOW_START"] = enriched_start
        enriched["ENRICHED_WINDOW_START"] = enriched_start
    return core, enriched


def latest_bank_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["REPDTE"] = pd.to_datetime(out["REPDTE"])
    return out.sort_values(["CERT", "REPDTE"]).groupby("CERT", as_index=False).tail(1).reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def _json_safe_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_nested(v) for v in value]
    return _json_safe(value)


def _build_run_risk_components(row: pd.Series) -> list[dict[str, Any]]:
    return _build_rank_components(row, RUN_RISK_COMPONENTS)


def _build_rank_components(row: pd.Series, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_weight = sum(float(spec["weight"]) for spec in specs)
    components: list[dict[str, Any]] = []
    for spec in specs:
        rank_value = pd.to_numeric(row.get(spec["rank_col"]), errors="coerce")
        if pd.isna(rank_value):
            continue
        percentile = 100.0 * float(rank_value)
        contribution = percentile * float(spec["weight"]) / total_weight
        components.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "weight": float(spec["weight"]),
                "percentile": percentile,
                "contribution": contribution,
            }
        )
    return components


def _build_contribution_components(row: pd.Series, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for spec in specs:
        contribution = pd.to_numeric(row.get(spec["contrib_col"]), errors="coerce")
        if pd.isna(contribution):
            continue
        components.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "weight": float(spec["weight"]),
                "contribution": float(contribution),
            }
        )
    return components


def _build_composite_components(row: pd.Series) -> list[dict[str, Any]]:
    run_risk = pd.to_numeric(row.get("RUN_RISK_INDEX"), errors="coerce")
    alm = pd.to_numeric(row.get("ALM_MISMATCH_INDEX"), errors="coerce")
    treasury_buffer = pd.to_numeric(row.get("TREASURY_BUFFER_INDEX"), errors="coerce")
    treasury_inverse = np.nan if pd.isna(treasury_buffer) else 100.0 - float(treasury_buffer)
    values = {
        "run_risk_index": run_risk,
        "alm_mismatch_index": alm,
        "inverse_treasury_buffer_index": treasury_inverse,
    }

    components: list[dict[str, Any]] = []
    for spec in COMPOSITE_COMPONENTS:
        value = values[spec["key"]]
        if pd.isna(value):
            continue
        components.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "weight": float(spec["weight"]),
                "score": float(value),
                "contribution": float(spec["weight"]) * float(value),
            }
        )
    return components


def _metric_row(validation_metrics: pd.DataFrame | None, score_col: str, horizon_quarters: int) -> pd.Series | None:
    if validation_metrics is None or validation_metrics.empty:
        return None
    rows = validation_metrics[
        (validation_metrics["SLICE"] == "full_sample")
        & (validation_metrics["HORIZON_QUARTERS"] == horizon_quarters)
        & (validation_metrics["SCORE_COL"] == score_col)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _latest_completed_quarter_end(reference: datetime) -> pd.Timestamp:
    quarter_start_month = ((reference.month - 1) // 3) * 3 + 1
    current_quarter_start = pd.Timestamp(datetime(reference.year, quarter_start_month, 1, tzinfo=UTC))
    return (current_quarter_start - pd.Timedelta(days=1)).normalize().tz_localize(None)


def _max_repdte_for_flag(mart: pd.DataFrame, column: str) -> pd.Timestamp | None:
    if column not in mart.columns:
        return None
    flag = pd.to_numeric(mart[column], errors="coerce").fillna(0).astype(int) > 0
    if not flag.any():
        return None
    return pd.to_datetime(mart.loc[flag, "REPDTE"]).max()


def _sod_snapshot_date(repdte: pd.Timestamp | None) -> pd.Timestamp | None:
    if repdte is None or pd.isna(repdte):
        return None
    year = repdte.year if repdte.month >= 6 else repdte.year - 1
    return pd.Timestamp(year=year, month=6, day=30)


def build_manifest_freshness(mart: pd.DataFrame, generated_at: datetime) -> dict[str, Any]:
    site_snapshot_as_of = pd.to_datetime(mart["REPDTE"]).max() if not mart.empty else None
    source_max_dates = {
        "fdic_financials": _json_safe(site_snapshot_as_of),
        "ffiec": _json_safe(_max_repdte_for_flag(mart, "HAS_FFIEC_FEATURES")),
        "sod": _json_safe(_sod_snapshot_date(_max_repdte_for_flag(mart, "HAS_SOD_FEATURES"))),
        "treasury": _json_safe(
            pd.to_datetime(mart["TREASURY_YIELD_DATE"]).max()
            if "TREASURY_YIELD_DATE" in mart.columns and mart["TREASURY_YIELD_DATE"].notna().any()
            else None
        ),
    }
    coverage_warnings: list[str] = []
    for source, max_date in source_max_dates.items():
        if max_date is None:
            coverage_warnings.append(f"{source} coverage is unavailable for the published site snapshot.")
            continue
        if site_snapshot_as_of is not None and pd.Timestamp(max_date) < site_snapshot_as_of:
            coverage_warnings.append(
                f"{source} coverage trails the published site snapshot ({max_date} vs {_json_safe(site_snapshot_as_of)})."
            )
    stale = site_snapshot_as_of is None or site_snapshot_as_of < _latest_completed_quarter_end(generated_at)
    return {
        "site_snapshot_as_of": _json_safe(site_snapshot_as_of),
        "generated_at": generated_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_max_dates": source_max_dates,
        "coverage_warnings": coverage_warnings,
        "stale": bool(stale),
    }


def build_site_manifest(
    mart: pd.DataFrame,
    full_history_core: pd.DataFrame | None = None,
    recent_history_enriched: pd.DataFrame | None = None,
    validation_metrics: pd.DataFrame | None = None,
    headline_horizon_quarters: int = 4,
) -> dict[str, Any]:
    latest = latest_bank_snapshot(mart)
    generated_at = datetime.now(UTC).replace(microsecond=0)
    freshness = build_manifest_freshness(mart, generated_at)

    run_risk_metric = _metric_row(validation_metrics, "RUN_RISK_INDEX", headline_horizon_quarters)
    failures_tested = int(run_risk_metric["N_FAILURES"]) if run_risk_metric is not None and pd.notna(run_risk_metric["N_FAILURES"]) else 0

    indices: list[dict[str, Any]] = []
    for site_id, meta in SITE_INDEX_META.items():
        row = _metric_row(validation_metrics, meta["score_col"], headline_horizon_quarters)
        top = latest.sort_values(meta["score_col"], ascending=False).head(3) if meta["score_col"] in latest.columns else latest.head(0)
        indices.append(
            {
                "id": site_id,
                "title": meta["title"],
                "accent": meta["accent"],
                "description": meta["description"],
                "failure_auc": _json_safe(row["AUC"]) if row is not None else None,
                "failure_recall_20": _json_safe(row["RECALL_AT_20PCT"]) if row is not None else None,
                "bank_count": int(latest["CERT"].nunique()),
                "headline_horizon_quarters": headline_horizon_quarters,
                "validation_status": "backtested" if row is not None else "not_backtested_yet",
                "status_note": (
                    f"Quarter-aligned failure backtest available at the {headline_horizon_quarters}-quarter horizon."
                    if row is not None
                    else "Experimental index. No failure backtest has been published yet."
                ),
                "top_banks": [
                    {
                        "name": _json_safe(b.get("NAMEFULL")),
                        "cert": _json_safe(b.get("CERT")),
                        "score": _json_safe(b.get(meta["score_col"])),
                        "peer_group": _json_safe(b.get("PEER_GROUP")),
                    }
                    for _, b in top.iterrows()
                ],
            }
        )

    peer_groups = []
    if "PEER_GROUP" in latest.columns:
        for peer_group, count in latest["PEER_GROUP"].fillna("unassigned").value_counts().items():
            peer_groups.append(
                {
                    "id": peer_group,
                    "label": str(peer_group).replace("_", " ").title(),
                    "count": int(count),
                }
            )

    published_panels: dict[str, Any] = {}
    for panel_name, panel_df in [
        ("full_history_core", full_history_core),
        ("recent_history_enriched", recent_history_enriched),
    ]:
        if panel_df is None or panel_df.empty:
            continue
        panel_df = panel_df.copy()
        panel_df["REPDTE"] = pd.to_datetime(panel_df["REPDTE"])
        published_panels[panel_name] = {
            "bank_quarters": int(len(panel_df)),
            "unique_banks": int(panel_df["CERT"].nunique()),
            "quarters": int(panel_df["REPDTE"].nunique()),
            "date_range": (
                f"{panel_df['REPDTE'].min():%Y-%m-%d} – {panel_df['REPDTE'].max():%Y-%m-%d}"
                if not panel_df.empty else None
            ),
        }

    return {
        "schema_version": 3,
        "generated_at": freshness["generated_at"],
        "freshness": freshness,
        "pipeline": {
            "bank_quarters": int(len(mart)),
            "unique_banks": int(mart["CERT"].nunique()),
            "quarters": int(mart["REPDTE"].nunique()),
            "date_range": (
                f"{mart['REPDTE'].min():%Y-%m-%d} – {mart['REPDTE'].max():%Y-%m-%d}"
                if not mart.empty else None
            ),
            "failures_tested": failures_tested,
            "headline_horizon_quarters": headline_horizon_quarters,
        },
        "published_panels": published_panels,
        "site_panel": "recent_history_enriched" if recent_history_enriched is not None and not recent_history_enriched.empty else "full_history_core",
        "treasury_regime": (
            {
                "yield_date": _json_safe(latest["TREASURY_YIELD_DATE"].dropna().max()) if "TREASURY_YIELD_DATE" in latest.columns and latest["TREASURY_YIELD_DATE"].notna().any() else None,
                "y2": _json_safe(latest.loc[latest["TREASURY_YIELD_DATE"].notna(), "YC_2YR"].iloc[-1]) if "TREASURY_YIELD_DATE" in latest.columns and latest["TREASURY_YIELD_DATE"].notna().any() and "YC_2YR" in latest.columns else None,
                "y10": _json_safe(latest.loc[latest["TREASURY_YIELD_DATE"].notna(), "YC_10YR"].iloc[-1]) if "TREASURY_YIELD_DATE" in latest.columns and latest["TREASURY_YIELD_DATE"].notna().any() and "YC_10YR" in latest.columns else None,
                "slope_10y_3m_bp": _json_safe(latest.loc[latest["TREASURY_YIELD_DATE"].notna(), "YC_10Y_3M_SLOPE_BP"].iloc[-1]) if "TREASURY_YIELD_DATE" in latest.columns and latest["TREASURY_YIELD_DATE"].notna().any() and "YC_10Y_3M_SLOPE_BP" in latest.columns else None,
            }
        ),
        "indices": indices,
        "index_methodology": _json_safe_nested(INDEX_METHODOLOGY),
        "peer_groups": peer_groups,
        "stress_episodes": [],
        "methodology_notes": {
            "positioning": "Exploratory public-data fragility screen, not a decision-grade failure predictor.",
            "supervised_overlay": "Experimental secondary overlay trained on next-quarter deposit weakness.",
            "deposit_life": "Deposit-life outputs are scenario proxies, not empirical decay estimates.",
            "alm": "ALM outputs are structural public-data heuristics, not a bank's internal risk system.",
        },
    }


def build_league_rows(mart: pd.DataFrame) -> list[dict[str, Any]]:
    latest = latest_bank_snapshot(mart)
    latest = latest.sort_values("FUNDING_FRAGILITY_INDEX", ascending=False) if "FUNDING_FRAGILITY_INDEX" in latest.columns else latest
    peer_group_sizes = latest["PEER_GROUP"].fillna("unassigned").value_counts().to_dict() if "PEER_GROUP" in latest.columns else {}

    rows: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        deposits = row.get("DEPDOM_EFFECTIVE", row.get("DEPDOM"))
        run_risk_components = _build_run_risk_components(row)
        deposit_competition_components = _build_rank_components(row, DEPOSIT_COMPETITION_COMPONENTS)
        alm_components = _build_contribution_components(row, ALM_COMPONENTS)
        treasury_buffer_components = _build_contribution_components(row, TREASURY_BUFFER_COMPONENTS)
        composite_components = _build_composite_components(row)
        rows.append(
            {
                "cert": _json_safe(row.get("CERT")),
                "name": _json_safe(row.get("NAMEFULL")),
                "peer_group": _json_safe(row.get("PEER_GROUP")),
                "peer_group_bank_count": int(peer_group_sizes.get(row.get("PEER_GROUP"), 0)),
                "assets": _json_safe(row.get("ASSET")),
                "deposits": _json_safe(deposits),
                "uninsured_pct": _json_safe(row.get("UNINSURED_SHARE")),
                "state": _json_safe(row.get("STALP")),
                "charter": _json_safe(row.get("BKCLASS")),
                "repdte": _json_safe(row.get("REPDTE")),
                "run_risk_score": _json_safe(row.get("RUN_RISK_SCORE")),
                "stickiness_score": _json_safe(row.get("STICKINESS_SCORE")),
                "run_risk": _json_safe(row.get("RUN_RISK_INDEX")),
                "alm_mismatch": _json_safe(row.get("ALM_MISMATCH_INDEX")),
                "treasury_buffer": _json_safe(row.get("TREASURY_BUFFER_INDEX")),
                "deposit_competition": _json_safe(row.get("DEPOSIT_COMPETITION_PRESSURE_INDEX")),
                "deposit_competition_resilience": _json_safe(row.get("DEPOSIT_COMPETITION_RESILIENCE_INDEX")),
                "deposit_competition_score": _json_safe(row.get("DEPOSIT_COMPETITION_PRESSURE_SCORE")),
                "funding_fragility": _json_safe(row.get("FUNDING_FRAGILITY_INDEX")),
                "run_risk_components": _json_safe_nested(run_risk_components),
                "deposit_competition_components": _json_safe_nested(deposit_competition_components),
                "alm_components": _json_safe_nested(alm_components),
                "treasury_buffer_components": _json_safe_nested(treasury_buffer_components),
                "composite_components": _json_safe_nested(composite_components),
                "supervised_outflow_score": _json_safe(row.get("SUPERVISED_OUTFLOW_SCORE")),
                "treasury_yield_date": _json_safe(row.get("TREASURY_YIELD_DATE")),
                "has_treasury_yield_history": _json_safe(row.get("HAS_TREASURY_YIELD_HISTORY")),
                "yc_2yr": _json_safe(row.get("YC_2YR")),
                "yc_10yr": _json_safe(row.get("YC_10YR")),
                "yc_10y_3m_slope_bp": _json_safe(row.get("YC_10Y_3M_SLOPE_BP")),
                "yc_10y_2y_slope_bp": _json_safe(row.get("YC_10Y_2Y_SLOPE_BP")),
                "yc_10yr_qoq_change_bp": _json_safe(row.get("YC_10YR_QOQ_CHANGE_BP")),
                "has_ffiec_features": _json_safe(row.get("HAS_FFIEC_FEATURES")),
                "has_sod_features": _json_safe(row.get("HAS_SOD_FEATURES")),
                "index_version": _json_safe(row.get("INDEX_VERSION")),
                "failed": False,
                "fail_date": None,
            }
        )
    return rows


def build_bank_summary_rows(mart: pd.DataFrame) -> list[dict[str, Any]]:
    detail_rows = build_league_rows(mart)
    summary_fields = [
        "cert",
        "name",
        "peer_group",
        "peer_group_bank_count",
        "assets",
        "deposits",
        "uninsured_pct",
        "state",
        "repdte",
        "run_risk",
        "alm_mismatch",
        "treasury_buffer",
        "deposit_competition",
        "funding_fragility",
    ]
    return [{field: row.get(field) for field in summary_fields} for row in detail_rows]


def write_site_exports(
    mart: pd.DataFrame,
    site_dir: str | Path,
    full_history_core: pd.DataFrame | None = None,
    recent_history_enriched: pd.DataFrame | None = None,
    validation_metrics: pd.DataFrame | None = None,
    headline_horizon_quarters: int = 4,
) -> None:
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)

    if full_history_core is None or recent_history_enriched is None:
        inferred_core, inferred_enriched = split_publishable_panels(mart)
        full_history_core = inferred_core if full_history_core is None else full_history_core
        recent_history_enriched = inferred_enriched if recent_history_enriched is None else recent_history_enriched

    site_mart = recent_history_enriched if recent_history_enriched is not None and not recent_history_enriched.empty else mart
    manifest = build_site_manifest(
        site_mart,
        full_history_core=full_history_core,
        recent_history_enriched=recent_history_enriched,
        validation_metrics=validation_metrics,
        headline_horizon_quarters=headline_horizon_quarters,
    )
    bank_detail_rows = build_league_rows(site_mart)
    bank_summary_rows = build_bank_summary_rows(site_mart)
    banks_dir = site_dir / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)

    for stale_file in banks_dir.glob("*.json"):
        stale_file.unlink()
    legacy_league = site_dir / "league.json"
    if legacy_league.exists():
        legacy_league.unlink()

    (site_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (banks_dir / "latest.json").write_text(json.dumps(bank_summary_rows, indent=2), encoding="utf-8")
    for row in bank_detail_rows:
        cert = row.get("cert")
        if cert is None:
            continue
        (banks_dir / f"{cert}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
