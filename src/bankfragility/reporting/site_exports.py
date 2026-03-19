"""Build publishable mart, split publishable panels, and site export artifacts."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def build_site_manifest(
    mart: pd.DataFrame,
    full_history_core: pd.DataFrame | None = None,
    recent_history_enriched: pd.DataFrame | None = None,
    validation_metrics: pd.DataFrame | None = None,
    headline_horizon_quarters: int = 4,
) -> dict[str, Any]:
    latest = latest_bank_snapshot(mart)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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
        "schema_version": 2,
        "generated_at": generated_at,
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

    rows: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        deposits = row.get("DEPDOM_EFFECTIVE", row.get("DEPDOM"))
        rows.append(
            {
                "cert": _json_safe(row.get("CERT")),
                "name": _json_safe(row.get("NAMEFULL")),
                "peer_group": _json_safe(row.get("PEER_GROUP")),
                "assets": _json_safe(row.get("ASSET")),
                "deposits": _json_safe(deposits),
                "uninsured_pct": _json_safe(row.get("UNINSURED_SHARE")),
                "state": _json_safe(row.get("STALP")),
                "charter": _json_safe(row.get("BKCLASS")),
                "repdte": _json_safe(row.get("REPDTE")),
                "run_risk": _json_safe(row.get("RUN_RISK_INDEX")),
                "alm_mismatch": _json_safe(row.get("ALM_MISMATCH_INDEX")),
                "treasury_buffer": _json_safe(row.get("TREASURY_BUFFER_INDEX")),
                "funding_fragility": _json_safe(row.get("FUNDING_FRAGILITY_INDEX")),
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
    league_rows = build_league_rows(site_mart)

    (site_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (site_dir / "league.json").write_text(json.dumps(league_rows, indent=2), encoding="utf-8")
