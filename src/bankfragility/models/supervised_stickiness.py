"""Experimental supervised deposit outflow overlay.

Trains a secondary model to predict severe *next-quarter* deposit weakness
using the same features as the transparent run-risk score. The transparent
score remains the primary product; the supervised layer is an experimental
overlay that produces a relative risk score, not a calibrated probability.

Training uses walk-forward out-of-time validation: for each quarter, the
model is trained on all prior quarters and predictions are made out-of-sample.
Rows without an observed next quarter are treated as censored and excluded
from supervised training labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from bankfragility.tables import read_table, save_table

# Features that are safe to use at prediction time (balance-sheet snapshot, no momentum leakage)
SNAPSHOT_FEATURES = [
    "UNINSURED_SHARE",
    "BROKERED_SHARE",
    "CORE_DEPOSIT_SHARE",
    "NONINTEREST_SHARE",
    "TIME_DEPOSIT_SHARE",
    "SHORT_FHLB_SHARE",
    "VOLATILE_TO_LIQUID_LOWER",
    "LOANS_TO_CORE_DEPOSITS",
    "DOMESTIC_DEPOSIT_COST",
]

# Momentum features — computed from deposit history up to time T.
# Label is T+1 outflow, so these are safe at prediction time.
MOMENTUM_FEATURES = [
    "DEP_GROWTH_VOL_4Q",
    "DEP_DRAWDOWN_4Q",
]

# SOD geographic concentration features
SOD_FEATURES = [
    "SOD_DEPOSIT_HHI_STATE",
    "SOD_DEPOSIT_HHI_COUNTY",
    "SOD_TOP_STATE_SHARE",
]

# Derivative hedge indicator
DERIVATIVE_FEATURES = [
    "HAS_IR_DERIVATIVES",
]

DEFAULT_FEATURES = SNAPSHOT_FEATURES + MOMENTUM_FEATURES + SOD_FEATURES + DERIVATIVE_FEATURES

FFIEC_FEATURES = [
    "DURATION_GAP_LITE",
    "LOAN_WAM_PROXY",
    "TD_WAM_PROXY",
]

MIN_TRAIN_QUARTERS = 4
MIN_TRAIN_EVENTS = 20
DEFAULT_LABEL_COL = "SEVERE_RELATIVE_OUTFLOW"

# Monotonic constraints: +1 = higher value → higher risk, -1 = lower risk, 0 = unconstrained
MONOTONE_CONSTRAINTS = {
    "UNINSURED_SHARE": 1,
    "BROKERED_SHARE": 1,
    "CORE_DEPOSIT_SHARE": 0,   # COREDEP doesn't exclude uninsured; not a clean stickiness proxy
    "NONINTEREST_SHARE": 0,    # Empirically ambiguous: higher NI share → higher outflow rate in data
    "TIME_DEPOSIT_SHARE": 0,
    "SHORT_FHLB_SHARE": 1,
    "VOLATILE_TO_LIQUID_LOWER": 1,
    "LOANS_TO_CORE_DEPOSITS": 1,
    "DOMESTIC_DEPOSIT_COST": 1,
    "DEP_GROWTH_VOL_4Q": 1,
    "DEP_DRAWDOWN_4Q": 1,
    "DURATION_GAP_LITE": 1,
    "LOAN_WAM_PROXY": 1,
    "TD_WAM_PROXY": -1,
    "SOD_DEPOSIT_HHI_STATE": 1,     # Higher concentration → more vulnerable
    "SOD_DEPOSIT_HHI_COUNTY": 1,
    "SOD_TOP_STATE_SHARE": 1,
    "HAS_IR_DERIVATIVES": 0,        # Unconstrained — having derivatives ≠ being hedged
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outflow-percentile", type=float, default=5.0)
    parser.add_argument("--walk-forward", action="store_true", default=True,
                        help="Use walk-forward out-of-time training (default)")
    parser.add_argument("--no-walk-forward", dest="walk_forward", action="store_false",
                        help="Train on all data at once (faster, but look-ahead bias)")
    parser.add_argument("--model", choices=["logistic", "gbm"], default="gbm",
                        help="Model type: logistic regression or monotonic GBM (default: gbm)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def compute_deposit_growth(df: pd.DataFrame) -> pd.Series:
    """Quarter-over-quarter domestic deposit growth: (DEPDOM_t - DEPDOM_{t-1}) / DEPDOM_{t-1}."""
    sorted_df = df.sort_values(["CERT", "REPDTE"])
    prev = sorted_df.groupby("CERT")["DEPDOM"].shift(1)
    return (sorted_df["DEPDOM"] - prev) / prev.replace(0, np.nan)


def compute_next_deposit_growth(df: pd.DataFrame) -> pd.Series:
    """Return next quarter's deposit growth aligned to the current row."""
    sorted_df = df.sort_values(["CERT", "REPDTE"]).copy()
    growth = compute_deposit_growth(sorted_df)
    sorted_df["_growth"] = growth.values
    next_growth = sorted_df.groupby("CERT")["_growth"].shift(-1)
    return next_growth.reindex(df.index)


def compute_next_quarter_observed(df: pd.DataFrame) -> pd.Series:
    """Flag rows where the bank has an observed consecutive next quarter."""
    sorted_df = df.sort_values(["CERT", "REPDTE"]).copy()
    sorted_df["REPDTE"] = pd.to_datetime(sorted_df["REPDTE"])
    next_repdte = sorted_df.groupby("CERT")["REPDTE"].shift(-1)
    expected_next = sorted_df["REPDTE"] + pd.offsets.QuarterEnd(1)
    observed = next_repdte.eq(expected_next)
    return observed.fillna(False).reindex(df.index)


def label_relative_outflow(
    next_growth: pd.Series,
    repdte: pd.Series,
    next_q_observed: pd.Series,
    percentile: float = 5.0,
) -> pd.Series:
    """Quarter-relative target for the weakest next-quarter growth observations."""
    labels = pd.Series(pd.NA, index=next_growth.index, dtype="Int64")
    observed_mask = next_q_observed.fillna(False) & next_growth.notna() & repdte.notna()
    if not observed_mask.any():
        return labels

    obs_df = pd.DataFrame(
        {"REPDTE": pd.to_datetime(repdte), "NEXT_DEP_GROWTH_QOQ": next_growth},
        index=next_growth.index,
    )

    def _label(group: pd.Series) -> pd.Series:
        thr = group.quantile(percentile / 100.0)
        return (group <= thr).astype("Int64")

    labels.loc[observed_mask] = (
        obs_df.loc[observed_mask]
        .groupby("REPDTE")["NEXT_DEP_GROWTH_QOQ"]
        .transform(_label)
        .astype("Int64")
    )
    return labels


def label_absolute_outflow(
    next_growth: pd.Series,
    repdte: pd.Series,
    next_q_observed: pd.Series,
    percentile: float = 5.0,
) -> pd.Series:
    """Quarter-relative severe weakness constrained to true negative deposit contraction."""
    relative = label_relative_outflow(next_growth, repdte, next_q_observed, percentile=percentile)
    out = pd.Series(pd.NA, index=next_growth.index, dtype="Int64")
    observed_mask = next_q_observed.fillna(False) & next_growth.notna()
    out.loc[observed_mask] = (
        (relative.loc[observed_mask].fillna(0).astype(int) == 1)
        & (next_growth.loc[observed_mask] < 0)
    ).astype("Int64")
    return out


def select_features(df: pd.DataFrame) -> list[str]:
    candidates = DEFAULT_FEATURES + FFIEC_FEATURES
    useful = []
    for col in candidates:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 10 and vals.std() > 0:
            useful.append(col)
    return useful


def _fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[Any, StandardScaler]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, l1_ratio=0, max_iter=1000,
                               class_weight="balanced", solver="lbfgs")
    model.fit(X_s, y)
    return model, scaler


def _fit_gbm(X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> tuple[Any, None]:
    if not _HAS_LIGHTGBM:
        raise ImportError("lightgbm required for GBM model. Install with: pip install lightgbm")
    constraints = [MONOTONE_CONSTRAINTS.get(f, 0) for f in feature_cols]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale = n_neg / max(n_pos, 1)
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        monotone_constraints=constraints,
        scale_pos_weight=scale,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
    )
    model.fit(X, y)
    return model, None  # GBM doesn't need a scaler


def _fit_model(X: np.ndarray, y: np.ndarray, feature_cols: list[str],
               model_type: str = "gbm") -> tuple[Any, StandardScaler | None]:
    if model_type == "gbm" and _HAS_LIGHTGBM:
        return _fit_gbm(X, y, feature_cols)
    return _fit_logistic(X, y)


def _impute_median(df: pd.DataFrame, feature_cols: list[str], train_medians: dict[str, float] | None = None) -> pd.DataFrame:
    """Impute missing values with medians. Uses train medians if provided to avoid leakage."""
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median = train_medians[col] if train_medians and col in train_medians else X[col].median()
        X[col] = X[col].fillna(median)
    return X


def _predict(df: pd.DataFrame, model: Any, scaler: StandardScaler | None,
             feature_cols: list[str], train_medians: dict[str, float] | None = None) -> np.ndarray:
    X = _impute_median(df, feature_cols, train_medians)
    X_arr = X.values.astype(float)
    if scaler is not None:
        X_arr = scaler.transform(X_arr)
    if hasattr(model, "booster_"):
        X_pred = pd.DataFrame(X_arr, columns=feature_cols)
    else:
        X_pred = X_arr
    return model.predict_proba(X_pred)[:, 1]


def _get_importances(model: Any, feature_cols: list[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)
    return pd.Series(dtype=float)


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str],
                label_col: str = DEFAULT_LABEL_COL) -> tuple[np.ndarray, np.ndarray, pd.Index, dict[str, float]]:
    """Prepare training data. Impute features with median (same approach as prediction).

    Returns X, y, index, and the median dict used for imputation (to pass to _predict).
    """
    has_label = df[label_col].notna()
    subset = df[has_label].copy()
    medians: dict[str, float] = {}
    for col in feature_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        med = float(subset[col].median()) if subset[col].notna().any() else 0.0
        medians[col] = med
        subset[col] = subset[col].fillna(med)
    X = subset[feature_cols].values.astype(float)
    y = subset[label_col].values.astype(int)
    return X, y, subset.index, medians


# ---------------------------------------------------------------------------
# Walk-forward out-of-time training
# ---------------------------------------------------------------------------

def walk_forward_train(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = DEFAULT_LABEL_COL,
    model_type: str = "gbm",
) -> pd.DataFrame:
    """For each quarter, train on all prior quarters, predict current quarter."""
    out = df.copy()
    out["SUPERVISED_OUTFLOW_SCORE"] = np.nan
    out["SUPERVISED_TRAIN_QUARTERS"] = np.nan

    quarters = sorted(out["REPDTE"].unique())
    auc_by_quarter: list[dict[str, Any]] = []
    last_model = None
    last_scaler = None
    last_medians: dict[str, float] | None = None

    for i, q in enumerate(quarters):
        train_mask = out["REPDTE"] < q
        test_mask = out["REPDTE"] == q
        train_df = out[train_mask]
        test_df = out[test_mask]

        n_train_q = train_df["REPDTE"].nunique()
        n_train_events = train_df[label_col].sum() if label_col in train_df.columns else 0

        if n_train_q < MIN_TRAIN_QUARTERS or n_train_events < MIN_TRAIN_EVENTS:
            if last_model is not None:
                probs = _predict(test_df, last_model, last_scaler, feature_cols, last_medians)
                out.loc[test_mask, "SUPERVISED_OUTFLOW_SCORE"] = probs
                out.loc[test_mask, "SUPERVISED_TRAIN_QUARTERS"] = n_train_q
            continue

        X_train, y_train, _, train_medians = _prepare_xy(train_df, feature_cols, label_col)
        if len(np.unique(y_train)) < 2:
            continue

        model, scaler = _fit_model(X_train, y_train, feature_cols, model_type)
        last_model = model
        last_scaler = scaler
        last_medians = train_medians

        probs = _predict(test_df, model, scaler, feature_cols, train_medians)
        out.loc[test_mask, "SUPERVISED_OUTFLOW_SCORE"] = probs
        out.loc[test_mask, "SUPERVISED_TRAIN_QUARTERS"] = n_train_q

        # Compute OOT AUC if we have both classes in test
        test_labels = test_df[label_col].dropna()
        test_probs = out.loc[test_mask, "SUPERVISED_OUTFLOW_SCORE"]
        valid = test_labels.notna() & test_probs.notna()
        if valid.sum() > 10 and test_labels[valid].nunique() == 2:
            auc = roc_auc_score(test_labels[valid], test_probs[valid])
            auc_by_quarter.append({"quarter": q, "auc": auc, "n_test": int(valid.sum()),
                                   "n_train_q": n_train_q, "n_events": int(n_train_events)})

    # Store walk-forward AUC metrics
    if auc_by_quarter:
        auc_df = pd.DataFrame(auc_by_quarter)
        mean_auc = auc_df["auc"].mean()
        out["SUPERVISED_OOT_AUC_MEAN"] = mean_auc
        print(f"  Walk-forward AUC: {mean_auc:.4f} (across {len(auc_df)} quarters)", file=sys.stderr)
        for _, row in auc_df.iterrows():
            q_str = pd.Timestamp(row["quarter"]).strftime("%Y-%m-%d")
            print(f"    {q_str}: AUC={row['auc']:.4f} (n={row['n_test']}, "
                  f"train_q={row['n_train_q']}, events={row['n_events']})", file=sys.stderr)

    # Store feature importances from the final model
    if last_model is not None:
        importance = _get_importances(last_model, feature_cols)
        for i, (feat, imp) in enumerate(importance.items()):
            out[f"SUPERVISED_IMPORTANCE_{i + 1}"] = feat
            out[f"SUPERVISED_COEF_{i + 1}"] = float(imp)

    out["SUPERVISED_MODEL_TYPE"] = model_type
    return out


# ---------------------------------------------------------------------------
# Full-sample training (legacy, faster)
# ---------------------------------------------------------------------------

def full_sample_train(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = DEFAULT_LABEL_COL,
    model_type: str = "gbm",
) -> pd.DataFrame:
    out = df.copy()
    X, y, idx, medians = _prepare_xy(out, feature_cols, label_col)
    if len(np.unique(y)) < 2 or len(y) < 50:
        out["SUPERVISED_OUTFLOW_SCORE"] = np.nan
        return out

    model, scaler = _fit_model(X, y, feature_cols, model_type)
    out["SUPERVISED_OUTFLOW_SCORE"] = _predict(out, model, scaler, feature_cols, medians)

    importance = _get_importances(model, feature_cols)
    for i, (feat, imp) in enumerate(importance.items()):
        out[f"SUPERVISED_IMPORTANCE_{i + 1}"] = feat
        out[f"SUPERVISED_COEF_{i + 1}"] = float(imp)

    out["SUPERVISED_MODEL_TYPE"] = model_type
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_supervised_overlay(
    df: pd.DataFrame,
    outflow_percentile: float = 5.0,
    walk_forward: bool = True,
    model_type: str = "gbm",
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    out = df.copy()
    out["REPDTE"] = pd.to_datetime(out["REPDTE"])
    out["DEPDOM"] = pd.to_numeric(out["DEPDOM"], errors="coerce")

    out["NEXT_DEP_GROWTH_QOQ"] = compute_next_deposit_growth(out)
    out["NEXT_Q_OBSERVED"] = compute_next_quarter_observed(out)
    out["SEVERE_RELATIVE_OUTFLOW"] = label_relative_outflow(
        out["NEXT_DEP_GROWTH_QOQ"],
        out["REPDTE"],
        out["NEXT_Q_OBSERVED"],
        percentile=outflow_percentile,
    )
    out["SEVERE_ABSOLUTE_OUTFLOW"] = label_absolute_outflow(
        out["NEXT_DEP_GROWTH_QOQ"],
        out["REPDTE"],
        out["NEXT_Q_OBSERVED"],
        percentile=outflow_percentile,
    )
    out["DEP_GROWTH_QOQ"] = compute_deposit_growth(out)

    feature_cols = select_features(out)
    if len(feature_cols) < 3:
        out["SUPERVISED_OUTFLOW_SCORE"] = np.nan
        out["SUPERVISED_RISK_SCORE"] = np.nan
        return out

    # Fall back to logistic if lightgbm not available
    effective_model = model_type
    if model_type == "gbm" and not _HAS_LIGHTGBM:
        print("  lightgbm not available, falling back to logistic regression", file=sys.stderr)
        effective_model = "logistic"

    if walk_forward:
        out = walk_forward_train(out, feature_cols, label_col=label_col, model_type=effective_model)
    else:
        out = full_sample_train(out, feature_cols, label_col=label_col, model_type=effective_model)

    out["SUPERVISED_LABEL_KIND"] = label_col
    out["SUPERVISED_RISK_SCORE"] = (out["SUPERVISED_OUTFLOW_SCORE"] * 100).clip(0, 100)
    return out


def run_supervised(args: argparse.Namespace) -> pd.DataFrame:
    df = read_table(args.input)
    out = build_supervised_overlay(df, outflow_percentile=args.outflow_percentile,
                                   walk_forward=args.walk_forward,
                                   model_type=args.model)
    save_table(out, args.out)
    n_severe = int(out["SEVERE_RELATIVE_OUTFLOW"].fillna(0).sum())
    n_total = int(out["NEXT_Q_OBSERVED"].fillna(False).sum())
    print(
        f"Supervised model: {n_severe}/{n_total} observed next-quarter severe-relative-outflow labels. "
        f"Saved to {args.out}",
        file=sys.stderr,
    )
    return out


def main() -> None:
    args = parse_args()
    run_supervised(args)


if __name__ == "__main__":
    main()
