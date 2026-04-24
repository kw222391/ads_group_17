#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three XGBoost models for LAD-level homelessness:
1) Growth-from-lag1 XGB
2) Rate model XGB
3) XGB + lag1 blend

This script is designed for PyCharm.

Expected input file in the same folder as this script:
    monthly_lad_panel_2000_2025_with_homelessness_2000_2025.csv

Main output folder:
    three_xgb_homelessness_outputs/

Core idea:
    Homelessness counts are very persistent, so a plain XGBoost count model is usually
    dominated by lag-1 homelessness and population scale. This script therefore:
    - uses a growth target for the main model:
          log1p(homelessness_t) - log1p(homelessness_t_minus_1)
    - limits direct lag-1 count features inside XGBoost
    - uses a rate target as a robustness check:
          log1p(homelessness_per_1000_population)
    - blends XGB growth predictions with the lag-1 baseline using validation-tuned weights

Install:
    pip install pandas numpy matplotlib scikit-learn xgboost joblib
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError(
        "xgboost is not installed. Please run: pip install xgboost"
    ) from e


# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "monthly_lad_panel_2000_2025_with_homelessness_2000_2025.csv"
OUTPUT_DIR = "three_xgb_homelessness_outputs"

# Homelessness Reduction Act implementation changed measurement framework.
# The main model uses post-HRA data only.
MODEL_START_QUARTER = "2019Q3"

# Fixed split. If a model has no 2025 test rows, the code uses a fallback split.
TRAIN_END_QUARTER = "2023Q4"
VALID_START_QUARTER = "2024Q1"
VALID_END_QUARTER = "2024Q4"
TEST_START_QUARTER = "2025Q1"

# Lag settings
LIVING_COST_LAGS = [1, 2, 4, 8]
TARGET_HISTORY_LAGS = [1, 2, 4, 8]

# This is the key switch to reduce lag-1 dominance inside the growth model.
# Lag1 is still used outside the model as the baseline / target transformation,
# but direct lag1 count features are not given to XGBoost by default.
EXCLUDE_DIRECT_LAG1_COUNT_FEATURES_IN_GROWTH_XGB = True

# Rate model uses rate lag-1 because this is a risk-persistence model, but it
# does not need raw count lag1 as a main driver.
EXCLUDE_RAW_COUNT_HISTORY_IN_RATE_MODEL = True

# Feature selection
USE_FEATURE_SELECTION = True
MAX_FEATURES_GROWTH = 160
MAX_FEATURES_RATE = 160
MIN_NON_MISSING_FRAC = 0.65
FORCE_KEEP_CPI_TOTAL_FEATURES = True
FORCE_KEEP_KEY_FEATURES = True

# Blend settings. Use constrained blends to prevent lag1 from taking almost all weight.
# w is XGB weight in: pred_blend = w * pred_xgb + (1 - w) * pred_lag1
BLEND_WEIGHT_GRID = np.round(np.arange(0.00, 1.001, 0.01), 2)
CONSTRAINED_BLEND_MIN_XGB_WEIGHT = 0.50

# XGBoost parameters
XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=1600,
    learning_rate=0.025,
    max_depth=3,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=3.0,
    tree_method="hist",
    random_state=42,
    n_jobs=4,
)

EARLY_STOPPING_ROUNDS = 80

# Plots
SHOW_PLOTS = True
SAVE_PLOTS = True

# Safety
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.copy_on_write = False


# =============================================================================
# Utility functions
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def find_input_file() -> Path:
    candidates = [
        Path.cwd() / INPUT_FILE,
        Path(__file__).resolve().parent / INPUT_FILE,
        Path("/mnt/data") / INPUT_FILE,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. Put it in the same folder as this script."
    )


def to_period(q: pd.Series | str) -> pd.Series | pd.Period:
    if isinstance(q, str):
        return pd.Period(q, freq="Q")
    return pd.PeriodIndex(q.astype(str), freq="Q")


def safe_log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def expm1_clip(x: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(x), 0, None)


def metrics_count_scale(
    actual: np.ndarray,
    pred: np.ndarray,
    model: str,
    split_or_model: str,
    extra: Optional[Dict] = None,
) -> Dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]

    if len(actual) == 0:
        out = dict(
            model=model,
            split_or_model=split_or_model,
            n=0,
            MAE=np.nan,
            RMSE=np.nan,
            R2=np.nan,
            SMAPE_percent=np.nan,
            mean_actual=np.nan,
            mean_predicted=np.nan,
            mean_bias_actual_minus_predicted=np.nan,
        )
        if extra:
            out.update(extra)
        return out

    mae = mean_absolute_error(actual, pred)
    rmse = math.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred) if len(np.unique(actual)) > 1 else np.nan
    smape = np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-9)) * 100

    out = dict(
        model=model,
        split_or_model=split_or_model,
        n=len(actual),
        MAE=mae,
        RMSE=rmse,
        R2=r2,
        SMAPE_percent=smape,
        mean_actual=float(np.mean(actual)),
        mean_predicted=float(np.mean(pred)),
        mean_bias_actual_minus_predicted=float(np.mean(actual - pred)),
    )
    if extra:
        out.update(extra)
    return out


def group_feature(feature: str) -> str:
    f = feature.lower()

    # Order matters: brent contains "rent" as characters, so energy must come before rent.
    if f.startswith("lad_code_"):
        return "LAD fixed effect"
    if f in {"year", "quarter_num", "quarter_index"} or "quarter_" in f or "post_2018" in f:
        return "time / seasonality"
    if "homelessness" in f:
        return "target history"
    if "cpi_00_all_items" in f:
        return "CPI total"
    if "brent" in f or "oil" in f or "energy" in f:
        return "oil / energy proxy"
    if (
        "private_rental" in f
        or "rental_price" in f
        or "average_private_rent" in f
        or "average_private_rental" in f
        or "annual_rent_to_income" in f
        or "rent_to_income" in f
    ):
        return "rent"
    if "house_price" in f or "house_sales" in f or "housing" in f or "hpi" in f:
        return "house price / housing"
    if "income" in f or "affordability" in f:
        return "income / affordability"
    if "unemployment" in f or "claimant" in f:
        return "unemployment"
    if "bank_rate" in f or "interest" in f:
        return "interest rate"
    if "gbp" in f or "ftse" in f or "exchange" in f:
        return "market / FX"
    if "migration" in f:
        return "migration"
    if "population" in f:
        return "population / scale"
    return "other controls"


def make_feature_importance(
    model: XGBRegressor,
    feature_names: List[str],
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    booster = model.get_booster()

    def score_df(importance_type: str) -> pd.DataFrame:
        scores = booster.get_score(importance_type=importance_type)
        rows = []
        for k, v in scores.items():
            if k.startswith("f"):
                try:
                    idx = int(k[1:])
                    feature = feature_names[idx]
                except Exception:
                    feature = k
            else:
                feature = k
            rows.append({"feature": feature, importance_type: float(v)})
        return pd.DataFrame(rows)

    gain = score_df("gain")
    weight = score_df("weight")
    cover = score_df("cover")

    if gain.empty:
        fi = pd.DataFrame(columns=["model", "feature", "group", "gain", "weight", "cover", "gain_share"])
        gi = pd.DataFrame(columns=["model", "group", "gain", "weight", "n_features_used", "gain_share"])
        return fi, gi

    fi = gain.merge(weight, on="feature", how="outer").merge(cover, on="feature", how="outer")
    fi["model"] = model_name
    fi["group"] = fi["feature"].apply(group_feature)
    fi[["gain", "weight", "cover"]] = fi[["gain", "weight", "cover"]].fillna(0.0)
    total_gain = fi["gain"].sum()
    fi["gain_share"] = fi["gain"] / total_gain if total_gain > 0 else 0.0
    fi = fi[["model", "feature", "group", "gain", "weight", "cover", "gain_share"]]
    fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)

    gi = (
        fi.groupby(["model", "group"], as_index=False)
        .agg(
            gain=("gain", "sum"),
            weight=("weight", "sum"),
            n_features_used=("feature", "nunique"),
        )
        .sort_values("gain", ascending=False)
    )
    gi["gain_share"] = gi["gain"] / gi["gain"].sum() if gi["gain"].sum() > 0 else 0.0
    gi = gi.reset_index(drop=True)
    return fi, gi


def plot_actual_vs_pred(df: pd.DataFrame, pred_col: str, title: str, out_path: Optional[Path] = None) -> None:
    plot_df = df[df["split"].eq("test")].copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(plot_df["actual_count"], plot_df[pred_col], alpha=0.45)
    max_v = np.nanmax([plot_df["actual_count"].max(), plot_df[pred_col].max()])
    plt.plot([0, max_v], [0, max_v], linestyle="--")
    plt.xlabel("Actual homelessness assessments")
    plt.ylabel("Predicted homelessness assessments")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")


def plot_england_aggregate(df: pd.DataFrame, pred_cols: List[str], title: str, out_path: Optional[Path] = None) -> None:
    plot_df = df[df["split"].isin(["validation", "test"])].copy()
    if plot_df.empty:
        return
    cols = ["actual_count"] + pred_cols
    agg = plot_df.groupby("quarter")[cols].sum(min_count=1).reset_index()
    agg = agg.sort_values("quarter")
    plt.figure(figsize=(11, 5))
    plt.plot(agg["quarter"].astype(str), agg["actual_count"], marker="o", label="Actual")
    for c in pred_cols:
        if c in agg.columns:
            plt.plot(agg["quarter"].astype(str), agg[c], marker="o", label=c)
    plt.xticks(rotation=45)
    plt.ylabel("England sum of LAD homelessness assessments")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")


def plot_feature_group_importance(gi: pd.DataFrame, title: str, out_path: Optional[Path] = None) -> None:
    if gi.empty:
        return
    plot_df = gi.sort_values("gain_share", ascending=True).tail(15)
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["group"], plot_df["gain_share"])
    plt.xlabel("Gain share")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")


# =============================================================================
# Data preparation
# =============================================================================

def load_and_make_quarterly() -> pd.DataFrame:
    section("Loading monthly data and aggregating to LAD-quarter panel")
    input_path = find_input_file()
    print(f"Input file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Raw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns):,}")

    # Keep real LADs. Exclude England aggregate and other non-LAD geography rows.
    df["lad_code"] = df["lad_code"].astype(str)
    real_lad_mask = df["lad_code"].str.startswith(("E06", "E07", "E08", "E09"), na=False)
    df = df.loc[real_lad_mask].copy()
    print(f"Rows after keeping real LADs: {len(df):,}")
    print(f"LADs: {df['lad_code'].nunique():,}")

    # Keep CPI total only, drop all other CPI category columns.
    cpi_cols = [c for c in df.columns if c.lower().startswith("cpi_")]
    drop_cpi = [c for c in cpi_cols if c != "cpi_00_all_items"]
    if drop_cpi:
        df = df.drop(columns=drop_cpi)
    print(f"CPI columns kept: {[c for c in df.columns if c.lower().startswith('cpi_')]}")

    # Standardize rent column name if needed.
    if "average_private_rental_price" in df.columns and "average_private_rent" not in df.columns:
        df = df.rename(columns={"average_private_rental_price": "average_private_rent"})

    # Date / quarter
    if "year" not in df.columns or "month" not in df.columns:
        raise ValueError("Input must contain year and month columns.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["date"] = pd.to_datetime(
        dict(year=df["year"].astype("Int64"), month=df["month"].astype("Int64"), day=1),
        errors="coerce",
    )
    df["quarter_period"] = df["date"].dt.to_period("Q")
    df["quarter"] = df["quarter_period"].astype(str)

    # Convert object columns that should be numeric.
    id_cols = {"lad_code", "lad_name", "quarter", "quarter_period", "date"}
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # Aggregate to quarterly. For most monthly features, mean is appropriate.
    # Homelessness values are quarterly values merged into monthly data; mean avoids triple-counting.
    numeric_cols = [c for c in df.columns if c not in ["lad_code", "lad_name", "date", "quarter", "quarter_period"]
                    and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(["lad_code", "quarter_period"], as_index=False)[numeric_cols].mean()
    names = (
        df.sort_values(["lad_code", "date"])
        .groupby(["lad_code", "quarter_period"])["lad_name"]
        .last()
        .reset_index()
    )
    qdf = agg.merge(names, on=["lad_code", "quarter_period"], how="left")
    qdf["quarter"] = qdf["quarter_period"].astype(str)
    qdf["quarter_date"] = qdf["quarter_period"].dt.start_time
    qdf["year"] = qdf["quarter_period"].dt.year
    qdf["quarter_num"] = qdf["quarter_period"].dt.quarter

    qdf = qdf.sort_values(["lad_code", "quarter_period"]).reset_index(drop=True)
    qdf["quarter_index"] = (qdf["year"] - qdf["year"].min()) * 4 + qdf["quarter_num"]
    qdf["post_2018_HRA"] = (qdf["quarter_period"] >= pd.Period("2018Q2", freq="Q")).astype(int)

    print(f"Quarterly rows: {len(qdf):,}")
    print(f"Quarter range: {qdf['quarter'].min()} to {qdf['quarter'].max()}")
    if "homelessness_total_assessments" in qdf.columns:
        print(
            "Rows with non-missing homelessness_total_assessments: "
            f"{qdf['homelessness_total_assessments'].notna().sum():,}"
        )
    else:
        raise ValueError("Input must contain homelessness_total_assessments.")

    return qdf


def add_features(qdf: pd.DataFrame) -> pd.DataFrame:
    section("Feature engineering")

    df = qdf.copy()
    df = df.sort_values(["lad_code", "quarter_period"]).reset_index(drop=True)

    # Derived living-cost and affordability variables.
    if "cpi_00_all_items" in df.columns:
        cpi = df["cpi_00_all_items"].replace(0, np.nan)
        if "income" in df.columns:
            df["real_income_cpi_adjusted"] = df["income"] / cpi * 100
        if "average_house_price" in df.columns:
            df["real_house_price_cpi_adjusted"] = df["average_house_price"] / cpi * 100
        if "average_private_rent" in df.columns:
            df["real_private_rent_cpi_adjusted"] = df["average_private_rent"] / cpi * 100

    if {"average_house_price", "income"}.issubset(df.columns):
        df["house_price_to_income"] = df["average_house_price"] / df["income"].replace(0, np.nan)

    if {"average_private_rent", "income"}.issubset(df.columns):
        df["annual_rent_to_income"] = (df["average_private_rent"] * 12) / df["income"].replace(0, np.nan)

    if {"unemployment_count", "population"}.issubset(df.columns):
        df["unemployment_per_1000"] = df["unemployment_count"] / df["population"].replace(0, np.nan) * 1000

    if {"homelessness_total_assessments", "population"}.issubset(df.columns):
        df["homelessness_rate_per_1000"] = (
            df["homelessness_total_assessments"] / df["population"].replace(0, np.nan) * 1000
        )

    # Candidate non-homelessness base features.
    forbidden_same_quarter_homelessness = [
        c for c in df.columns
        if c.startswith("homelessness_") and not any(s in c for s in ["lag", "rolling"])
    ]

    id_like = {
        "lad_code", "lad_name", "quarter", "quarter_period", "quarter_date",
        "date", "month"
    }

    base_numeric = [
        c for c in df.columns
        if c not in id_like
        and c not in forbidden_same_quarter_homelessness
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Drop other CPI categories if any survived.
    base_numeric = [
        c for c in base_numeric
        if not (c.lower().startswith("cpi_") and c != "cpi_00_all_items")
    ]

    # Add qoq/yoy percent change for non-homelessness features.
    for c in list(base_numeric):
        if c in ["year", "quarter_num", "quarter_index", "post_2018_HRA"]:
            continue
        by = df.groupby("lad_code")[c]
        df[f"{c}_qoq_pct"] = by.pct_change(1) * 100
        df[f"{c}_yoy_pct"] = by.pct_change(4) * 100

    # Recompute base after adding changes.
    all_non_homeless_numeric = [
        c for c in df.columns
        if c not in id_like
        and not c.startswith("homelessness_")
        and pd.api.types.is_numeric_dtype(df[c])
        and not (c.lower().startswith("cpi_") and c != "cpi_00_all_items")
    ]

    # Create lagged living-cost / economic features.
    created_lag_cols = []
    for c in all_non_homeless_numeric:
        # Do not lag pure time dummies/indices except macro features.
        if c in ["year", "quarter_num", "quarter_index", "post_2018_HRA"]:
            continue
        for lag in LIVING_COST_LAGS:
            new_c = f"{c}_lag{lag}q"
            df[new_c] = df.groupby("lad_code")[c].shift(lag)
            created_lag_cols.append(new_c)

    # Target-history features.
    target = "homelessness_total_assessments"
    if target in df.columns:
        df["log1p_homelessness_total_assessments"] = safe_log1p(df[target])
        for lag in TARGET_HISTORY_LAGS:
            df[f"{target}_lag{lag}q"] = df.groupby("lad_code")[target].shift(lag)
            df[f"log1p_{target}_lag{lag}q"] = df.groupby("lad_code")[
                "log1p_homelessness_total_assessments"
            ].shift(lag)

        # Rolling target history using only past values.
        shifted = df.groupby("lad_code")[target].shift(1)
        df["homelessness_total_assessments_rolling4_mean_lag1q"] = (
            shifted.groupby(df["lad_code"]).rolling(4, min_periods=2).mean().reset_index(level=0, drop=True)
        )
        df["homelessness_total_assessments_rolling4_std_lag1q"] = (
            shifted.groupby(df["lad_code"]).rolling(4, min_periods=3).std().reset_index(level=0, drop=True)
        )

        # Change history
        df["homelessness_log_growth_lag1q"] = (
            df.groupby("lad_code")["log1p_homelessness_total_assessments"].diff(1).shift(1)
        )
        df["homelessness_log_growth_lag4q"] = (
            df.groupby("lad_code")["log1p_homelessness_total_assessments"].diff(4).shift(1)
        )

    if "homelessness_rate_per_1000" in df.columns:
        df["log1p_homelessness_rate_per_1000"] = safe_log1p(df["homelessness_rate_per_1000"])
        for lag in TARGET_HISTORY_LAGS:
            df[f"homelessness_rate_per_1000_lag{lag}q"] = df.groupby("lad_code")[
                "homelessness_rate_per_1000"
            ].shift(lag)
            df[f"log1p_homelessness_rate_per_1000_lag{lag}q"] = df.groupby("lad_code")[
                "log1p_homelessness_rate_per_1000"
            ].shift(lag)

    # Replace inf from pct change.
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"Created lagged living-cost/economic features: {len(created_lag_cols):,}")
    target_hist_cols = [c for c in df.columns if "homelessness" in c and ("lag" in c or "rolling" in c)]
    print(f"Created target-history features: {len(target_hist_cols):,}")

    return df


# =============================================================================
# Feature construction and selection
# =============================================================================

def add_lad_dummies(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["lad_code"], prefix="lad_code", dtype=np.uint8)
    return pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def base_exclusion_columns() -> set:
    return {
        "lad_code",
        "lad_name",
        "quarter",
        "quarter_period",
        "quarter_date",
        "date",
        "month",
        # current same-quarter target columns must not be features
        "homelessness_total_assessments",
        "log1p_homelessness_total_assessments",
        "homelessness_rate_per_1000",
        "log1p_homelessness_rate_per_1000",
        "homelessness_total_owed",
        "homelessness_threatened",
        "homelessness_relief",
        "homelessness_per_1000",
    }


def get_feature_columns(
    df: pd.DataFrame,
    model_kind: str,
) -> List[str]:
    exclude = base_exclusion_columns()

    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # CPI safety: keep only total CPI columns and derived total-CPI columns.
    numeric_cols = [
        c for c in numeric_cols
        if not (c.lower().startswith("cpi_") and not c.lower().startswith("cpi_00_all_items"))
    ]

    # No same-quarter homelessness features.
    numeric_cols = [
        c for c in numeric_cols
        if not (
            c.startswith("homelessness_")
            and ("lag" not in c)
            and ("rolling" not in c)
            and ("growth" not in c)
        )
    ]

    if model_kind == "growth":
        if EXCLUDE_DIRECT_LAG1_COUNT_FEATURES_IN_GROWTH_XGB:
            # The lag1 count level is already used outside the model as the baseline.
            # Excluding it from XGB reduces feature-importance domination by lag1.
            remove_patterns = [
                "homelessness_total_assessments_lag1q",
                "log1p_homelessness_total_assessments_lag1q",
            ]
            numeric_cols = [c for c in numeric_cols if c not in remove_patterns]

    if model_kind == "rate":
        if EXCLUDE_RAW_COUNT_HISTORY_IN_RATE_MODEL:
            numeric_cols = [
                c for c in numeric_cols
                if not (
                    ("homelessness_total_assessments_lag" in c)
                    or ("log1p_homelessness_total_assessments_lag" in c)
                    or ("homelessness_total_assessments_rolling" in c)
                )
            ]

    return numeric_cols


def split_fixed_or_fallback(model_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.copy()
    q = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")

    train_end = pd.Period(TRAIN_END_QUARTER, freq="Q")
    valid_start = pd.Period(VALID_START_QUARTER, freq="Q")
    valid_end = pd.Period(VALID_END_QUARTER, freq="Q")
    test_start = pd.Period(TEST_START_QUARTER, freq="Q")

    df["split"] = np.where(
        q <= train_end,
        "train",
        np.where((q >= valid_start) & (q <= valid_end), "validation", np.where(q >= test_start, "test", "drop")),
    )

    if (df["split"].eq("test").sum() >= 50) and (df["split"].eq("validation").sum() >= 50):
        return df[df["split"].ne("drop")].copy()

    # Fallback: last available quarter = test; previous 4 quarters = validation; rest = train.
    available = sorted(pd.PeriodIndex(df["quarter"].dropna().astype(str).unique(), freq="Q"))
    if len(available) < 8:
        raise ValueError("Not enough quarters for fallback split.")

    test_q = available[-1]
    valid_qs = set(available[-5:-1])
    df["split"] = "train"
    df.loc[pd.PeriodIndex(df["quarter"].astype(str), freq="Q").isin(valid_qs), "split"] = "validation"
    df.loc[pd.PeriodIndex(df["quarter"].astype(str), freq="Q") == test_q, "split"] = "test"

    print("\nWarning: fixed 2025 test split was empty or too small.")
    print("Using fallback split:")
    print(f"  test quarter: {test_q}")
    print(f"  validation quarters: {[str(x) for x in sorted(valid_qs)]}")

    return df


def prepare_model_table(
    df: pd.DataFrame,
    model_kind: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if "lad_code_" not in " ".join(df.columns[:0]):
        pass

    data = df.copy()
    data = data[data["quarter_period"] >= pd.Period(MODEL_START_QUARTER, freq="Q")].copy()

    if model_kind == "growth":
        required = [
            "homelessness_total_assessments",
            "homelessness_total_assessments_lag1q",
            "log1p_homelessness_total_assessments_lag1q",
        ]
        for r in required:
            if r not in data.columns:
                raise ValueError(f"Missing required column for growth model: {r}")
        data["model_target"] = (
            safe_log1p(data["homelessness_total_assessments"])
            - data["log1p_homelessness_total_assessments_lag1q"].astype(float)
        )
        data = data.dropna(subset=["model_target", "homelessness_total_assessments_lag1q"])

    elif model_kind == "rate":
        required = [
            "homelessness_total_assessments",
            "population",
            "homelessness_rate_per_1000",
        ]
        for r in required:
            if r not in data.columns:
                raise ValueError(f"Missing required column for rate model: {r}")
        data["model_target"] = safe_log1p(data["homelessness_rate_per_1000"])
        data = data.dropna(subset=["model_target", "homelessness_total_assessments", "population"])
        data = data[data["population"] > 0].copy()

    else:
        raise ValueError("model_kind must be either 'growth' or 'rate'.")

    data = add_lad_dummies(data)
    feature_cols = get_feature_columns(data, model_kind=model_kind)

    # Drop rows without actual count.
    data = data.dropna(subset=["homelessness_total_assessments"]).copy()
    data = split_fixed_or_fallback(data)

    # Drop features with too much missingness across the model data.
    non_missing_frac = data[feature_cols].notna().mean()
    feature_cols = [c for c in feature_cols if non_missing_frac.get(c, 0) >= MIN_NON_MISSING_FRAC]

    # Drop constant / near-constant features using train only.
    train_data = data[data["split"].eq("train")]
    nunique = train_data[feature_cols].nunique(dropna=True)
    feature_cols = [c for c in feature_cols if nunique.get(c, 0) > 1]

    return data, feature_cols


def select_features(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_features: int,
    model_name: str,
) -> Tuple[List[str], pd.DataFrame]:
    if not USE_FEATURE_SELECTION or len(feature_cols) <= max_features:
        selected = feature_cols
        sel_info = pd.DataFrame({
            "model": model_name,
            "feature": selected,
            "selection_reason": "all_features_kept",
            "score": np.nan,
            "group": [group_feature(f) for f in selected],
        })
        return selected, sel_info

    train = model_df[model_df["split"].eq("train")].copy()
    X_train = train[feature_cols].copy()
    y_train = train[target_col].astype(float).copy()

    # Impute just for scoring.
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)

    # Scale not required for f_regression, but it makes numeric stability better.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp)

    scores, pvals = f_regression(X_scaled, y_train)
    score_df = pd.DataFrame({
        "feature": feature_cols,
        "score": np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
        "pvalue": pvals,
    })
    score_df["group"] = score_df["feature"].apply(group_feature)
    score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

    selected = list(score_df.head(max_features)["feature"])

    forced = []
    if FORCE_KEEP_CPI_TOTAL_FEATURES:
        forced += [c for c in feature_cols if "cpi_00_all_items" in c]

    if FORCE_KEEP_KEY_FEATURES:
        key_patterns = [
            "average_private_rent",
            "private_rental_price_index",
            "annual_rent_to_income",
            "house_price_to_income",
            "real_income_cpi_adjusted",
            "unemployment_per_1000",
            "uk_bank_rate",
            "brent_oil_price",
            "quarter_index",
        ]
        forced += [c for c in feature_cols if any(p in c for p in key_patterns)]

    # Keep forced features even if not top-scored, then trim only non-forced if too large.
    forced = sorted(set(forced))
    selected = list(dict.fromkeys(selected + forced))

    if len(selected) > max_features + len(forced):
        selected = selected[:max_features + len(forced)]

    sel_info = score_df[score_df["feature"].isin(selected)].copy()
    sel_info["model"] = model_name
    sel_info["selection_reason"] = np.where(
        sel_info["feature"].isin(forced),
        "forced_or_selected",
        "univariate_selected",
    )
    sel_info = sel_info[["model", "feature", "selection_reason", "score", "pvalue", "group"]]
    return selected, sel_info


def fit_xgb_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> XGBRegressor:
    try:
        model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    except TypeError:
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
    return model


def transform_features(
    model_df: pd.DataFrame,
    selected_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, List[int], List[int], List[int]]:
    train_idx = model_df["split"].eq("train").to_numpy()
    valid_idx = model_df["split"].eq("validation").to_numpy()
    test_idx = model_df["split"].eq("test").to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(model_df.loc[train_idx, selected_features])
    X_valid = imputer.transform(model_df.loc[valid_idx, selected_features])
    X_test = imputer.transform(model_df.loc[test_idx, selected_features])
    return X_train, X_valid, X_test, imputer, train_idx, valid_idx, test_idx


# =============================================================================
# Model 1: Growth-from-lag1 XGB
# =============================================================================

def run_growth_model(df: pd.DataFrame, out_dir: Path) -> Dict:
    section("Model 1: Growth-from-lag1 XGB")

    model_name = "growth_from_lag1_xgb"
    model_df, feature_cols = prepare_model_table(df, model_kind="growth")

    print(f"Rows: {len(model_df):,}")
    print(f"Quarter range: {model_df['quarter'].min()} to {model_df['quarter'].max()}")
    print("Rows by split:")
    print(model_df["split"].value_counts().reindex(["train", "validation", "test"]).to_string())
    print(f"Candidate features before selection: {len(feature_cols):,}")

    selected_features, selected_info = select_features(
        model_df,
        feature_cols,
        target_col="model_target",
        max_features=MAX_FEATURES_GROWTH,
        model_name=model_name,
    )
    print(f"Selected features: {len(selected_features):,}")

    X_train, X_valid, X_test, imputer, train_idx, valid_idx, test_idx = transform_features(
        model_df, selected_features
    )
    y_train = model_df.loc[train_idx, "model_target"].to_numpy(dtype=float)
    y_valid = model_df.loc[valid_idx, "model_target"].to_numpy(dtype=float)
    y_test = model_df.loc[test_idx, "model_target"].to_numpy(dtype=float)

    print("Training XGBoost growth model...")
    model = fit_xgb_with_early_stopping(X_train, y_train, X_valid, y_valid)
    print(f"Best iteration: {getattr(model, 'best_iteration', None)}")
    print(f"Best validation score: {getattr(model, 'best_score', None)}")

    # Predict growth and convert to count scale.
    pred_growth = np.full(len(model_df), np.nan)
    pred_growth[train_idx] = model.predict(X_train)
    pred_growth[valid_idx] = model.predict(X_valid)
    pred_growth[test_idx] = model.predict(X_test)

    lag1_log = model_df["log1p_homelessness_total_assessments_lag1q"].to_numpy(dtype=float)
    pred_count_xgb = expm1_clip(lag1_log + pred_growth)

    actual = model_df["homelessness_total_assessments"].to_numpy(dtype=float)
    lag1_pred = model_df["homelessness_total_assessments_lag1q"].to_numpy(dtype=float)

    if "homelessness_total_assessments_lag4q" in model_df.columns:
        lag4_pred = model_df["homelessness_total_assessments_lag4q"].to_numpy(dtype=float)
    else:
        lag4_pred = np.full(len(model_df), np.nan)

    train_mean = np.nanmean(actual[train_idx])
    mean_pred = np.full(len(model_df), train_mean)

    pred_df = model_df[[
        "lad_code", "lad_name", "quarter", "quarter_period", "split",
        "homelessness_total_assessments",
        "homelessness_total_assessments_lag1q",
    ]].copy()
    pred_df = pred_df.rename(columns={"homelessness_total_assessments": "actual_count"})
    pred_df["model"] = model_name
    pred_df["pred_xgb_growth_count"] = pred_count_xgb
    pred_df["pred_lag1_count"] = lag1_pred
    pred_df["pred_lag4_count"] = lag4_pred
    pred_df["pred_train_mean_count"] = mean_pred
    pred_df["predicted_growth_log"] = pred_growth

    # Metrics
    metrics_rows = []
    for split in ["train", "validation", "test"]:
        mask = pred_df["split"].eq(split).to_numpy()
        metrics_rows.append(metrics_count_scale(actual[mask], pred_count_xgb[mask], model_name, f"xgb_{split}"))
        metrics_rows.append(metrics_count_scale(actual[mask], lag1_pred[mask], model_name, f"lag1_baseline_{split}"))
        metrics_rows.append(metrics_count_scale(actual[mask], lag4_pred[mask], model_name, f"lag4_baseline_{split}"))
        metrics_rows.append(metrics_count_scale(actual[mask], mean_pred[mask], model_name, f"train_mean_baseline_{split}"))

    metrics = pd.DataFrame(metrics_rows)

    # Feature importance
    fi, gi = make_feature_importance(model, selected_features, model_name)

    print("\nTest metrics:")
    print(metrics[metrics["split_or_model"].str.contains("_test")].to_string(index=False))

    print("\nTop feature groups:")
    print(gi.head(12).to_string(index=False))

    # Save model-specific files
    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(model_out / "predictions_growth_xgb.csv", index=False)
    metrics.to_csv(model_out / "metrics_growth_xgb.csv", index=False)
    fi.to_csv(model_out / "feature_importance_growth_xgb.csv", index=False)
    gi.to_csv(model_out / "feature_group_importance_growth_xgb.csv", index=False)
    selected_info.to_csv(model_out / "selected_features_growth_xgb.csv", index=False)

    # Plots
    if SAVE_PLOTS:
        plot_actual_vs_pred(
            pred_df,
            "pred_xgb_growth_count",
            "Growth-from-lag1 XGB: actual vs predicted, test",
            model_out / "actual_vs_predicted_growth_xgb_test.png",
        )
        plot_england_aggregate(
            pred_df,
            ["pred_xgb_growth_count", "pred_lag1_count"],
            "Growth XGB vs lag1 baseline: England aggregate",
            model_out / "england_aggregate_growth_xgb_vs_lag1.png",
        )
        plot_feature_group_importance(
            gi,
            "Growth-from-lag1 XGB: feature group importance",
            model_out / "feature_group_importance_growth_xgb.png",
        )

    return dict(
        model_name=model_name,
        model=model,
        model_df=model_df,
        selected_features=selected_features,
        selected_info=selected_info,
        predictions=pred_df,
        metrics=metrics,
        feature_importance=fi,
        feature_group_importance=gi,
        pred_xgb_count=pred_count_xgb,
        pred_lag1_count=lag1_pred,
        actual_count=actual,
    )


# =============================================================================
# Model 2: Rate model XGB
# =============================================================================

def run_rate_model(df: pd.DataFrame, out_dir: Path) -> Dict:
    section("Model 2: Rate model XGB")

    model_name = "rate_model_xgb"
    model_df, feature_cols = prepare_model_table(df, model_kind="rate")

    print(f"Rows: {len(model_df):,}")
    print(f"Quarter range: {model_df['quarter'].min()} to {model_df['quarter'].max()}")
    print("Rows by split:")
    print(model_df["split"].value_counts().reindex(["train", "validation", "test"]).to_string())
    print(f"Candidate features before selection: {len(feature_cols):,}")

    selected_features, selected_info = select_features(
        model_df,
        feature_cols,
        target_col="model_target",
        max_features=MAX_FEATURES_RATE,
        model_name=model_name,
    )
    print(f"Selected features: {len(selected_features):,}")

    X_train, X_valid, X_test, imputer, train_idx, valid_idx, test_idx = transform_features(
        model_df, selected_features
    )
    y_train = model_df.loc[train_idx, "model_target"].to_numpy(dtype=float)
    y_valid = model_df.loc[valid_idx, "model_target"].to_numpy(dtype=float)
    y_test = model_df.loc[test_idx, "model_target"].to_numpy(dtype=float)

    print("Training XGBoost rate model...")
    model = fit_xgb_with_early_stopping(X_train, y_train, X_valid, y_valid)
    print(f"Best iteration: {getattr(model, 'best_iteration', None)}")
    print(f"Best validation score: {getattr(model, 'best_score', None)}")

    # Predict rate and convert to count scale.
    pred_log_rate = np.full(len(model_df), np.nan)
    pred_log_rate[train_idx] = model.predict(X_train)
    pred_log_rate[valid_idx] = model.predict(X_valid)
    pred_log_rate[test_idx] = model.predict(X_test)

    pred_rate = expm1_clip(pred_log_rate)
    population = model_df["population"].to_numpy(dtype=float)
    pred_count_xgb = pred_rate * population / 1000.0

    actual_count = model_df["homelessness_total_assessments"].to_numpy(dtype=float)

    # Count lag1 baseline if available.
    if "homelessness_total_assessments_lag1q" in model_df.columns:
        lag1_count_pred = model_df["homelessness_total_assessments_lag1q"].to_numpy(dtype=float)
    else:
        lag1_count_pred = np.full(len(model_df), np.nan)

    # Rate lag1 baseline: last quarter's risk applied to current population.
    if "homelessness_rate_per_1000_lag1q" in model_df.columns:
        lag1_rate_pred = model_df["homelessness_rate_per_1000_lag1q"].to_numpy(dtype=float) * population / 1000.0
    else:
        lag1_rate_pred = np.full(len(model_df), np.nan)

    train_mean = np.nanmean(actual_count[train_idx])
    mean_pred = np.full(len(model_df), train_mean)

    pred_df = model_df[[
        "lad_code", "lad_name", "quarter", "quarter_period", "split",
        "homelessness_total_assessments",
        "homelessness_rate_per_1000",
        "population",
    ]].copy()
    pred_df = pred_df.rename(columns={"homelessness_total_assessments": "actual_count"})
    pred_df["model"] = model_name
    pred_df["pred_xgb_rate_per_1000"] = pred_rate
    pred_df["pred_xgb_rate_count"] = pred_count_xgb
    pred_df["pred_lag1_count"] = lag1_count_pred
    pred_df["pred_lag1_rate_count"] = lag1_rate_pred
    pred_df["pred_train_mean_count"] = mean_pred

    metrics_rows = []
    for split in ["train", "validation", "test"]:
        mask = pred_df["split"].eq(split).to_numpy()
        metrics_rows.append(metrics_count_scale(actual_count[mask], pred_count_xgb[mask], model_name, f"xgb_rate_{split}"))
        metrics_rows.append(metrics_count_scale(actual_count[mask], lag1_count_pred[mask], model_name, f"lag1_count_baseline_{split}"))
        metrics_rows.append(metrics_count_scale(actual_count[mask], lag1_rate_pred[mask], model_name, f"lag1_rate_baseline_{split}"))
        metrics_rows.append(metrics_count_scale(actual_count[mask], mean_pred[mask], model_name, f"train_mean_baseline_{split}"))

    metrics = pd.DataFrame(metrics_rows)
    fi, gi = make_feature_importance(model, selected_features, model_name)

    print("\nTest metrics:")
    print(metrics[metrics["split_or_model"].str.contains("_test")].to_string(index=False))

    print("\nTop feature groups:")
    print(gi.head(12).to_string(index=False))

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(model_out / "predictions_rate_xgb.csv", index=False)
    metrics.to_csv(model_out / "metrics_rate_xgb.csv", index=False)
    fi.to_csv(model_out / "feature_importance_rate_xgb.csv", index=False)
    gi.to_csv(model_out / "feature_group_importance_rate_xgb.csv", index=False)
    selected_info.to_csv(model_out / "selected_features_rate_xgb.csv", index=False)

    if SAVE_PLOTS:
        plot_actual_vs_pred(
            pred_df,
            "pred_xgb_rate_count",
            "Rate model XGB: actual vs predicted count, test",
            model_out / "actual_vs_predicted_rate_xgb_test.png",
        )
        plot_england_aggregate(
            pred_df,
            ["pred_xgb_rate_count", "pred_lag1_count"],
            "Rate model XGB vs lag1 baseline: England aggregate",
            model_out / "england_aggregate_rate_xgb_vs_lag1.png",
        )
        plot_feature_group_importance(
            gi,
            "Rate model XGB: feature group importance",
            model_out / "feature_group_importance_rate_xgb.png",
        )

    return dict(
        model_name=model_name,
        model=model,
        model_df=model_df,
        selected_features=selected_features,
        selected_info=selected_info,
        predictions=pred_df,
        metrics=metrics,
        feature_importance=fi,
        feature_group_importance=gi,
        pred_xgb_count=pred_count_xgb,
        pred_lag1_count=lag1_count_pred,
        actual_count=actual_count,
    )


# =============================================================================
# Model 3: XGB + lag1 blend
# =============================================================================

def choose_blend_weight(
    actual: np.ndarray,
    pred_xgb: np.ndarray,
    pred_lag1: np.ndarray,
    metric: str = "MAE",
    min_xgb_weight: float = 0.0,
) -> Tuple[float, float]:
    best_w = None
    best_score = np.inf

    for w in BLEND_WEIGHT_GRID:
        if w < min_xgb_weight:
            continue
        pred = w * pred_xgb + (1.0 - w) * pred_lag1
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.sum() == 0:
            continue
        if metric.upper() == "RMSE":
            score = math.sqrt(mean_squared_error(actual[mask], pred[mask]))
        else:
            score = mean_absolute_error(actual[mask], pred[mask])

        if score < best_score:
            best_score = score
            best_w = float(w)

    return float(best_w), float(best_score)


def run_blend_model(growth_result: Dict, out_dir: Path) -> Dict:
    section("Model 3: XGB + lag1 blend")

    model_name = "xgb_lag1_blend"
    pred_df = growth_result["predictions"].copy()

    pred_xgb = pred_df["pred_xgb_growth_count"].to_numpy(dtype=float)
    pred_lag1 = pred_df["pred_lag1_count"].to_numpy(dtype=float)
    actual = pred_df["actual_count"].to_numpy(dtype=float)

    valid_mask = pred_df["split"].eq("validation").to_numpy()
    test_mask = pred_df["split"].eq("test").to_numpy()

    # Unconstrained blend: pure validation optimum.
    w_mae, valid_mae = choose_blend_weight(
        actual[valid_mask],
        pred_xgb[valid_mask],
        pred_lag1[valid_mask],
        metric="MAE",
        min_xgb_weight=0.0,
    )
    w_rmse, valid_rmse = choose_blend_weight(
        actual[valid_mask],
        pred_xgb[valid_mask],
        pred_lag1[valid_mask],
        metric="RMSE",
        min_xgb_weight=0.0,
    )

    # Constrained blend: XGB must contribute at least CONSTRAINED_BLEND_MIN_XGB_WEIGHT.
    w_mae_con, valid_mae_con = choose_blend_weight(
        actual[valid_mask],
        pred_xgb[valid_mask],
        pred_lag1[valid_mask],
        metric="MAE",
        min_xgb_weight=CONSTRAINED_BLEND_MIN_XGB_WEIGHT,
    )
    w_rmse_con, valid_rmse_con = choose_blend_weight(
        actual[valid_mask],
        pred_xgb[valid_mask],
        pred_lag1[valid_mask],
        metric="RMSE",
        min_xgb_weight=CONSTRAINED_BLEND_MIN_XGB_WEIGHT,
    )

    print(f"Validation-tuned XGB weight, unconstrained MAE:  {w_mae:.2f}")
    print(f"Validation-tuned XGB weight, unconstrained RMSE: {w_rmse:.2f}")
    print(f"Validation-tuned XGB weight, constrained MAE:    {w_mae_con:.2f}")
    print(f"Validation-tuned XGB weight, constrained RMSE:   {w_rmse_con:.2f}")
    print(f"Constrained blend requires XGB weight >= {CONSTRAINED_BLEND_MIN_XGB_WEIGHT:.2f}")

    pred_df["model"] = model_name
    pred_df["pred_blend_unconstrained_valid_MAE"] = w_mae * pred_xgb + (1.0 - w_mae) * pred_lag1
    pred_df["pred_blend_unconstrained_valid_RMSE"] = w_rmse * pred_xgb + (1.0 - w_rmse) * pred_lag1
    pred_df["pred_blend_constrained_valid_MAE"] = w_mae_con * pred_xgb + (1.0 - w_mae_con) * pred_lag1
    pred_df["pred_blend_constrained_valid_RMSE"] = w_rmse_con * pred_xgb + (1.0 - w_rmse_con) * pred_lag1

    metrics_rows = []

    blend_specs = [
        ("blend_unconstrained_valid_MAE", "pred_blend_unconstrained_valid_MAE", w_mae, 0.0),
        ("blend_unconstrained_valid_RMSE", "pred_blend_unconstrained_valid_RMSE", w_rmse, 0.0),
        ("blend_constrained_valid_MAE", "pred_blend_constrained_valid_MAE", w_mae_con, CONSTRAINED_BLEND_MIN_XGB_WEIGHT),
        ("blend_constrained_valid_RMSE", "pred_blend_constrained_valid_RMSE", w_rmse_con, CONSTRAINED_BLEND_MIN_XGB_WEIGHT),
        ("xgb_growth_only", "pred_xgb_growth_count", 1.0, 1.0),
        ("lag1_baseline", "pred_lag1_count", 0.0, 0.0),
    ]

    for split in ["train", "validation", "test"]:
        split_mask = pred_df["split"].eq(split).to_numpy()
        for label, col, w, min_w in blend_specs:
            metrics_rows.append(metrics_count_scale(
                actual[split_mask],
                pred_df.loc[split_mask, col].to_numpy(dtype=float),
                model_name,
                f"{label}_{split}",
                extra={
                    "xgb_weight": w,
                    "lag1_weight": 1.0 - w,
                    "min_xgb_weight_constraint": min_w,
                },
            ))

    metrics = pd.DataFrame(metrics_rows)

    print("\nTest metrics:")
    print(metrics[metrics["split_or_model"].str.endswith("_test")].to_string(index=False))

    # Save
    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(model_out / "predictions_xgb_lag1_blend.csv", index=False)
    metrics.to_csv(model_out / "metrics_xgb_lag1_blend.csv", index=False)

    weight_summary = pd.DataFrame([
        {"blend": "unconstrained_valid_MAE", "xgb_weight": w_mae, "lag1_weight": 1 - w_mae, "valid_score": valid_mae, "metric_used": "MAE", "min_xgb_weight": 0.0},
        {"blend": "unconstrained_valid_RMSE", "xgb_weight": w_rmse, "lag1_weight": 1 - w_rmse, "valid_score": valid_rmse, "metric_used": "RMSE", "min_xgb_weight": 0.0},
        {"blend": "constrained_valid_MAE", "xgb_weight": w_mae_con, "lag1_weight": 1 - w_mae_con, "valid_score": valid_mae_con, "metric_used": "MAE", "min_xgb_weight": CONSTRAINED_BLEND_MIN_XGB_WEIGHT},
        {"blend": "constrained_valid_RMSE", "xgb_weight": w_rmse_con, "lag1_weight": 1 - w_rmse_con, "valid_score": valid_rmse_con, "metric_used": "RMSE", "min_xgb_weight": CONSTRAINED_BLEND_MIN_XGB_WEIGHT},
    ])
    weight_summary.to_csv(model_out / "blend_weight_summary.csv", index=False)

    if SAVE_PLOTS:
        plot_actual_vs_pred(
            pred_df,
            "pred_blend_constrained_valid_MAE",
            "XGB + lag1 blend: actual vs predicted, test",
            model_out / "actual_vs_predicted_blend_constrained_MAE_test.png",
        )
        plot_england_aggregate(
            pred_df,
            [
                "pred_blend_constrained_valid_MAE",
                "pred_xgb_growth_count",
                "pred_lag1_count",
            ],
            "Blend vs XGB growth vs lag1: England aggregate",
            model_out / "england_aggregate_blend_vs_xgb_lag1.png",
        )

    return dict(
        model_name=model_name,
        predictions=pred_df,
        metrics=metrics,
        weight_summary=weight_summary,
    )


# =============================================================================
# Combined summaries
# =============================================================================

def save_combined_outputs(
    out_dir: Path,
    growth_result: Dict,
    rate_result: Dict,
    blend_result: Dict,
) -> None:
    section("Saving combined outputs")

    all_metrics = pd.concat(
        [
            growth_result["metrics"],
            rate_result["metrics"],
            blend_result["metrics"],
        ],
        ignore_index=True,
    )
    all_metrics.to_csv(out_dir / "all_model_metrics.csv", index=False)

    all_predictions = pd.concat(
        [
            growth_result["predictions"],
            rate_result["predictions"],
            blend_result["predictions"],
        ],
        ignore_index=True,
        sort=False,
    )
    all_predictions.to_csv(out_dir / "all_model_predictions.csv", index=False)

    all_fi = pd.concat(
        [
            growth_result["feature_importance"],
            rate_result["feature_importance"],
        ],
        ignore_index=True,
    )
    all_fi.to_csv(out_dir / "all_xgb_feature_importance.csv", index=False)

    all_gi = pd.concat(
        [
            growth_result["feature_group_importance"],
            rate_result["feature_group_importance"],
        ],
        ignore_index=True,
    )
    all_gi.to_csv(out_dir / "all_xgb_feature_group_importance.csv", index=False)

    all_selected = pd.concat(
        [
            growth_result["selected_info"],
            rate_result["selected_info"],
        ],
        ignore_index=True,
    )
    all_selected.to_csv(out_dir / "selected_features.csv", index=False)

    blend_result["weight_summary"].to_csv(out_dir / "blend_weight_summary.csv", index=False)

    # Final concise metrics table: test rows only.
    test_metrics = all_metrics[all_metrics["split_or_model"].str.contains("_test", na=False)].copy()
    test_metrics = test_metrics.sort_values(["MAE", "RMSE"], ascending=[True, True])
    test_metrics.to_csv(out_dir / "final_test_metrics_ranked.csv", index=False)

    print(f"Outputs saved to: {out_dir.resolve()}")
    print("\nFinal test metrics ranked by MAE:")
    print(test_metrics[[
        "model", "split_or_model", "n", "MAE", "RMSE", "R2",
        "SMAPE_percent", "mean_actual", "mean_predicted",
        "mean_bias_actual_minus_predicted",
    ]].to_string(index=False))

    # Bar plot of test MAE.
    if SAVE_PLOTS and not test_metrics.empty:
        plot_df = test_metrics.head(12).iloc[::-1]
        labels = plot_df["model"] + " | " + plot_df["split_or_model"]
        plt.figure(figsize=(11, 7))
        plt.barh(labels, plot_df["MAE"])
        plt.xlabel("Test MAE")
        plt.title("Final test MAE comparison")
        plt.tight_layout()
        plt.savefig(out_dir / "final_test_mae_comparison.png", dpi=160, bbox_inches="tight")


def save_run_config(out_dir: Path) -> None:
    config = {
        "MODEL_START_QUARTER": MODEL_START_QUARTER,
        "TRAIN_END_QUARTER": TRAIN_END_QUARTER,
        "VALID_START_QUARTER": VALID_START_QUARTER,
        "VALID_END_QUARTER": VALID_END_QUARTER,
        "TEST_START_QUARTER": TEST_START_QUARTER,
        "LIVING_COST_LAGS": LIVING_COST_LAGS,
        "TARGET_HISTORY_LAGS": TARGET_HISTORY_LAGS,
        "EXCLUDE_DIRECT_LAG1_COUNT_FEATURES_IN_GROWTH_XGB": EXCLUDE_DIRECT_LAG1_COUNT_FEATURES_IN_GROWTH_XGB,
        "EXCLUDE_RAW_COUNT_HISTORY_IN_RATE_MODEL": EXCLUDE_RAW_COUNT_HISTORY_IN_RATE_MODEL,
        "USE_FEATURE_SELECTION": USE_FEATURE_SELECTION,
        "MAX_FEATURES_GROWTH": MAX_FEATURES_GROWTH,
        "MAX_FEATURES_RATE": MAX_FEATURES_RATE,
        "MIN_NON_MISSING_FRAC": MIN_NON_MISSING_FRAC,
        "CONSTRAINED_BLEND_MIN_XGB_WEIGHT": CONSTRAINED_BLEND_MIN_XGB_WEIGHT,
        "XGB_PARAMS": XGB_PARAMS,
        "EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(out_dir)

    qdf = load_and_make_quarterly()
    fdf = add_features(qdf)

    # Run the three requested models.
    growth_result = run_growth_model(fdf, out_dir)
    rate_result = run_rate_model(fdf, out_dir)
    blend_result = run_blend_model(growth_result, out_dir)

    save_combined_outputs(out_dir, growth_result, rate_result, blend_result)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
