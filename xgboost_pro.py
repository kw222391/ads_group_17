"""
Report-only XGBoost homelessness models
=======================================

This script keeps only the XGBoost-related model section used in the report:

1. Lag-1 persistence baseline
2. Growth-from-lag1 XGBoost
3. XGBoost + lag-1 blended model

Important report rule:
- Use only total CPI: cpi_00_all_items.
- Do NOT use CPI category columns.
- Do NOT use same-quarter homelessness outcomes as features.
- For the Growth-from-lag1 XGBoost model, lag-1 homelessness count/log-count is
  used to reconstruct the final prediction, so direct lag-1 target features are
  excluded from the XGBoost feature matrix.

Default data paths match the user's Windows folder. You can also run with:
    python report_xgboost_three_models.py --monthly "D:\\...\\monthly_lad_panel_2000_2025_new.csv" --homeless "D:\\...\\homeless_19_25_cleaned.csv"

Required packages:
    pip install pandas numpy matplotlib scikit-learn xgboost joblib

Outputs are saved to:
    report_xgboost_outputs/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required. Install it with: pip install xgboost") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# =============================================================================
# 0. CONFIG
# =============================================================================

DEFAULT_MONTHLY_FILE = r"D:\UOB\ads_group_17\ads_group_17\monthly_lad_panel_2000_2025_new.csv"
DEFAULT_HOMELESS_FILE = r"D:\UOB\ads_group_17\ads_group_17\homeless_19_25_cleaned.csv"

OUTPUT_DIR = Path("report_xgboost_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = max(1, min(4, os.cpu_count() or 4))

TARGET_COL = "homelessness_total_assessments"
TOTAL_CPI_COL = "cpi_00_all_items"

# Same chronological split described in the report.
MODELLING_START_QUARTER = "2019Q3"
TRAIN_END_Q = "2023Q4"
VALID_START_Q = "2024Q1"
VALID_END_Q = "2024Q4"
TEST_START_Q = "2025Q1"
TEST_END_Q = "2025Q4"

# Feature settings.
LAGS_Q = [1, 2, 4, 8]
ROLL_WINDOWS_Q = [4, 8]
INCLUDE_SAME_QUARTER_EXOG = True  # report-style explanatory/nowcasting setup
ADD_LAD_DUMMIES = True            # LAD fixed effects for known local authorities
ADD_TIME_FEATURES = True
DROP_FEATURES_MISSING_ABOVE = 0.80
TOP_K_FEATURES = 160

# XGBoost hyperparameters described in the report.
XGB_N_ESTIMATORS = 1600
XGB_LEARNING_RATE = 0.025
XGB_MAX_DEPTH = 3
XGB_MIN_CHILD_WEIGHT = 10
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_REG_ALPHA = 0.05
XGB_REG_LAMBDA = 3.0
XGB_EARLY_STOPPING_ROUNDS = 100

SAVE_PLOTS = True
SHOW_PLOTS = False


# =============================================================================
# 1. UTILITIES
# =============================================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def quarter_to_period(q: str | pd.Period) -> pd.Period:
    if isinstance(q, pd.Period):
        return q.asfreq("Q")
    return pd.Period(str(q), freq="Q")


def is_real_english_lad_code(s: pd.Series) -> pd.Series:
    """Keep LAD codes and drop England/region aggregate rows such as E92000001/E12000007."""
    return s.astype(str).str.match(r"^E0[6789]", na=False)


def safe_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "SMAPE_percent": np.nan}
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "SMAPE_percent": smape(y_true, y_pred),
        "mean_actual": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
        "bias_actual_minus_predicted": float(np.mean(y_true - y_pred)),
    }


def ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def save_figure(path: Path) -> None:
    if not SAVE_PLOTS:
        plt.close()
        return
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def resolve_input_path(user_path: str, fallback_filename: str) -> Path:
    """Use the given path; if it is not found, try the same folder as this script."""
    p = Path(user_path)
    if p.exists():
        return p
    local = Path(__file__).resolve().parent / fallback_filename
    if local.exists():
        return local
    raise FileNotFoundError(
        f"Cannot find input file. Tried:\n  {p}\n  {local}\n"
        "Set the path in the config section or pass --monthly / --homeless."
    )


# =============================================================================
# 2. DATA LOADING AND MERGING
# =============================================================================

def monthly_columns_to_read(csv_path: Path) -> List[str]:
    """Read only variables used by the report model. CPI categories are deliberately excluded."""
    report_cols = [
        "year", "month", "lad_code", "lad_name",
        "average_house_price",
        "average_house_price_monthly_change",
        "average_house_price_annual_change",
        "seasonally_adjusted_average_house_price",
        "house_price_index",
        "house_sales_volume",
        "unemployment_count",
        "private_rental_price_index",
        "private_rental_price_monthly_change",
        "private_rental_price_annual_change",
        "average_private_rental_price",
        "gbp_index",
        "ftse_100",
        "income",
        "uk_bank_rate",
        "brent_oil_price",
        "population",
        "internal_net_migration",
        "international_net_migration",
        TOTAL_CPI_COL,
    ]
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in report_cols if c in available]
    missing = [c for c in ["year", "month", "lad_code", "lad_name", TOTAL_CPI_COL] if c not in usecols]
    if missing:
        raise ValueError(f"Monthly panel is missing required columns: {missing}")
    return usecols


def load_monthly_panel(monthly_file: Path) -> pd.DataFrame:
    print_section("Loading monthly feature panel")
    usecols = monthly_columns_to_read(monthly_file)
    df = pd.read_csv(monthly_file, usecols=usecols)

    df = df[is_real_english_lad_code(df["lad_code"])].copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q")

    for c in df.columns:
        if c not in {"lad_code", "lad_name", "date", "quarter"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"Monthly rows after keeping real LADs: {len(df):,}")
    print(f"LADs: {df['lad_code'].nunique():,}")
    print(f"Quarter range from monthly panel: {df['quarter'].min()} to {df['quarter'].max()}")
    print(f"CPI columns read: {[c for c in df.columns if c.startswith('cpi_')]}")
    return df


def monthly_to_quarterly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    print_section("Aggregating monthly features to quarterly LAD panel")
    numeric_cols = [c for c in monthly.columns if c not in {"lad_code", "lad_name", "date", "quarter"}]
    agg = {c: "mean" for c in numeric_cols}
    agg["lad_name"] = "last"

    qdf = monthly.groupby(["lad_code", "quarter"], as_index=False).agg(agg)

    # Reindex to a full calendar-quarter grid. This prevents lag1 from jumping across missing quarters.
    lad_names = qdf.groupby("lad_code")["lad_name"].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    all_lads = sorted(qdf["lad_code"].unique())
    all_quarters = pd.period_range(qdf["quarter"].min(), qdf["quarter"].max(), freq="Q")
    full_index = pd.MultiIndex.from_product([all_lads, all_quarters], names=["lad_code", "quarter"])
    qdf = qdf.set_index(["lad_code", "quarter"]).reindex(full_index).reset_index()
    qdf["lad_name"] = qdf["lad_code"].map(lad_names)

    qdf["quarter_date"] = qdf["quarter"].dt.start_time
    qdf["year"] = qdf["quarter"].dt.year.astype(int)
    qdf["quarter_num"] = qdf["quarter"].dt.quarter.astype(int)
    qdf["quarter_index"] = (qdf["year"] - qdf["year"].min()) * 4 + qdf["quarter_num"]

    print(f"Quarterly feature-grid rows: {len(qdf):,}")
    print(f"Quarterly feature range: {qdf['quarter'].min()} to {qdf['quarter'].max()}")
    return qdf


def load_homelessness(homeless_file: Path) -> pd.DataFrame:
    print_section("Loading quarterly homelessness target data")
    h = pd.read_csv(homeless_file)
    rename = {
        "LAD_code": "lad_code",
        "LA_name": "homeless_lad_name",
        "Total_owed": TARGET_COL,
        "Threatened": "homelessness_threatened",
        "Homeless_relief": "homelessness_relief",
        "Homeless_per_1000": "homelessness_per_1000",
        "Year": "year_homeless",
        "Quarter": "quarter_num_homeless",
    }
    h = h.rename(columns={k: v for k, v in rename.items() if k in h.columns})

    required = ["lad_code", TARGET_COL, "year_homeless", "quarter_num_homeless"]
    missing = [c for c in required if c not in h.columns]
    if missing:
        raise ValueError(f"Homelessness file is missing required columns: {missing}")

    h = h[is_real_english_lad_code(h["lad_code"])].copy()
    h["quarter"] = [pd.Period(f"{int(y)}Q{int(q)}", freq="Q") for y, q in zip(h["year_homeless"], h["quarter_num_homeless"])]

    value_cols = [TARGET_COL, "homelessness_threatened", "homelessness_relief", "homelessness_per_1000"]
    for c in value_cols:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")

    keep_cols = ["lad_code", "quarter", "homeless_lad_name"] + [c for c in value_cols if c in h.columns]
    h = h[keep_cols].drop_duplicates(["lad_code", "quarter"])

    print(f"Homelessness rows after keeping real LADs: {len(h):,}")
    print(f"LADs: {h['lad_code'].nunique():,}")
    print(f"Quarter range from homelessness data: {h['quarter'].min()} to {h['quarter'].max()}")
    return h


def build_quarterly_panel(monthly_file: Path, homeless_file: Path) -> pd.DataFrame:
    monthly = load_monthly_panel(monthly_file)
    features_q = monthly_to_quarterly_features(monthly)
    homeless_q = load_homelessness(homeless_file)

    print_section("Merging quarterly features with homelessness target")
    qdf = features_q.merge(homeless_q, on=["lad_code", "quarter"], how="left")
    qdf["lad_name"] = qdf["lad_name"].fillna(qdf.get("homeless_lad_name"))
    if "homeless_lad_name" in qdf.columns:
        qdf = qdf.drop(columns=["homeless_lad_name"])

    print(f"Merged panel rows: {len(qdf):,}")
    print(f"Rows with non-missing target: {qdf[TARGET_COL].notna().sum():,}")
    print(f"Merged quarter range: {qdf['quarter'].min()} to {qdf['quarter'].max()}")
    return qdf


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def add_report_features(qdf: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    print_section("Feature engineering")
    qdf = qdf.sort_values(["lad_code", "quarter"]).copy()

    # Enforce CPI rule one more time: only cpi_00_all_items can remain.
    cpi_to_drop = [c for c in qdf.columns if c.startswith("cpi_") and c != TOTAL_CPI_COL]
    if cpi_to_drop:
        qdf = qdf.drop(columns=cpi_to_drop)

    base_exog = [
        "average_house_price",
        "average_house_price_monthly_change",
        "average_house_price_annual_change",
        "seasonally_adjusted_average_house_price",
        "house_price_index",
        "house_sales_volume",
        "unemployment_count",
        "private_rental_price_index",
        "private_rental_price_monthly_change",
        "private_rental_price_annual_change",
        "average_private_rental_price",
        "gbp_index",
        "ftse_100",
        "income",
        "uk_bank_rate",
        "brent_oil_price",
        "population",
        "internal_net_migration",
        "international_net_migration",
        TOTAL_CPI_COL,
    ]
    base_exog = [c for c in base_exog if c in qdf.columns]

    # Derived affordability/living-cost features.
    constructed = []
    if TOTAL_CPI_COL in qdf.columns and "income" in qdf.columns:
        qdf["real_income_cpi_adjusted"] = safe_divide(qdf["income"] * 100.0, qdf[TOTAL_CPI_COL])
        constructed.append("real_income_cpi_adjusted")
    if TOTAL_CPI_COL in qdf.columns and "average_house_price" in qdf.columns:
        qdf["real_house_price_cpi_adjusted"] = safe_divide(qdf["average_house_price"] * 100.0, qdf[TOTAL_CPI_COL])
        constructed.append("real_house_price_cpi_adjusted")
    if "average_house_price" in qdf.columns and "income" in qdf.columns:
        qdf["house_price_to_income"] = safe_divide(qdf["average_house_price"], qdf["income"])
        constructed.append("house_price_to_income")
    if "average_private_rental_price" in qdf.columns and "income" in qdf.columns:
        qdf["annual_rent_to_income"] = safe_divide(qdf["average_private_rental_price"] * 12.0, qdf["income"])
        constructed.append("annual_rent_to_income")
    if "unemployment_count" in qdf.columns and "population" in qdf.columns:
        qdf["unemployment_per_1000"] = safe_divide(qdf["unemployment_count"] * 1000.0, qdf["population"])
        constructed.append("unemployment_per_1000")
    if TARGET_COL in qdf.columns and "population" in qdf.columns:
        qdf["computed_homelessness_rate_per_1000"] = safe_divide(qdf[TARGET_COL] * 1000.0, qdf["population"])

    same_quarter_exog = base_exog + constructed

    # Log transforms for positive levels.
    positive_for_log = [
        "average_house_price", "seasonally_adjusted_average_house_price", "house_price_index",
        "house_sales_volume", "unemployment_count", "private_rental_price_index",
        "average_private_rental_price", "income", "brent_oil_price", "population",
        TOTAL_CPI_COL, "real_income_cpi_adjusted", "real_house_price_cpi_adjusted",
        "house_price_to_income", "annual_rent_to_income", "unemployment_per_1000",
    ]
    log_features = []
    for c in positive_for_log:
        if c in qdf.columns:
            new_c = f"log1p_{c}"
            qdf[new_c] = safe_log1p(qdf[c])
            log_features.append(new_c)
    same_quarter_exog += log_features

    # Quarter-on-quarter and year-on-year changes by LAD.
    pct_features = []
    for c in base_exog + constructed:
        if c not in qdf.columns:
            continue
        prev1 = qdf.groupby("lad_code")[c].shift(1)
        prev4 = qdf.groupby("lad_code")[c].shift(4)
        qoq = f"{c}_qoq_pct"
        yoy = f"{c}_yoy_pct"
        qdf[qoq] = np.where((prev1.notna()) & (prev1 != 0), (qdf[c] / prev1 - 1.0) * 100.0, np.nan)
        qdf[yoy] = np.where((prev4.notna()) & (prev4 != 0), (qdf[c] / prev4 - 1.0) * 100.0, np.nan)
        pct_features.extend([qoq, yoy])
    same_quarter_exog += pct_features

    # Lag exogenous variables by 1, 2, 4, and 8 quarters.
    lagged_exog = []
    for c in same_quarter_exog:
        if c not in qdf.columns:
            continue
        for lag in LAGS_Q:
            new_c = f"{c}_lag{lag}q"
            qdf[new_c] = qdf.groupby("lad_code")[c].shift(lag)
            lagged_exog.append(new_c)

    # Target history. Same-quarter target is never a feature.
    target_history = []
    qdf["log1p_homelessness_total"] = safe_log1p(qdf[TARGET_COL])
    for lag in LAGS_Q:
        count_col = f"{TARGET_COL}_lag{lag}q"
        log_col = f"log1p_{TARGET_COL}_lag{lag}q"
        qdf[count_col] = qdf.groupby("lad_code")[TARGET_COL].shift(lag)
        qdf[log_col] = safe_log1p(qdf[count_col])
        target_history += [count_col, log_col]

        if "computed_homelessness_rate_per_1000" in qdf.columns:
            rate_col = f"computed_homelessness_rate_per_1000_lag{lag}q"
            log_rate_col = f"log1p_computed_homelessness_rate_per_1000_lag{lag}q"
            qdf[rate_col] = qdf.groupby("lad_code")["computed_homelessness_rate_per_1000"].shift(lag)
            qdf[log_rate_col] = safe_log1p(qdf[rate_col])
            target_history += [rate_col, log_rate_col]

    # Past rolling means/stds are shifted by one quarter to avoid leakage.
    for win in ROLL_WINDOWS_Q:
        shifted_target = qdf.groupby("lad_code")[TARGET_COL].shift(1)
        roll_mean = (
            shifted_target.groupby(qdf["lad_code"])
            .rolling(win, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )
        roll_std = (
            shifted_target.groupby(qdf["lad_code"])
            .rolling(win, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
        mean_col = f"{TARGET_COL}_rolling{win}_mean_lag1q"
        std_col = f"{TARGET_COL}_rolling{win}_std_lag1q"
        qdf[mean_col] = roll_mean
        qdf[std_col] = roll_std
        target_history += [mean_col, std_col]

    # Previous observed log-growth features. The shift is done within LAD, not globally.
    growth_1 = qdf.groupby("lad_code")["log1p_homelessness_total"].diff(1)
    growth_4 = qdf.groupby("lad_code")["log1p_homelessness_total"].diff(4)
    qdf["target_log_growth_lag1q"] = growth_1.groupby(qdf["lad_code"]).shift(1)
    qdf["target_log_growth_lag4q"] = growth_4.groupby(qdf["lad_code"]).shift(1)
    target_history += ["target_log_growth_lag1q", "target_log_growth_lag4q"]

    # Time controls.
    qdf["quarter_sin"] = np.sin(2.0 * np.pi * qdf["quarter_num"] / 4.0)
    qdf["quarter_cos"] = np.cos(2.0 * np.pi * qdf["quarter_num"] / 4.0)
    qdf["post_covid_2020plus"] = (qdf["quarter"] >= pd.Period("2020Q2", freq="Q")).astype(int)
    qdf["cost_of_living_shock_2022plus"] = (qdf["quarter"] >= pd.Period("2022Q1", freq="Q")).astype(int)

    qdf = qdf.replace([np.inf, -np.inf], np.nan)
    same_quarter_exog = list(dict.fromkeys([c for c in same_quarter_exog if c in qdf.columns]))
    lagged_exog = list(dict.fromkeys([c for c in lagged_exog if c in qdf.columns]))
    target_history = list(dict.fromkeys([c for c in target_history if c in qdf.columns]))

    print(f"Same-quarter exogenous features: {len(same_quarter_exog):,}")
    print(f"Lagged exogenous features: {len(lagged_exog):,}")
    print(f"Target-history features before report exclusions: {len(target_history):,}")
    print(f"CPI check after feature engineering: {[c for c in qdf.columns if c.startswith('cpi_')][:10]}")
    return qdf, same_quarter_exog, lagged_exog, target_history


def feature_group(feature: str) -> str:
    f = feature.lower()
    if "homelessness_total_assessments" in f or "homelessness_rate" in f or "target_log_growth" in f:
        return "target history"
    if f.startswith("lad_"):
        return "LAD fixed effect"
    if f in {"year", "quarter_num", "quarter_index", "quarter_sin", "quarter_cos"} or "covid" in f or "shock" in f:
        return "time / policy"
    if "cpi_00_all_items" in f:
        return "CPI total"
    if "brent" in f or "oil" in f or "energy" in f:
        return "oil / energy proxy"
    if "rent" in f or "rental" in f:
        return "rent"
    if "house_price" in f or "housing" in f or "sales_volume" in f:
        return "house price / housing"
    if "income" in f or "afford" in f:
        return "income / affordability"
    if "unemployment" in f:
        return "unemployment"
    if "bank_rate" in f or "interest" in f:
        return "interest rate"
    if "gbp" in f or "ftse" in f:
        return "market / FX"
    if "migration" in f:
        return "migration"
    if "population" in f:
        return "population / scale"
    return "other controls"


# =============================================================================
# 4. MODELLING TABLE AND FEATURE SELECTION
# =============================================================================

def add_lad_dummies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if not ADD_LAD_DUMMIES:
        return df, []
    dummies = pd.get_dummies(df["lad_code"].astype(str), prefix="lad", dtype=np.int8)
    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return out, dummies.columns.tolist()


def build_report_model_table(
    qdf: pd.DataFrame,
    same_quarter_exog: List[str],
    lagged_exog: List[str],
    target_history: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    print_section("Building report modelling table")
    df = qdf.copy()
    start_q = quarter_to_period(MODELLING_START_QUARTER)
    df = df[(df["quarter"] >= start_q) & df[TARGET_COL].notna()].copy()

    lag1_col = f"{TARGET_COL}_lag1q"
    log_lag1_col = f"log1p_{TARGET_COL}_lag1q"
    if lag1_col not in df.columns or log_lag1_col not in df.columns:
        raise ValueError("Lag-1 target features were not created correctly.")

    # Growth target from the report: g_it = log(1 + y_it) - log(1 + y_i,t-1).
    df = df[df[lag1_col].notna()].copy()
    df["target_growth_from_lag1"] = safe_log1p(df[TARGET_COL]) - safe_log1p(df[lag1_col])

    # Candidate features.
    feature_cols: List[str] = []
    if INCLUDE_SAME_QUARTER_EXOG:
        feature_cols += same_quarter_exog
    feature_cols += lagged_exog
    feature_cols += target_history

    if ADD_TIME_FEATURES:
        feature_cols += [
            c for c in [
                "year", "quarter_num", "quarter_index", "quarter_sin", "quarter_cos",
                "post_covid_2020plus", "cost_of_living_shock_2022plus",
            ] if c in df.columns
        ]

    df, lad_dummy_cols = add_lad_dummies(df)
    feature_cols += lad_dummy_cols

    # Leakage and report exclusions.
    forbidden_exact = {
        TARGET_COL,
        "homelessness_threatened",
        "homelessness_relief",
        "homelessness_per_1000",
        "computed_homelessness_rate_per_1000",
        "log1p_homelessness_total",
        "target_growth_from_lag1",
    }

    # In the report model, lag-1 count/log-count is already used in the reconstruction,
    # so it is not fed directly to XGBoost.
    direct_lag1_target_features = {
        f"{TARGET_COL}_lag1q",
        f"log1p_{TARGET_COL}_lag1q",
        "computed_homelessness_rate_per_1000_lag1q",
        "log1p_computed_homelessness_rate_per_1000_lag1q",
    }

    feature_cols = [c for c in feature_cols if c not in forbidden_exact]
    feature_cols = [c for c in feature_cols if c not in direct_lag1_target_features]

    # Final CPI rule: only total CPI and its derived/lagged versions are allowed.
    # Because we only read cpi_00_all_items, this is mostly a safety check.
    feature_cols = [c for c in feature_cols if "cpi_" not in c or TOTAL_CPI_COL in c]

    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))

    train_end = quarter_to_period(TRAIN_END_Q)
    valid_start = quarter_to_period(VALID_START_Q)
    valid_end = quarter_to_period(VALID_END_Q)
    test_start = quarter_to_period(TEST_START_Q)
    test_end = quarter_to_period(TEST_END_Q)

    df["split"] = "unused"
    df.loc[df["quarter"] <= train_end, "split"] = "train"
    df.loc[(df["quarter"] >= valid_start) & (df["quarter"] <= valid_end), "split"] = "valid"
    df.loc[(df["quarter"] >= test_start) & (df["quarter"] <= test_end), "split"] = "test"
    df = df[df["split"].isin(["train", "valid", "test"])].copy()

    split_counts = df["split"].value_counts().reindex(["train", "valid", "test"]).fillna(0).astype(int)
    if (split_counts == 0).any():
        raise ValueError(f"At least one split is empty. Split counts:\n{split_counts}")

    print(f"Rows in modelling table: {len(df):,}")
    print(f"Feature columns before selection: {len(feature_cols):,}")
    print(f"Target period: {df['quarter'].min()} to {df['quarter'].max()}")
    print("Rows by split:")
    print(split_counts.to_string())
    return df, feature_cols


def protected_feature_mask(feature: str) -> bool:
    """Forced-inclusion variables from the report.

    The report says feature selection keeps a maximum of 160 selected features and
    forces in key CPI, rent, housing, income, unemployment, interest-rate and
    oil-price variables. To keep that cap meaningful, this protects the core
    level/log variables and their quarter lags, not every derived qoq/yoy feature
    containing those words.
    """
    core_bases = [
        TOTAL_CPI_COL,
        "average_private_rental_price",
        "private_rental_price_index",
        "average_house_price",
        "house_price_index",
        "house_sales_volume",
        "income",
        "real_income_cpi_adjusted",
        "real_house_price_cpi_adjusted",
        "house_price_to_income",
        "annual_rent_to_income",
        "unemployment_count",
        "unemployment_per_1000",
        "uk_bank_rate",
        "brent_oil_price",
    ]
    for base in core_bases:
        protected_patterns = [base, f"log1p_{base}"]
        for pattern in protected_patterns:
            if feature == pattern or feature.startswith(pattern + "_lag"):
                return True
    protected_time = {"quarter_num", "quarter_index", "quarter_sin", "quarter_cos"}
    return feature in protected_time


def clean_and_select_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
    print_section("Feature cleaning and selection")
    train_mask = df["split"] == "train"
    X_train = ensure_numeric_df(df.loc[train_mask, feature_cols])
    y_train = df.loc[train_mask, "target_growth_from_lag1"].astype(float).values

    protected = {c for c in feature_cols if protected_feature_mask(c)}

    # Drop features completely missing or constant in the training split.
    missing_rate = X_train.isna().mean()
    nunique = X_train.nunique(dropna=True)
    keep = []
    for c in feature_cols:
        if missing_rate.get(c, 1.0) >= 1.0:
            continue
        if nunique.get(c, 0) <= 1:
            continue
        if c in protected or missing_rate.get(c, 1.0) <= DROP_FEATURES_MISSING_ABOVE:
            keep.append(c)

    X_train = X_train[keep]
    protected = {c for c in keep if c in protected}

    # Univariate feature selection on training data only.
    med = X_train.median(numeric_only=True)
    X_filled = X_train.fillna(med).fillna(0.0)
    try:
        scores, pvals = f_regression(X_filled, y_train)
    except Exception:
        scores = np.zeros(len(keep))
        pvals = np.ones(len(keep))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)

    ranking = pd.DataFrame({"feature": keep, "univariate_f": scores, "univariate_p": pvals})
    ranking["group"] = ranking["feature"].map(feature_group)
    ranking["protected_report_feature"] = ranking["feature"].isin(protected)
    ranking = ranking.sort_values("univariate_f", ascending=False)

    selected = list(protected)
    remaining_slots = max(TOP_K_FEATURES - len(selected), 0)
    selected += [f for f in ranking[~ranking["feature"].isin(selected)].head(remaining_slots)["feature"].tolist()]

    # Preserve original order for reproducibility and easier feature inspection.
    selected_set = set(selected)
    selected_features = [c for c in feature_cols if c in selected_set]

    ranking.to_csv(OUTPUT_DIR / "feature_selection_ranking.csv", index=False)
    pd.DataFrame({"feature": selected_features, "group": [feature_group(f) for f in selected_features]}).to_csv(
        OUTPUT_DIR / "selected_features.csv", index=False
    )

    print(f"Features after missing/constant cleaning: {len(keep):,}")
    print(f"Forced report features retained: {len(protected):,}")
    print(f"Selected feature count: {len(selected_features):,}")
    return selected_features, ranking


# =============================================================================
# 5. TRAINING, PREDICTION, BLENDING
# =============================================================================

def xgb_params() -> Dict[str, object]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": XGB_N_ESTIMATORS,
        "learning_rate": XGB_LEARNING_RATE,
        "max_depth": XGB_MAX_DEPTH,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "reg_alpha": XGB_REG_ALPHA,
        "reg_lambda": XGB_REG_LAMBDA,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
    }


def fit_xgboost_growth_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBRegressor:
    print_section("Training Growth-from-lag1 XGBoost")
    params = xgb_params()
    for k, v in params.items():
        print(f"  {k}: {v}")

    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    except TypeError:
        # Compatibility for older xgboost versions where early_stopping_rounds is passed to fit().
        es_rounds = int(params.pop("early_stopping_rounds"))
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=es_rounds)

    print(f"Best iteration: {getattr(model, 'best_iteration', None)}")
    print(f"Best validation RMSE: {getattr(model, 'best_score', None)}")
    return model


def growth_to_count(pred_growth: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    lag1 = rows[f"{TARGET_COL}_lag1q"].astype(float).values
    pred = np.expm1(np.log1p(np.clip(lag1, 0, None)) + np.asarray(pred_growth, dtype=float))
    return np.clip(pred, 0.0, None)


def optimize_blend_weight(
    y_true_valid: np.ndarray,
    xgb_pred_valid: np.ndarray,
    lag1_valid: np.ndarray,
    metric: str = "MAE",
    min_weight: float = 0.0,
) -> Tuple[float, float]:
    best_w = float(min_weight)
    best_score = np.inf
    for w in np.linspace(min_weight, 1.0, int(round((1.0 - min_weight) * 100)) + 1):
        pred = w * xgb_pred_valid + (1.0 - w) * lag1_valid
        if metric.upper() == "RMSE":
            score = math.sqrt(mean_squared_error(y_true_valid, pred))
        else:
            score = mean_absolute_error(y_true_valid, pred)
        if score < best_score:
            best_score = float(score)
            best_w = float(w)
    return best_w, best_score


def evaluate_by_split(pred_df: pd.DataFrame, pred_col: str, model_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in ["train", "valid", "test"]:
        sub = pred_df[pred_df["split"] == split]
        if len(sub) == 0:
            continue
        metrics = compute_metrics(sub[TARGET_COL].astype(float).values, sub[pred_col].astype(float).values)
        rows.append({"model": model_name, "split": split, "prediction_column": pred_col, **metrics})
    return rows


def get_xgb_importance(model: XGBRegressor, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    booster = model.get_booster()
    out = pd.DataFrame({"feature": feature_cols})
    for imp_type in ["gain", "weight", "cover"]:
        scores = booster.get_score(importance_type=imp_type)
        out[imp_type] = out["feature"].map(scores).fillna(0.0)
    out["group"] = out["feature"].map(feature_group)
    out["gain_share"] = out["gain"] / out["gain"].sum() if out["gain"].sum() > 0 else 0.0
    out = out.sort_values("gain", ascending=False)

    grp = (
        out.groupby("group", as_index=False)
        .agg(gain=("gain", "sum"), weight=("weight", "sum"), n_features_used=("feature", lambda s: int((out.loc[s.index, "gain"] > 0).sum())))
    )
    grp["gain_share"] = grp["gain"] / grp["gain"].sum() if grp["gain"].sum() > 0 else 0.0
    grp = grp.sort_values("gain_share", ascending=False)
    return out, grp


def run_three_report_models(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    selected_features, _ranking = clean_and_select_features(df, feature_cols)

    train_mask = df["split"] == "train"
    valid_mask = df["split"] == "valid"
    test_mask = df["split"] == "test"

    X_train = ensure_numeric_df(df.loc[train_mask, selected_features])
    y_train = df.loc[train_mask, "target_growth_from_lag1"].astype(float)
    X_valid = ensure_numeric_df(df.loc[valid_mask, selected_features])
    y_valid = df.loc[valid_mask, "target_growth_from_lag1"].astype(float)
    X_test = ensure_numeric_df(df.loc[test_mask, selected_features])

    model = fit_xgboost_growth_model(X_train, y_train, X_valid, y_valid)

    # Base prediction table.
    pred_df = df[[
        "lad_code", "lad_name", "quarter", "quarter_date", "year", "quarter_num",
        TARGET_COL, f"{TARGET_COL}_lag1q", "split",
    ]].copy()
    pred_df["quarter"] = pred_df["quarter"].astype(str)

    # Model 1: lag-1 persistence baseline.
    pred_df["pred_lag1_baseline"] = pred_df[f"{TARGET_COL}_lag1q"].astype(float).clip(lower=0)

    # Model 2: Growth-from-lag1 XGBoost.
    pred_growth_all = np.full(len(df), np.nan)
    for mask, X_part in [(train_mask, X_train), (valid_mask, X_valid), (test_mask, X_test)]:
        pred_growth_all[np.where(mask.values)[0]] = model.predict(X_part)
    pred_df["pred_growth_from_lag1_xgboost"] = growth_to_count(pred_growth_all, df)

    # Model 3: XGBoost + lag-1 blended model, with w selected on validation MAE.
    y_val = pred_df.loc[valid_mask.values, TARGET_COL].astype(float).values
    xgb_val = pred_df.loc[valid_mask.values, "pred_growth_from_lag1_xgboost"].astype(float).values
    lag1_val = pred_df.loc[valid_mask.values, "pred_lag1_baseline"].astype(float).values

    w_mae, valid_mae = optimize_blend_weight(y_val, xgb_val, lag1_val, metric="MAE", min_weight=0.0)
    w_rmse, valid_rmse = optimize_blend_weight(y_val, xgb_val, lag1_val, metric="RMSE", min_weight=0.0)
    w_mae_constrained, valid_mae_constrained = optimize_blend_weight(y_val, xgb_val, lag1_val, metric="MAE", min_weight=0.50)

    pred_df["pred_xgboost_lag1_blend"] = np.clip(
        w_mae * pred_df["pred_growth_from_lag1_xgboost"] + (1.0 - w_mae) * pred_df["pred_lag1_baseline"],
        0.0,
        None,
    )

    blend_summary = {
        "w_xgb_valid_MAE": float(w_mae),
        "valid_MAE_at_w": float(valid_mae),
        "w_xgb_valid_RMSE": float(w_rmse),
        "valid_RMSE_at_w": float(valid_rmse),
        "w_xgb_valid_MAE_constrained_min_0_50": float(w_mae_constrained),
        "valid_MAE_constrained_at_w": float(valid_mae_constrained),
        "w_lag1_in_main_blend": float(1.0 - w_mae),
    }

    metrics_rows: List[Dict[str, object]] = []
    metrics_rows += evaluate_by_split(pred_df, "pred_lag1_baseline", "Lag-1 baseline")
    metrics_rows += evaluate_by_split(pred_df, "pred_growth_from_lag1_xgboost", "Growth-from-lag1 XGBoost")
    metrics_rows += evaluate_by_split(pred_df, "pred_xgboost_lag1_blend", "XGBoost + lag-1 blend")
    metrics_df = pd.DataFrame(metrics_rows)

    # Add deltas relative to lag-1 within each split.
    baseline_by_split = metrics_df[metrics_df["model"] == "Lag-1 baseline"].set_index("split")
    metrics_df["delta_MAE_vs_lag1"] = metrics_df.apply(
        lambda r: r["MAE"] - baseline_by_split.loc[r["split"], "MAE"] if r["split"] in baseline_by_split.index else np.nan,
        axis=1,
    )
    metrics_df["delta_RMSE_vs_lag1"] = metrics_df.apply(
        lambda r: r["RMSE"] - baseline_by_split.loc[r["split"], "RMSE"] if r["split"] in baseline_by_split.index else np.nan,
        axis=1,
    )

    imp_df, group_imp_df = get_xgb_importance(model, selected_features)

    # Blend component-level importance: lag-1 component plus XGB gain groups scaled by w.
    component_rows = [{"component": "lag-1 baseline", "share": 1.0 - w_mae}]
    for _, row in group_imp_df.iterrows():
        component_rows.append({"component": f"XGB: {row['group']}", "share": w_mae * float(row["gain_share"])})
    component_imp_df = pd.DataFrame(component_rows).sort_values("share", ascending=False)

    # Save artefacts.
    pred_df.to_csv(OUTPUT_DIR / "report_three_model_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "report_three_model_metrics_all_splits.csv", index=False)
    metrics_df[metrics_df["split"] == "test"].to_csv(OUTPUT_DIR / "report_table_test_metrics.csv", index=False)
    imp_df.to_csv(OUTPUT_DIR / "xgb_feature_importance_gain.csv", index=False)
    group_imp_df.to_csv(OUTPUT_DIR / "xgb_feature_group_importance.csv", index=False)
    component_imp_df.to_csv(OUTPUT_DIR / "blend_component_importance.csv", index=False)
    pd.DataFrame([blend_summary]).to_csv(OUTPUT_DIR / "blend_weight_summary.csv", index=False)

    model.save_model(str(OUTPUT_DIR / "growth_from_lag1_xgboost_model.json"))
    joblib.dump(model, OUTPUT_DIR / "growth_from_lag1_xgboost_model.joblib")
    with open(OUTPUT_DIR / "xgb_selected_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)
    with open(OUTPUT_DIR / "report_model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "target": TARGET_COL,
            "total_cpi_feature_used": TOTAL_CPI_COL,
            "cpi_categories_used": False,
            "model_split": {
                "train_end": TRAIN_END_Q,
                "validation": f"{VALID_START_Q}-{VALID_END_Q}",
                "test": f"{TEST_START_Q}-{TEST_END_Q}",
            },
            "xgb_params": xgb_params(),
            "blend_summary": blend_summary,
            "excluded_direct_lag1_target_features": [
                f"{TARGET_COL}_lag1q",
                f"log1p_{TARGET_COL}_lag1q",
                "computed_homelessness_rate_per_1000_lag1q",
                "log1p_computed_homelessness_rate_per_1000_lag1q",
            ],
        }, f, indent=2)

    return pred_df, metrics_df, imp_df, group_imp_df, blend_summary


# =============================================================================
# 6. PLOTS FOR REPORT SECTION
# =============================================================================

def make_report_plots(metrics_df: pd.DataFrame, group_imp_df: pd.DataFrame, blend_summary: Dict[str, float]) -> None:
    print_section("Saving report plots")
    test = metrics_df[metrics_df["split"] == "test"].copy()
    test = test.set_index("model").loc[["Lag-1 baseline", "Growth-from-lag1 XGBoost", "XGBoost + lag-1 blend"]].reset_index()

    # Figure 1: test MAE comparison.
    plt.figure(figsize=(8, 5))
    plt.bar(test["model"], test["MAE"])
    plt.ylabel("Test MAE")
    plt.title("Test-set MAE comparison")
    plt.xticks(rotation=20, ha="right")
    save_figure(OUTPUT_DIR / "fig_test_mae_comparison.png")

    # Figure 2: grouped MAE/RMSE comparison.
    x = np.arange(len(test))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, test["MAE"], width, label="MAE")
    plt.bar(x + width / 2, test["RMSE"], width, label="RMSE")
    plt.xticks(x, test["model"], rotation=20, ha="right")
    plt.ylabel("Error on count scale")
    plt.title("Test MAE and RMSE by model")
    plt.legend()
    save_figure(OUTPUT_DIR / "fig_test_mae_rmse_grouped.png")

    # Figure 3: deltas against lag-1 baseline.
    delta = test[test["model"] != "Lag-1 baseline"]
    x = np.arange(len(delta))
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, delta["delta_MAE_vs_lag1"], width, label="Delta MAE")
    plt.bar(x + width / 2, delta["delta_RMSE_vs_lag1"], width, label="Delta RMSE")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, delta["model"], rotation=20, ha="right")
    plt.ylabel("Change versus lag-1 baseline")
    plt.title("Change in test errors relative to lag-1")
    plt.legend()
    save_figure(OUTPUT_DIR / "fig_delta_metrics_vs_lag1.png")

    # Figure 4: XGBoost feature group gain importance.
    grp = group_imp_df[group_imp_df["gain_share"] > 0].head(12).copy()
    if len(grp) > 0:
        plt.figure(figsize=(8, 6))
        plt.barh(grp["group"][::-1], grp["gain_share"][::-1] * 100.0)
        plt.xlabel("Gain share (%)")
        plt.title("Growth-from-lag1 XGBoost feature-group importance")
        save_figure(OUTPUT_DIR / "fig_xgb_feature_group_importance.png")

    # Figure 5: blend component shares.
    w_xgb = blend_summary["w_xgb_valid_MAE"]
    w_lag1 = 1.0 - w_xgb
    component = pd.DataFrame({"component": ["lag-1 baseline", "Growth-from-lag1 XGBoost"], "share": [w_lag1, w_xgb]})
    plt.figure(figsize=(7, 4))
    plt.bar(component["component"], component["share"] * 100.0)
    plt.ylabel("Blend share (%)")
    plt.title("Blend component weights selected on validation MAE")
    plt.xticks(rotation=15, ha="right")
    save_figure(OUTPUT_DIR / "fig_blend_component_weights.png")


# =============================================================================
# 7. MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three report XGBoost-section homelessness models.")
    parser.add_argument("--monthly", type=str, default=DEFAULT_MONTHLY_FILE, help="Path to monthly_lad_panel_2000_2025_new.csv")
    parser.add_argument("--homeless", type=str, default=DEFAULT_HOMELESS_FILE, help="Path to homeless_19_25_cleaned.csv")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output folder")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively as well as saving them")
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR, SHOW_PLOTS
    args = parse_args()
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHOW_PLOTS = bool(args.show_plots)

    monthly_file = resolve_input_path(args.monthly, "monthly_lad_panel_2000_2025_new.csv")
    homeless_file = resolve_input_path(args.homeless, "homeless_19_25_cleaned.csv")

    print_section("Report XGBoost-only modelling script")
    print(f"Monthly feature file: {monthly_file}")
    print(f"Homelessness file:   {homeless_file}")
    print(f"Output directory:    {OUTPUT_DIR.resolve()}")
    print("CPI rule: using only cpi_00_all_items; all CPI category columns are excluded.")

    qdf = build_quarterly_panel(monthly_file, homeless_file)
    qdf, same_quarter_exog, lagged_exog, target_history = add_report_features(qdf)
    model_df, feature_cols = build_report_model_table(qdf, same_quarter_exog, lagged_exog, target_history)

    # Optional save for checking the engineered modelling table.
    check_df = model_df[["lad_code", "lad_name", "quarter", TARGET_COL, f"{TARGET_COL}_lag1q", "split", "target_growth_from_lag1"]].copy()
    check_df["quarter"] = check_df["quarter"].astype(str)
    check_df.to_csv(OUTPUT_DIR / "modelling_rows_check.csv", index=False)

    pred_df, metrics_df, imp_df, group_imp_df, blend_summary = run_three_report_models(model_df, feature_cols)
    make_report_plots(metrics_df, group_imp_df, blend_summary)

    print_section("Final report test table")
    test_table = metrics_df[metrics_df["split"] == "test"].copy()
    test_table = test_table[["model", "n", "MAE", "delta_MAE_vs_lag1", "RMSE", "delta_RMSE_vs_lag1", "R2", "SMAPE_percent"]]
    print(test_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print_section("Blend weight")
    print(f"Main validation-MAE blend weight on XGBoost: {blend_summary['w_xgb_valid_MAE']:.2f}")
    print(f"Main validation-MAE blend weight on lag-1:    {blend_summary['w_lag1_in_main_blend']:.2f}")
    print(f"RMSE-selected XGBoost weight also saved:     {blend_summary['w_xgb_valid_RMSE']:.2f}")
    print(f"Constrained min-0.50 XGBoost weight saved:  {blend_summary['w_xgb_valid_MAE_constrained_min_0_50']:.2f}")

    print_section("Top XGBoost feature groups")
    print(group_imp_df.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_section("Outputs saved")
    print(f"Folder: {OUTPUT_DIR.resolve()}")
    print("Main files:")
    for name in [
        "report_table_test_metrics.csv",
        "report_three_model_metrics_all_splits.csv",
        "report_three_model_predictions.csv",
        "xgb_feature_group_importance.csv",
        "blend_weight_summary.csv",
        "growth_from_lag1_xgboost_model.json",
        "xgb_selected_feature_columns.json",
    ]:
        print(f"  - {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
