"""Forecast homelessness with a lagged-feature XGBoost model.

The script builds a quarterly LAD panel, trains a growth-from-lag1 model,
calibrates/blends it on the validation split, and saves report-ready figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required. Install it with: pip install xgboost") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# Config

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MONTHLY_FILE = str(BASE_DIR / "data" / "clean" / "Final all Junxi.gz")

OUTPUT_DIR = Path("xgboost_improved_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = max(1, min(4, os.cpu_count() or 4))

TARGET_COL = "homelessness_total_assessments"
AUX_HOMELESSNESS_COLS = ["homelessness_relief", "homelessness_per_1000"]
TOTAL_CPI_COL = "cpi_00_all_items"

# Chronological split.
MODELLING_START_QUARTER = "2019Q3"
TRAIN_END_Q = "2023Q4"
VALID_START_Q = "2024Q1"
VALID_END_Q = "2024Q4"
TEST_START_Q = "2025Q1"
TEST_END_Q = "2025Q4"

# Feature engineering.
LAGS_Q = [1, 2, 4, 8]
ROLL_WINDOWS_Q = [4, 8]
ADD_LAD_DUMMIES = True
FORCE_KEEP_LAD_DUMMIES = False
ADD_TIME_FEATURES = True

# Feature selection.
DROP_FEATURES_MISSING_ABOVE = 0.80
PAIRWISE_CORR_PRUNE_THRESHOLD = 0.985
TARGET_CORR_MIN_ABS = 0.005
TOP_K_FEATURES: Optional[int] = None
MIN_AUTO_FEATURES = 50
MAX_AUTO_FEATURES: Optional[int] = None

# Main XGBoost settings.
XGB_N_ESTIMATORS = 1600
XGB_LEARNING_RATE = 0.025
XGB_MAX_DEPTH = 3
XGB_MIN_CHILD_WEIGHT = 10
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_REG_ALPHA = 0.05
XGB_REG_LAMBDA = 3.0
XGB_EARLY_STOPPING_ROUNDS = 100

# Smaller XGBoost used only for feature selection.
QUICK_XGB_N_ESTIMATORS = 350
QUICK_XGB_LEARNING_RATE = 0.05
QUICK_XGB_EARLY_STOPPING_ROUNDS = 50

# Plot settings.
SAVE_PLOTS = True
SHOW_PLOTS = True

ACADEMIC_COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "red": "#C1121F",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "black": "#111827",
}
ACADEMIC_PALETTE = [
    ACADEMIC_COLORS["blue"],
    ACADEMIC_COLORS["vermillion"],
    ACADEMIC_COLORS["green"],
    ACADEMIC_COLORS["purple"],
    ACADEMIC_COLORS["orange"],
    ACADEMIC_COLORS["gray"],
]
SCENARIO_LABELS = {
    "lagged_only_no_same_quarter_exog_forecast": "Forecast",
}
MODEL_LABELS = {
    "Lag-1 baseline": "Lag-1 baseline",
    "Growth-from-lag1 XGBoost raw": "XGBoost raw",
    "Growth-from-lag1 XGBoost + bias correction": "XGBoost + bias correction",
    "XGBoost-bc + lag-1 blend, unrestricted": "XGBoost + lag-1 blend",
    "Final blend + bias correction": "Final blend + correction",
}
SERIES_LABELS = {
    "actual": "Actual",
    "predicted": "Predicted",
}
PLOT_MODEL_ORDER = [
    "Lag-1 baseline",
    "Growth-from-lag1 XGBoost + bias correction",
    "Final blend + bias correction",
]
PLOT_MODEL_LABELS = {
    "Lag-1 baseline": "Baseline",
    "Growth-from-lag1 XGBoost + bias correction": "XGBoost",
    "Final blend + bias correction": "Blend",
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Palatino", "serif"],
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.linewidth": 0.8,
    "axes.edgecolor": ACADEMIC_COLORS["black"],
    "axes.labelcolor": ACADEMIC_COLORS["black"],
    "xtick.color": ACADEMIC_COLORS["black"],
    "ytick.color": ACADEMIC_COLORS["black"],
    "axes.unicode_minus": False,
})

sns.set_theme(
    context="paper",
    style="whitegrid",
    palette=ACADEMIC_PALETTE,
    rc={
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Palatino", "serif"],
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)


# Utilities

def log(message: str, level: str = "INFO") -> None:
    """Print a timestamped log message."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{level}] {message}")


def print_section(title: str) -> None:
    """Print a visible section heading in the console."""
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def plot_text(value: object) -> str:
    """LaTeX-safe plot text with underscores shown as spaces."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = text.replace("_", " ")
    text = text.replace("—", "-").replace("–", "-")
    for src, dst in {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }.items():
        text = text.replace(src, dst)
    return text


def scenario_plot_label(scenario_name: str) -> str:
    """Return the short label used in plot titles and legends."""
    return plot_text(SCENARIO_LABELS.get(scenario_name, scenario_name))


def model_plot_label(model_name: str) -> str:
    """Return the short display name for a model."""
    return plot_text(PLOT_MODEL_LABELS.get(model_name, MODEL_LABELS.get(model_name, model_name)))


def series_plot_label(series_name: str) -> str:
    """Return the display name for a plotted series."""
    return plot_text(SERIES_LABELS.get(series_name, series_name))


def prepare_plot_category(series: pd.Series) -> pd.Series:
    """Clean category labels before sending them to Matplotlib."""
    return series.astype(str).map(plot_text)


def midpoint_time(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    """Return the midpoint between two timestamps."""
    return start + (end - start) / 2


def polish_axes(ax: plt.Axes) -> None:
    """Apply the shared report-figure axis style."""
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(False, axis="x")
    sns.despine(ax=ax)


def quarter_to_period(q: str | pd.Period) -> pd.Period:
    """Convert a quarter string to a quarterly Period."""
    if isinstance(q, pd.Period):
        return q.asfreq("Q")
    return pd.Period(str(q), freq="Q")


def is_real_english_lad_code(s: pd.Series) -> pd.Series:
    """Identify real English LAD rows."""
    return s.astype(str).str.match(r"^E0[6789]", na=False)


def safe_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Apply log1p after clipping negative values to zero."""
    return np.log1p(np.clip(x, 0, None))


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    """Divide arrays while returning NaN for invalid denominators."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the metrics used throughout the report."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "SMAPE_percent": np.nan,
            "mean_actual": np.nan,
            "mean_predicted": np.nan,
            "bias_actual_minus_predicted": np.nan,
        }
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


def infer_intended_feature_type(feature: str) -> str:
    """Label a feature with its intended numeric type."""
    if feature.startswith("lad_"):
        return "int8_dummy"
    if feature in {"post_covid_2020plus", "cost_of_living_shock_2022plus"}:
        return "int8_binary"
    if feature in {"year", "quarter_num", "quarter_index"}:
        return "int16_time"
    if feature in {"quarter_sin", "quarter_cos"}:
        return "float32_time"
    return "float32_numeric"


def ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a feature table to numeric dtypes for XGBoost."""
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        intended = infer_intended_feature_type(c)
        if intended.startswith("int8") and out[c].isna().sum() == 0:
            out[c] = out[c].astype(np.int8)
        elif intended.startswith("int16") and out[c].isna().sum() == 0:
            out[c] = out[c].astype(np.int16)
        else:
            out[c] = out[c].astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan)


def save_figure(path: Path) -> None:
    """Save the current figure and optionally show it interactively."""
    for ax in plt.gcf().axes:
        polish_axes(ax)
    if SAVE_PLOTS:
        plt.tight_layout(pad=0.4)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close()


def resolve_input_path(user_path: str, fallback_filename: str) -> Path:
    """Resolve an input path with local fallbacks."""
    candidates = [
        Path(user_path),
        BASE_DIR / "data" / "clean" / "Final all Junxi.gz",
        Path(__file__).resolve().parent / fallback_filename,
        Path.cwd() / fallback_filename,
        Path.cwd() / "data" / "clean" / "Final all Junxi.gz",
        Path("/mnt/data") / fallback_filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find input file. Tried:\n  {tried}\nPass --monthly with the correct path.")


def write_json(path: Path, obj: Dict[str, object]) -> None:
    """Write a JSON file with stable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# Data loading

def monthly_columns_to_read(csv_path: Path) -> List[str]:
    """Choose the monthly columns needed for this model."""
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
        TARGET_COL,
        *AUX_HOMELESSNESS_COLS,
    ]
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in report_cols if c in available]
    missing = [c for c in ["year", "month", "lad_code", "lad_name", TOTAL_CPI_COL, TARGET_COL] if c not in usecols]
    if missing:
        raise ValueError(f"Latest monthly panel is missing required columns: {missing}")
    excluded_cpi = [c for c in available if c.startswith("cpi_") and c != TOTAL_CPI_COL]
    log(f"Available columns in latest CSV: {len(available):,}")
    log(f"Columns read for this model: {len(usecols):,}")
    log(f"CPI category columns intentionally excluded: {len(excluded_cpi):,}")
    return usecols


def load_monthly_panel(monthly_file: Path) -> pd.DataFrame:
    """Load the monthly file and add dates, quarters, and basic types."""
    print_section("Loading latest integrated monthly panel")
    usecols = monthly_columns_to_read(monthly_file)
    df = pd.read_csv(monthly_file, usecols=usecols)

    log(f"Raw monthly rows read: {len(df):,}")
    df = df[is_real_english_lad_code(df["lad_code"])].copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q")

    df["lad_code"] = df["lad_code"].astype("category")
    df["lad_name"] = df["lad_name"].astype("category")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int8")

    for c in df.columns:
        if c not in {"lad_code", "lad_name", "date", "quarter", "year", "month"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    target_nonmissing = int(df[TARGET_COL].notna().sum())
    target_quarters = df.loc[df[TARGET_COL].notna(), "quarter"]

    print(f"Monthly rows after keeping real LADs: {len(df):,}")
    print(f"LADs: {df['lad_code'].nunique():,}")
    print(f"Month range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Quarter range: {df['quarter'].min()} to {df['quarter'].max()}")
    print(f"Rows with non-missing target: {target_nonmissing:,}")
    if target_nonmissing:
        print(f"Target quarter range: {target_quarters.min()} to {target_quarters.max()}")

    # Guard against inconsistent target values within the same LAD-quarter.
    target_check = (
        df[df[TARGET_COL].notna()]
        .groupby(["lad_code", "quarter"], observed=True)[TARGET_COL]
        .nunique(dropna=True)
    )
    inconsistent = int((target_check > 1).sum()) if len(target_check) else 0
    print(f"LAD-quarter target consistency check: {inconsistent:,} groups have >1 unique monthly target value")
    print(f"CPI columns read: {[c for c in df.columns if c.startswith('cpi_')]}")
    return df


def monthly_to_quarterly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly rows into a full quarterly LAD panel."""
    print_section("Aggregating monthly features to quarterly LAD panel")
    numeric_cols = [c for c in monthly.columns if c not in {"lad_code", "lad_name", "date", "quarter"}]
    agg = {c: "mean" for c in numeric_cols}
    agg["lad_name"] = "last"

    qdf = monthly.groupby(["lad_code", "quarter"], observed=True, as_index=False).agg(agg)

    # Keep lags from jumping over missing quarters.
    lad_names = qdf.groupby("lad_code", observed=True)["lad_name"].agg(
        lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan
    )
    all_lads = sorted(qdf["lad_code"].astype(str).unique())
    all_quarters = pd.period_range(qdf["quarter"].min(), qdf["quarter"].max(), freq="Q")
    full_index = pd.MultiIndex.from_product([all_lads, all_quarters], names=["lad_code", "quarter"])
    qdf = qdf.assign(lad_code=qdf["lad_code"].astype(str)).set_index(["lad_code", "quarter"]).reindex(full_index).reset_index()
    qdf["lad_name"] = qdf["lad_code"].map(lad_names.astype(str).to_dict())

    qdf["quarter_date"] = qdf["quarter"].dt.start_time
    qdf["year"] = qdf["quarter"].dt.year.astype(np.int16)
    qdf["quarter_num"] = qdf["quarter"].dt.quarter.astype(np.int8)
    qdf["quarter_index"] = ((qdf["year"] - int(qdf["year"].min())) * 4 + qdf["quarter_num"]).astype(np.int16)
    qdf["lad_code"] = qdf["lad_code"].astype("category")
    qdf["lad_name"] = qdf["lad_name"].astype("category")

    print(f"Quarterly feature-grid rows: {len(qdf):,}")
    print(f"Quarterly feature range: {qdf['quarter'].min()} to {qdf['quarter'].max()}")
    print(f"Rows with non-missing quarterly target: {qdf[TARGET_COL].notna().sum():,}")
    target_by_q = qdf.groupby("quarter", observed=True)[TARGET_COL].apply(lambda s: int(s.notna().sum()))
    print("Target rows by quarter, tail:")
    print(target_by_q[target_by_q > 0].tail(12).to_string())
    return qdf


def build_quarterly_panel(monthly_file: Path) -> pd.DataFrame:
    """Build the quarterly panel from the monthly input file."""
    monthly = load_monthly_panel(monthly_file)
    qdf = monthly_to_quarterly_features(monthly)
    return qdf


# Feature engineering

def group_shift(df: pd.DataFrame, value_col: str, lag: int) -> pd.Series:
    """Shift a column within each LAD."""
    return df.groupby("lad_code", observed=True, sort=False)[value_col].shift(lag)


def add_report_features(qdf: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Create model features and return the feature groups."""
    print_section("Feature engineering with group-safe lags")
    qdf = qdf.sort_values(["lad_code", "quarter"]).copy()

    # Keep only the total CPI series.
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

    # Affordability and living-cost ratios.
    constructed: List[str] = []
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

    # Log versions of positive-valued controls.
    positive_for_log = [
        "average_house_price", "seasonally_adjusted_average_house_price", "house_price_index",
        "house_sales_volume", "unemployment_count", "private_rental_price_index",
        "average_private_rental_price", "income", "brent_oil_price", "population",
        TOTAL_CPI_COL, "real_income_cpi_adjusted", "real_house_price_cpi_adjusted",
        "house_price_to_income", "annual_rent_to_income", "unemployment_per_1000",
    ]
    log_features: List[str] = []
    for c in positive_for_log:
        if c in qdf.columns:
            new_c = f"log1p_{c}"
            qdf[new_c] = safe_log1p(qdf[c])
            log_features.append(new_c)
    same_quarter_exog += log_features

    # LAD-level growth rates.
    pct_features: List[str] = []
    for c in base_exog + constructed:
        if c not in qdf.columns:
            continue
        prev1 = group_shift(qdf, c, 1)
        prev4 = group_shift(qdf, c, 4)
        qoq = f"{c}_qoq_pct"
        yoy = f"{c}_yoy_pct"
        qdf[qoq] = np.where((prev1.notna()) & (prev1 != 0), (qdf[c] / prev1 - 1.0) * 100.0, np.nan)
        qdf[yoy] = np.where((prev4.notna()) & (prev4 != 0), (qdf[c] / prev4 - 1.0) * 100.0, np.nan)
        pct_features.extend([qoq, yoy])
    same_quarter_exog += pct_features

    # Lagged external predictors.
    lagged_exog: List[str] = []
    for c in same_quarter_exog:
        if c not in qdf.columns:
            continue
        for lag in LAGS_Q:
            new_c = f"{c}_lag{lag}q"
            qdf[new_c] = group_shift(qdf, c, lag)
            lagged_exog.append(new_c)

    # Past target values only.
    target_history: List[str] = []
    qdf["log1p_homelessness_total"] = safe_log1p(qdf[TARGET_COL])
    for lag in LAGS_Q:
        count_col = f"{TARGET_COL}_lag{lag}q"
        log_col = f"log1p_{TARGET_COL}_lag{lag}q"
        qdf[count_col] = group_shift(qdf, TARGET_COL, lag)
        qdf[log_col] = safe_log1p(qdf[count_col])
        target_history += [count_col, log_col]

        if "computed_homelessness_rate_per_1000" in qdf.columns:
            rate_col = f"computed_homelessness_rate_per_1000_lag{lag}q"
            log_rate_col = f"log1p_computed_homelessness_rate_per_1000_lag{lag}q"
            qdf[rate_col] = group_shift(qdf, "computed_homelessness_rate_per_1000", lag)
            qdf[log_rate_col] = safe_log1p(qdf[rate_col])
            target_history += [rate_col, log_rate_col]

        if "homelessness_per_1000" in qdf.columns:
            official_rate_col = f"homelessness_per_1000_lag{lag}q"
            qdf[official_rate_col] = group_shift(qdf, "homelessness_per_1000", lag)
            target_history += [official_rate_col]

        if "homelessness_relief" in qdf.columns:
            relief_col = f"homelessness_relief_lag{lag}q"
            log_relief_col = f"log1p_homelessness_relief_lag{lag}q"
            qdf[relief_col] = group_shift(qdf, "homelessness_relief", lag)
            qdf[log_relief_col] = safe_log1p(qdf[relief_col])
            target_history += [relief_col, log_relief_col]

    # Rolling target history.
    grouped = qdf.groupby("lad_code", observed=True, sort=False)
    for win in ROLL_WINDOWS_Q:
        mean_col = f"{TARGET_COL}_rolling{win}_mean_lag1q"
        std_col = f"{TARGET_COL}_rolling{win}_std_lag1q"
        qdf[mean_col] = grouped[TARGET_COL].transform(lambda s: s.shift(1).rolling(win, min_periods=2).mean())
        qdf[std_col] = grouped[TARGET_COL].transform(lambda s: s.shift(1).rolling(win, min_periods=2).std())
        target_history += [mean_col, std_col]

    # Previous target growth.
    qdf["target_log_growth_lag1q"] = grouped["log1p_homelessness_total"].transform(lambda s: s.diff(1).shift(1))
    qdf["target_log_growth_lag4q"] = grouped["log1p_homelessness_total"].transform(lambda s: s.diff(4).shift(1))
    target_history += ["target_log_growth_lag1q", "target_log_growth_lag4q"]

    # Time controls.
    qdf["quarter_sin"] = np.sin(2.0 * np.pi * qdf["quarter_num"].astype(float) / 4.0).astype(np.float32)
    qdf["quarter_cos"] = np.cos(2.0 * np.pi * qdf["quarter_num"].astype(float) / 4.0).astype(np.float32)
    qdf["post_covid_2020plus"] = (qdf["quarter"] >= pd.Period("2020Q2", freq="Q")).astype(np.int8)
    qdf["cost_of_living_shock_2022plus"] = (qdf["quarter"] >= pd.Period("2022Q1", freq="Q")).astype(np.int8)

    qdf = qdf.replace([np.inf, -np.inf], np.nan)
    same_quarter_exog = list(dict.fromkeys([c for c in same_quarter_exog if c in qdf.columns]))
    lagged_exog = list(dict.fromkeys([c for c in lagged_exog if c in qdf.columns]))
    target_history = list(dict.fromkeys([c for c in target_history if c in qdf.columns]))

    print(f"Same-quarter exogenous features: {len(same_quarter_exog):,}")
    print(f"Lagged exogenous features: {len(lagged_exog):,}")
    print(f"Target-history features before exclusions: {len(target_history):,}")
    print(f"CPI check after feature engineering: {[c for c in qdf.columns if c.startswith('cpi_')][:10]}")
    log("All lag, rolling, diff and growth features were created after groupby('lad_code').")
    return qdf, same_quarter_exog, lagged_exog, target_history


def feature_group(feature: str) -> str:
    """Map a feature name to a readable group."""
    f = feature.lower()
    if "homelessness_total_assessments" in f or "homelessness_rate" in f or "homelessness_per_1000" in f or "homelessness_relief" in f or "target_log_growth" in f:
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


# Modelling table

def add_lad_dummies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add LAD fixed-effect dummy columns when requested."""
    if not ADD_LAD_DUMMIES:
        return df, []
    dummies = pd.get_dummies(df["lad_code"].astype(str), prefix="lad", dtype=np.int8)
    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return out, dummies.columns.tolist()


def build_model_table(
    qdf: pd.DataFrame,
    same_quarter_exog: List[str],
    lagged_exog: List[str],
    target_history: List[str],
    include_same_quarter_exog: bool,
    scenario_name: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the train/validation/test table for one forecast setup."""
    print_section(f"Building modelling table: {scenario_name}")
    df = qdf.copy()
    start_q = quarter_to_period(MODELLING_START_QUARTER)
    df = df[(df["quarter"] >= start_q) & df[TARGET_COL].notna()].copy()

    lag1_col = f"{TARGET_COL}_lag1q"
    log_lag1_col = f"log1p_{TARGET_COL}_lag1q"
    if lag1_col not in df.columns or log_lag1_col not in df.columns:
        raise ValueError("Lag-1 target features were not created correctly.")

    # Predict growth from the previous observed quarter.
    df = df[df[lag1_col].notna()].copy()
    df["target_growth_from_lag1"] = safe_log1p(df[TARGET_COL]) - safe_log1p(df[lag1_col])

    # Candidate features.
    feature_cols: List[str] = []
    if include_same_quarter_exog:
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

    # Drop contemporaneous outcome information.
    forbidden_exact = {
        TARGET_COL,
        "homelessness_threatened",
        "homelessness_relief",
        "homelessness_per_1000",
        "computed_homelessness_rate_per_1000",
        "log1p_homelessness_total",
        "target_growth_from_lag1",
    }

    # Lag-1 count is used after prediction, so keep it out of XGBoost.
    direct_lag1_target_features = {
        f"{TARGET_COL}_lag1q",
        f"log1p_{TARGET_COL}_lag1q",
        "computed_homelessness_rate_per_1000_lag1q",
        "log1p_computed_homelessness_rate_per_1000_lag1q",
    }

    feature_cols = [c for c in feature_cols if c not in forbidden_exact]
    feature_cols = [c for c in feature_cols if c not in direct_lag1_target_features]
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
    print(f"Include same-quarter exogenous features: {include_same_quarter_exog}")
    print(f"Target period after dropping missing target/lag1: {df['quarter'].min()} to {df['quarter'].max()}")
    print("Rows by split:")
    print(split_counts.to_string())
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        print(f"  {split}: quarters {sub['quarter'].min()} to {sub['quarter'].max()}, LADs {sub['lad_code'].nunique():,}")

    feature_type_report = make_feature_type_report(df, feature_cols)
    feature_type_report.to_csv(out_dir / f"candidate_feature_types_{scenario_name}.csv", index=False)
    print("Candidate feature intended-type counts:")
    print(feature_type_report["intended_type"].value_counts().to_string())
    print("Candidate feature group counts:")
    print(feature_type_report["feature_group"].value_counts().head(15).to_string())
    return df, feature_cols


def make_feature_type_report(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Summarise feature types and missingness."""
    rows = []
    for c in features:
        s = df[c] if c in df.columns else pd.Series(dtype=float)
        rows.append({
            "feature": c,
            "intended_type": infer_intended_feature_type(c),
            "actual_pandas_dtype": str(s.dtype),
            "feature_group": feature_group(c),
            "missing_rate_all_rows": float(s.isna().mean()) if len(s) else np.nan,
            "nunique_all_rows": int(s.nunique(dropna=True)) if len(s) else 0,
        })
    return pd.DataFrame(rows)


def infer_panel_column_type(column: str, series: pd.Series) -> str:
    """Label a panel column with a compact type name."""
    if column in {"lad_code", "lad_name"}:
        return "category_identifier"
    if column == "quarter":
        return "period_quarter"
    if column == "quarter_date" or pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if column in {"year", "month", "quarter_num", "quarter_index"}:
        return "int_time"
    if column in {TARGET_COL, *AUX_HOMELESSNESS_COLS, "computed_homelessness_rate_per_1000", "log1p_homelessness_total"}:
        return "float_target_or_target_derived"
    if column.startswith("lad_"):
        return "int8_dummy"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return "float"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return "other"


def make_panel_column_type_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise all columns in the engineered panel."""
    rows = []
    for c in df.columns:
        s = df[c]
        if c in {"lad_code", "lad_name", "quarter", "quarter_date"}:
            role = "identifier_or_time_index"
        elif c == TARGET_COL:
            role = "target"
        elif c in AUX_HOMELESSNESS_COLS or "homelessness" in c or "target_log_growth" in c:
            role = "target_history_or_auxiliary"
        else:
            role = "predictor_or_engineered_predictor"
        rows.append({
            "column": c,
            "role": role,
            "explicit_type": infer_panel_column_type(c, s),
            "actual_pandas_dtype": str(s.dtype),
            "missing_rate": float(s.isna().mean()) if len(s) else np.nan,
            "nunique": int(s.nunique(dropna=True)) if len(s) else 0,
        })
    return pd.DataFrame(rows)


# Feature selection

def protected_feature_mask(feature: str) -> bool:
    """Mark core features that should survive cleaning when possible."""
    if FORCE_KEEP_LAD_DUMMIES and feature.startswith("lad_"):
        return True
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


def quick_xgb_params() -> Dict[str, object]:
    """Return parameters for the feature-selection XGBoost."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": QUICK_XGB_N_ESTIMATORS,
        "learning_rate": QUICK_XGB_LEARNING_RATE,
        "max_depth": XGB_MAX_DEPTH,
        "min_child_weight": max(3, XGB_MIN_CHILD_WEIGHT // 2),
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "reg_alpha": XGB_REG_ALPHA,
        "reg_lambda": XGB_REG_LAMBDA,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "early_stopping_rounds": QUICK_XGB_EARLY_STOPPING_ROUNDS,
    }


def fit_quick_xgb_selector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBRegressor:
    """Fit the smaller XGBoost model used to rank features."""
    params = quick_xgb_params()
    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    except TypeError:
        es_rounds = int(params.pop("early_stopping_rounds"))
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=es_rounds)
    return model


def correlation_prune_features(
    X_filled: pd.DataFrame,
    ranking: pd.DataFrame,
    protected: set[str],
    out_path: Path,
) -> Tuple[List[str], pd.DataFrame]:
    """Drop redundant correlated features."""
    if X_filled.shape[1] <= 1:
        return X_filled.columns.tolist(), pd.DataFrame()

    ordered = ranking.copy()
    ordered["protected"] = ordered["feature"].isin(protected)
    ordered = ordered.sort_values(
        ["protected", "abs_target_corr", "univariate_f", "missing_rate"],
        ascending=[False, False, False, True],
    )
    ordered_features = [f for f in ordered["feature"].tolist() if f in X_filled.columns]
    corr = X_filled[ordered_features].corr().abs()

    kept: List[str] = []
    dropped_rows: List[Dict[str, object]] = []
    for f in ordered_features:
        if f in protected:
            kept.append(f)
            continue
        if not kept:
            kept.append(f)
            continue
        max_corr = corr.loc[f, kept].max()
        if pd.notna(max_corr) and max_corr >= PAIRWISE_CORR_PRUNE_THRESHOLD:
            matched = corr.loc[f, kept].idxmax()
            dropped_rows.append({
                "dropped_feature": f,
                "kept_correlated_feature": matched,
                "abs_pairwise_corr": float(max_corr),
                "threshold": PAIRWISE_CORR_PRUNE_THRESHOLD,
                "dropped_group": feature_group(f),
                "kept_group": feature_group(matched),
            })
        else:
            kept.append(f)

    dropped_df = pd.DataFrame(dropped_rows)
    dropped_df.to_csv(out_path, index=False)
    return kept, dropped_df


def clean_and_select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scenario_name: str,
    out_dir: Path,
) -> Tuple[List[str], pd.DataFrame]:
    """Clean, rank, and select the final feature set."""
    print_section(f"Feature cleaning and selection: {scenario_name}")
    train_mask = df["split"] == "train"
    valid_mask = df["split"] == "valid"

    X_train = ensure_numeric_df(df.loc[train_mask, feature_cols])
    y_train = df.loc[train_mask, "target_growth_from_lag1"].astype(float)
    X_valid = ensure_numeric_df(df.loc[valid_mask, feature_cols])
    y_valid = df.loc[valid_mask, "target_growth_from_lag1"].astype(float)

    protected = {c for c in feature_cols if protected_feature_mask(c)}

    # Screen on the training split only.
    missing_rate = X_train.isna().mean()
    nunique = X_train.nunique(dropna=True)
    keep_after_missing: List[str] = []
    dropped_missing_constant: List[Dict[str, object]] = []
    for c in feature_cols:
        reason = None
        if missing_rate.get(c, 1.0) >= 1.0:
            reason = "all_missing_in_train"
        elif nunique.get(c, 0) <= 1:
            reason = "constant_in_train"
        elif c not in protected and missing_rate.get(c, 1.0) > DROP_FEATURES_MISSING_ABOVE:
            reason = f"missing_rate_above_{DROP_FEATURES_MISSING_ABOVE}"

        if reason is None:
            keep_after_missing.append(c)
        else:
            dropped_missing_constant.append({
                "feature": c,
                "reason": reason,
                "missing_rate_train": float(missing_rate.get(c, np.nan)),
                "nunique_train": int(nunique.get(c, 0)),
                "feature_group": feature_group(c),
            })

    pd.DataFrame(dropped_missing_constant).to_csv(out_dir / f"dropped_missing_constant_{scenario_name}.csv", index=False)
    X_train = X_train[keep_after_missing]
    X_valid = X_valid[keep_after_missing]
    protected = {c for c in keep_after_missing if c in protected}

    # Rank simple target relationships.
    med = X_train.median(numeric_only=True)
    X_filled = X_train.fillna(med).fillna(0.0)
    y_aligned = pd.Series(y_train.values, index=X_filled.index, name="target_growth_from_lag1")

    target_corr = X_filled.corrwith(y_aligned, method="pearson").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        spearman_corr = X_filled.corrwith(y_aligned, method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    except Exception:
        spearman_corr = pd.Series(0.0, index=X_filled.columns)

    try:
        scores, pvals = f_regression(X_filled, y_train.values)
    except Exception:
        scores = np.zeros(len(keep_after_missing))
        pvals = np.ones(len(keep_after_missing))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)

    ranking = pd.DataFrame({
        "feature": keep_after_missing,
        "missing_rate": [float(missing_rate.get(c, np.nan)) for c in keep_after_missing],
        "nunique_train": [int(nunique.get(c, 0)) for c in keep_after_missing],
        "target_corr": [float(target_corr.get(c, 0.0)) for c in keep_after_missing],
        "abs_target_corr": [float(abs(target_corr.get(c, 0.0))) for c in keep_after_missing],
        "spearman_target_corr": [float(spearman_corr.get(c, 0.0)) for c in keep_after_missing],
        "abs_spearman_target_corr": [float(abs(spearman_corr.get(c, 0.0))) for c in keep_after_missing],
        "univariate_f": scores,
        "univariate_p": pvals,
    })
    ranking["group"] = ranking["feature"].map(feature_group)
    ranking["intended_type"] = ranking["feature"].map(infer_intended_feature_type)
    ranking["protected_feature"] = ranking["feature"].isin(protected)

    # Remove near-duplicates.
    kept_after_prune, dropped_corr_df = correlation_prune_features(
        X_filled,
        ranking,
        protected,
        out_dir / f"correlation_pruning_dropped_{scenario_name}.csv",
    )
    ranking["kept_after_correlation_pruning"] = ranking["feature"].isin(kept_after_prune)

    # Add quick-XGBoost signal.
    X_train_pruned = ensure_numeric_df(df.loc[train_mask, kept_after_prune])
    X_valid_pruned = ensure_numeric_df(df.loc[valid_mask, kept_after_prune])
    quick_model = fit_quick_xgb_selector(X_train_pruned, y_train, X_valid_pruned, y_valid)
    booster = quick_model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    quick_gain = pd.Series({f: float(gain_scores.get(f, 0.0)) for f in kept_after_prune})
    quick_weight = pd.Series({f: float(weight_scores.get(f, 0.0)) for f in kept_after_prune})
    ranking["quick_xgb_gain"] = ranking["feature"].map(quick_gain).fillna(0.0)
    ranking["quick_xgb_weight"] = ranking["feature"].map(quick_weight).fillna(0.0)
    ranking["quick_xgb_gain_positive"] = ranking["quick_xgb_gain"] > 0

    # Combine the ranking signals.
    for col in ["abs_target_corr", "abs_spearman_target_corr", "univariate_f", "quick_xgb_gain"]:
        ranking[f"{col}_rank_pct"] = ranking[col].rank(pct=True, method="average")
    ranking["combined_selection_score"] = (
        ranking["abs_target_corr_rank_pct"]
        + ranking["abs_spearman_target_corr_rank_pct"]
        + ranking["univariate_f_rank_pct"]
        + 2.0 * ranking["quick_xgb_gain_rank_pct"]
    )

    # Pick the final set.
    if TOP_K_FEATURES is not None:
        selected = list(protected)
        ordered = ranking[ranking["kept_after_correlation_pruning"]].sort_values("combined_selection_score", ascending=False)
        selected += [f for f in ordered["feature"].tolist() if f not in selected]
        selected = selected[: max(TOP_K_FEATURES, len(protected))]
        selection_mode = f"fixed_top_k_{TOP_K_FEATURES}"
    else:
        selected_set = set(protected)
        auto_candidates = ranking[
            ranking["kept_after_correlation_pruning"]
            & (
                (ranking["quick_xgb_gain_positive"])
                | (ranking["abs_target_corr"] >= TARGET_CORR_MIN_ABS)
            )
        ]["feature"].tolist()
        selected_set.update(auto_candidates)
        selection_mode = "auto_quick_xgb_plus_target_corr"

        if len(selected_set) < MIN_AUTO_FEATURES:
            ordered = ranking[ranking["kept_after_correlation_pruning"]].sort_values("combined_selection_score", ascending=False)
            for f in ordered["feature"].tolist():
                selected_set.add(f)
                if len(selected_set) >= MIN_AUTO_FEATURES:
                    break

        if MAX_AUTO_FEATURES is not None and len(selected_set) > MAX_AUTO_FEATURES:
            ordered = ranking[ranking["feature"].isin(selected_set)].sort_values(
                ["protected_feature", "combined_selection_score"], ascending=[False, False]
            )
            selected_set = set(ordered.head(max(MAX_AUTO_FEATURES, len(protected)))["feature"].tolist())

        selected = [c for c in feature_cols if c in selected_set and c in kept_after_prune]
        for c in protected:
            if c in kept_after_prune and c not in selected:
                selected.append(c)

    ranking["selected_final"] = ranking["feature"].isin(selected)
    ranking["selection_mode"] = selection_mode
    ranking = ranking.sort_values(["selected_final", "quick_xgb_gain", "abs_target_corr"], ascending=[False, False, False])

    selected_type_report = make_feature_type_report(df, selected)
    selected_type_report.to_csv(out_dir / f"selected_feature_types_{scenario_name}.csv", index=False)
    ranking.to_csv(out_dir / f"feature_selection_ranking_{scenario_name}.csv", index=False)
    pd.DataFrame({"feature": selected, "group": [feature_group(f) for f in selected]}).to_csv(
        out_dir / f"selected_features_{scenario_name}.csv", index=False
    )

    print(f"Initial candidate features: {len(feature_cols):,}")
    print(f"Dropped by missing/constant screen: {len(dropped_missing_constant):,}")
    print(f"Features after missing/constant cleaning: {len(keep_after_missing):,}")
    print(f"Dropped by pairwise correlation pruning: {len(dropped_corr_df):,}")
    print(f"Features after correlation pruning: {len(kept_after_prune):,}")
    print(f"Quick XGBoost features with positive gain: {int(ranking['quick_xgb_gain_positive'].sum()):,}")
    print(f"Protected features retained: {len([f for f in selected if f in protected]):,}")
    print(f"Final selected feature count: {len(selected):,} ({selection_mode})")
    print(f"Selected LAD dummy count: {sum(f.startswith('lad_') for f in selected):,}")
    print("Top 15 selected features by quick XGB gain:")
    top_cols = ["feature", "group", "quick_xgb_gain", "abs_target_corr", "univariate_f", "protected_feature"]
    print(ranking[ranking["selected_final"]][top_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("Selected feature groups:")
    print(pd.Series([feature_group(f) for f in selected]).value_counts().head(15).to_string())
    return selected, ranking


# Training and prediction

def xgb_params() -> Dict[str, object]:
    """Return parameters for the final XGBoost model."""
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
    scenario_name: str,
) -> XGBRegressor:
    """Fit the final growth-from-lag1 XGBoost model."""
    print_section(f"Training final Growth-from-lag1 XGBoost: {scenario_name}")
    params = xgb_params()
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"  X_train shape: {X_train.shape}, X_valid shape: {X_valid.shape}")

    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    except TypeError:
        es_rounds = int(params.pop("early_stopping_rounds"))
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=es_rounds)

    print(f"Best iteration: {getattr(model, 'best_iteration', None)}")
    print(f"Best validation RMSE: {getattr(model, 'best_score', None)}")
    return model


def growth_to_count(pred_growth: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """Convert predicted log-growth back to assessment counts."""
    lag1 = rows[f"{TARGET_COL}_lag1q"].astype(float).values
    pred = np.expm1(np.log1p(np.clip(lag1, 0, None)) + np.asarray(pred_growth, dtype=float))
    return np.clip(pred, 0.0, None)


def fit_bias_correction(y_true_valid: np.ndarray, pred_valid: np.ndarray, model_name: str) -> Dict[str, float | str]:
    """Choose a simple validation calibration by MAE."""
    y = np.asarray(y_true_valid, dtype=float)
    p = np.asarray(pred_valid, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 5:
        return {"model_name": model_name, "method": "raw", "intercept": 0.0, "slope": 1.0, "valid_MAE": np.nan, "valid_RMSE": np.nan}

    candidates: List[Dict[str, float | str]] = []

    def add_candidate(method: str, intercept: float, slope: float) -> None:
        """Store one calibration option."""
        corrected = np.clip(intercept + slope * p, 0.0, None)
        candidates.append({
            "model_name": model_name,
            "method": method,
            "intercept": float(intercept),
            "slope": float(slope),
            "valid_MAE": float(mean_absolute_error(y, corrected)),
            "valid_RMSE": float(math.sqrt(mean_squared_error(y, corrected))),
            "valid_bias_actual_minus_predicted": float(np.mean(y - corrected)),
        })

    add_candidate("raw", 0.0, 1.0)
    additive_bias = float(np.mean(y - p))
    add_candidate("additive_mean_residual", additive_bias, 1.0)

    if len(np.unique(p)) > 2:
        slope, intercept = np.polyfit(p, y, deg=1)
        # Avoid extreme validation-only corrections.
        slope = float(np.clip(slope, 0.50, 1.50))
        intercept = float(np.mean(y) - slope * np.mean(p))
        add_candidate("linear_validation_calibration", intercept, slope)

    best = min(candidates, key=lambda d: (d["valid_MAE"], d["valid_RMSE"]))
    best["n_valid_for_correction"] = int(len(y))
    return best


def apply_bias_correction(pred: np.ndarray | pd.Series, correction: Dict[str, float | str]) -> np.ndarray:
    """Apply a fitted bias correction."""
    p = np.asarray(pred, dtype=float)
    intercept = float(correction.get("intercept", 0.0))
    slope = float(correction.get("slope", 1.0))
    return np.clip(intercept + slope * p, 0.0, None)


def optimize_blend_weight(
    y_true_valid: np.ndarray,
    xgb_pred_valid: np.ndarray,
    lag1_valid: np.ndarray,
    metric: str = "MAE",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[float, float, pd.DataFrame]:
    """Choose the validation blend weight."""
    rows = []
    best_w = float(min_weight)
    best_score = np.inf
    grid = np.linspace(min_weight, max_weight, int(round((max_weight - min_weight) * 100)) + 1)
    for w in grid:
        pred = w * xgb_pred_valid + (1.0 - w) * lag1_valid
        mae = float(mean_absolute_error(y_true_valid, pred))
        rmse = float(math.sqrt(mean_squared_error(y_true_valid, pred)))
        score = rmse if metric.upper() == "RMSE" else mae
        rows.append({"w_xgb": float(w), "w_lag1": float(1.0 - w), "valid_MAE": mae, "valid_RMSE": rmse})
        if score < best_score:
            best_score = score
            best_w = float(w)
    return best_w, float(best_score), pd.DataFrame(rows)


def evaluate_by_split(pred_df: pd.DataFrame, pred_col: str, model_name: str) -> List[Dict[str, object]]:
    """Evaluate one prediction column on each split."""
    rows: List[Dict[str, object]] = []
    for split in ["train", "valid", "test"]:
        sub = pred_df[pred_df["split"] == split]
        if len(sub) == 0:
            continue
        metrics = compute_metrics(sub[TARGET_COL].astype(float).values, sub[pred_col].astype(float).values)
        rows.append({"model": model_name, "split": split, "prediction_column": pred_col, **metrics})
    return rows


def get_xgb_importance(model: XGBRegressor, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export feature and feature-group importance."""
    booster = model.get_booster()
    out = pd.DataFrame({"feature": feature_cols})
    for imp_type in ["gain", "weight", "cover"]:
        scores = booster.get_score(importance_type=imp_type)
        out[imp_type] = out["feature"].map(scores).fillna(0.0)
    out["group"] = out["feature"].map(feature_group)
    out["intended_type"] = out["feature"].map(infer_intended_feature_type)
    out["gain_share"] = out["gain"] / out["gain"].sum() if out["gain"].sum() > 0 else 0.0
    out = out.sort_values("gain", ascending=False)

    grp = (
        out.groupby("group", as_index=False)
        .agg(
            gain=("gain", "sum"),
            weight=("weight", "sum"),
            n_features=("feature", "count"),
            n_features_used=("gain", lambda s: int((s > 0).sum())),
        )
    )
    grp["gain_share"] = grp["gain"] / grp["gain"].sum() if grp["gain"].sum() > 0 else 0.0
    grp = grp.sort_values("gain_share", ascending=False)
    return out, grp


def choose_final_prediction_column(metrics_df: pd.DataFrame) -> str:
    """Pick the validation-best prediction column."""
    valid = metrics_df[metrics_df["split"] == "valid"].copy()
    valid = valid.sort_values(["MAE", "RMSE"], ascending=True)
    return str(valid.iloc[0]["prediction_column"])


def run_three_report_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    scenario_name: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], str]:
    """Train the model family and save the main artefacts."""
    selected_features, selection_ranking = clean_and_select_features(df, feature_cols, scenario_name, out_dir)

    train_mask = df["split"] == "train"
    valid_mask = df["split"] == "valid"
    test_mask = df["split"] == "test"

    X_train = ensure_numeric_df(df.loc[train_mask, selected_features])
    y_train = df.loc[train_mask, "target_growth_from_lag1"].astype(float)
    X_valid = ensure_numeric_df(df.loc[valid_mask, selected_features])
    y_valid = df.loc[valid_mask, "target_growth_from_lag1"].astype(float)
    X_test = ensure_numeric_df(df.loc[test_mask, selected_features])

    model = fit_xgboost_growth_model(X_train, y_train, X_valid, y_valid, scenario_name)

    # Prediction table.
    pred_df = df[[
        "lad_code", "lad_name", "quarter", "quarter_date", "year", "quarter_num",
        TARGET_COL, f"{TARGET_COL}_lag1q", "split",
    ]].copy()
    pred_df["quarter"] = pred_df["quarter"].astype(str)

    # Lag-1 baseline.
    pred_df["pred_lag1_baseline"] = pred_df[f"{TARGET_COL}_lag1q"].astype(float).clip(lower=0)

    # XGBoost growth model.
    pred_growth_all = np.full(len(df), np.nan)
    for mask, X_part in [(train_mask, X_train), (valid_mask, X_valid), (test_mask, X_test)]:
        pred_growth_all[np.where(mask.values)[0]] = model.predict(X_part)
    pred_df["pred_growth_from_lag1_xgboost_raw"] = growth_to_count(pred_growth_all, df)

    # Validation calibration.
    y_val = pred_df.loc[valid_mask.values, TARGET_COL].astype(float).values
    xgb_val_raw = pred_df.loc[valid_mask.values, "pred_growth_from_lag1_xgboost_raw"].astype(float).values
    xgb_correction = fit_bias_correction(y_val, xgb_val_raw, "Growth-from-lag1 XGBoost raw")
    pred_df["pred_growth_from_lag1_xgboost_bias_corrected"] = apply_bias_correction(
        pred_df["pred_growth_from_lag1_xgboost_raw"].values,
        xgb_correction,
    )

    # Validation blend.
    xgb_val_bc = pred_df.loc[valid_mask.values, "pred_growth_from_lag1_xgboost_bias_corrected"].astype(float).values
    lag1_val = pred_df.loc[valid_mask.values, "pred_lag1_baseline"].astype(float).values
    w_mae, valid_mae, blend_curve = optimize_blend_weight(y_val, xgb_val_bc, lag1_val, metric="MAE", min_weight=0.0, max_weight=1.0)
    w_rmse, valid_rmse, _ = optimize_blend_weight(y_val, xgb_val_bc, lag1_val, metric="RMSE", min_weight=0.0, max_weight=1.0)
    blend_curve.to_csv(out_dir / f"blend_weight_curve_{scenario_name}.csv", index=False)

    pred_df["pred_xgboost_bc_lag1_blend_unrestricted"] = np.clip(
        w_mae * pred_df["pred_growth_from_lag1_xgboost_bias_corrected"] + (1.0 - w_mae) * pred_df["pred_lag1_baseline"],
        0.0,
        None,
    )

    # Final calibration.
    blend_val = pred_df.loc[valid_mask.values, "pred_xgboost_bc_lag1_blend_unrestricted"].astype(float).values
    blend_correction = fit_bias_correction(y_val, blend_val, "Unrestricted XGBoost-bc + lag1 blend")
    pred_df["pred_xgboost_bc_lag1_blend_bias_corrected"] = apply_bias_correction(
        pred_df["pred_xgboost_bc_lag1_blend_unrestricted"].values,
        blend_correction,
    )

    bias_corrections_df = pd.DataFrame([xgb_correction, blend_correction])
    bias_corrections_df.to_csv(out_dir / f"bias_correction_summary_{scenario_name}.csv", index=False)

    blend_summary: Dict[str, object] = {
        "scenario": scenario_name,
        "w_xgb_valid_MAE_unrestricted": float(w_mae),
        "w_lag1_valid_MAE_unrestricted": float(1.0 - w_mae),
        "valid_MAE_at_w": float(valid_mae),
        "w_xgb_valid_RMSE_unrestricted": float(w_rmse),
        "valid_RMSE_at_w": float(valid_rmse),
        "forced_50_percent_weight_used": False,
        "xgb_bias_correction_method": xgb_correction.get("method", "raw"),
        "blend_bias_correction_method": blend_correction.get("method", "raw"),
        "selected_feature_count": int(len(selected_features)),
        "selected_lad_dummy_count": int(sum(f.startswith("lad_") for f in selected_features)),
        "top_k_features_fixed": TOP_K_FEATURES,
    }

    metrics_rows: List[Dict[str, object]] = []
    metrics_rows += evaluate_by_split(pred_df, "pred_lag1_baseline", "Lag-1 baseline")
    metrics_rows += evaluate_by_split(pred_df, "pred_growth_from_lag1_xgboost_raw", "Growth-from-lag1 XGBoost raw")
    metrics_rows += evaluate_by_split(pred_df, "pred_growth_from_lag1_xgboost_bias_corrected", "Growth-from-lag1 XGBoost + bias correction")
    metrics_rows += evaluate_by_split(pred_df, "pred_xgboost_bc_lag1_blend_unrestricted", "XGBoost-bc + lag-1 blend, unrestricted")
    metrics_rows += evaluate_by_split(pred_df, "pred_xgboost_bc_lag1_blend_bias_corrected", "Final blend + bias correction")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.insert(0, "scenario", scenario_name)

    # Compare each model to lag-1.
    baseline_by_split = metrics_df[metrics_df["model"] == "Lag-1 baseline"].set_index("split")
    metrics_df["delta_MAE_vs_lag1"] = metrics_df.apply(
        lambda r: r["MAE"] - baseline_by_split.loc[r["split"], "MAE"] if r["split"] in baseline_by_split.index else np.nan,
        axis=1,
    )
    metrics_df["delta_RMSE_vs_lag1"] = metrics_df.apply(
        lambda r: r["RMSE"] - baseline_by_split.loc[r["split"], "RMSE"] if r["split"] in baseline_by_split.index else np.nan,
        axis=1,
    )

    final_pred_col = choose_final_prediction_column(metrics_df)
    final_model_name = metrics_df.loc[metrics_df["prediction_column"] == final_pred_col, "model"].iloc[0]
    blend_summary["final_prediction_column_selected_by_valid_MAE"] = final_pred_col
    blend_summary["final_model_name_selected_by_valid_MAE"] = final_model_name

    imp_df, group_imp_df = get_xgb_importance(model, selected_features)

    # Blend-level importance.
    component_rows = [{"component": "lag-1 baseline", "share": 1.0 - w_mae}]
    for _, row in group_imp_df.iterrows():
        component_rows.append({"component": f"XGB: {row['group']}", "share": w_mae * float(row["gain_share"])})
    component_imp_df = pd.DataFrame(component_rows).sort_values("share", ascending=False)

    # Worst LADs by test error.
    final_errors = pred_df.copy()
    final_errors["final_prediction_column"] = final_pred_col
    final_errors["final_pred"] = final_errors[final_pred_col]
    final_errors["abs_error"] = (final_errors[TARGET_COL] - final_errors["final_pred"]).abs()
    final_errors["signed_error_actual_minus_pred"] = final_errors[TARGET_COL] - final_errors["final_pred"]
    worst_lad = (
        final_errors[final_errors["split"] == "test"]
        .groupby(["lad_code", "lad_name"], observed=True, as_index=False)
        .agg(
            n=(TARGET_COL, "size"),
            mean_actual=(TARGET_COL, "mean"),
            mean_pred=("final_pred", "mean"),
            MAE=("abs_error", "mean"),
            RMSE=("abs_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            mean_signed_error_actual_minus_pred=("signed_error_actual_minus_pred", "mean"),
        )
        .sort_values("MAE", ascending=False)
    )

    # Save outputs.
    pred_df.to_csv(out_dir / f"predictions_{scenario_name}.csv", index=False)
    metrics_df.to_csv(out_dir / f"metrics_all_splits_{scenario_name}.csv", index=False)
    metrics_df[metrics_df["split"] == "test"].to_csv(out_dir / f"test_metrics_{scenario_name}.csv", index=False)
    imp_df.to_csv(out_dir / f"xgb_feature_importance_gain_{scenario_name}.csv", index=False)
    group_imp_df.to_csv(out_dir / f"xgb_feature_group_importance_{scenario_name}.csv", index=False)
    component_imp_df.to_csv(out_dir / f"blend_component_importance_{scenario_name}.csv", index=False)
    worst_lad.to_csv(out_dir / f"worst_lad_error_summary_{scenario_name}.csv", index=False)
    pd.DataFrame([blend_summary]).to_csv(out_dir / f"blend_weight_summary_{scenario_name}.csv", index=False)

    model.save_model(str(out_dir / f"growth_from_lag1_xgboost_model_{scenario_name}.json"))
    joblib.dump(model, out_dir / f"growth_from_lag1_xgboost_model_{scenario_name}.joblib")
    write_json(out_dir / f"xgb_selected_feature_columns_{scenario_name}.json", {"selected_features": selected_features})
    write_json(out_dir / f"model_config_{scenario_name}.json", {
        "scenario": scenario_name,
        "target": TARGET_COL,
        "total_cpi_feature_used": TOTAL_CPI_COL,
        "cpi_categories_used": False,
        "model_split": {
            "train_end": TRAIN_END_Q,
            "validation": f"{VALID_START_Q}-{VALID_END_Q}",
            "test": f"{TEST_START_Q}-{TEST_END_Q}",
        },
        "feature_selection": {
            "drop_features_missing_above": DROP_FEATURES_MISSING_ABOVE,
            "pairwise_corr_prune_threshold": PAIRWISE_CORR_PRUNE_THRESHOLD,
            "target_corr_min_abs": TARGET_CORR_MIN_ABS,
            "top_k_features": TOP_K_FEATURES,
            "min_auto_features": MIN_AUTO_FEATURES,
            "max_auto_features": MAX_AUTO_FEATURES,
            "force_keep_lad_dummies": FORCE_KEEP_LAD_DUMMIES,
        },
        "xgb_params": xgb_params(),
        "quick_xgb_params": quick_xgb_params(),
        "blend_summary": blend_summary,
        "bias_corrections": bias_corrections_df.to_dict(orient="records"),
        "excluded_direct_lag1_target_features": [
            f"{TARGET_COL}_lag1q",
            f"log1p_{TARGET_COL}_lag1q",
            "computed_homelessness_rate_per_1000_lag1q",
            "log1p_computed_homelessness_rate_per_1000_lag1q",
        ],
    })

    print_section(f"Scenario summary: {scenario_name}")
    print("Validation-selected final model:", final_model_name)
    print("Validation-selected final prediction column:", final_pred_col)
    print(f"Unrestricted blend weight on XGBoost: {w_mae:.2f}; lag-1: {1.0 - w_mae:.2f}")
    print("Bias correction summary:")
    print(bias_corrections_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Test metrics:")
    test_table = metrics_df[metrics_df["split"] == "test"][
        ["model", "n", "MAE", "delta_MAE_vs_lag1", "RMSE", "delta_RMSE_vs_lag1", "R2", "SMAPE_percent", "bias_actual_minus_predicted"]
    ]
    print(test_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Worst LAD test errors, top 10:")
    print(worst_lad.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return pred_df, metrics_df, imp_df, group_imp_df, blend_summary, final_pred_col


# Plots

def make_report_plots(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    group_imp_df: pd.DataFrame,
    blend_summary: Dict[str, object],
    scenario_name: str,
    out_dir: Path,
    final_pred_col: str,
) -> None:
    """Create the figures used in the report."""
    print_section(f"Saving and showing plots: {scenario_name}")
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    test_metrics = test_metrics[test_metrics["model"].isin(PLOT_MODEL_ORDER)].copy()
    test_metrics["model"] = pd.Categorical(test_metrics["model"], categories=PLOT_MODEL_ORDER, ordered=True)
    test_metrics = test_metrics.sort_values("model")

    # Test MAE.
    test_metrics["model_display"] = test_metrics["model"].astype(str).map(model_plot_label)
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(
        data=test_metrics,
        x="model_display",
        y="MAE",
        color=ACADEMIC_COLORS["blue"],
        edgecolor=ACADEMIC_COLORS["black"],
        linewidth=0.6,
    )
    ax.set_ylabel(plot_text("Test MAE"))
    plt.xlabel("")
    plt.title(plot_text("Test-set MAE comparison"))
    plt.xticks(rotation=25, ha="right")
    save_figure(out_dir / f"fig_test_mae_comparison_{scenario_name}.png")

    # Actual vs predicted.
    test_pred = pred_df[pred_df["split"] == "test"].copy()
    test_pred["final_pred"] = test_pred[final_pred_col]
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=test_pred,
        x=TARGET_COL,
        y="final_pred",
        alpha=0.55,
        color=ACADEMIC_COLORS["blue"],
        edgecolor="white",
        linewidth=0.25,
    )
    max_val = np.nanmax([test_pred[TARGET_COL].max(), test_pred["final_pred"].max()])
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1, color=ACADEMIC_COLORS["red"])
    plt.xlabel(plot_text("Actual homelessness assessments"))
    plt.ylabel(plot_text("Predicted homelessness assessments"))
    plt.title(plot_text("Actual vs predicted, test"))
    save_figure(out_dir / f"fig_actual_vs_predicted_scatter_{scenario_name}.png")

    # England aggregate timeline.
    agg_source = pred_df.copy().assign(final_pred=lambda d: d[final_pred_col])
    agg_source["quarter_period"] = pd.PeriodIndex(agg_source["quarter"].astype(str), freq="Q")
    agg = (
        agg_source
        .groupby("quarter_period", as_index=False)
        .agg(actual=(TARGET_COL, "sum"), predicted=("final_pred", "sum"))
        .sort_values("quarter_period")
        .reset_index(drop=True)
    )
    agg["quarter"] = agg["quarter_period"].astype(str)
    agg["quarter_start"] = agg["quarter_period"].apply(lambda p: p.start_time)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        agg["quarter_start"],
        agg["actual"],
        marker="o",
        linewidth=1.8,
        color=ACADEMIC_COLORS["blue"],
        label=plot_text("Actual"),
    )
    ax.plot(
        agg["quarter_start"],
        agg["predicted"],
        marker="o",
        linestyle="--",
        linewidth=1.8,
        color=ACADEMIC_COLORS["orange"],
        label=plot_text("Predicted"),
    )
    valid_start = quarter_to_period(VALID_START_Q).start_time
    test_start = quarter_to_period(TEST_START_Q).start_time
    for boundary in [valid_start, test_start]:
        ax.axvline(boundary, linestyle="--", linewidth=1, color=ACADEMIC_COLORS["gray"], alpha=0.8)
    x_start = agg["quarter_start"].min()
    x_end = agg["quarter_start"].max()
    ax.text(midpoint_time(x_start, valid_start), 0.98, plot_text("Train"), transform=ax.get_xaxis_transform(), ha="center", va="top", color=ACADEMIC_COLORS["gray"])
    ax.text(midpoint_time(valid_start, test_start), 0.98, plot_text("Validation"), transform=ax.get_xaxis_transform(), ha="center", va="top", color=ACADEMIC_COLORS["gray"])
    ax.text(midpoint_time(test_start, x_end), 0.98, plot_text("Test"), transform=ax.get_xaxis_transform(), ha="center", va="top", color=ACADEMIC_COLORS["gray"])
    plt.xticks(agg["quarter_start"].tolist(), agg["quarter"].tolist(), rotation=45, ha="right")
    plt.title(plot_text("England aggregate actual vs predicted"))
    plt.xlabel(plot_text("Quarter"))
    plt.ylabel(plot_text("England aggregate"))
    plt.legend(title="")
    save_figure(out_dir / f"fig_england_aggregate_prediction_{scenario_name}.png")
    agg[["quarter", "quarter_start", "actual", "predicted"]].to_csv(out_dir / f"england_aggregate_prediction_{scenario_name}.csv", index=False)

    # Worst LADs.
    worst_lad = (
        test_pred.assign(abs_error=lambda d: (d[TARGET_COL] - d["final_pred"]).abs())
        .groupby(["lad_code", "lad_name"], observed=True, as_index=False)
        .agg(MAE=("abs_error", "mean"), mean_actual=(TARGET_COL, "mean"), mean_pred=("final_pred", "mean"), n=(TARGET_COL, "size"))
        .sort_values("MAE", ascending=False)
        .head(20)
    )
    if len(worst_lad) > 0:
        plt.figure(figsize=(10, 7))
        worst_lad["lad_label"] = worst_lad["lad_name"].astype(str) + " (" + worst_lad["lad_code"].astype(str) + ")"
        worst_lad["lad_label_display"] = prepare_plot_category(worst_lad["lad_label"])
        sns.barplot(
            data=worst_lad,
            y="lad_label_display",
            x="MAE",
            orient="h",
            color=ACADEMIC_COLORS["blue"],
            edgecolor=ACADEMIC_COLORS["black"],
            linewidth=0.5,
        )
        plt.xlabel(plot_text("Mean absolute error on test quarters"))
        plt.ylabel("")
        plt.title(plot_text("Worst LAD test error summary"))
        save_figure(out_dir / f"fig_worst_lad_error_summary_{scenario_name}.png")

    # Feature-group importance.
    grp = group_imp_df[group_imp_df["gain_share"] > 0].head(15).copy()
    if len(grp) > 0:
        plt.figure(figsize=(9, 6))
        grp["gain_share_percent"] = grp["gain_share"] * 100.0
        grp["group_display"] = prepare_plot_category(grp["group"])
        sns.barplot(
            data=grp,
            y="group_display",
            x="gain_share_percent",
            orient="h",
            color=ACADEMIC_COLORS["blue"],
            edgecolor=ACADEMIC_COLORS["black"],
            linewidth=0.5,
        )
        plt.xlabel(plot_text("Gain share (%)"))
        plt.ylabel("")
        plt.title(plot_text("XGBoost feature-group importance"))
        save_figure(out_dir / f"fig_xgb_feature_group_importance_{scenario_name}.png")

    # Blend curve.
    curve_path = out_dir / f"blend_weight_curve_{scenario_name}.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=curve, x="w_xgb", y="valid_MAE", color=ACADEMIC_COLORS["blue"], linewidth=1.8)
        best_w = float(blend_summary["w_xgb_valid_MAE_unrestricted"])
        plt.axvline(best_w, linestyle="--", linewidth=1, color=ACADEMIC_COLORS["vermillion"])
        plt.xlabel(plot_text("Blend weight on XGBoost prediction"))
        plt.ylabel(plot_text("Validation MAE"))
        plt.title(plot_text("Unrestricted blend weight curve"))
        save_figure(out_dir / f"fig_blend_weight_curve_{scenario_name}.png")


def make_combined_scenario_plots(combined_metrics: pd.DataFrame, root_dir: Path) -> None:
    """Create comparison plots when more than one scenario is present."""
    print_section("Saving and showing combined scenario comparison plots")
    test = combined_metrics[combined_metrics["split"] == "test"].copy()
    test = test[test["model"].isin(PLOT_MODEL_ORDER)].copy()
    test["model"] = pd.Categorical(test["model"], categories=PLOT_MODEL_ORDER, ordered=True)
    test = test.sort_values("model")
    test["model_display"] = test["model"].astype(str).map(model_plot_label)
    test["scenario_display"] = test["scenario"].astype(str).map(scenario_plot_label)
    plt.figure(figsize=(11, 5.5))
    sns.barplot(
        data=test,
        x="model_display",
        y="MAE",
        hue="scenario_display",
        palette=ACADEMIC_PALETTE,
        edgecolor=ACADEMIC_COLORS["black"],
        linewidth=0.5,
    )
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(plot_text("Test MAE"))
    plt.xlabel("")
    plt.title(plot_text("Forecast model comparison"))
    plt.legend(title="")
    save_figure(root_dir / "fig_combined_scenario_test_mae.png")

    valid = combined_metrics[combined_metrics["split"] == "valid"].copy()
    valid = valid[valid["model"].isin(PLOT_MODEL_ORDER)].copy()
    valid["model"] = pd.Categorical(valid["model"], categories=PLOT_MODEL_ORDER, ordered=True)
    valid = valid.sort_values("model")
    valid["model_display"] = valid["model"].astype(str).map(model_plot_label)
    valid["scenario_display"] = valid["scenario"].astype(str).map(scenario_plot_label)
    plt.figure(figsize=(11, 5.5))
    sns.barplot(
        data=valid,
        x="model_display",
        y="MAE",
        hue="scenario_display",
        palette=ACADEMIC_PALETTE,
        edgecolor=ACADEMIC_COLORS["black"],
        linewidth=0.5,
    )
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(plot_text("Validation MAE"))
    plt.xlabel("")
    plt.title(plot_text("Validation MAE by forecast model"))
    plt.legend(title="")
    save_figure(root_dir / "fig_combined_scenario_valid_mae.png")


# Main

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run forecast-only improved XGBoost homelessness models on the latest integrated monthly panel.")
    parser.add_argument("--monthly", type=str, default=DEFAULT_MONTHLY_FILE, help="Path to latest integrated monthly LAD panel CSV")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output folder")
    parser.add_argument("--show-plots", dest="show_plots", action="store_true", default=SHOW_PLOTS, help="Show plots interactively as well as saving them")
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false", help="Save plots but do not open/show them")
    parser.add_argument("--top-k", type=int, default=None, help="Optional fixed top-k features. Omit to let quick XGBoost determine the count")
    parser.add_argument("--no-lad-dummies", dest="add_lad_dummies", action="store_false", default=ADD_LAD_DUMMIES, help="Do not include LAD one-hot dummy features")
    parser.add_argument("--force-keep-lad", dest="force_keep_lad", action="store_true", default=FORCE_KEEP_LAD_DUMMIES, help="Force all LAD dummy features to survive feature selection")
    return parser.parse_args()


def main() -> None:
    """Run the full modelling pipeline."""
    global OUTPUT_DIR, SHOW_PLOTS, TOP_K_FEATURES, ADD_LAD_DUMMIES, FORCE_KEEP_LAD_DUMMIES
    args = parse_args()
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHOW_PLOTS = bool(args.show_plots)
    TOP_K_FEATURES = args.top_k
    ADD_LAD_DUMMIES = bool(args.add_lad_dummies)
    FORCE_KEEP_LAD_DUMMIES = bool(args.force_keep_lad)

    monthly_file = resolve_input_path(args.monthly, "monthly_lad_panel_2000_2025_replaced_homelessness_final.csv")

    print_section("Forecast-only improved XGBoost homelessness modelling script")
    print(f"Latest monthly integrated file: {monthly_file}")
    print(f"Output directory:                {OUTPUT_DIR.resolve()}")
    print(f"Show plots interactively:        {SHOW_PLOTS}")
    print(f"LAD dummies included:            {ADD_LAD_DUMMIES}")
    print(f"Force-keep all LAD dummies:      {FORCE_KEEP_LAD_DUMMIES}")
    print(f"Top-k feature cap:               {TOP_K_FEATURES if TOP_K_FEATURES is not None else 'None, auto-selected'}")
    print("CPI rule: using only cpi_00_all_items; all CPI category columns are excluded.")
    print("Forecast rule: same-quarter exogenous nowcasting inputs are removed.")
    print("Blend rule: unrestricted 0%-100% XGBoost weight; no forced 50% weight.")

    qdf = build_quarterly_panel(monthly_file)
    qdf, same_quarter_exog, lagged_exog, target_history = add_report_features(qdf)

    panel_type_report = make_panel_column_type_report(qdf)
    panel_type_report.to_csv(OUTPUT_DIR / "engineered_panel_column_types.csv", index=False)
    print_section("Engineered panel explicit type summary")
    print(panel_type_report["explicit_type"].value_counts().to_string())
    print(f"Saved all-column type report: {OUTPUT_DIR / 'engineered_panel_column_types.csv'}")

    # Single forecast scenario for the report.
    scenarios: List[Tuple[str, bool]] = [("lagged_only_no_same_quarter_exog_forecast", False)]

    all_metrics: List[pd.DataFrame] = []
    scenario_summaries: List[Dict[str, object]] = []

    for scenario_name, include_same_quarter in scenarios:
        scenario_dir = OUTPUT_DIR / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        model_df, feature_cols = build_model_table(
            qdf,
            same_quarter_exog,
            lagged_exog,
            target_history,
            include_same_quarter_exog=include_same_quarter,
            scenario_name=scenario_name,
            out_dir=scenario_dir,
        )

        check_df = model_df[["lad_code", "lad_name", "quarter", TARGET_COL, f"{TARGET_COL}_lag1q", "split", "target_growth_from_lag1"]].copy()
        check_df["quarter"] = check_df["quarter"].astype(str)
        check_df.to_csv(scenario_dir / f"modelling_rows_check_{scenario_name}.csv", index=False)

        pred_df, metrics_df, imp_df, group_imp_df, blend_summary, final_pred_col = run_three_report_models(
            model_df,
            feature_cols,
            scenario_name,
            scenario_dir,
        )
        make_report_plots(pred_df, metrics_df, group_imp_df, blend_summary, scenario_name, scenario_dir, final_pred_col)
        all_metrics.append(metrics_df)
        scenario_summaries.append(blend_summary)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    scenario_summary_df = pd.DataFrame(scenario_summaries)
    # Keep old filenames alongside forecast-only aliases.
    combined_metrics.to_csv(OUTPUT_DIR / "combined_scenario_metrics_all_splits.csv", index=False)
    scenario_summary_df.to_csv(OUTPUT_DIR / "combined_scenario_model_summaries.csv", index=False)
    combined_metrics.to_csv(OUTPUT_DIR / "forecast_metrics_all_splits.csv", index=False)
    scenario_summary_df.to_csv(OUTPUT_DIR / "forecast_model_summary.csv", index=False)
    print_section("Final forecast validation/test comparison")
    display_cols = [
        "scenario", "model", "split", "n", "MAE", "delta_MAE_vs_lag1", "RMSE", "delta_RMSE_vs_lag1", "R2", "SMAPE_percent", "bias_actual_minus_predicted",
    ]
    comparison = combined_metrics[combined_metrics["split"].isin(["valid", "test"])][display_cols].copy()
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    valid_best = combined_metrics[combined_metrics["split"] == "valid"].sort_values(["MAE", "RMSE"]).iloc[0]
    print_section("Forecast model selected by validation MAE")
    print(f"Scenario:          {valid_best['scenario']}")
    print(f"Model:             {valid_best['model']}")
    print(f"Prediction column: {valid_best['prediction_column']}")
    print(f"Validation MAE:    {valid_best['MAE']:.3f}")
    test_match = combined_metrics[
        (combined_metrics["scenario"] == valid_best["scenario"])
        & (combined_metrics["prediction_column"] == valid_best["prediction_column"])
        & (combined_metrics["split"] == "test")
    ]
    if len(test_match):
        row = test_match.iloc[0]
        print(f"Test MAE:          {row['MAE']:.3f}")
        print(f"Test RMSE:         {row['RMSE']:.3f}")
        print(f"Test R2:           {row['R2']:.3f}")

    print_section("Outputs saved")
    print(f"Root folder: {OUTPUT_DIR.resolve()}")
    print("Main root files:")
    for name in [
        "forecast_metrics_all_splits.csv",
        "forecast_model_summary.csv",
        "combined_scenario_metrics_all_splits.csv",
        "combined_scenario_model_summaries.csv",
    ]:
        print(f"  - {OUTPUT_DIR / name}")
    print("The forecast scenario folder contains predictions, metrics, feature selection details, bias correction, blend curves, diagnostics and model files.")


if __name__ == "__main__":
    main()
