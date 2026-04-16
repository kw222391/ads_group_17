from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

DEFAULT_RANDOM_STATE = 42
DEFAULT_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DEFAULT_L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_EXCLUDE_COLS = ["lad_code", "lad_name", "date", "quarter"]
DEFAULT_ALWAYS_KEEP_COLS = [
    "homelessness_post_2018_indicator",
    "year_num",
    "quarter_num",
]

SELECTOR_MANIFEST_NAME = "selector_manifest.json"
DATASET_MANIFEST_NAME = "dataset_manifest.json"


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def slugify_float(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def combo_folder_name(alpha: float, l1_ratio: float) -> str:
    return f"alpha_{slugify_float(alpha)}__l1_{slugify_float(l1_ratio)}"


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffixes = path.suffixes
    if suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path, compression="gzip")
    if suffixes[-1:] == [".csv"]:
        return pd.read_csv(path)
    if suffixes[-1:] == [".parquet"]:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file format for {path}. Expected .csv, .csv.gz, or .parquet"
    )


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_series_csv(values: Iterable, path: str | Path, name: str = "value") -> None:
    pd.Series(list(values), name=name).to_csv(path, index=False)


def require_parquet_support() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Saving parquet requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc


def save_dataframe_parquet(df: pd.DataFrame, path: str | Path) -> None:
    require_parquet_support()
    df.to_parquet(path, index=False)


def set_random_seed(random_state: int = DEFAULT_RANDOM_STATE) -> None:
    random.seed(random_state)
    np.random.seed(random_state)


def ensure_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        return df

    for year_col, month_col in [("year", "month"), ("Year", "Month")]:
        if year_col in df.columns and month_col in df.columns:
            out = df.copy()
            out[date_col] = pd.to_datetime(
                {"year": out[year_col], "month": out[month_col], "day": 1},
                errors="coerce",
            )
            return out

    raise KeyError(
        f"Date column '{date_col}' not found and no year/month columns were available to build it."
    )


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def validate_search_space(alpha_grid: Iterable[float], l1_ratio_grid: Iterable[float]) -> None:
    if any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("All alpha values must be > 0.")
    if any((ratio < 0) or (ratio > 1) for ratio in l1_ratio_grid):
        raise ValueError("All l1_ratio values must be within [0, 1].")


def validate_model_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    stage: str,
) -> None:
    split_rows = {
        "train": (len(X_train), len(y_train)),
        "val": (len(X_val), len(y_val)),
        "test": (len(X_test), len(y_test)),
    }
    empty_splits = [
        name for name, (x_rows, y_rows) in split_rows.items() if x_rows == 0 or y_rows == 0
    ]
    if empty_splits:
        raise ValueError(
            f"Empty data split(s) after {stage}: {empty_splits}. "
            "Check the date boundaries and missing-target rows."
        )

    if X_train.shape[1] == 0:
        raise ValueError(
            f"No features remain after {stage}. Relax the filtering thresholds or review the input data."
        )


def make_time_split_masks(
    df: pd.DataFrame,
    date_col: str = "date",
    train_start: str | None = None,
    train_end: str = "2023-09-01",
    val_end: str = "2024-09-30",
):
    date_series = pd.to_datetime(df[date_col], errors="coerce")
    train_start_ts = pd.Timestamp(train_start) if train_start is not None else None
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    if train_start_ts is not None and train_start_ts >= train_end_ts:
        raise ValueError("train_start must be earlier than train_end.")
    if train_end_ts > val_end_ts:
        raise ValueError("train_end must be earlier than or equal to val_end.")

    train_mask = date_series < train_end_ts
    if train_start_ts is not None:
        train_mask &= date_series >= train_start_ts
    val_mask = (date_series >= train_end_ts) & (date_series <= val_end_ts)
    test_mask = date_series > val_end_ts
    return train_mask, val_mask, test_mask


def select_numeric_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Iterable[str] | None = None,
) -> list[str]:
    exclude_cols = list(exclude_cols or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != target_col and c not in exclude_cols]


def infer_target_family_exclusions(df: pd.DataFrame, target_col: str) -> list[str]:
    family_prefix = "homelessness_total_assessments"
    if not target_col.startswith(family_prefix):
        return []

    target_lower = target_col.lower()
    use_log_diff = "log_diff" in target_lower
    use_change_rate = "change_rate" in target_lower

    exclusions: list[str] = []
    for col in df.columns:
        if col == target_col or not col.startswith(family_prefix):
            continue

        col_lower = col.lower()
        is_lag_like = (
            ("lag" in col_lower)
            or ("mean" in col_lower)
            or ("rolling" in col_lower)
        )
        is_log_diff_family = "log_diff" in col_lower
        is_change_rate_family = "change_rate" in col_lower

        if not is_lag_like:
            exclusions.append(col)
            continue
        if use_log_diff and is_change_rate_family:
            exclusions.append(col)
            continue
        if use_change_rate and is_log_diff_family:
            exclusions.append(col)
            continue

    return sorted(set(exclusions))


def summarize_feature_groups(feature_cols: Iterable[str]) -> dict[str, int]:
    feature_cols = list(feature_cols)
    summary = {
        "total_features": len(feature_cols),
        "lag_like": 0,
        "mean_or_rolling_like": 0,
        "homeless_family": 0,
        "other": 0,
    }

    for col in feature_cols:
        col_lower = col.lower()
        is_lag_like = "lag" in col_lower
        is_mean_like = ("mean" in col_lower) or ("rolling" in col_lower)
        is_homeless_family = col.startswith("homelessness_total_assessments")

        if is_lag_like:
            summary["lag_like"] += 1
        elif is_mean_like:
            summary["mean_or_rolling_like"] += 1
        elif is_homeless_family:
            summary["homeless_family"] += 1
        else:
            summary["other"] += 1

    return summary


def apply_train_based_feature_filters(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    missing_threshold: float = 0.4,
    variance_threshold: float = 1e-8,
):
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    missing_ratio = X_train.isna().mean()
    keep_after_missing = missing_ratio[missing_ratio < missing_threshold].index.tolist()

    X_train = X_train[keep_after_missing].copy()
    X_val = X_val[keep_after_missing].copy()
    X_test = X_test[keep_after_missing].copy()

    var_series = X_train.var(numeric_only=True)
    keep_after_variance = var_series[var_series > variance_threshold].index.tolist()

    X_train = X_train[keep_after_variance].copy()
    X_val = X_val[keep_after_variance].copy()
    X_test = X_test[keep_after_variance].copy()

    return X_train, X_val, X_test, keep_after_missing, keep_after_variance


def clean_features(X: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def drop_rows_with_missing_target(X: pd.DataFrame, y: pd.Series):
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_mask = y.notna()
    return X.loc[valid_mask].copy(), y.loc[valid_mask].copy()


def build_final_feature_frames(
    X_train_filtered: pd.DataFrame,
    X_val_filtered: pd.DataFrame,
    X_test_filtered: pd.DataFrame,
    X_train_keep: pd.DataFrame,
    X_val_keep: pd.DataFrame,
    X_test_keep: pd.DataFrame,
    selected_candidate_features: Iterable[str],
    always_keep_feature_cols: Iterable[str],
):
    selected_candidate_features = list(selected_candidate_features)
    always_keep_feature_cols = list(always_keep_feature_cols)

    X_train_final = pd.concat(
        [
            X_train_keep[always_keep_feature_cols].copy() if always_keep_feature_cols else pd.DataFrame(index=X_train_filtered.index),
            X_train_filtered[selected_candidate_features].copy() if selected_candidate_features else pd.DataFrame(index=X_train_filtered.index),
        ],
        axis=1,
    )
    X_val_final = pd.concat(
        [
            X_val_keep[always_keep_feature_cols].copy() if always_keep_feature_cols else pd.DataFrame(index=X_val_filtered.index),
            X_val_filtered[selected_candidate_features].copy() if selected_candidate_features else pd.DataFrame(index=X_val_filtered.index),
        ],
        axis=1,
    )
    X_test_final = pd.concat(
        [
            X_test_keep[always_keep_feature_cols].copy() if always_keep_feature_cols else pd.DataFrame(index=X_test_filtered.index),
            X_test_filtered[selected_candidate_features].copy() if selected_candidate_features else pd.DataFrame(index=X_test_filtered.index),
        ],
        axis=1,
    )

    final_feature_cols = list(dict.fromkeys(always_keep_feature_cols + selected_candidate_features))
    return X_train_final, X_val_final, X_test_final, final_feature_cols


def discover_manifest_paths(root_dir: str | Path, manifest_name: str) -> list[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    return sorted(root_dir.rglob(manifest_name))
