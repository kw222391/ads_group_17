import argparse
import os
import random
from itertools import product

import numpy as np
import pandas as pd

try:
    import cudf
    import cupy as cp

    from cuml.preprocessing import StandardScaler
    from cuml.linear_model import ElasticNet
    from cuml.metrics import mean_squared_error, r2_score
except ModuleNotFoundError as exc:
    cudf = None
    cp = None
    StandardScaler = None
    ElasticNet = None
    mean_squared_error = None
    r2_score = None
    RAPIDS_IMPORT_ERROR = exc
else:
    RAPIDS_IMPORT_ERROR = None


DEFAULT_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DEFAULT_L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_RANDOM_STATE = 42


def load_gzipped_csv(gz_path: str) -> pd.DataFrame:
    """Load a gzipped CSV file."""
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Gzipped CSV file not found: {gz_path}")

    if not gz_path.endswith(".gz"):
        raise ValueError(f"Expected a .gz file, got: {gz_path}")

    return pd.read_csv(gz_path, compression="gzip")


def require_rapids() -> None:
    """Raise a clear error when RAPIDS dependencies are unavailable."""
    if RAPIDS_IMPORT_ERROR is not None:
        raise ImportError(
            "This script requires RAPIDS GPU packages: cudf, cupy, and cuml. "
            "Install a compatible RAPIDS/CUDA environment before training."
        ) from RAPIDS_IMPORT_ERROR


def set_random_seed(random_state: int = DEFAULT_RANDOM_STATE) -> None:
    """Set global random seeds used by NumPy, Python, and CuPy."""
    random.seed(random_state)
    np.random.seed(random_state)
    if cp is not None:
        cp.random.seed(random_state)


def ensure_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensure a usable datetime column exists.

    If `date_col` is missing and the dataframe contains `year` and `month`,
    construct a first-of-month timestamp column.
    """
    if date_col in df.columns:
        return df

    year_month_pairs = [
        ("year", "month"),
        ("Year", "Month"),
    ]

    for year_col, month_col in year_month_pairs:
        if year_col in df.columns and month_col in df.columns:
            df = df.copy()
            df[date_col] = pd.to_datetime(
                {
                    "year": df[year_col],
                    "month": df[month_col],
                    "day": 1,
                },
                errors="coerce",
            )
            return df

    raise KeyError(
        f"Date column '{date_col}' not found and no year/month columns were available to build it."
    )


def validate_required_columns(df: pd.DataFrame, required_cols) -> None:
    """Raise an error if required columns are missing."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")


def validate_search_space(alpha_grid, l1_ratio_grid) -> None:
    """Validate the hyperparameter search space."""
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
    """Fail early when a split becomes empty or filtering removes all features."""
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
            f"No features remain after {stage}. "
            "Relax the missing/variance thresholds or review the input data."
        )


def make_time_split_masks(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = "2023-09-01",
    val_end: str = "2024-09-30",
):
    """
    Split rule:
      train: date < train_end
      val  : train_end <= date <= val_end
      test : date > val_end
    """
    date_series = pd.to_datetime(df[date_col], errors="coerce")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train_mask = date_series < train_end_ts
    val_mask = (date_series >= train_end_ts) & (date_series <= val_end_ts)
    test_mask = date_series > val_end_ts

    return train_mask, val_mask, test_mask


def select_numeric_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols=None,
):
    """Select numeric feature columns excluding target and specified columns."""
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col and c not in exclude_cols]
    return feature_cols


# ==== Helper functions for family exclusions and group summaries ====

def infer_target_family_exclusions(
    df: pd.DataFrame,
    target_col: str,
):
    """
    Exclude leakage-prone homelessness target-family columns based on the chosen target.

    Rules:
    - Always exclude contemporaneous homelessness_total_assessments family columns
      (except the target itself).
    - If target uses log_diff, also exclude change_rate-related history columns.
    - If target uses change_rate, also exclude log_diff-related history columns.
    - Keep same-family lag/mean history only for the selected target type.
    """
    family_prefix = "homelessness_total_assessments"
    if not target_col.startswith(family_prefix):
        return []

    target_lower = target_col.lower()
    use_log_diff = "log_diff" in target_lower
    use_change_rate = "change_rate" in target_lower

    exclusions = []
    for col in df.columns:
        if col == target_col:
            continue
        if not col.startswith(family_prefix):
            continue

        col_lower = col.lower()
        is_lag_like = ("lag" in col_lower) or ("mean" in col_lower) or ("rolling" in col_lower)
        is_log_diff_family = "log_diff" in col_lower
        is_change_rate_family = "change_rate" in col_lower

        # Exclude all contemporaneous same-family columns other than the target itself.
        if not is_lag_like:
            exclusions.append(col)
            continue

        # If target is log_diff, do not use change_rate-related history columns.
        if use_log_diff and is_change_rate_family:
            exclusions.append(col)
            continue

        # If target is change_rate, do not use log_diff-related history columns.
        if use_change_rate and is_log_diff_family:
            exclusions.append(col)
            continue

    return sorted(set(exclusions))


def summarize_feature_groups(feature_cols):
    """Create a lightweight grouped summary to inspect what enters selection."""
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
    """
    Apply missing-ratio and low-variance filtering based on train only.
    """
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Missing filter on train only
    missing_ratio = X_train.isna().mean()
    keep_after_missing = missing_ratio[missing_ratio < missing_threshold].index.tolist()

    X_train = X_train[keep_after_missing].copy()
    X_val = X_val[keep_after_missing].copy()
    X_test = X_test[keep_after_missing].copy()

    # Variance filter on train only
    var_series = X_train.var(numeric_only=True)
    keep_after_variance = var_series[var_series > variance_threshold].index.tolist()

    X_train = X_train[keep_after_variance].copy()
    X_val = X_val[keep_after_variance].copy()
    X_test = X_test[keep_after_variance].copy()

    return X_train, X_val, X_test, keep_after_missing, keep_after_variance


def clean_features(X: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """Replace inf with NaN and fill missing values."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(fill_value)
    return X


def drop_rows_with_missing_target(X: pd.DataFrame, y: pd.Series):
    """Drop rows where target is missing."""
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_mask = y.notna()

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()
    return X, y


def to_gpu(X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd):
    """Move pandas data to GPU."""
    require_rapids()

    X_train = cudf.DataFrame.from_pandas(X_train_pd.astype(np.float32))
    X_val = cudf.DataFrame.from_pandas(X_val_pd.astype(np.float32))
    X_test = cudf.DataFrame.from_pandas(X_test_pd.astype(np.float32))

    y_train = cudf.Series(y_train_pd.astype(np.float32).values)
    y_val = cudf.Series(y_val_pd.astype(np.float32).values)
    y_test = cudf.Series(y_test_pd.astype(np.float32).values)

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_single_elastic_net(
    X_train_gpu,
    X_val_gpu,
    X_test_gpu,
    y_train_gpu,
    y_val_gpu,
    y_test_gpu,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 5000,
    tol: float = 1e-4,
    selection: str = "cyclic",
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Fit one Elastic Net model and return metrics + objects.
    """
    require_rapids()

    # cuML ElasticNet does not expose an estimator-level random_state.
    # Reseeding before each fit keeps `selection="random"` as reproducible as possible.
    set_random_seed(random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_gpu)
    X_val_scaled = scaler.transform(X_val_gpu)
    X_test_scaled = scaler.transform(X_test_gpu)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        selection=selection,
    )
    model.fit(X_train_scaled, y_train_gpu)

    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    val_rmse = float(cp.sqrt(mean_squared_error(y_val_gpu, val_pred)).get())
    test_rmse = float(cp.sqrt(mean_squared_error(y_test_gpu, test_pred)).get())

    val_r2 = float(r2_score(y_val_gpu, val_pred).get())
    test_r2 = float(r2_score(y_test_gpu, test_pred).get())

    return {
        "model": model,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "val_r2": val_r2,
        "test_r2": test_r2,
    }


def run_elastic_net_search(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = "date",
    train_end: str = "2023-09-01",
    val_end: str = "2024-09-30",
    exclude_cols=None,
    missing_threshold: float = 0.4,
    variance_threshold: float = 1e-8,
    fill_value: float = 0.0,
    alpha_grid=None,
    l1_ratio_grid=None,
    max_iter: int = 5000,
    tol: float = 1e-4,
    selection: str = "cyclic",
    sort_cols=None,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
    always_keep_cols=None,
):
    """
    Full pipeline:
      1) sort
      2) time split
      3) train-based feature filtering
      4) clean data
      5) move to GPU
      6) grid search over alpha / l1_ratio
      7) return best model + selected features
    """
    if exclude_cols is None:
        exclude_cols = ["lad_code", "lad_name", "date", "quarter"]

    if always_keep_cols is None:
        always_keep_cols = [
            "homelessness_post_2018_indicator",
            "year_num",
            "quarter_num",
        ]

    if alpha_grid is None:
        alpha_grid = DEFAULT_ALPHA_GRID

    if l1_ratio_grid is None:
        l1_ratio_grid = DEFAULT_L1_RATIO_GRID

    validate_search_space(alpha_grid, l1_ratio_grid)

    df = ensure_date_column(df.copy(), date_col)
    validate_required_columns(df, [target_col, date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    invalid_date_count = int(df[date_col].isna().sum())
    if invalid_date_count > 0 and verbose:
        print(
            f"Warning: {invalid_date_count} rows have invalid '{date_col}' values "
            "and will be excluded from train/val/test splits."
        )

    if sort_cols is None:
        sort_cols = [c for c in ["lad_code", date_col] if c in df.columns]
    if len(sort_cols) > 0:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    auto_exclude_cols = infer_target_family_exclusions(
        df=df,
        target_col=target_col,
    )
    effective_exclude_cols = sorted(set(list(exclude_cols) + list(auto_exclude_cols)))

    feature_cols = select_numeric_features(
        df=df,
        target_col=target_col,
        exclude_cols=effective_exclude_cols,
    )
    if not feature_cols:
        raise ValueError("No numeric features were available after applying exclude_cols.")

    always_keep_feature_cols = [c for c in always_keep_cols if c in feature_cols]
    candidate_feature_cols = [c for c in feature_cols if c not in always_keep_feature_cols]

    if not candidate_feature_cols:
        raise ValueError(
            "No candidate features remain for Elastic Net after reserving always_keep_cols."
        )

    X_all = df[candidate_feature_cols].copy()
    X_all_keep = df[always_keep_feature_cols].copy() if always_keep_feature_cols else pd.DataFrame(index=df.index)
    y_all = df[target_col].copy()

    train_mask, val_mask, test_mask = make_time_split_masks(
        df=df,
        date_col=date_col,
        train_end=train_end,
        val_end=val_end,
    )

    X_train_pd = X_all.loc[train_mask].copy()
    X_val_pd = X_all.loc[val_mask].copy()
    X_test_pd = X_all.loc[test_mask].copy()
    X_train_keep_pd = X_all_keep.loc[train_mask].copy()
    X_val_keep_pd = X_all_keep.loc[val_mask].copy()
    X_test_keep_pd = X_all_keep.loc[test_mask].copy()

    y_train_pd = y_all.loc[train_mask].copy()
    y_val_pd = y_all.loc[val_mask].copy()
    y_test_pd = y_all.loc[test_mask].copy()

    validate_model_inputs(
        X_train_pd,
        X_val_pd,
        X_test_pd,
        y_train_pd,
        y_val_pd,
        y_test_pd,
        stage="time-based splitting",
    )

    if verbose:
        print("Effective exclude cols:", effective_exclude_cols)
        if auto_exclude_cols:
            print("Auto-excluded homelessness family cols:", auto_exclude_cols)
        print("Always-keep feature cols:", always_keep_feature_cols)
        print("Initial candidate feature group summary:", summarize_feature_groups(candidate_feature_cols))
        print("Initial total feature group summary:", summarize_feature_groups(feature_cols))
        print("Initial shapes:")
        print("  X_train:", X_train_pd.shape)
        print("  X_val  :", X_val_pd.shape)
        print("  X_test :", X_test_pd.shape)

    X_train_pd, X_val_pd, X_test_pd, keep_after_missing, keep_after_variance = apply_train_based_feature_filters(
        X_train=X_train_pd,
        X_val=X_val_pd,
        X_test=X_test_pd,
        missing_threshold=missing_threshold,
        variance_threshold=variance_threshold,
    )

    X_train_pd = clean_features(X_train_pd, fill_value=fill_value)
    X_train_keep_pd = clean_features(X_train_keep_pd, fill_value=fill_value)
    X_val_pd = clean_features(X_val_pd, fill_value=fill_value)
    X_val_keep_pd = clean_features(X_val_keep_pd, fill_value=fill_value)
    X_test_pd = clean_features(X_test_pd, fill_value=fill_value)
    X_test_keep_pd = clean_features(X_test_keep_pd, fill_value=fill_value)

    X_train_pd, y_train_pd = drop_rows_with_missing_target(X_train_pd, y_train_pd)
    X_train_keep_pd = X_train_keep_pd.loc[y_train_pd.index].copy()
    X_val_pd, y_val_pd = drop_rows_with_missing_target(X_val_pd, y_val_pd)
    X_val_keep_pd = X_val_keep_pd.loc[y_val_pd.index].copy()
    X_test_pd, y_test_pd = drop_rows_with_missing_target(X_test_pd, y_test_pd)
    X_test_keep_pd = X_test_keep_pd.loc[y_test_pd.index].copy()

    validate_model_inputs(
        X_train_pd,
        X_val_pd,
        X_test_pd,
        y_train_pd,
        y_val_pd,
        y_test_pd,
        stage="feature filtering and target cleaning",
    )

    elastic_net_candidate_cols = X_train_pd.columns.tolist()
    final_feature_cols = always_keep_feature_cols + elastic_net_candidate_cols

    if verbose:
        print("After filters:")
        print("  features after missing filter :", len(keep_after_missing))
        print("  Elastic Net candidate features after variance filter:", len(elastic_net_candidate_cols))
        print("  always-keep features:", len(always_keep_feature_cols))
        print("  final total features before downstream merge:", len(final_feature_cols))
        print("  final candidate X_train:", X_train_pd.shape)
        print("  final candidate X_val  :", X_val_pd.shape)
        print("  final candidate X_test :", X_test_pd.shape)

    X_train_gpu, X_val_gpu, X_test_gpu, y_train_gpu, y_val_gpu, y_test_gpu = to_gpu(
        X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd
    )

    all_results = []
    best_result = None
    best_score = np.inf
    best_n_selected = np.inf

    for alpha, l1_ratio in product(alpha_grid, l1_ratio_grid):
        result = fit_single_elastic_net(
            X_train_gpu=X_train_gpu,
            X_val_gpu=X_val_gpu,
            X_test_gpu=X_test_gpu,
            y_train_gpu=y_train_gpu,
            y_val_gpu=y_val_gpu,
            y_test_gpu=y_test_gpu,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
            random_state=random_state,
        )

        coef = cp.asnumpy(result["model"].coef_)
        n_selected = int(np.sum(coef != 0))

        row = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "val_rmse": result["val_rmse"],
            "test_rmse": result["test_rmse"],
            "val_r2": result["val_r2"],
            "test_r2": result["test_r2"],
            "n_selected_features": n_selected,
        }
        all_results.append(row)

        if verbose:
            print(
                f"alpha={alpha:.5f}, l1_ratio={l1_ratio:.3f} | "
                f"val_rmse={result['val_rmse']:.4f}, "
                f"val_r2={result['val_r2']:.4f}, "
                f"selected={n_selected}"
            )

        if (result["val_rmse"] < best_score) or (
            np.isclose(result["val_rmse"], best_score) and n_selected < best_n_selected
        ):
            best_score = result["val_rmse"]
            best_n_selected = n_selected
            best_result = {
                **result,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "coef": coef,
                "n_selected_features": n_selected,
            }

    if best_result is None:
        raise RuntimeError("Elastic Net search did not produce any fitted models.")

    results_df = pd.DataFrame(all_results).sort_values(
        ["val_rmse", "n_selected_features"], ascending=[True, True]
    ).reset_index(drop=True)

    coef_df = pd.DataFrame({
        "feature": elastic_net_candidate_cols,
        "coef": best_result["coef"],
        "abs_coef": np.abs(best_result["coef"]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    selected_candidate_features = coef_df.loc[coef_df["coef"] != 0, "feature"].tolist()
    selected_features = always_keep_feature_cols + selected_candidate_features
    selected_features = list(dict.fromkeys(selected_features))

    output = {
        "best_alpha": best_result["alpha"],
        "best_l1_ratio": best_result["l1_ratio"],
        "best_val_rmse": best_result["val_rmse"],
        "best_test_rmse": best_result["test_rmse"],
        "best_val_r2": best_result["val_r2"],
        "best_test_r2": best_result["test_r2"],
        "random_state": random_state,
        "results_df": results_df,
        "coef_df": coef_df,
        "selected_features": selected_features,
        "selected_candidate_features": selected_candidate_features,
        "best_model": best_result["model"],
        "best_scaler": best_result["scaler"],
        "best_n_selected_features": best_result["n_selected_features"],
        "X_train_filtered": X_train_pd,
        "X_val_filtered": X_val_pd,
        "X_test_filtered": X_test_pd,
        "X_train_keep": X_train_keep_pd,
        "X_val_keep": X_val_keep_pd,
        "X_test_keep": X_test_keep_pd,
        "y_train": y_train_pd,
        "y_val": y_val_pd,
        "y_test": y_test_pd,
        "final_feature_cols": final_feature_cols,
        "effective_exclude_cols": effective_exclude_cols,
        "auto_exclude_cols": auto_exclude_cols,
        "always_keep_feature_cols": always_keep_feature_cols,
    }

    return output

def parse_args():
    parser = argparse.ArgumentParser(description="cuML Elastic Net search for homelessness prediction")

    # ===== 数据输入 =====
    parser.add_argument(
        "--gz_path",
        type=str,
        required=True,
        help="Path to a gzipped CSV file (.csv.gz)"
    )

    # ===== 基本列设置 =====
    parser.add_argument(
        "--target_col",
        type=str,
        default="homelessness_total_assessments",
        help="Target column name"
    )
    parser.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Date column name"
    )

    # ===== 时间切分 =====
    parser.add_argument(
        "--train_end",
        type=str,
        default="2023-09-01",
        help="Train end boundary: train uses date < train_end"
    )
    parser.add_argument(
        "--val_end",
        type=str,
        default="2024-09-30",
        help="Validation end boundary: val uses train_end <= date <= val_end"
    )

    # ===== 特征筛选 =====
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.4,
        help="Drop features with missing ratio >= this threshold, based on train only"
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=1e-8,
        help="Drop features with variance <= this threshold, based on train only"
    )
    parser.add_argument(
        "--fill_value",
        type=float,
        default=0.0,
        help="Fill value for remaining missing values"
    )

    # ===== Elastic Net 参数 =====
    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_GRID,
        help="List of alpha values to search"
    )
    parser.add_argument(
        "--l1_ratio_grid",
        type=float,
        nargs="+",
        default=DEFAULT_L1_RATIO_GRID,
        help="List of l1_ratio values to search"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5000,
        help="Maximum iterations for Elastic Net"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerance for Elastic Net"
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="cyclic",
        choices=["cyclic", "random"],
        help="Coordinate descent selection strategy"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Global random seed used for NumPy/CuPy/Python RNGs"
    )

    # ===== 可选排除列 =====
    parser.add_argument(
        "--exclude_cols",
        type=str,
        nargs="*",
        default=["lad_code", "lad_name", "date", "quarter"],
        help="Columns to exclude from features"
    )
    parser.add_argument(
        "--always_keep_cols",
        type=str,
        nargs="*",
        default=["homelessness_post_2018_indicator", "year_num", "quarter_num"],
        help="Columns kept outside Elastic Net screening and merged back for downstream models"
    )

    # ===== 输出 =====
    parser.add_argument(
        "--output_dir",
        type=str,
        default="elastic_net_outputs",
        help="Directory to save results"
    )

    # ===== 日志 =====
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose printing"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    verbose = not args.quiet
    set_random_seed(args.random_state)

    os.makedirs(args.output_dir, exist_ok=True)

    if verbose:
        print("Loading data...")
        print(f"  gz_path      : {args.gz_path}")
        print(f"  random_state : {args.random_state}")

    df = load_gzipped_csv(args.gz_path)

    if verbose:
        print("Data loaded.")
        print("Shape:", df.shape)
        print("Target:", args.target_col)

    result = run_elastic_net_search(
        df=df,
        target_col=args.target_col,
        date_col=args.date_col,
        train_end=args.train_end,
        val_end=args.val_end,
        exclude_cols=args.exclude_cols,
        missing_threshold=args.missing_threshold,
        variance_threshold=args.variance_threshold,
        fill_value=args.fill_value,
        alpha_grid=args.alpha_grid,
        l1_ratio_grid=args.l1_ratio_grid,
        max_iter=args.max_iter,
        tol=args.tol,
        selection=args.selection,
        always_keep_cols=args.always_keep_cols,
        random_state=args.random_state,
        verbose=verbose,
    )

    print("\n===== BEST RESULT =====")
    print(f"best_alpha      : {result['best_alpha']}")
    print(f"best_l1_ratio   : {result['best_l1_ratio']}")
    print(f"best_val_rmse   : {result['best_val_rmse']:.6f}")
    print(f"best_test_rmse  : {result['best_test_rmse']:.6f}")
    print(f"best_val_r2     : {result['best_val_r2']:.6f}")
    print(f"best_test_r2    : {result['best_test_r2']:.6f}")
    print(f"selected_count  : {len(result['selected_features'])}")

    # ===== 保存结果 =====
    results_path = os.path.join(args.output_dir, "elastic_net_grid_results.csv")
    coef_path = os.path.join(args.output_dir, "elastic_net_coefficients.csv")
    selected_path = os.path.join(args.output_dir, "elastic_net_selected_features.csv")
    summary_path = os.path.join(args.output_dir, "elastic_net_best_summary.txt")
    initial_features_path = os.path.join(args.output_dir, "elastic_net_initial_features.csv")
    exclude_cols_path = os.path.join(args.output_dir, "elastic_net_effective_exclude_cols.csv")
    always_keep_path = os.path.join(args.output_dir, "elastic_net_always_keep_cols.csv")

    result["results_df"].to_csv(results_path, index=False)
    result["coef_df"].to_csv(coef_path, index=False)
    pd.Series(result["selected_features"], name="feature").to_csv(selected_path, index=False)
    pd.Series(result["final_feature_cols"], name="feature").to_csv(initial_features_path, index=False)
    pd.Series(result["effective_exclude_cols"], name="feature").to_csv(exclude_cols_path, index=False)
    pd.Series(result["always_keep_feature_cols"], name="feature").to_csv(always_keep_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Best Elastic Net Result\n")
        f.write(f"best_alpha: {result['best_alpha']}\n")
        f.write(f"best_l1_ratio: {result['best_l1_ratio']}\n")
        f.write(f"best_val_rmse: {result['best_val_rmse']}\n")
        f.write(f"best_test_rmse: {result['best_test_rmse']}\n")
        f.write(f"best_val_r2: {result['best_val_r2']}\n")
        f.write(f"best_test_r2: {result['best_test_r2']}\n")
        f.write(f"random_state: {result['random_state']}\n")
        f.write(f"selected_feature_count: {len(result['selected_features'])}\n")
        f.write(f"selected_candidate_feature_count: {len(result['selected_candidate_features'])}\n")
        f.write(f"best_n_selected_features: {result['best_n_selected_features']}\n")
        f.write(f"auto_exclude_cols_count: {len(result['auto_exclude_cols'])}\n")
        f.write(f"always_keep_feature_cols_count: {len(result['always_keep_feature_cols'])}\n")

    print("\nSaved files:")
    print(" ", results_path)
    print(" ", coef_path)
    print(" ", selected_path)
    print(" ", summary_path)
    print(" ", initial_features_path)
    print(" ", exclude_cols_path)
    print(" ", always_keep_path)


if __name__ == "__main__":
    main()
