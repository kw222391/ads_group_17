from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from selector_utils import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_ALWAYS_KEEP_COLS,
    DEFAULT_EXCLUDE_COLS,
    DEFAULT_L1_RATIO_GRID,
    DEFAULT_RANDOM_STATE,
    SELECTOR_MANIFEST_NAME,
    apply_train_based_feature_filters,
    clean_features,
    combo_folder_name,
    drop_rows_with_missing_target,
    ensure_date_column,
    ensure_directory,
    infer_target_family_exclusions,
    load_table,
    make_time_split_masks,
    save_json,
    save_series_csv,
    select_numeric_features,
    set_random_seed,
    summarize_feature_groups,
    timestamp_string,
    validate_model_inputs,
    validate_required_columns,
    validate_search_space,
)

try:
    import cudf
    import cupy as cp
    from cuml.linear_model import ElasticNet
    from cuml.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover
    cudf = None
    cp = None
    ElasticNet = None
    StandardScaler = None
    RAPIDS_IMPORT_ERROR = exc
else:
    RAPIDS_IMPORT_ERROR = None


def require_rapids() -> None:
    if RAPIDS_IMPORT_ERROR is not None:
        raise ImportError(
            "This script requires RAPIDS GPU packages: cudf, cupy, and cuml. "
            "Install a compatible RAPIDS/CUDA environment before training."
        ) from RAPIDS_IMPORT_ERROR


def to_gpu_train(X_train_pd: pd.DataFrame, y_train_pd: pd.Series):
    require_rapids()
    X_train = cudf.from_pandas(X_train_pd.astype(np.float32))
    y_train = cudf.from_pandas(y_train_pd.astype(np.float32))
    return X_train, y_train


def fit_single_elastic_net_selector(
    X_train_gpu,
    y_train_gpu,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    selection: str,
    random_state: int,
):
    """
    Fit Elastic Net only as a feature selector.
    No validation/test linear evaluation is performed here.
    """
    require_rapids()
    set_random_seed(random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_gpu)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        selection=selection,
    )
    model.fit(X_train_scaled, y_train_gpu)

    coef = cp.asnumpy(model.coef_)
    n_selected = int(np.sum(coef != 0))

    return {
        "model": model,
        "scaler": scaler,
        "coef": coef,
        "n_selected_features": n_selected,
    }


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    train_start: str | None,
    train_end: str,
    val_end: str,
    exclude_cols: list[str],
    always_keep_cols: list[str],
    missing_threshold: float,
    variance_threshold: float,
    fill_value: float,
    verbose: bool,
):
    df = ensure_date_column(df.copy(), date_col)
    validate_required_columns(df, [target_col, date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    auto_exclude_cols = infer_target_family_exclusions(df=df, target_col=target_col)
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
    X_all_keep = (
        df[always_keep_feature_cols].copy()
        if always_keep_feature_cols
        else pd.DataFrame(index=df.index)
    )
    y_all = df[target_col].copy()

    train_mask, val_mask, test_mask = make_time_split_masks(
        df=df,
        date_col=date_col,
        train_start=train_start,
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
        X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd, stage="time-based splitting"
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
        X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd, stage="feature filtering and target cleaning"
    )

    candidate_feature_cols_filtered = X_train_pd.columns.tolist()

    if verbose:
        print("After filters:")
        print("  features after missing filter :", len(keep_after_missing))
        print("  Elastic Net candidate features after variance filter:", len(candidate_feature_cols_filtered))
        print("  always-keep features:", len(always_keep_feature_cols))

    return {
        "df": df,
        "target_col": target_col,
        "date_col": date_col,
        "train_start": train_start,
        "train_end": train_end,
        "val_end": val_end,
        "exclude_cols": exclude_cols,
        "auto_exclude_cols": auto_exclude_cols,
        "effective_exclude_cols": effective_exclude_cols,
        "always_keep_feature_cols": always_keep_feature_cols,
        "candidate_feature_cols": candidate_feature_cols_filtered,
        "fill_value": fill_value,
        "missing_threshold": missing_threshold,
        "variance_threshold": variance_threshold,
        "X_train_pd": X_train_pd,
        "X_val_pd": X_val_pd,
        "X_test_pd": X_test_pd,
        "X_train_keep_pd": X_train_keep_pd,
        "X_val_keep_pd": X_val_keep_pd,
        "X_test_keep_pd": X_test_keep_pd,
        "y_train_pd": y_train_pd,
        "y_val_pd": y_val_pd,
        "y_test_pd": y_test_pd,
        "keep_after_missing": keep_after_missing,
        "keep_after_variance": candidate_feature_cols_filtered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Elastic Net feature selection and save one folder per parameter combo.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw data (.csv, .csv.gz, or .parquet)")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name")
    parser.add_argument("--date_col", type=str, default="date", help="Date column name")
    parser.add_argument("--train_start", type=str, default=None, help="Optional train start boundary: train uses train_start <= date < train_end")
    parser.add_argument("--train_end", type=str, default="2023-09-01")
    parser.add_argument("--val_end", type=str, default="2024-09-30")
    parser.add_argument("--exclude_cols", type=str, nargs="*", default=DEFAULT_EXCLUDE_COLS)
    parser.add_argument("--always_keep_cols", type=str, nargs="*", default=DEFAULT_ALWAYS_KEEP_COLS)
    parser.add_argument("--missing_threshold", type=float, default=0.4)
    parser.add_argument("--variance_threshold", type=float, default=1e-8)
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--l1_ratio_grid", type=float, nargs="+", default=DEFAULT_L1_RATIO_GRID)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--selection", type=str, default="cyclic", choices=["cyclic", "random"])
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--output_root", type=str, default="selector_runs")
    parser.add_argument("--run_name", type=str, default=None, help="Optional custom parent run folder name")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    require_rapids()
    set_random_seed(args.random_state)
    validate_search_space(args.alpha_grid, args.l1_ratio_grid)

    df = load_table(args.input_path)
    run_name = args.run_name or f"{args.target_col}__{timestamp_string()}"
    parent_run_dir = ensure_directory(Path(args.output_root) / run_name)

    if verbose:
        print(f"Loading data from: {args.input_path}")
        print(f"Output root: {parent_run_dir}")
        print(f"Shape: {df.shape}")

    prepared = prepare_data(
        df=df,
        target_col=args.target_col,
        date_col=args.date_col,
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        exclude_cols=args.exclude_cols,
        always_keep_cols=args.always_keep_cols,
        missing_threshold=args.missing_threshold,
        variance_threshold=args.variance_threshold,
        fill_value=args.fill_value,
        verbose=verbose,
    )

    X_train_gpu, y_train_gpu = to_gpu_train(
        prepared["X_train_pd"],
        prepared["y_train_pd"],
    )

    results_rows: list[dict] = []

    for alpha in args.alpha_grid:
        for l1_ratio in args.l1_ratio_grid:
            fit_result = fit_single_elastic_net_selector(
                X_train_gpu=X_train_gpu,
                y_train_gpu=y_train_gpu,
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=args.max_iter,
                tol=args.tol,
                selection=args.selection,
                random_state=args.random_state,
            )

            coef_df = pd.DataFrame(
                {
                    "feature": prepared["candidate_feature_cols"],
                    "coef": fit_result["coef"],
                    "abs_coef": np.abs(fit_result["coef"]),
                }
            ).sort_values("abs_coef", ascending=False).reset_index(drop=True)
            selected_candidate_features = coef_df.loc[coef_df["coef"] != 0, "feature"].tolist()
            selected_features = list(
                dict.fromkeys(prepared["always_keep_feature_cols"] + selected_candidate_features)
            )

            combo_dir = ensure_directory(parent_run_dir / combo_folder_name(alpha, l1_ratio))
            coef_df.to_csv(combo_dir / "elastic_net_coefficients.csv", index=False)
            save_series_csv(selected_candidate_features, combo_dir / "selected_candidate_features.csv", name="feature")
            save_series_csv(selected_features, combo_dir / "selected_features.csv", name="feature")
            save_series_csv(prepared["always_keep_feature_cols"], combo_dir / "always_keep_feature_cols.csv", name="feature")
            save_series_csv(prepared["effective_exclude_cols"], combo_dir / "effective_exclude_cols.csv", name="feature")

            manifest = {
                "selector_type": "elastic_net",
                "raw_data_path": str(Path(args.input_path).resolve()),
                "run_name": run_name,
                "combo_name": combo_dir.name,
                "target_col": args.target_col,
                "date_col": args.date_col,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "val_end": args.val_end,
                "exclude_cols": list(args.exclude_cols),
                "auto_exclude_cols": prepared["auto_exclude_cols"],
                "effective_exclude_cols": prepared["effective_exclude_cols"],
                "always_keep_feature_cols": prepared["always_keep_feature_cols"],
                "selected_candidate_features": selected_candidate_features,
                "selected_features": selected_features,
                "candidate_feature_cols": prepared["candidate_feature_cols"],
                "fill_value": args.fill_value,
                "missing_threshold": args.missing_threshold,
                "variance_threshold": args.variance_threshold,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "n_selected_candidate_features": fit_result["n_selected_features"],
                "n_selected_total_features": len(selected_features),
                "random_state": args.random_state,
                "selection": args.selection,
                "max_iter": args.max_iter,
                "tol": args.tol,
            }
            save_json(manifest, combo_dir / SELECTOR_MANIFEST_NAME)

            row = {
                "run_name": run_name,
                "combo_name": combo_dir.name,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "val_end": args.val_end,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "n_selected_candidate_features": fit_result["n_selected_features"],
                "n_selected_total_features": len(selected_features),
                "combo_dir": str(combo_dir.resolve()),
            }
            results_rows.append(row)

            if verbose:
                print(
                    f"alpha={alpha:.5g}, l1_ratio={l1_ratio:.3f} | "
                    f"selected={fit_result['n_selected_features']}"
                )

    results_df = pd.DataFrame(results_rows).sort_values(
        ["n_selected_candidate_features", "alpha", "l1_ratio"],
        ascending=[True, True, True],
    )
    results_df.to_csv(parent_run_dir / "elastic_net_grid_results.csv", index=False)

    summary = {
        "run_name": run_name,
        "target_col": args.target_col,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "n_combos": int(len(results_rows)),
        "output_root": str(parent_run_dir.resolve()),
        "note": "No linear validation/test ranking is used. Final model selection should be done downstream with XGBoost.",
    }
    save_json(summary, parent_run_dir / "selector_run_summary.json")
    (parent_run_dir / "selector_run_summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()), encoding="utf-8"
    )

    print("\n===== ELASTIC NET FEATURE SELECTION COMPLETE =====")
    print(f"Saved {len(results_rows)} selector combos under: {parent_run_dir}")
    print("No selector-level linear best is chosen here; downstream XGBoost should decide the final best combo.")


if __name__ == "__main__":
    main()
