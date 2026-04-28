"""Tune the growth-from-lag1 XGBoost model from xgboost_latest_zhou with Optuna.

This script reuses the data preparation, leakage controls, and feature-selection
logic from xgboost_latest_zhou.py, but does not save Optuna trial or best-param
files. Results are printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
_XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "xdg-cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

import xgboost_latest_zhou as zhou
from xgboost import XGBRegressor


SCENARIO_NAME = "lagged_only_no_same_quarter_exog_forecast"


@dataclass
class PreparedOptunaData:
    model_df: pd.DataFrame
    selected_features: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Optuna run."""
    parser = argparse.ArgumentParser(
        description="Run Optuna tuning for the growth-from-lag1 XGBoost model."
    )
    parser.add_argument(
        "--monthly",
        type=str,
        default=zhou.DEFAULT_MONTHLY_FILE,
        help="Path to latest integrated monthly LAD panel CSV",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=zhou.RANDOM_STATE,
        help="Random seed for XGBoost and Optuna sampler",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional fixed top-k features; omit to reuse auto feature selection",
    )
    parser.add_argument(
        "--no-lad-dummies",
        dest="add_lad_dummies",
        action="store_false",
        default=zhou.ADD_LAD_DUMMIES,
        help="Do not include LAD one-hot dummy features",
    )
    parser.add_argument(
        "--force-keep-lad",
        dest="force_keep_lad",
        action="store_true",
        default=zhou.FORCE_KEEP_LAD_DUMMIES,
        help="Force all LAD dummy features to survive feature selection",
    )
    parser.add_argument(
        "--include-same-quarter-exog",
        action="store_true",
        help="Include same-quarter exogenous predictors. Off by default to keep the forecast setup.",
    )
    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna jobs. Keep at 1 unless you have enough CPU headroom.",
    )
    return parser.parse_args()


def fit_xgb(
    params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBRegressor:
    """Fit XGBoost while supporting both old and new early-stopping APIs."""
    params = dict(params)
    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    except TypeError:
        early_stopping_rounds = int(params.pop("early_stopping_rounds"))
        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
    return model


def rmse_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE without relying on sklearn's version-specific squared flag."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def prepare_optuna_data(args: argparse.Namespace) -> PreparedOptunaData:
    """Build the model table and selected feature matrix using source logic."""
    zhou.SHOW_PLOTS = False
    zhou.SAVE_PLOTS = False
    zhou.TOP_K_FEATURES = args.top_k
    zhou.ADD_LAD_DUMMIES = bool(args.add_lad_dummies)
    zhou.FORCE_KEEP_LAD_DUMMIES = bool(args.force_keep_lad)

    monthly_file = zhou.resolve_input_path(
        args.monthly,
        "monthly_lad_panel_2000_2025_replaced_homelessness_final.csv",
    )

    zhou.print_section("Optuna XGBoost growth tuning setup")
    print(f"Monthly file:                 {monthly_file}")
    print(f"Scenario:                     {SCENARIO_NAME}")
    print(f"Include same-quarter exog:    {args.include_same_quarter_exog}")
    print(f"LAD dummies included:         {zhou.ADD_LAD_DUMMIES}")
    print(f"Force-keep all LAD dummies:   {zhou.FORCE_KEEP_LAD_DUMMIES}")
    print(f"Top-k feature cap:            {zhou.TOP_K_FEATURES if zhou.TOP_K_FEATURES is not None else 'None, auto-selected'}")
    print("Optuna artefacts:             not saved; all tuning results are printed")

    qdf = zhou.build_quarterly_panel(monthly_file)
    qdf, same_quarter_exog, lagged_exog, target_history = zhou.add_report_features(qdf)

    with tempfile.TemporaryDirectory(prefix="xgb_growth_optuna_") as tmp:
        tmp_dir = Path(tmp)
        model_df, feature_cols = zhou.build_model_table(
            qdf,
            same_quarter_exog,
            lagged_exog,
            target_history,
            include_same_quarter_exog=args.include_same_quarter_exog,
            scenario_name=SCENARIO_NAME,
            out_dir=tmp_dir,
        )
        selected_features, _ = zhou.clean_and_select_features(
            model_df,
            feature_cols,
            SCENARIO_NAME,
            tmp_dir,
        )

    train_mask = model_df["split"] == "train"
    valid_mask = model_df["split"] == "valid"
    test_mask = model_df["split"] == "test"

    X_train = zhou.ensure_numeric_df(model_df.loc[train_mask, selected_features])
    y_train = model_df.loc[train_mask, "target_growth_from_lag1"].astype(float)
    X_valid = zhou.ensure_numeric_df(model_df.loc[valid_mask, selected_features])
    y_valid = model_df.loc[valid_mask, "target_growth_from_lag1"].astype(float)
    X_test = zhou.ensure_numeric_df(model_df.loc[test_mask, selected_features])

    print("\nPrepared matrices:")
    print(f"  X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")
    print(f"  Selected features: {len(selected_features):,}")

    return PreparedOptunaData(
        model_df=model_df,
        selected_features=selected_features,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
    )


def trial_params(trial: optuna.Trial, seed: int) -> Dict[str, object]:
    """Suggest XGBoost parameters for the growth target."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": zhou.XGB_N_ESTIMATORS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0, log=True),
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": zhou.N_JOBS,
        "early_stopping_rounds": zhou.XGB_EARLY_STOPPING_ROUNDS,
    }


def objective_factory(data: PreparedOptunaData, seed: int):
    """Create an Optuna objective over the prepared growth matrices."""
    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial, seed)
        model = fit_xgb(params, data.X_train, data.y_train, data.X_valid, data.y_valid)
        pred_valid_growth = model.predict(data.X_valid)
        rmse = rmse_score(data.y_valid, pred_valid_growth)
        trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))
        return rmse

    return objective


def print_trial_result(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print one compact line after each completed trial."""
    if trial.value is None:
        return
    best = study.best_value
    best_marker = " *best*" if trial.number == study.best_trial.number else ""
    print(f"Trial {trial.number:03d}: valid_growth_RMSE={trial.value:.6f}, best={best:.6f}{best_marker}")


def full_params_from_best(best_params: Dict[str, object], seed: int) -> Dict[str, object]:
    """Merge tuned parameters with the fixed XGBoost settings."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": zhou.XGB_N_ESTIMATORS,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": zhou.N_JOBS,
        "early_stopping_rounds": zhou.XGB_EARLY_STOPPING_ROUNDS,
    }
    params.update(best_params)
    return params


def print_best_model_count_metrics(data: PreparedOptunaData, params: Dict[str, object]) -> None:
    """Refit the best model and print count-scale validation/test metrics."""
    valid_mask = data.model_df["split"] == "valid"
    test_mask = data.model_df["split"] == "test"

    best_model = fit_xgb(params, data.X_train, data.y_train, data.X_valid, data.y_valid)

    pred_valid_growth = best_model.predict(data.X_valid)
    pred_test_growth = best_model.predict(data.X_test)

    valid_rows = data.model_df.loc[valid_mask].copy()
    test_rows = data.model_df.loc[test_mask].copy()

    pred_valid_count = zhou.growth_to_count(pred_valid_growth, valid_rows)
    pred_test_count = zhou.growth_to_count(pred_test_growth, test_rows)

    valid_growth_rmse = rmse_score(data.y_valid, pred_valid_growth)
    test_growth_rmse = rmse_score(
        data.model_df.loc[test_mask, "target_growth_from_lag1"].astype(float),
        pred_test_growth,
    )
    valid_count_metrics = zhou.compute_metrics(
        valid_rows[zhou.TARGET_COL].astype(float).values,
        pred_valid_count,
    )
    test_count_metrics = zhou.compute_metrics(
        test_rows[zhou.TARGET_COL].astype(float).values,
        pred_test_count,
    )

    zhou.print_section("Best model refit metrics")
    print(f"Best iteration:             {getattr(best_model, 'best_iteration', None)}")
    print(f"Validation growth RMSE:     {valid_growth_rmse:.6f}")
    print(f"Test growth RMSE:           {test_growth_rmse:.6f}")
    print(
        "Validation count metrics:  "
        f"MAE={valid_count_metrics['MAE']:.3f}, "
        f"RMSE={valid_count_metrics['RMSE']:.3f}, "
        f"R2={valid_count_metrics['R2']:.3f}, "
        f"SMAPE={valid_count_metrics['SMAPE_percent']:.3f}"
    )
    print(
        "Test count metrics:        "
        f"MAE={test_count_metrics['MAE']:.3f}, "
        f"RMSE={test_count_metrics['RMSE']:.3f}, "
        f"R2={test_count_metrics['R2']:.3f}, "
        f"SMAPE={test_count_metrics['SMAPE_percent']:.3f}"
    )


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = prepare_optuna_data(args)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    zhou.print_section("Running Optuna")
    print(f"Trials:                       {args.n_trials}")
    print(f"Timeout seconds:              {args.timeout}")
    print(f"Optuna parallel jobs:          {args.optuna_n_jobs}")
    print(f"XGBoost n_jobs per trial:      {zhou.N_JOBS}")

    study.optimize(
        objective_factory(data, args.seed),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.optuna_n_jobs,
        callbacks=[print_trial_result],
        show_progress_bar=False,
    )

    best_full_params = full_params_from_best(study.best_params, args.seed)

    zhou.print_section("Optuna best result")
    print(f"Best validation growth RMSE: {study.best_value:.6f}")
    print("Best tuned parameters:")
    print(json.dumps(study.best_params, indent=2, sort_keys=True))
    print("Full XGBoost parameters for the tuned growth model:")
    print(json.dumps(best_full_params, indent=2, sort_keys=True))

    print_best_model_count_metrics(data, best_full_params)


if __name__ == "__main__":
    main()
