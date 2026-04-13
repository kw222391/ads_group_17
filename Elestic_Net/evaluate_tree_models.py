from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from selector_utils import DATASET_MANIFEST_NAME, discover_manifest_paths, ensure_directory, load_json, save_json

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def require_xgboost() -> None:
    if XGBOOST_IMPORT_ERROR is not None:
        raise ImportError(
            "This script requires xgboost. Install it with `pip install xgboost`."
        ) from XGBOOST_IMPORT_ERROR


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model(args: argparse.Namespace) -> XGBRegressor:
    require_xgboost()

    base_kwargs = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="reg:squarederror",
        random_state=args.random_state,
        tree_method="hist",
        eval_metric="rmse",
    )

    if args.device == "cuda":
        base_kwargs["device"] = "cuda"
    elif args.device == "cpu":
        pass
    else:  # auto
        base_kwargs["device"] = "cuda"

    return XGBRegressor(**base_kwargs)


def fit_and_evaluate_dataset(manifest_path: Path, args: argparse.Namespace, verbose: bool) -> dict:
    dataset_dir = manifest_path.parent
    run_dir = dataset_dir.parent
    manifest = load_json(manifest_path)

    train_df = pd.read_parquet(manifest["train_path"])
    val_df = pd.read_parquet(manifest["val_path"])
    test_df = pd.read_parquet(manifest["test_path"])

    target_col = manifest["target_col"]
    feature_cols = manifest["final_feature_cols"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = build_model(args)

    fit_kwargs = {}
    if args.early_stopping_rounds > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
        fit_kwargs["early_stopping_rounds"] = args.early_stopping_rounds

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except Exception as exc:
        if args.device == "auto":
            if verbose:
                print(f"CUDA XGBoost failed for {dataset_dir}, retrying on CPU: {exc}")
            args_cpu = argparse.Namespace(**vars(args))
            args_cpu.device = "cpu"
            model = build_model(args_cpu)
            model.fit(X_train, y_train, **{k: v for k, v in fit_kwargs.items() if k != "early_stopping_rounds"})
        else:
            raise

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    metrics = {
        "selector_dir": str(run_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "alpha": manifest.get("alpha"),
        "l1_ratio": manifest.get("l1_ratio"),
        "n_final_features": len(feature_cols),
        "val_rmse": rmse(y_val, val_pred),
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "val_r2": float(r2_score(y_val, val_pred)),
        "test_rmse": rmse(y_test, test_pred),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
    }

    out_dir = ensure_directory(run_dir / "xgboost_eval")
    save_json(metrics, out_dir / "metrics.json")

    booster = model.get_booster()
    try:
        booster.save_model(str(out_dir / "xgboost_model.json"))
    except Exception:
        pass

    importance_gain = booster.get_score(importance_type="gain")
    importance_weight = booster.get_score(importance_type="weight")
    importance_df = pd.DataFrame({"feature": feature_cols})
    importance_df["gain"] = importance_df["feature"].map(importance_gain).fillna(0.0)
    importance_df["weight"] = importance_df["feature"].map(importance_weight).fillna(0.0)
    importance_df = importance_df.sort_values("gain", ascending=False)
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    if verbose:
        print(
            f"Evaluated {run_dir.name}: val_rmse={metrics['val_rmse']:.4f}, test_rmse={metrics['test_rmse']:.4f}"
        )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively evaluate XGBoost on all built train/val/test datasets.")
    parser.add_argument("--selector_root", type=str, required=True, help="Root directory containing selector folders and built datasets")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    manifest_paths = discover_manifest_paths(args.selector_root, DATASET_MANIFEST_NAME)
    if not manifest_paths:
        raise FileNotFoundError(
            f"No {DATASET_MANIFEST_NAME} files were found under {args.selector_root}. "
            "Run build_selected_dataset.py first."
        )

    rows = []
    for manifest_path in manifest_paths:
        rows.append(fit_and_evaluate_dataset(manifest_path, args, verbose))

    summary_df = pd.DataFrame(rows).sort_values(["val_rmse", "test_rmse"])
    summary_path = Path(args.selector_root) / "xgboost_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nEvaluated {len(rows)} datasets.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
