from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selector_utils import (
    DATASET_MANIFEST_NAME,
    SELECTOR_MANIFEST_NAME,
    build_final_feature_frames,
    clean_features,
    discover_manifest_paths,
    drop_rows_with_missing_target,
    ensure_date_column,
    ensure_directory,
    load_json,
    load_table,
    make_time_split_masks,
    save_dataframe_parquet,
    save_json,
)


def build_dataset_for_manifest(
    manifest_path: Path,
    raw_data_path_override: str | None,
    dataset_dir_name: str,
    verbose: bool,
) -> dict:
    selector_dir = manifest_path.parent
    manifest = load_json(manifest_path)
    raw_data_path = raw_data_path_override or manifest["raw_data_path"]
    df = load_table(raw_data_path)
    df = ensure_date_column(df, manifest["date_col"])
    df[manifest["date_col"]] = pd.to_datetime(df[manifest["date_col"]], errors="coerce")

    train_mask, val_mask, test_mask = make_time_split_masks(
        df=df,
        date_col=manifest["date_col"],
        train_end=manifest["train_end"],
        val_end=manifest["val_end"],
    )

    selected_candidate_features = [
        c for c in manifest["selected_candidate_features"] if c in df.columns
    ]
    always_keep_feature_cols = [
        c for c in manifest["always_keep_feature_cols"] if c in df.columns
    ]
    target_col = manifest["target_col"]
    fill_value = float(manifest.get("fill_value", 0.0))

    X_candidates = df[selected_candidate_features].copy() if selected_candidate_features else pd.DataFrame(index=df.index)
    X_keep = df[always_keep_feature_cols].copy() if always_keep_feature_cols else pd.DataFrame(index=df.index)
    y_all = df[target_col].copy()

    X_train_filtered = clean_features(X_candidates.loc[train_mask].copy(), fill_value=fill_value)
    X_val_filtered = clean_features(X_candidates.loc[val_mask].copy(), fill_value=fill_value)
    X_test_filtered = clean_features(X_candidates.loc[test_mask].copy(), fill_value=fill_value)

    X_train_keep = clean_features(X_keep.loc[train_mask].copy(), fill_value=fill_value)
    X_val_keep = clean_features(X_keep.loc[val_mask].copy(), fill_value=fill_value)
    X_test_keep = clean_features(X_keep.loc[test_mask].copy(), fill_value=fill_value)

    y_train = y_all.loc[train_mask].copy()
    y_val = y_all.loc[val_mask].copy()
    y_test = y_all.loc[test_mask].copy()

    X_train_filtered, y_train = drop_rows_with_missing_target(X_train_filtered, y_train)
    X_train_keep = X_train_keep.loc[y_train.index].copy()
    X_val_filtered, y_val = drop_rows_with_missing_target(X_val_filtered, y_val)
    X_val_keep = X_val_keep.loc[y_val.index].copy()
    X_test_filtered, y_test = drop_rows_with_missing_target(X_test_filtered, y_test)
    X_test_keep = X_test_keep.loc[y_test.index].copy()

    X_train_final, X_val_final, X_test_final, final_feature_cols = build_final_feature_frames(
        X_train_filtered=X_train_filtered,
        X_val_filtered=X_val_filtered,
        X_test_filtered=X_test_filtered,
        X_train_keep=X_train_keep,
        X_val_keep=X_val_keep,
        X_test_keep=X_test_keep,
        selected_candidate_features=selected_candidate_features,
        always_keep_feature_cols=always_keep_feature_cols,
    )

    train_df = X_train_final.copy()
    train_df[target_col] = y_train.values
    val_df = X_val_final.copy()
    val_df[target_col] = y_val.values
    test_df = X_test_final.copy()
    test_df[target_col] = y_test.values

    dataset_dir = ensure_directory(selector_dir / dataset_dir_name)
    train_path = dataset_dir / "train_selected.parquet"
    val_path = dataset_dir / "val_selected.parquet"
    test_path = dataset_dir / "test_selected.parquet"

    save_dataframe_parquet(train_df, train_path)
    save_dataframe_parquet(val_df, val_path)
    save_dataframe_parquet(test_df, test_path)

    dataset_manifest = {
        "source_selector_manifest": str(manifest_path.resolve()),
        "raw_data_path": str(Path(raw_data_path).resolve()),
        "target_col": target_col,
        "date_col": manifest["date_col"],
        "train_end": manifest["train_end"],
        "val_end": manifest["val_end"],
        "alpha": manifest["alpha"],
        "l1_ratio": manifest["l1_ratio"],
        "val_rmse_selector": manifest.get("val_rmse"),
        "test_rmse_selector": manifest.get("test_rmse"),
        "always_keep_feature_cols": always_keep_feature_cols,
        "selected_candidate_features": selected_candidate_features,
        "final_feature_cols": final_feature_cols,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "n_final_features": int(len(final_feature_cols)),
        "train_path": str(train_path.resolve()),
        "val_path": str(val_path.resolve()),
        "test_path": str(test_path.resolve()),
    }
    save_json(dataset_manifest, dataset_dir / DATASET_MANIFEST_NAME)

    if verbose:
        print(f"Built dataset for: {selector_dir}")
        print(f"  final features: {len(final_feature_cols)}")
        print(f"  saved under   : {dataset_dir}")

    return {
        "selector_dir": str(selector_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "alpha": manifest["alpha"],
        "l1_ratio": manifest["l1_ratio"],
        "n_final_features": len(final_feature_cols),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_test_rows": len(test_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test parquet datasets for all selector result folders.")
    parser.add_argument("--selector_root", type=str, required=True, help="Root directory containing selector run folders")
    parser.add_argument("--raw_data_path", type=str, default=None, help="Optional override for the raw input data path")
    parser.add_argument("--dataset_dir_name", type=str, default="dataset", help="Name of the dataset subfolder inside each selector folder")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    manifest_paths = discover_manifest_paths(args.selector_root, SELECTOR_MANIFEST_NAME)
    if not manifest_paths:
        raise FileNotFoundError(
            f"No {SELECTOR_MANIFEST_NAME} files were found under {args.selector_root}"
        )

    rows = []
    for manifest_path in manifest_paths:
        rows.append(
            build_dataset_for_manifest(
                manifest_path=manifest_path,
                raw_data_path_override=args.raw_data_path,
                dataset_dir_name=args.dataset_dir_name,
                verbose=verbose,
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(["alpha", "l1_ratio"])
    summary_path = Path(args.selector_root) / "build_selected_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nBuilt datasets for {len(rows)} selector folders.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
