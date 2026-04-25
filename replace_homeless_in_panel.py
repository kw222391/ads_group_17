from pathlib import Path
import argparse
import pandas as pd

NEW_COLS = ["homelessness_relief", "homelessness_total_assessments", "homelessness_per_1000"]


def quarter_from_month(s):
    m = pd.to_numeric(s, errors="coerce")
    return "Q" + (((m - 1) // 3 + 1).astype("Int64")).astype(str)


def first_non_null(x):
    y = x.dropna()
    return y.iloc[0] if len(y) else pd.NA


def read_csv_auto(path):
    return pd.read_csv(path, compression="infer", low_memory=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-homeless", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    homeless_path = Path(args.new_homeless)
    panel_path = Path(args.panel)
    output_path = Path(args.output)

    homeless = read_csv_auto(homeless_path)
    panel = read_csv_auto(panel_path)
    original_rows = len(panel)

    homeless = homeless.rename(columns={
        "LAD_code": "lad_code",
        "LA_name": "la_name",
        "Year": "year",
        "Quarter": "_merge_quarter",
        "Homeless_relief": "homelessness_relief",
        "Total_assessments": "homelessness_total_assessments",
        "Homeless_per_1000": "homelessness_per_1000",
    })

    need_homeless = ["lad_code", "year", "_merge_quarter"] + NEW_COLS
    missing_homeless = [c for c in need_homeless if c not in homeless.columns]
    if missing_homeless:
        raise ValueError(f"new homeless file missing columns: {missing_homeless}")

    need_panel = ["lad_code", "year", "month"]
    missing_panel = [c for c in need_panel if c not in panel.columns]
    if missing_panel:
        raise ValueError(f"panel file missing columns: {missing_panel}")

    homeless = homeless[need_homeless].copy()
    homeless["lad_code"] = homeless["lad_code"].astype("string").str.strip().str.upper()
    homeless["year"] = pd.to_numeric(homeless["year"], errors="coerce").astype("Int64")
    homeless["_merge_quarter"] = homeless["_merge_quarter"].astype("string").str.strip().str.upper()
    for c in NEW_COLS:
        homeless[c] = pd.to_numeric(homeless[c], errors="coerce")

    homeless = homeless.groupby(["lad_code", "year", "_merge_quarter"], as_index=False, dropna=False).agg({c: first_non_null for c in NEW_COLS})

    old_homeless_cols = [c for c in panel.columns if "homeless" in c.lower()]
    insert_at = min([panel.columns.get_loc(c) for c in old_homeless_cols], default=len(panel.columns))
    base_cols = [c for c in panel.columns if c not in old_homeless_cols]
    before_cols = [c for c in base_cols if panel.columns.get_loc(c) < insert_at]
    after_cols = [c for c in base_cols if panel.columns.get_loc(c) >= insert_at]

    panel = panel.drop(columns=old_homeless_cols)
    panel["lad_code"] = panel["lad_code"].astype("string").str.strip().str.upper()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["_merge_quarter"] = quarter_from_month(panel["month"])

    out = panel.merge(homeless, on=["lad_code", "year", "_merge_quarter"], how="left", validate="many_to_one")
    out = out.drop(columns=["_merge_quarter"])
    out = out[before_cols + NEW_COLS + after_cols]

    if len(out) != original_rows:
        raise RuntimeError(f"row count changed from {original_rows} to {len(out)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_compression = {"method": "gzip", "compresslevel": 1} if str(output_path).lower().endswith(".gz") else None
    out.to_csv(output_path, index=False, compression=write_compression)

    print(f"saved: {output_path}")
    print(f"rows: {len(out):,}")
    print(f"dropped old homeless columns: {old_homeless_cols}")
    print("new homeless non-null counts:")
    print(out[NEW_COLS].notna().sum().to_string())


if __name__ == "__main__":
    main()
