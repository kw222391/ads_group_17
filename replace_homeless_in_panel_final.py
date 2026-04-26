from pathlib import Path
import argparse
import pandas as pd

OLD_HOMELESS_COLUMNS = [
    "homelessness_total_owed",
    "homelessness_threatened",
    "homelessness_total_assessments",
    "homelessness_relief",
    "homelessness_per_1000",
    # possible earlier/raw names, in case they were accidentally merged before
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]

NEW_HOMELESS_COLUMNS = [
    "homelessness_relief",
    "homelessness_total_assessments",
    "homelessness_per_1000",
]

REQUIRED_HOMELESS_COLUMNS = [
    "LAD_code",
    "LA_name",
    "Year",
    "Quarter",
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]


QUARTER_MAP = {
    1: "Q1", 2: "Q1", 3: "Q1",
    4: "Q2", 5: "Q2", 6: "Q2",
    7: "Q3", 8: "Q3", 9: "Q3",
    10: "Q4", 11: "Q4", 12: "Q4",
}


def read_csv_auto(path):
    path = Path(path)
    return pd.read_csv(path, compression="infer", low_memory=False)


def write_csv_auto(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="infer", encoding="utf-8-sig")


def first_existing_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find {label} column. Tried: {candidates}. "
        f"Available columns include: {list(df.columns[:30])}"
    )


def month_to_quarter(month_value):
    if pd.isna(month_value):
        return pd.NA

    # numeric month: 1-12
    try:
        m = int(float(month_value))
        if m in QUARTER_MAP:
            return QUARTER_MAP[m]
    except Exception:
        pass

    # date-like string: 2000-01-01, 2000/01, Jan 2000, etc.
    dt = pd.to_datetime(month_value, errors="coerce")
    if pd.notna(dt):
        return QUARTER_MAP[int(dt.month)]

    return pd.NA


def clean_key_series(s):
    return s.astype("string").str.strip()


def make_match_summary(panel_with_helpers, merged, output_path):
    summary = []

    total_rows = len(merged)
    matched_rows = merged["_new_homeless_key_matched"].sum()

    summary.append({
        "metric": "panel_rows",
        "value": total_rows,
    })
    summary.append({
        "metric": "rows_with_matching_homeless_key",
        "value": int(matched_rows),
    })
    summary.append({
        "metric": "match_rate_percent",
        "value": round(float(matched_rows) / total_rows * 100, 2) if total_rows else 0,
    })

    for col in NEW_HOMELESS_COLUMNS:
        summary.append({
            "metric": f"{col}_nonnull",
            "value": int(merged[col].notna().sum()),
        })

    summary_df = pd.DataFrame(summary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove old homelessness variables from the monthly LAD panel and "
            "merge in the final cleaned quarterly homelessness file."
        )
    )
    parser.add_argument(
        "--new-homeless",
        default="data/clean/homeless_lad_2009_2025_final.csv",
        help="Final cleaned homelessness CSV with LAD_code, Year, Quarter columns.",
    )
    parser.add_argument(
        "--panel",
        default="data_new/monthly_lad_panel_2000_2025_with_homelessness_plus_england_london_2014_2018.csv.gz",
        help="Original monthly panel CSV or CSV.GZ.",
    )
    parser.add_argument(
        "--output",
        default="data/clean/monthly_lad_panel_2000_2025_replaced_homelessness_final.csv.gz",
        help="Output path for merged monthly panel.",
    )
    parser.add_argument(
        "--audit",
        default="data/clean/monthly_lad_panel_2000_2025_replaced_homelessness_final_audit.csv",
        help="Small merge audit CSV.",
    )
    args = parser.parse_args()

    panel = read_csv_auto(args.panel)
    homeless = read_csv_auto(args.new_homeless)

    missing = [c for c in REQUIRED_HOMELESS_COLUMNS if c not in homeless.columns]
    if missing:
        raise ValueError(f"New homeless file is missing columns: {missing}")

    original_row_count = len(panel)
    original_columns = list(panel.columns)

    lad_col = first_existing_col(
        panel,
        ["lad_code", "LAD_code", "lad_code_current", "LAD21CD", "LAD22CD", "LAD23CD", "LAD24CD"],
        "LAD code",
    )
    year_col = first_existing_col(panel, ["year", "Year"], "year")
    month_col = first_existing_col(panel, ["month", "Month", "date", "Date"], "month/date")

    # Remove all old homelessness columns before merging the new final version.
    dropped_old_cols = [c for c in OLD_HOMELESS_COLUMNS if c in panel.columns]
    panel = panel.drop(columns=dropped_old_cols)

    # Prepare final homelessness data.
    homeless_merge = homeless[[
        "LAD_code",
        "Year",
        "Quarter",
        "Homeless_relief",
        "Total_assessments",
        "Homeless_per_1000",
    ]].copy()

    homeless_merge = homeless_merge.rename(columns={
        "LAD_code": "_merge_lad_code",
        "Year": "_merge_year",
        "Quarter": "_merge_quarter",
        "Homeless_relief": "homelessness_relief",
        "Total_assessments": "homelessness_total_assessments",
        "Homeless_per_1000": "homelessness_per_1000",
    })

    homeless_merge["_merge_lad_code"] = clean_key_series(homeless_merge["_merge_lad_code"])
    homeless_merge["_merge_year"] = pd.to_numeric(homeless_merge["_merge_year"], errors="coerce").astype("Int64")
    homeless_merge["_merge_quarter"] = clean_key_series(homeless_merge["_merge_quarter"])

    dup_mask = homeless_merge.duplicated(["_merge_lad_code", "_merge_year", "_merge_quarter"], keep=False)
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        dup_preview = homeless_merge.loc[dup_mask, ["_merge_lad_code", "_merge_year", "_merge_quarter"]].head(20)
        raise ValueError(
            f"New homeless file has duplicate LAD-year-quarter keys: {dup_count} rows. "
            f"Preview:\n{dup_preview}"
        )

    homeless_merge["_new_homeless_key_matched"] = True

    # Prepare monthly panel merge keys. Homeless data is quarterly, so it is repeated
    # across the three months in each matching quarter.
    panel["_merge_lad_code"] = clean_key_series(panel[lad_col])
    panel["_merge_year"] = pd.to_numeric(panel[year_col], errors="coerce").astype("Int64")
    panel["_merge_quarter"] = panel[month_col].apply(month_to_quarter).astype("string")

    before_merge_cols = list(panel.columns)

    merged = panel.merge(
        homeless_merge,
        how="left",
        on=["_merge_lad_code", "_merge_year", "_merge_quarter"],
        validate="many_to_one",
    )

    if len(merged) != original_row_count:
        raise RuntimeError(
            f"Row count changed after merge: before={original_row_count}, after={len(merged)}"
        )

    merged["_new_homeless_key_matched"] = merged["_new_homeless_key_matched"].fillna(False).astype(bool)

    make_match_summary(panel, merged, args.audit)

    helper_cols = ["_merge_lad_code", "_merge_year", "_merge_quarter", "_new_homeless_key_matched"]
    merged = merged.drop(columns=[c for c in helper_cols if c in merged.columns])

    write_csv_auto(merged, args.output)

    print(f"Original panel rows: {original_row_count:,}")
    print(f"Output panel rows:   {len(merged):,}")
    print(f"Original panel cols: {len(original_columns):,}")
    print(f"Output panel cols:   {len(merged.columns):,}")
    print(f"LAD column used:     {lad_col}")
    print(f"Year column used:    {year_col}")
    print(f"Month/date used:     {month_col}")
    print(f"Dropped old homeless columns: {dropped_old_cols}")
    print("New homeless non-null counts:")
    for col in NEW_HOMELESS_COLUMNS:
        print(f"  {col}: {merged[col].notna().sum():,}")
    print(f"Saved output: {args.output}")
    print(f"Saved audit:  {args.audit}")


if __name__ == "__main__":
    main()
