from pathlib import Path
import argparse
import pandas as pd


VALUE_COLUMNS = [
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]

OUTPUT_COLUMNS = [
    "LAD_code",
    "LA_name",
    "Year",
    "Quarter",
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]


def qnum(q):
    return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(str(q).strip(), 99)


def period_key(year, quarter):
    return int(year) * 10 + qnum(quarter)


def clean_value(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "na", "n/a"}:
        return ""
    return s


def make_coverage(df):
    rows = []

    for (year, quarter), g in df.groupby(["Year", "Quarter"], dropna=False):
        row = {
            "Year": int(year),
            "Quarter": quarter,
            "LAD_rows": len(g),
        }

        for col in VALUE_COLUMNS:
            row[f"{col}_nonnull"] = g[col].apply(clean_value).ne("").sum()

        rows.append(row)

    rows = sorted(rows, key=lambda x: (int(x["Year"]), qnum(x["Quarter"])))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-sources",
        default="data/clean/homeless_lad_2009_2025_all_sources.csv",
    )
    parser.add_argument(
        "--conflicts",
        default="data/clean/homeless_lad_2009_2025_conflicts.csv",
    )
    parser.add_argument(
        "--output",
        default="data/clean/homeless_lad_2009_2025_final.csv",
    )
    parser.add_argument(
        "--audit",
        default="data/clean/homeless_lad_final_overlap_resolution_audit.csv",
    )
    parser.add_argument(
        "--coverage",
        default="data/clean/homeless_lad_final_coverage_by_quarter.csv",
    )
    args = parser.parse_args()

    all_sources = pd.read_csv(args.all_sources, dtype=str)
    conflicts = pd.read_csv(args.conflicts, dtype=str)

    for col in OUTPUT_COLUMNS:
        if col not in all_sources.columns:
            all_sources[col] = ""

    all_sources["Year"] = all_sources["Year"].astype(int)

    key_cols = ["LAD_code", "Year", "Quarter"]
    all_sources = all_sources.set_index(key_cols, drop=False)

    audit_rows = []

    start = period_key(2014, "Q2")
    end = period_key(2016, "Q4")

    for _, row in conflicts.iterrows():
        lad = row["LAD_code"]
        year = int(row["Year"])
        quarter = row["Quarter"]
        column = row["Column"]

        if column != "Total_assessments":
            continue

        p = period_key(year, quarter)

        # 只处理 2014Q2-2016Q4 的重叠冲突
        if not (start <= p <= end):
            continue

        other_source = clean_value(row.get("Other_source", ""))

        # 只接受 Homelessness 14-18 的值覆盖旧值
        if "14-18" not in other_source and "14_18" not in other_source:
            continue

        key = (lad, year, quarter)

        if key not in all_sources.index:
            continue

        old_value = clean_value(all_sources.at[key, column])
        new_value = clean_value(row["Other_value"])

        if new_value == "":
            continue

        if old_value != new_value:
            all_sources.at[key, column] = new_value

            audit_rows.append(
                {
                    "LAD_code": lad,
                    "Year": year,
                    "Quarter": quarter,
                    "Column": column,
                    "Old_value": old_value,
                    "New_value": new_value,
                    "Reason": "Use Homelessness 14-18 for overlap period 2014Q2-2016Q4",
                    "Source_used": other_source,
                }
            )

    final_df = all_sources.reset_index(drop=True)
    final_df = final_df[OUTPUT_COLUMNS]

    final_df["Year"] = final_df["Year"].astype(int)
    final_df["_q"] = final_df["Quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
    final_df = final_df.sort_values(["Year", "_q", "LAD_code"]).drop(columns=["_q"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(args.audit, index=False, encoding="utf-8-sig")

    coverage = make_coverage(final_df)
    coverage.to_csv(args.coverage, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(all_sources):,}")
    print(f"Final rows: {len(final_df):,}")
    print(f"Overlap values replaced: {len(audit_df):,}")
    print(f"Saved final: {args.output}")
    print(f"Saved audit: {args.audit}")
    print(f"Saved coverage: {args.coverage}")


if __name__ == "__main__":
    main()