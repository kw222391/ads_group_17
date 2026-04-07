import os
import pandas as pd
from functools import reduce

ROOT_DIR = r"D:\UOB\ads_group_17\ads_group_17\data"
OUTPUT_FILE = os.path.join(ROOT_DIR, "merged_house.csv")

target_files = [
    "Average-prices-2025-12_house_lad.csv",
    "Indices-2025-12_house_lad.csv",
    "Sales-2025-12_house_lad.csv"
]

csv_files = [f for f in target_files if os.path.exists(os.path.join(ROOT_DIR, f))]

print(f"Found {len(csv_files)} house CSV files:")
for f in csv_files:
    print(" -", f)


def standardise_df(df, filename):
    print("\nProcessing:", filename)
    print("Original columns:", df.columns.tolist())

    df.columns = [c.lower().strip() for c in df.columns]

    date_col = None
    for col in df.columns:
        if "month" in col or "date" in col or "time" in col:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No date column in {filename}")

    df = df.rename(columns={date_col: "month"})
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    lad_col = None
    for col in df.columns:
        if ("lad" in col and "cd" in col) or ("area" in col and "code" in col):
            lad_col = col
            break
    if lad_col is None:
        raise ValueError(f"No LAD column in {filename}")

    df = df.rename(columns={lad_col: "lad_code"})
    df["lad_code"] = df["lad_code"].astype(str).str.strip().str.upper()

    region_col = None
    for col in df.columns:
        if "region_name" in col:
            region_col = col
            break
    if region_col and region_col != "region_name":
        df = df.rename(columns={region_col: "region_name"})

    df = df.dropna(subset=["month", "lad_code"])

    print("Standardised columns:", df.columns.tolist())
    print("Shape:", df.shape)

    return df


dfs = []

for file in csv_files:
    path = os.path.join(ROOT_DIR, file)
    try:
        df = pd.read_csv(path)
        df = standardise_df(df, file)
        dfs.append(df)
    except Exception as e:
        print(f"Skipped {file}: {e}")

if not dfs:
    raise ValueError("No valid files loaded")

merged_df = reduce(
    lambda l, r: pd.merge(l, r, on=["lad_code", "month"], how="outer"),
    dfs
)

print("\nFinal merged shape:", merged_df.shape)
print("\nColumns before cleaning:")
print(merged_df.columns.tolist())

region_cols = [c for c in merged_df.columns if "region_name" in c]
print("\nRegion name columns:", region_cols)

if region_cols:
    merged_df["region_name"] = pd.Series([None] * len(merged_df), index=merged_df.index)
    for col in region_cols:
        merged_df["region_name"] = merged_df["region_name"].fillna(merged_df[col])

    drop_cols = [c for c in region_cols if c != "region_name"]
    merged_df = merged_df.drop(columns=drop_cols, errors="ignore")

preferred_cols = [
    "lad_code",
    "region_name",
    "month",
    "average_price",
    "monthly_change",
    "annual_change",
    "average_price_sa",
    "index",
    "sales_volume"
]

existing = [c for c in preferred_cols if c in merged_df.columns]
others = [c for c in merged_df.columns if c not in existing]

merged_df = merged_df[existing + others]

merged_df = merged_df.sort_values(["lad_code", "month"]).reset_index(drop=True)

dup_count = merged_df.duplicated(subset=["lad_code", "month"]).sum()
print(f"\nDuplicate rows: {dup_count}")

merged_df.to_csv(OUTPUT_FILE, index=False)

print("\nFinal columns:")
print(merged_df.columns.tolist())

print("\nSaved to:", OUTPUT_FILE)