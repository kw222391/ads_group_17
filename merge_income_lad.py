import os
import pandas as pd

ROOT_DIR = r"D:\UOB\ads_group_17\ads_group_17\data\income_uk"
OUTPUT_LONG = os.path.join(ROOT_DIR, "income_uk_combined_long.csv")
OUTPUT_WIDE = os.path.join(ROOT_DIR, "income_uk_combined_wide.csv")
TARGET_SHEET = "Table 3"

files = [f for f in os.listdir(ROOT_DIR) if f.lower().endswith(".xlsx")]

print(f"Found {len(files)} files:")
for f in files:
    print(" -", f)


def load_one_file(file_path, sheet_name=TARGET_SHEET):
    print("\nReading:", file_path)

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, engine="openpyxl")

    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    df = df.dropna(axis=1, how="all")

    df = df.rename(columns={
        "Region": "region",
        "LAD code": "lad_code",
        "Region name": "lad_name"
    })

    required_cols = ["region", "lad_code", "lad_name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    year_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.isdigit():
            year_cols.append(col_str)

    print("Detected year columns:", year_cols[:5], "...", year_cols[-5:])

    if not year_cols:
        raise ValueError("No year columns found.")

    df = df[["region", "lad_code", "lad_name"] + year_cols].copy()
    df = df.dropna(subset=["lad_code", "lad_name"])

    df = df[
        (~df["lad_code"].astype(str).str.strip().eq("")) &
        (~df["lad_name"].astype(str).str.strip().eq(""))
    ]

    long_df = df.melt(
        id_vars=["region", "lad_code", "lad_name"],
        value_vars=year_cols,
        var_name="year",
        value_name="income"
    )

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["income"] = pd.to_numeric(long_df["income"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "income"])
    long_df["year"] = long_df["year"].astype(int)

    print("Cleaned long shape:", long_df.shape)
    print(long_df.head())

    return df, long_df


wide_list = []
long_list = []

for f in files:
    path = os.path.join(ROOT_DIR, f)
    try:
        df_wide, df_long = load_one_file(path, TARGET_SHEET)
        wide_list.append(df_wide)
        long_list.append(df_long)
        print(f"Loaded: {f}")
    except Exception as e:
        print(f"Error in {f}: {e}")

if not long_list:
    raise ValueError("No files were loaded successfully.")

combined_wide = pd.concat(wide_list, ignore_index=True).drop_duplicates()
combined_long = pd.concat(long_list, ignore_index=True).drop_duplicates()

print("\nFINAL WIDE SHAPE:", combined_wide.shape)
print("FINAL LONG SHAPE:", combined_long.shape)

print("\nWide preview:")
print(combined_wide.head())

print("\nLong preview:")
print(combined_long.head())

combined_wide.to_csv(OUTPUT_WIDE, index=False)
combined_long.to_csv(OUTPUT_LONG, index=False)

print("\nSaved wide CSV to:", OUTPUT_WIDE)
print("Saved long CSV to:", OUTPUT_LONG)