import os
import pandas as pd

ROOT_DIR = r"D:\UOB\ads_group_17\ads_group_17\data"

CPI_FILE = os.path.join(ROOT_DIR, "cpi.xlsx")
RENT_FILE = os.path.join(ROOT_DIR, "private_rent_price_index_monthly.xlsx")
BRENT_FILE = os.path.join(ROOT_DIR, "brent_oil_price.xls")
UNEMPLOYMENT_FILE = os.path.join(ROOT_DIR, "unemplolyement.xlsx")

OUTPUT_ALL_CSV = os.path.join(ROOT_DIR, "merged_monthly_data_all.csv")
OUTPUT_VALID_CSV = os.path.join(ROOT_DIR, "merged_monthly_data_valid.csv")


def quarter_to_months(q):
    q = str(q).strip()
    year, quarter = q.split()
    year = int(year)
    quarter_num = int(quarter.replace("Q", ""))
    start_month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter_num]
    return [
        pd.Timestamp(year=year, month=start_month, day=1),
        pd.Timestamp(year=year, month=start_month + 1, day=1),
        pd.Timestamp(year=year, month=start_month + 2, day=1),
    ]


print("CPI_FILE:", CPI_FILE, os.path.exists(CPI_FILE))
print("RENT_FILE:", RENT_FILE, os.path.exists(RENT_FILE))
print("BRENT_FILE:", BRENT_FILE, os.path.exists(BRENT_FILE))
print("UNEMPLOYMENT_FILE:", UNEMPLOYMENT_FILE, os.path.exists(UNEMPLOYMENT_FILE))

# Read source sheets
cpi_raw = pd.read_excel(CPI_FILE, sheet_name="Table 37", header=None)
rent_raw = pd.read_excel(RENT_FILE, sheet_name="Table 1", header=None)
brent_raw = pd.read_excel(BRENT_FILE, sheet_name="Data 1", header=None, engine="xlrd")
unemployment_raw = pd.read_excel(UNEMPLOYMENT_FILE, sheet_name="2", header=None)

# CPIH
cpi_header_row = 6
cpi_headers = cpi_raw.iloc[cpi_header_row].tolist()
cpi_df = cpi_raw.iloc[cpi_header_row + 1:].copy()
cpi_df.columns = cpi_headers

cpih_df = cpi_df[["name", "CPIH ALL ITEMS"]].copy()
cpih_df.columns = ["month", "cpih_index"]
cpih_df["month"] = pd.to_datetime(cpih_df["month"], errors="coerce")
cpih_df["cpih_index"] = pd.to_numeric(cpih_df["cpih_index"], errors="coerce")
cpih_df = cpih_df.dropna(subset=["month", "cpih_index"]).copy()
cpih_df["month"] = cpih_df["month"].dt.to_period("M").dt.to_timestamp()
cpih_df = cpih_df.sort_values("month").drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)

# Rent index
rent_header_row = 2
rent_headers = rent_raw.iloc[rent_header_row].tolist()
rent_df = rent_raw.iloc[rent_header_row + 1:].copy()
rent_df.columns = rent_headers

rent_index_df = rent_df[["Time period", "Area name", "Index"]].copy()
rent_index_df = rent_index_df[rent_index_df["Area name"] == "United Kingdom"].copy()
rent_index_df.columns = ["month", "area_name", "rent_index"]
rent_index_df["month"] = pd.to_datetime(rent_index_df["month"], errors="coerce")
rent_index_df["rent_index"] = pd.to_numeric(rent_index_df["rent_index"], errors="coerce")
rent_index_df = rent_index_df.dropna(subset=["month", "rent_index"]).copy()
rent_index_df["month"] = rent_index_df["month"].dt.to_period("M").dt.to_timestamp()
rent_index_df = rent_index_df.sort_values("month").drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)
rent_index_df = rent_index_df[["month", "rent_index"]].copy()

# Brent monthly first available price
brent_header_row = 2
brent_headers = brent_raw.iloc[brent_header_row].tolist()
brent_df = brent_raw.iloc[brent_header_row + 1:].copy()
brent_df.columns = brent_headers

brent_monthly_df = brent_df[["Date", "Europe Brent Spot Price FOB (Dollars per Barrel)"]].copy()
brent_monthly_df.columns = ["date", "brent_price_daily"]
brent_monthly_df["date"] = pd.to_datetime(brent_monthly_df["date"], errors="coerce")
brent_monthly_df["brent_price_daily"] = pd.to_numeric(brent_monthly_df["brent_price_daily"], errors="coerce")
brent_monthly_df = brent_monthly_df.dropna(subset=["date", "brent_price_daily"]).copy()
brent_monthly_df["month"] = brent_monthly_df["date"].dt.to_period("M").dt.to_timestamp()
brent_monthly_df = brent_monthly_df.sort_values("date").groupby("month", as_index=False).first()
brent_monthly_df = brent_monthly_df[["month", "date", "brent_price_daily"]].copy()
brent_monthly_df = brent_monthly_df.rename(columns={"date": "brent_observation_date"})

# Unemployment quarterly to monthly
unemployment_header_row = 3
unemployment_headers = unemployment_raw.iloc[unemployment_header_row].tolist()
unemployment_df = unemployment_raw.iloc[unemployment_header_row + 1:].copy()
unemployment_df.columns = unemployment_headers

uk_col = "United Kingdom [note 1, 2]"
unemployment_df = unemployment_df[["Date", uk_col]].copy()
unemployment_df.columns = ["quarter", "unemployment_rate"]
unemployment_df["quarter"] = unemployment_df["quarter"].astype(str).str.strip()
unemployment_df["unemployment_rate"] = pd.to_numeric(unemployment_df["unemployment_rate"], errors="coerce")
unemployment_df = unemployment_df.dropna(subset=["quarter", "unemployment_rate"]).copy()
unemployment_df = unemployment_df[unemployment_df["quarter"].str.match(r"^\d{4}\sQ[1-4]$")].copy()

monthly_rows = []
for _, row in unemployment_df.iterrows():
    for month in quarter_to_months(row["quarter"]):
        monthly_rows.append(
            {
                "month": month,
                "unemployment_rate": row["unemployment_rate"],
                "source_quarter": row["quarter"],
            }
        )

unemployment_monthly_df = pd.DataFrame(monthly_rows)
unemployment_monthly_df = unemployment_monthly_df.sort_values("month").drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)

# Merge
merged_df = cpih_df.merge(rent_index_df, on="month", how="outer")
merged_df = merged_df.merge(brent_monthly_df, on="month", how="outer")
merged_df = merged_df.merge(unemployment_monthly_df, on="month", how="outer")
merged_df = merged_df.sort_values("month").reset_index(drop=True)

merged_valid_df = merged_df.dropna(
    subset=["cpih_index", "rent_index", "brent_price_daily", "unemployment_rate"]
).copy()

# Keep only core output columns
merged_df_core = merged_df[
    ["month", "cpih_index", "rent_index", "brent_price_daily", "unemployment_rate"]
].copy()

merged_valid_df = merged_valid_df[
    ["month", "cpih_index", "rent_index", "brent_price_daily", "unemployment_rate"]
].copy()

print("\nCPIH preview:")
print(cpih_df.head(10).to_string(index=False))
print(cpih_df.tail(10).to_string(index=False))
print(cpih_df.shape)

print("\nRent preview:")
print(rent_index_df.head(10).to_string(index=False))
print(rent_index_df.tail(10).to_string(index=False))
print(rent_index_df.shape)

print("\nBrent preview:")
print(brent_monthly_df.head(10).to_string(index=False))
print(brent_monthly_df.tail(10).to_string(index=False))
print(brent_monthly_df.shape)

print("\nUnemployment preview:")
print(unemployment_monthly_df.head(10).to_string(index=False))
print(unemployment_monthly_df.tail(10).to_string(index=False))
print(unemployment_monthly_df.shape)

print("\nMerged all preview:")
print(merged_df_core.head(20).to_string(index=False))
print(merged_df_core.tail(20).to_string(index=False))
print(merged_df_core.shape)

print("\nMerged valid preview:")
print(merged_valid_df.head(20).to_string(index=False))
print(merged_valid_df.tail(20).to_string(index=False))
print(merged_valid_df.shape)

merged_df_core.to_csv(OUTPUT_ALL_CSV, index=False, encoding="utf-8-sig")
merged_valid_df.to_csv(OUTPUT_VALID_CSV, index=False, encoding="utf-8-sig")

print("\nDone.")
print("All-month CSV saved to:", OUTPUT_ALL_CSV)
print("Valid-only CSV saved to:", OUTPUT_VALID_CSV)

print("\nSelected sources:")
print("CPI -> sheet: Table 37 | columns: name, CPIH ALL ITEMS")
print("Rent -> sheet: Table 1 | columns: Time period, Area name, Index | filter: Area name == United Kingdom")
print("Brent -> sheet: Data 1 | columns: Date, Europe Brent Spot Price FOB (Dollars per Barrel)")
print("Unemployment -> sheet: 2 | columns: Date, United Kingdom [note 1, 2]")

# with homeless
HOMELESS_FILE = os.path.join(ROOT_DIR, "Statutory_Homelessness_England_Time_Series_202509 (1).ods")

OUTPUT_ALL_HOMELESS_CSV = os.path.join(ROOT_DIR, "merged_monthly_data_all_with_homeless.csv")
OUTPUT_VALID_HOMELESS_CSV = os.path.join(ROOT_DIR, "merged_monthly_data_valid_with_homeless.csv")

print("\nHOMELESS_FILE:", HOMELESS_FILE, os.path.exists(HOMELESS_FILE))
print("Homelessness -> sheet: A1 | column: Homeless - Relief duty owed")

homeless_raw = pd.read_excel(HOMELESS_FILE, sheet_name="A1", header=None)

print("\nHomeless raw head:")
print(homeless_raw.head(20).to_string())
print()

homeless_df = homeless_raw.iloc[14:, [0, 1, 8]].copy()
homeless_df.columns = ["year", "quarter", "homeless_households"]

homeless_df["year"] = pd.to_numeric(homeless_df["year"], errors="coerce")
homeless_df["year"] = homeless_df["year"].ffill()
homeless_df["quarter"] = homeless_df["quarter"].astype(str).str.strip()

homeless_df["homeless_households"] = pd.to_numeric(
    homeless_df["homeless_households"], errors="coerce"
)

homeless_df = homeless_df.dropna(subset=["year", "homeless_households"]).copy()
homeless_df = homeless_df[homeless_df["quarter"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()

homeless_df["quarter_str"] = (
    homeless_df["year"].astype(int).astype(str) + " " + homeless_df["quarter"]
)

homeless_monthly_rows = []
for _, row in homeless_df.iterrows():
    for month in quarter_to_months(row["quarter_str"]):
        homeless_monthly_rows.append(
            {
                "month": month,
                "homeless_households": row["homeless_households"],
                "homeless_source_quarter": row["quarter_str"],
            }
        )

homeless_monthly_df = pd.DataFrame(homeless_monthly_rows)
homeless_monthly_df = homeless_monthly_df.sort_values("month").drop_duplicates(
    subset=["month"], keep="last"
).reset_index(drop=True)

print("Homeless monthly preview:")
print(homeless_monthly_df.head(12).to_string(index=False))
print(homeless_monthly_df.tail(12).to_string(index=False))
print(homeless_monthly_df.shape)
print(homeless_monthly_df["month"].min(), homeless_monthly_df["month"].max())

all_with_homeless_df = merged_df_core.copy()
all_with_homeless_df["month"] = pd.to_datetime(all_with_homeless_df["month"], errors="coerce")

all_with_homeless_df = all_with_homeless_df.merge(
    homeless_monthly_df[["month", "homeless_households"]],
    on="month",
    how="left"
).sort_values("month").reset_index(drop=True)

valid_with_homeless_df = all_with_homeless_df.dropna(
    subset=[
        "cpih_index",
        "rent_index",
        "brent_price_daily",
        "unemployment_rate",
        "homeless_households",
    ]
).copy()

print("\nAll-month with homelessness preview:")
print(all_with_homeless_df.head(20).to_string(index=False))
print(all_with_homeless_df.tail(20).to_string(index=False))
print(all_with_homeless_df.shape)

print("\nValid-only with homelessness preview:")
print(valid_with_homeless_df.head(20).to_string(index=False))
print(valid_with_homeless_df.tail(20).to_string(index=False))
print(valid_with_homeless_df.shape)

all_with_homeless_df.to_csv(OUTPUT_ALL_HOMELESS_CSV, index=False, encoding="utf-8-sig")
valid_with_homeless_df.to_csv(OUTPUT_VALID_HOMELESS_CSV, index=False, encoding="utf-8-sig")

print("\nDone.")
print("All-month CSV with homelessness saved to:", OUTPUT_ALL_HOMELESS_CSV)
print("Valid-only CSV with homelessness saved to:", OUTPUT_VALID_HOMELESS_CSV)

print("Homelessness selected column: A1 -> Homeless - Relief duty owed")
