import pandas as pd

house_path = r"D:\UOB\ads_group_17\ads_group_17\data\merged_house.csv"
unemp_path = r"D:\UOB\ads_group_17\ads_group_17\data\unemployment_lad.csv"
output_path = r"D:\UOB\ads_group_17\ads_group_17\data\merged_unemployment_house.csv"

house = pd.read_csv(house_path)
unemp_wide = pd.read_csv(unemp_path, skiprows=8)

house.columns = [c.lower().strip() for c in house.columns]

house["month"] = pd.to_datetime(house["month"], errors="coerce")
house["region_name"] = (
    house["region_name"]
    .astype(str)
    .str.lower()
    .str.replace(",", "")
    .str.strip()
)

print("House shape:", house.shape)

unemp_wide.columns = [c.strip() for c in unemp_wide.columns]
unemp_wide = unemp_wide.rename(columns={"Date": "month"})

unemp_wide["month"] = pd.to_datetime(
    unemp_wide["month"],
    format="%B %Y",
    errors="coerce"
)

unemp_wide = unemp_wide[unemp_wide["month"].notna()].copy()

print("Filtered unemployment shape:", unemp_wide.shape)

unemp_long = unemp_wide.melt(
    id_vars="month",
    var_name="region_name",
    value_name="unemployment_count"
)

unemp_long["region_name"] = (
    unemp_long["region_name"]
    .astype(str)
    .str.lower()
    .str.replace(",", "")
    .str.strip()
)

unemp_long["unemployment_count"] = pd.to_numeric(
    unemp_long["unemployment_count"].replace("-", pd.NA),
    errors="coerce"
)

unemp_long = (
    unemp_long
    .groupby(["region_name", "month"], as_index=False)["unemployment_count"]
    .max()
)

print("Unemployment after dedup:", unemp_long.shape)

merged = pd.merge(
    house,
    unemp_long,
    on=["region_name", "month"],
    how="left"
)

print("Merged shape:", merged.shape)

print("\nMissing unemployment:")
print(merged["unemployment_count"].isna().sum())

merged.to_csv(output_path, index=False)

print("\nSaved to:", output_path)