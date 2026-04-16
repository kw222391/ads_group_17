import pandas as pd

input_path = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/clean_claimant_data.csv"
output_path = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/claimant_quarterly.csv"

df = pd.read_csv(input_path)

# Parse date properly
df["date"] = pd.to_datetime(df["date"])

# Create year and quarter columns to match the homelessness file
df["year"] = df["date"].dt.year
df["quarter"] = "Q" + df["date"].dt.quarter.astype(str)

# Quarterly average claimant count per LAD
claimant_quarterly = (
    df.groupby(["year", "quarter", "lad_code"], as_index=False)["claimant_count"]
      .mean()
)

# Optional: round to nearest whole number
claimant_quarterly["claimant_count"] = claimant_quarterly["claimant_count"].round().astype(int)

# Reorder columns for easy merge
claimant_quarterly = claimant_quarterly[["year", "quarter", "lad_code", "claimant_count"]]

# Save
claimant_quarterly.to_csv(output_path, index=False)

print(claimant_quarterly.head())
print(claimant_quarterly["quarter"].value_counts().sort_index())