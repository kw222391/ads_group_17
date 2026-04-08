from pathlib import Path

import pandas as pd


input_path = Path("data") / "NBGBBIS_gbp_index.csv"
output_dir = Path("data_new")
output_path = output_dir / "gbp_index_clean.csv"

# Load the raw GBP index data from the data folder.
gbp = pd.read_csv(input_path)

# Convert the source date column to a standard datetime format.
gbp["date"] = pd.to_datetime(gbp["observation_date"])

# Convert the source index values to a numeric GBP index column.
gbp["gbp_index"] = pd.to_numeric(gbp["NBGBBIS"], errors="coerce")

# Keep only the required output columns.
gbp_clean = gbp[["date", "gbp_index"]].copy()

# Sort by date and keep the last record for each month.
gbp_clean = gbp_clean.sort_values("date").reset_index(drop=True)
gbp_clean["year_month"] = gbp_clean["date"].dt.to_period("M")
gbp_clean = gbp_clean.groupby("year_month", as_index=False).last()

# Restore the monthly timestamp and keep only the final output columns.
gbp_clean["date"] = gbp_clean["year_month"].dt.to_timestamp()
gbp_clean = gbp_clean[["date", "gbp_index"]]

# Export the cleaned data to the data_new folder.
output_dir.mkdir(parents=True, exist_ok=True)
gbp_clean.to_csv(output_path, index=False)

print(gbp_clean.head())
print(gbp_clean.tail())
print(f"Saved: {output_path}")
