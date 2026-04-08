from pathlib import Path

import pandas as pd


input_path = Path("data") / "FTSE 100 Historical Results Price Data.csv"
output_dir = Path("data_new")
output_path = output_dir / "ftse_100_monthly_clean.csv"

# Load the raw stock data from the data folder.
ftse = pd.read_csv(input_path)

# Convert the source date column to a standard datetime format.
ftse["date"] = pd.to_datetime(ftse["Date"], dayfirst=True)

# Remove thousands separators and convert prices to numeric values.
ftse["Price"] = (
    ftse["Price"].astype(str).str.replace(",", "", regex=False)
)
ftse["Price"] = pd.to_numeric(ftse["Price"], errors="coerce")

# Keep only the required columns and rename Price to ftse_100.
ftse_clean = ftse[["date", "Price"]].copy()
ftse_clean = ftse_clean.rename(columns={"Price": "ftse_100"})

# Sort by date and keep the last record for each month.
ftse_clean = ftse_clean.sort_values("date").reset_index(drop=True)
ftse_clean["year_month"] = ftse_clean["date"].dt.to_period("M")
ftse_clean = ftse_clean.groupby("year_month", as_index=False).last()

# Restore the monthly timestamp and keep only the final output columns.
ftse_clean["date"] = ftse_clean["year_month"].dt.to_timestamp()
ftse_clean = ftse_clean[["date", "ftse_100"]]

# Export the cleaned data to the data_new folder.
output_dir.mkdir(parents=True, exist_ok=True)
ftse_clean.to_csv(output_path, index=False)

print(ftse_clean.head())
print(ftse_clean.tail())
print(f"Saved: {output_path}")
