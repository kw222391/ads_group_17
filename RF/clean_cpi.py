import pandas as pd

filename = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/consumerpriceinflationdetailedreferencetables.xlsx"
df = pd.read_excel(filename, sheet_name="To use")

# Clean column names
df.columns = df.columns.str.strip()

# Rename first column
df = df.rename(columns={df.columns[0]: "year"})

# Drop average column if present
if "average" in df.columns:
    df = df.drop(columns=["average"])

# Melt wide -> long
df_long = df.melt(
    id_vars="year",
    var_name="month",
    value_name="cpi"
)

# Clean month names
df_long["month"] = df_long["month"].str.strip()

# Convert CPI to numeric; strings like ".." become NaN
df_long["cpi"] = pd.to_numeric(df_long["cpi"], errors="coerce")

month_to_quarter = {
    "Jan": "Q1", "Feb": "Q1", "Mar": "Q1",
    "Apr": "Q2", "May": "Q2", "Jun": "Q2",
    "Jul": "Q3", "Aug": "Q3", "Sep": "Q3",
    "Oct": "Q4", "Nov": "Q4", "Dec": "Q4"
}

df_long["quarter"] = df_long["month"].map(month_to_quarter)

# Drop bad rows
df_long = df_long.dropna(subset=["cpi", "quarter", "year"])

# Optional: make year integer
df_long["year"] = df_long["year"].astype(int)

# Aggregate quarterly CPI
df_q = (
    df_long
    .groupby(["year", "quarter"], as_index=False)["cpi"]
    .mean()
)

# Create merge key
df_q["date"] = df_q["year"].astype(str) + " " + df_q["quarter"]

# Final output
# Keep columns aligned with homelessness dataset
df_final = df_q[["year", "quarter", "cpi"]]

print(df_final.head(12))
print(df_final.head(12))

# Save for merging
df_final.to_csv("/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/cpi_quarterly.csv", index=False)