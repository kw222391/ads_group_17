import pandas as pd

hpi_file = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/hpi_wide_lad.csv"
out_file = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/hpi_quarterly_long.csv"

df = pd.read_csv(hpi_file)

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")

df_long = df.melt(id_vars="date", var_name="lad_code", value_name="hpi")

df_long["year"] = df_long["date"].dt.year
df_long["quarter"] = "Q" + df_long["date"].dt.quarter.astype(str)

df_quarterly = (
    df_long
    .groupby(["year", "quarter", "lad_code"], as_index=False)["hpi"]
    .mean()
)



df_quarterly = df_quarterly[["year", "quarter", "lad_code", "hpi"]]
df_quarterly = df_quarterly.sort_values(["year", "quarter", "lad_code"]).reset_index(drop=True)

df_quarterly.to_csv(out_file, index=False)

print(df_quarterly.head())
print("Saved to:", out_file)