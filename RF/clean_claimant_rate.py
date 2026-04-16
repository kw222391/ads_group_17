# import pandas as pd

# input_path = "data_raw/claimaint_count_rate.xlsx"
# output_path = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/claimant_rate_long.csv"

# df = pd.read_excel(input_path)

# df = df.rename(columns={
#     df.columns[0]: "local_authority",
#     df.columns[1]: "lad_code"
# })

# df = df[df["lad_code"] != "Column Total"].copy()

# df_long = df.melt(
#     id_vars=["lad_code"],
#     value_vars=df.columns[2:],
#     var_name="date",
#     value_name="claimant_rate"
# )

# df_long["date"] = pd.to_datetime(df_long["date"], format="%B %Y", errors="coerce")

# df_long = df_long[["lad_code", "date", "claimant_rate"]]

# df_long = df_long.sort_values(["lad_code", "date"]).reset_index(drop=True)

# df_long.to_csv(output_path, index=False)

# print(df_long.head())
# print(df_long.dtypes)
# print(df_long["date"].min(), df_long["date"].max())

import pandas as pd

input_path = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/claimant_rate_long.csv"
output_path = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/claimant_rate_quarterly.csv"

df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"], dayfirst=False)# Force claimant_rate to numeric
df["claimant_rate"] = pd.to_numeric(df["claimant_rate"], errors="coerce")

df = df.dropna(subset=["date", "claimant_rate"]).copy()

df["year"] = df["date"].dt.year
df["quarter"] = "Q" + df["date"].dt.quarter.astype(str)

claimant_rate_quarterly = (
    df.groupby(["year", "quarter", "lad_code"], as_index=False)["claimant_rate"]
      .mean()
)

claimant_rate_quarterly["claimant_rate"] = claimant_rate_quarterly["claimant_rate"].round(3)

claimant_rate_quarterly = claimant_rate_quarterly[
    ["year", "quarter", "lad_code", "claimant_rate"]
]

claimant_rate_quarterly.to_csv(output_path, index=False)

# print(claimant_rate_quarterly.head())
# print(claimant_rate_quarterly.dtypes)

df_check = df[
    (df["lad_code"] == "E06000001") &
    (df["year"] == 2018) &
    (df["quarter"] == "Q1")
]

print(df_check)
print("Mean:", df_check["claimant_rate"].mean())
print("Count:", len(df_check))