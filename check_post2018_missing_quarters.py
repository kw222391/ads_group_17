import pandas as pd

file = "data/clean/homeless_lad_2009_2025_final.csv"
df = pd.read_csv(file)

df["Year"] = df["Year"].astype(int)
df["Quarter"] = df["Quarter"].astype(str)

# 只看 2018 以后
post = df[df["Year"] >= 2018].copy()

# 每个季度有多少 LAD 行、有多少 Total_assessments 非空
summary = (
    post.groupby(["Year", "Quarter"])
    .agg(
        LAD_rows=("LAD_code", "nunique"),
        Total_assessments_nonnull=("Total_assessments", lambda x: x.notna().sum()),
        Homeless_relief_nonnull=("Homeless_relief", lambda x: x.notna().sum()),
        Homeless_per_1000_nonnull=("Homeless_per_1000", lambda x: x.notna().sum()),
        Total_assessments_sum=("Total_assessments", "sum"),
    )
    .reset_index()
)

summary["_q"] = summary["Quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
summary = summary.sort_values(["Year", "_q"]).drop(columns="_q")

print("\n=== 2018 onwards coverage ===")
print(summary.to_string(index=False))

# 生成 2018-2025 所有季度
all_quarters = pd.MultiIndex.from_product(
    [range(2018, 2026), ["Q1", "Q2", "Q3", "Q4"]],
    names=["Year", "Quarter"]
).to_frame(index=False)

merged = all_quarters.merge(summary, on=["Year", "Quarter"], how="left")

missing_quarters = merged[
    merged["LAD_rows"].isna() |
    (merged["Total_assessments_nonnull"].fillna(0) == 0)
]

print("\n=== Missing / no Total_assessments quarters after 2018 ===")
print(missing_quarters[["Year", "Quarter", "LAD_rows", "Total_assessments_nonnull"]].to_string(index=False))

# 保存检查结果
summary.to_csv("data/clean/post2018_homeless_coverage.csv", index=False)
missing_quarters.to_csv("data/clean/post2018_missing_quarters.csv", index=False)

print("\nSaved:")
print("data/clean/post2018_homeless_coverage.csv")
print("data/clean/post2018_missing_quarters.csv")