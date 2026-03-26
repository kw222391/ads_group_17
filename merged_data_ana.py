import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FILE_PATH = r"D:\UOB\ads_group_17\ads_group_17\data\merged_monthly_data_valid_with_homeless.csv"

df = pd.read_csv(FILE_PATH)

# convert month column
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month").reset_index(drop=True)

print("Data shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

print("\nMonth check:")
print(df["month"].head(12))
print(df["month"].tail(12))
print("Unique months:", df["month"].nunique())
print("Total rows:", len(df))

numeric_cols = [
    "cpih_index",
    "rent_index",
    "brent_price_daily",
    "unemployment_rate",
    "homeless_households"
]

# plot each variable over time

for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["month"],
        df[col],
        "-o",
        linewidth=1.5,
        markersize=3
    )

    ax.set_title(f"{col} over time")
    ax.set_xlabel("Month")
    ax.set_ylabel(col)

    # make x-axis show real monthly timeline
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#standardized trends on one plot

z_df = df.copy()
for col in numeric_cols:
    z_df[col] = (z_df[col] - z_df[col].mean()) / z_df[col].std(ddof=0)

fig, ax = plt.subplots(figsize=(12, 6))

for col in numeric_cols:
    ax.plot(
        z_df["month"],
        z_df[col],
        linewidth=2,
        label=col
    )

ax.set_title("Standardized trends of all variables")
ax.set_xlabel("Month")
ax.set_ylabel("Z-score")
ax.legend()

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# scatter plots: predictors vs homeless
predictors = [
    "cpih_index",
    "rent_index",
    "brent_price_daily",
    "unemployment_rate"
]

for col in predictors:
    x = df[col].values
    y = df["homeless_households"].values

    corr = np.corrcoef(x, y)[0, 1]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.8)

    # linear fit line
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = coef[0] * x_line + coef[1]
    plt.plot(x_line, y_line, linewidth=2)

    plt.title(f"homeless_households vs {col}\nPearson r = {corr:.3f}")
    plt.xlabel(col)
    plt.ylabel("homeless_households")
    plt.tight_layout()
    plt.show()

# correlation matrix heatmap

corr_matrix = df[numeric_cols].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

plt.figure(figsize=(7, 6))
im = plt.imshow(corr_matrix, interpolation="nearest")
plt.colorbar(im)

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(corr_matrix.index)),
    corr_matrix.index
)

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

from scipy.stats import pearsonr

#correlation

target = "homeless_households"
predictors = [
    "cpih_index",
    "rent_index",
    "brent_price_daily",
    "unemployment_rate"
]

print("\n" + "=" * 50)
print("Correlation analysis with homeless_households")
print("=" * 50)

corr_results = []

for col in predictors:
    r, p = pearsonr(df[col], df[target])
    corr_results.append({
        "variable": col,
        "pearson_r": r,
        "p_value": p
    })

corr_df = pd.DataFrame(corr_results)
corr_df["abs_r"] = corr_df["pearson_r"].abs()
corr_df = corr_df.sort_values("abs_r", ascending=False).drop(columns="abs_r").reset_index(drop=True)

print(corr_df)

print("\nInterpretation:")
for _, row in corr_df.iterrows():
    var = row["variable"]
    r = row["pearson_r"]
    p = row["p_value"]

    if abs(r) >= 0.7:
        strength = "strong"
    elif abs(r) >= 0.4:
        strength = "moderate"
    elif abs(r) >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if r > 0 else "negative"
    significance = "statistically significant" if p < 0.05 else "not statistically significant"

    print(f"{var}: r = {r:.3f}, p = {p:.4f} -> {strength} {direction} correlation, {significance}.")