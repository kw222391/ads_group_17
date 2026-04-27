import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Palatino", "serif"],
    "font.size": 24,
    "legend.fontsize": 19
})


def latex_label(label):
    return label.replace("_", r" ")


TICK_LABEL_SIZE = 16
TITLE_PAD = 24
COLORBAR_LABEL_PAD = 18


base_dir = Path(__file__).resolve().parent
file_path = base_dir / "data" / "clean" / "Final all Junxi.gz"

df = pd.read_csv(file_path)

df = df[df["year"] >= 2018].copy()

target = "homelessness_total_assessments"

features = [
    "average_house_price",
    "average_house_price_monthly_change",
    "average_house_price_annual_change",
    "seasonally_adjusted_average_house_price",
    "house_price_index",
    "house_sales_volume",
    "unemployment_count",
    "private_rental_price_index",
    "private_rental_price_monthly_change",
    "private_rental_price_annual_change",
    "average_private_rental_price",
    "gbp_index",
    "ftse_100",
    "income",
    "uk_bank_rate",
    "brent_oil_price",
    "population",
    "internal_net_migration",
    "international_net_migration",
    "cpi_00_all_items"
]

required_cols = [target] + features
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"These columns are missing from the data: {missing_cols}")

data = df[required_cols].copy()

data = data.rename(columns={
    "cpi_00_all_items": "cpitotal"
})

data = data.apply(pd.to_numeric, errors="coerce")

data = data.dropna(axis=1, how="all")

constant_cols = [
    col for col in data.columns
    if data[col].nunique(dropna=True) <= 1
]

if constant_cols:
    print("Constant columns removed:")
    print(constant_cols)
    data = data.drop(columns=constant_cols)

corr_matrix = data.corr(method="pearson")

target_corr = corr_matrix[target].drop(target).sort_values(
    key=lambda x: x.abs(),
    ascending=False
)

print("\nCorrelation with target after 2018:")
print(target_corr)

print("\nFull correlation matrix:")
print(corr_matrix)

plt.figure(figsize=(16, 14))

plt.imshow(
    corr_matrix.values,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

cbar = plt.colorbar()
cbar.set_label("Pearson correlation", labelpad=COLORBAR_LABEL_PAD)

plt.xticks(
    ticks=range(len(corr_matrix.columns)),
    labels=[latex_label(col) for col in corr_matrix.columns],
    rotation=90,
    fontsize=TICK_LABEL_SIZE
)

plt.yticks(
    ticks=range(len(corr_matrix.index)),
    labels=[latex_label(index) for index in corr_matrix.index],
    fontsize=TICK_LABEL_SIZE
)

plt.title(
    "Correlation Matrix: Homelessness Target and Economic Features After 2018",
    pad=TITLE_PAD
)
plt.tight_layout()
plt.show()
