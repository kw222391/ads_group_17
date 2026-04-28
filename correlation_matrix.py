import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Palatino", "serif"],
    "font.size": 50,
    "legend.fontsize": 19
})


DISPLAY_LABELS = {
    "homelessness_total_assessments": "Homelessness",
    "average_house_price": "House price",
    "average_house_price_monthly_change": "House price MoM",
    "house_price_index": "HPI",
    "house_sales_volume": "Sales volume",
    "unemployment_count": "Unemployment",
    "private_rental_price_index": "Rent index",
    "private_rental_price_monthly_change": "Rent MoM",
    "average_private_rental_price": "Avg. rent",
    "gbp_index": "GBP index",
    "ftse_100": "FTSE 100",
    "income": "Income",
    "uk_bank_rate": "Bank rate",
    "brent_oil_price": "Brent oil",
    "population": "Population",
    "internal_net_migration": "Internal mig.",
    "international_net_migration": "Intl. mig.",
    "cpitotal": "CPI",
}


def latex_label(label):
    return DISPLAY_LABELS.get(label, label).replace("_", " ")


def plot_labels(labels):
    return [latex_label(label) for label in labels]



FIGURE_SIZE = (18, 16)
TICK_LABEL_SIZE = 28
TITLE_SIZE = 34
TITLE_PAD = 26
COLORBAR_TICK_SIZE = 24
COLORBAR_LABEL_SIZE = 30
COLORBAR_LABEL_PAD = 22


base_dir = Path(__file__).resolve().parent
file_path = base_dir / "data" / "clean" / "Final all Junxi.gz"

df = pd.read_csv(file_path)

df = df[df["year"] >= 2018].copy()

target = "homelessness_total_assessments"

features = [
    "average_house_price",
    "average_house_price_monthly_change",
    "house_price_index",
    "house_sales_volume",
    "unemployment_count",
    "private_rental_price_index",
    "private_rental_price_monthly_change",
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

plt.figure(figsize=FIGURE_SIZE)

plt.imshow(
    corr_matrix.values,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
cbar.set_label(
    "Pearson correlation",
    fontsize=COLORBAR_LABEL_SIZE,
    labelpad=COLORBAR_LABEL_PAD
)

plt.xticks(
    ticks=range(len(corr_matrix.columns)),
    labels=plot_labels(corr_matrix.columns),
    rotation=90,
    fontsize=TICK_LABEL_SIZE
)

plt.yticks(
    ticks=range(len(corr_matrix.index)),
    labels=plot_labels(corr_matrix.index),
    fontsize=TICK_LABEL_SIZE
)

plt.title(
    "Correlation Matrix: Homelessness Target and\nEconomic Features After 2018",
    fontsize=TITLE_SIZE,
    pad=TITLE_PAD
)
plt.tight_layout()

output_pdf = base_dir / "correlation_matrix.pdf"
plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
print(f"\nSaved PDF figure to: {output_pdf}")

plt.show()
