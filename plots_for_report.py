import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


FILE_PATH = r"D:\UOB\ads_group_17\ads_group_17\data_new\all_data_for_ana.csv"

BOUNDARY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
    "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)

plt.rcParams["figure.figsize"] = (11.5, 13)
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 21
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


df = pd.read_csv(FILE_PATH)

unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
df = df.drop(columns=unnamed_cols)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["month"] = pd.to_numeric(df["month"], errors="coerce")
df = df.dropna(subset=["year", "month"]).copy()

df["date"] = pd.to_datetime(
    dict(
        year=df["year"].astype(int),
        month=df["month"].astype(int),
        day=1
    )
)

df["homelessness_per_1000"] = pd.to_numeric(
    df["homelessness_per_1000"],
    errors="coerce"
)

df["lad_code"] = df["lad_code"].astype(str)
df["lad_name"] = df["lad_name"].astype(str)

df = df[df["lad_code"].str.startswith("E", na=False)].copy()

latest_date = df.loc[df["homelessness_per_1000"].notna(), "date"].max()

latest = df[df["date"] == latest_date].copy()
latest = latest.drop_duplicates(subset="lad_code")

print("Latest date:", latest_date)

lad_map = gpd.read_file(BOUNDARY_URL)

code_col = [c for c in lad_map.columns if "CD" in c][0]
lad_map[code_col] = lad_map[code_col].astype(str)

lad_map = lad_map[lad_map[code_col].str.startswith("E", na=False)].copy()

plot_gdf = lad_map.merge(
    latest[["lad_code", "homelessness_per_1000"]],
    left_on=code_col,
    right_on="lad_code",
    how="left"
)

print("Matched:", plot_gdf["homelessness_per_1000"].notna().sum())

vmin = plot_gdf["homelessness_per_1000"].quantile(0.02)
vmax = plot_gdf["homelessness_per_1000"].quantile(0.98)


# Map of homelessness per 1,000 population across England LADs
fig, ax = plt.subplots()

plot_gdf.plot(
    column="homelessness_per_1000",
    cmap="OrRd",
    linewidth=0.4,
    edgecolor="white",
    vmin=vmin,
    vmax=vmax,
    legend=True,
    legend_kwds={
        "label": "Homelessness per 1,000 population",
        "orientation": "vertical",
        "shrink": 0.75,
        "pad": 0.02
    },
    missing_kwds={
        "color": "#d9d9d9",
        "edgecolor": "white",
        "hatch": "///",
        "label": "No data"
    },
    ax=ax
)

cbar = ax.get_figure().axes[-1]

cbar.tick_params(labelsize=14)
cbar.set_ylabel(
    "Homelessness per 1,000 population",
    fontsize=20,
    labelpad=12
)

ax.set_title(
    f"Homelessness per 1,000 population across England LADs\n"
    f"Latest available date: {latest_date.strftime('%Y-%m')}",
    fontsize=22,
    pad=20
)

ax.set_axis_off()

for spine in ax.spines.values():
    spine.set_visible(False)

minx, miny, maxx, maxy = plot_gdf.total_bounds
pad_x = (maxx - minx) * 0.04
pad_y = (maxy - miny) * 0.04

ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

file_path = r"D:\UOB\ads_group_17\ads_group_17\data\homelessness_integrated_09_25_zhou.csv"
output_path = r"D:\UOB\ads_group_17\ads_group_17\data_new\homelessness_total_assessments_england_quarterly.png"

policy_year = 2018
target_col = "Total_assessments"

df = pd.read_csv(file_path)

df["LAD_code"] = df["LAD_code"].astype(str)
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Quarter can be either 1,2,3,4 or Q1,Q2,Q3,Q4
if pd.api.types.is_numeric_dtype(df["Quarter"]):
    df["quarter_num"] = pd.to_numeric(df["Quarter"], errors="coerce")
else:
    df["quarter_num"] = (
        df["Quarter"]
        .astype(str)
        .str.extract(r"([1-4])")[0]
        .astype(float)
    )

df = df.dropna(subset=["Year", "quarter_num"])
df["year"] = df["Year"].astype(int)
df["quarter_num"] = df["quarter_num"].astype(int)

df = df[
    (df["LAD_code"].str.startswith("E")) &
    (df["year"] >= 2009) &
    (df["year"] <= 2025)
].copy()

quarterly = (
    df.groupby(["year", "quarter_num"])[target_col]
    .sum(min_count=1)
    .reset_index(name="total_homeless_assessments")
)

full_quarters = pd.MultiIndex.from_product(
    [range(2009, 2026), [1, 2, 3, 4]],
    names=["year", "quarter_num"]
)

quarterly = (
    quarterly
    .set_index(["year", "quarter_num"])
    .reindex(full_quarters)
    .reset_index()
)

quarterly = quarterly.sort_values(["year", "quarter_num"]).reset_index(drop=True)
quarterly["x"] = range(len(quarterly))

before_2018 = quarterly[quarterly["year"] < policy_year]
after_2018 = quarterly[quarterly["year"] >= policy_year]

plt.figure(figsize=(16, 7))

plt.plot(
    before_2018["x"],
    before_2018["total_homeless_assessments"],
    color="blue",
    marker="o",
    linewidth=2.2,
    markersize=4.5,
    label="Before 2018"
)

plt.plot(
    after_2018["x"],
    after_2018["total_homeless_assessments"],
    color="red",
    marker="o",
    linewidth=2.2,
    markersize=4.5,
    label="2018 and After"
)

policy_x = quarterly[
    (quarterly["year"] == policy_year) &
    (quarterly["quarter_num"] == 1)
]["x"].iloc[0]

plt.axvline(
    x=policy_x,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Policy Change (2018)"
)

year_ticks = quarterly[quarterly["quarter_num"] == 1]["x"]
year_labels = quarterly[quarterly["quarter_num"] == 1]["year"]

plt.xticks(year_ticks, year_labels, rotation=45)

plt.title(
    "Quarterly Trend of Total Homelessness Assessments in England (2009-2025)",
    fontsize=18
)

plt.xlabel("Year", fontsize=14)
plt.ylabel("Total Homelessness Assessments", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()