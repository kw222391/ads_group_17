import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


FILE_PATH = r"D:\UOB\ads_group_17\ads_group_17\data_new\all_data_for_ana.csv"

BOUNDARY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
    "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)

plt.rcParams["figure.figsize"] = (11, 13)
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["font.size"] = 14
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def format_time_axis(ax, every_years=5):
    ax.xaxis.set_major_locator(mdates.YearLocator(every_years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)


def first_valid_by_date(df, col):
    return df.groupby("date")[col].first()


def sum_by_date(df, col):
    return df.groupby("date")[col].sum(min_count=1)


def weighted_mean_by_date(df, value_col, weight_col="population"):
    temp = df[["date", value_col, weight_col]].dropna().copy()
    if temp.empty:
        return pd.Series(dtype=float)
    weighted_sum = (temp[value_col] * temp[weight_col]).groupby(temp["date"]).sum()
    weight_sum = temp.groupby("date")[weight_col].sum()
    return weighted_sum / weight_sum


def fallback_mean_by_date(df, value_col):
    return df.groupby("date")[value_col].mean()


def combine_weighted_or_mean(df, value_col, weight_col="population"):
    weighted = weighted_mean_by_date(df, value_col, weight_col)
    if weighted.empty or weighted.notna().sum() == 0:
        return fallback_mean_by_date(df, value_col)
    mean_series = fallback_mean_by_date(df, value_col)
    return weighted.reindex(mean_series.index).combine_first(mean_series)


def safe_zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def normalize_to_100(series, base_date=None):
    s = series.dropna().copy()
    if s.empty:
        return series * np.nan
    if base_date is not None and base_date in s.index:
        base = s.loc[base_date]
    else:
        base = s.iloc[0]
    if pd.isna(base) or base == 0:
        return series * np.nan
    return series / base * 100


def year_mean_by_lad(df, value_col):
    temp = df[["lad_code", "lad_name", "year", value_col]].dropna().copy()
    return temp.groupby(["lad_code", "lad_name", "year"], as_index=False)[value_col].mean()


def plot_heatmap(ax, data, title, xlabels, ylabels, cmap="YlGnBu", annotate=False, fmt=".2f"):
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if pd.notna(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    print("Loading CSV")
    print("Original shape:", df.shape)

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)
    print("Shape after dropping unnamed columns:", df.shape)

    for c in ["year", "month"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1),
        errors="coerce"
    )
    df = df.dropna(subset=["date"]).copy()

    id_cols = {"lad_code", "lad_name", "date"}
    for col in df.columns:
        if col not in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["lad_code"] = df["lad_code"].astype(str)
    df["lad_name"] = df["lad_name"].astype(str)

    england_df = df[df["lad_code"].str.startswith("E", na=False)].copy()

    print("Date range:", england_df["date"].min(), "to", england_df["date"].max())
    print("Number of England LADs:", england_df["lad_code"].nunique())
    print("Columns:")
    print(england_df.columns.tolist())

    return england_df


def build_england_monthly_series(df):
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    england = pd.DataFrame(index=all_dates)
    england.index.name = "date"

    macro_cols = [
        "cpi_00_all_items", "uk_bank_rate", "brent_oil_price",
        "gbp_index", "ftse_100"
    ]

    weighted_cols = [
        "average_house_price",
        "seasonally_adjusted_average_house_price",
        "house_price_index",
        "private_rental_price_index",
        "average_private_rental_price",
        "income",
    ]

    sum_cols = [
        "house_sales_volume",
        "unemployment_count",
        "homelessness_total_owed",
        "homelessness_threatened",
        "homelessness_total_assessments",
        "homelessness_relief",
        "population",
        "internal_net_migration",
        "international_net_migration",
    ]

    for col in macro_cols:
        if col in df.columns:
            england[col] = first_valid_by_date(df, col).reindex(all_dates)

    for col in weighted_cols:
        if col in df.columns:
            england[col] = combine_weighted_or_mean(df, col).reindex(all_dates)

    for col in sum_cols:
        if col in df.columns:
            england[col] = sum_by_date(df, col).reindex(all_dates)

    england["house_price_yoy"] = england["average_house_price"].pct_change(12) * 100
    england["rent_price_yoy"] = england["average_private_rental_price"].pct_change(12) * 100
    england["income_yoy"] = england["income"].pct_change(12) * 100
    england["cpi_yoy"] = england["cpi_00_all_items"].pct_change(12) * 100
    england["bank_rate_change_12m"] = england["uk_bank_rate"].diff(12)
    england["oil_yoy"] = england["brent_oil_price"].pct_change(12) * 100
    england["ftse_yoy"] = england["ftse_100"].pct_change(12) * 100
    england["gbp_yoy"] = england["gbp_index"].pct_change(12) * 100

    england["real_income_index"] = england["income"] / england["cpi_00_all_items"] * 100
    england["real_house_price_index"] = england["average_house_price"] / england["cpi_00_all_items"] * 100
    england["real_rent_index"] = england["average_private_rental_price"] / england["cpi_00_all_items"] * 100

    england["rent_to_income_ratio"] = (england["average_private_rental_price"] * 12) / england["income"]
    england["house_to_income_ratio"] = england["average_house_price"] / england["income"]
    england["unemployment_per_1000"] = england["unemployment_count"] / england["population"] * 1000
    england["homelessness_assessments_per_1000"] = (
        england["homelessness_total_assessments"] / england["population"] * 1000
    )

    england["living_cost_pressure_index"] = (
        safe_zscore(england["cpi_yoy"]) +
        safe_zscore(england["rent_price_yoy"]) +
        safe_zscore(england["house_price_yoy"]) -
        safe_zscore(england["income_yoy"])
    )

    return england


def print_quick_facts(england):
    print("\nQuick facts")

    if england["cpi_00_all_items"].dropna().shape[0] > 1:
        cpi_start = england["cpi_00_all_items"].dropna().iloc[0]
        cpi_end = england["cpi_00_all_items"].dropna().iloc[-1]
        print(f"CPI all-items index: {cpi_start:.2f} -> {cpi_end:.2f} ({(cpi_end / cpi_start - 1) * 100:.1f}%)")

    if england["average_house_price"].dropna().shape[0] > 1:
        hp_start = england["average_house_price"].dropna().iloc[0]
        hp_end = england["average_house_price"].dropna().iloc[-1]
        print(f"Average house price: {hp_start:,.0f} -> {hp_end:,.0f} ({(hp_end / hp_start - 1) * 100:.1f}%)")

    if england["average_private_rental_price"].dropna().shape[0] > 1:
        rent_start = england["average_private_rental_price"].dropna().iloc[0]
        rent_end = england["average_private_rental_price"].dropna().iloc[-1]
        print(f"Average private rent: {rent_start:,.0f} -> {rent_end:,.0f} ({(rent_end / rent_start - 1) * 100:.1f}%)")

    if england["income"].dropna().shape[0] > 1:
        inc_start = england["income"].dropna().iloc[0]
        inc_end = england["income"].dropna().iloc[-1]
        print(f"Income: {inc_start:,.0f} -> {inc_end:,.0f} ({(inc_end / inc_start - 1) * 100:.1f}%)")

    for col, label in [
        ("cpi_yoy", "Peak CPI YoY"),
        ("house_price_yoy", "Peak house price YoY"),
        ("rent_price_yoy", "Peak rent YoY"),
    ]:
        s = england[col].dropna()
        if not s.empty:
            peak_date = s.idxmax()
            peak_val = s.max()
            print(f"{label}: {peak_val:.2f}% at {peak_date.strftime('%Y-%m')}")

    if england["rent_to_income_ratio"].dropna().shape[0] > 1:
        rti_start = england["rent_to_income_ratio"].dropna().iloc[0]
        rti_end = england["rent_to_income_ratio"].dropna().iloc[-1]
        print(f"Rent-to-income ratio: {rti_start:.3f} -> {rti_end:.3f}")

    if england["house_to_income_ratio"].dropna().shape[0] > 1:
        hti_start = england["house_to_income_ratio"].dropna().iloc[0]
        hti_end = england["house_to_income_ratio"].dropna().iloc[-1]
        print(f"House-price-to-income ratio: {hti_start:.3f} -> {hti_end:.3f}")


def plot_analysis_figures(df, england):
    england_2015 = england.loc[england.index >= "2015-01-01"].copy()
    england_2020 = england.loc[england.index >= "2020-01-01"].copy()

    latest_common_date = df.dropna(
        subset=["average_house_price", "average_private_rental_price", "income"]
    )["date"].max()
    print("Latest common date for house/rent/income:", latest_common_date)

    coverage_cols = [
        "cpi_00_all_items", "average_house_price", "average_private_rental_price",
        "income", "uk_bank_rate", "brent_oil_price", "unemployment_count",
        "population", "homelessness_total_assessments"
    ]
    coverage_labels = [c for c in coverage_cols if c in england.columns]
    coverage_data = england[coverage_labels].notna().astype(int).T.values

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1.2]})
    plot_heatmap(
        axes[0],
        coverage_data,
        "FIGURE 1A. DATA AVAILABILITY OVER TIME (1 = AVAILABLE)",
        [d.strftime("%Y") if d.month == 1 else "" for d in england.index],
        coverage_labels,
        cmap="YlGn"
    )
    non_null_counts = england[coverage_labels].notna().sum()
    axes[1].bar(non_null_counts.index, non_null_counts.values)
    axes[1].set_title("FIGURE 1B. NUMBER OF AVAILABLE MONTHS BY VARIABLE", fontweight="bold")
    axes[1].set_ylabel("Months available")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(england.index, england["cpi_00_all_items"], linewidth=2)
    axes[0, 0].set_title("FIGURE 2A. CPI ALL-ITEMS INDEX", fontweight="bold")
    axes[0, 0].set_ylabel("Index")
    format_time_axis(axes[0, 0], 5)

    axes[0, 1].plot(england.index, england["average_house_price"], linewidth=2)
    axes[0, 1].set_title("FIGURE 2B. ENGLAND AVERAGE HOUSE PRICE", fontweight="bold")
    axes[0, 1].set_ylabel("Price")
    format_time_axis(axes[0, 1], 5)

    axes[1, 0].plot(england.index, england["average_private_rental_price"], linewidth=2)
    axes[1, 0].set_title("FIGURE 2C. ENGLAND AVERAGE PRIVATE RENT", fontweight="bold")
    axes[1, 0].set_ylabel("Price")
    format_time_axis(axes[1, 0], 5)

    axes[1, 1].plot(england.index, england["income"], linewidth=2)
    axes[1, 1].set_title("FIGURE 2D. INCOME (MONTHLY PANEL VALUE)", fontweight="bold")
    axes[1, 1].set_ylabel("Income")
    format_time_axis(axes[1, 1], 5)

    for ax in axes.flatten():
        ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-01"), alpha=0.15)

    plt.tight_layout()
    plt.show()

    base_date = pd.Timestamp("2015-01-01")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(england_2015.index, normalize_to_100(england_2015["cpi_00_all_items"], base_date), label="CPI")
    ax.plot(england_2015.index, normalize_to_100(england_2015["average_house_price"], base_date), label="House price")
    ax.plot(england_2015.index, normalize_to_100(england_2015["average_private_rental_price"], base_date), label="Private rent")
    ax.plot(england_2015.index, normalize_to_100(england_2015["income"], base_date), label="Income")
    ax.set_title("FIGURE 3. NORMALIZED INDICES (2015 = 100)", fontweight="bold")
    ax.set_ylabel("Index (2015 = 100)")
    format_time_axis(ax, 1)
    ax.legend()
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-01"), alpha=0.15)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    series_map = {
        "CPI YoY (%)": "cpi_yoy",
        "House price YoY (%)": "house_price_yoy",
        "Private rent YoY (%)": "rent_price_yoy",
        "Income YoY (%)": "income_yoy",
    }
    for ax, (title, col) in zip(axes.flatten(), series_map.items()):
        ax.plot(england.index, england[col], linewidth=2)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-01"), alpha=0.15)
        ax.set_title(f"FIGURE 4. {title}", fontweight="bold")
        ax.set_ylabel("Percent")
        format_time_axis(ax, 5)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(england.index, normalize_to_100(england["real_income_index"]), linewidth=2)
    axes[0, 0].set_title("FIGURE 5A. REAL INCOME INDEX", fontweight="bold")
    axes[0, 0].set_ylabel("Index")
    format_time_axis(axes[0, 0], 5)

    axes[0, 1].plot(england.index, normalize_to_100(england["real_house_price_index"]), linewidth=2)
    axes[0, 1].set_title("FIGURE 5B. REAL HOUSE PRICE INDEX", fontweight="bold")
    axes[0, 1].set_ylabel("Index")
    format_time_axis(axes[0, 1], 5)

    axes[1, 0].plot(england.index, england["rent_to_income_ratio"], linewidth=2)
    axes[1, 0].set_title("FIGURE 5C. RENT-TO-INCOME RATIO", fontweight="bold")
    axes[1, 0].set_ylabel("Ratio")
    format_time_axis(axes[1, 0], 5)

    axes[1, 1].plot(england.index, england["house_to_income_ratio"], linewidth=2)
    axes[1, 1].set_title("FIGURE 5D. HOUSE-PRICE-TO-INCOME RATIO", fontweight="bold")
    axes[1, 1].set_ylabel("Ratio")
    format_time_axis(axes[1, 1], 5)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(england_2020.index, england_2020["cpi_yoy"], label="CPI YoY", linewidth=2)
    ax.plot(england_2020.index, england_2020["rent_price_yoy"], label="Rent YoY", linewidth=2)
    ax.plot(england_2020.index, england_2020["house_price_yoy"], label="House price YoY", linewidth=2)
    ax.plot(england_2020.index, england_2020["income_yoy"], label="Income YoY", linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("FIGURE 6. COST-OF-LIVING SHOCK AFTER 2020", fontweight="bold")
    ax.set_ylabel("YoY growth (%)")
    format_time_axis(ax, 1)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(england.index, england["uk_bank_rate"], linewidth=2)
    axes[0, 0].set_title("FIGURE 7A. UK BANK RATE", fontweight="bold")
    axes[0, 0].set_ylabel("Rate (%)")
    format_time_axis(axes[0, 0], 5)

    axes[0, 1].plot(england.index, england["brent_oil_price"], linewidth=2)
    axes[0, 1].set_title("FIGURE 7B. BRENT OIL PRICE", fontweight="bold")
    axes[0, 1].set_ylabel("Price")
    format_time_axis(axes[0, 1], 5)

    axes[1, 0].plot(england.index, england["ftse_100"], linewidth=2)
    axes[1, 0].set_title("FIGURE 7C. FTSE 100", fontweight="bold")
    axes[1, 0].set_ylabel("Index")
    format_time_axis(axes[1, 0], 5)

    axes[1, 1].plot(england.index, england["gbp_index"], linewidth=2)
    axes[1, 1].set_title("FIGURE 7D. GBP INDEX", fontweight="bold")
    axes[1, 1].set_ylabel("Index")
    format_time_axis(axes[1, 1], 5)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(england.index, england["unemployment_count"], linewidth=2)
    axes[0, 0].set_title("FIGURE 8A. TOTAL UNEMPLOYMENT COUNT", fontweight="bold")
    axes[0, 0].set_ylabel("Count")
    format_time_axis(axes[0, 0], 5)

    axes[0, 1].plot(england.index, england["unemployment_per_1000"], linewidth=2)
    axes[0, 1].set_title("FIGURE 8B. UNEMPLOYMENT PER 1000", fontweight="bold")
    axes[0, 1].set_ylabel("Per 1000")
    format_time_axis(axes[0, 1], 5)

    axes[1, 0].plot(england.index, england["homelessness_total_assessments"], linewidth=2)
    axes[1, 0].set_title("FIGURE 8C. HOMELESSNESS TOTAL ASSESSMENTS", fontweight="bold")
    axes[1, 0].set_ylabel("Count")
    format_time_axis(axes[1, 0], 5)

    axes[1, 1].plot(england.index, england["homelessness_assessments_per_1000"], linewidth=2)
    axes[1, 1].set_title("FIGURE 8D. HOMELESSNESS ASSESSMENTS PER 1000", fontweight="bold")
    axes[1, 1].set_ylabel("Per 1000")
    format_time_axis(axes[1, 1], 5)
    plt.tight_layout()
    plt.show()

    correlation_df = england[[
        "cpi_yoy", "house_price_yoy", "rent_price_yoy", "income_yoy",
        "uk_bank_rate", "oil_yoy", "ftse_yoy", "gbp_yoy",
        "unemployment_per_1000", "homelessness_assessments_per_1000"
    ]].dropna()

    corr = correlation_df.corr()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_heatmap(
        axes[0],
        corr.values,
        "FIGURE 9A. CORRELATION HEATMAP OF ENGLAND MONTHLY INDICATORS",
        corr.columns.tolist(),
        corr.index.tolist(),
        cmap="coolwarm",
        annotate=True,
        fmt=".2f"
    )

    rolling_corr_house = england["cpi_yoy"].rolling(24).corr(england["house_price_yoy"])
    rolling_corr_rent = england["cpi_yoy"].rolling(24).corr(england["rent_price_yoy"])
    rolling_corr_income = england["cpi_yoy"].rolling(24).corr(england["income_yoy"])
    axes[1].plot(england.index, rolling_corr_house, label="CPI vs house price")
    axes[1].plot(england.index, rolling_corr_rent, label="CPI vs rent")
    axes[1].plot(england.index, rolling_corr_income, label="CPI vs income")
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_title("FIGURE 9B. 24-MONTH ROLLING CORRELATION WITH CPI", fontweight="bold")
    axes[1].set_ylabel("Correlation")
    format_time_axis(axes[1], 5)
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    lags = range(-24, 25)
    lag_corrs = []
    for lag in lags:
        oil_shifted = england["oil_yoy"].shift(lag)
        pair = pd.concat([england["cpi_yoy"], oil_shifted], axis=1).dropna()
        if pair.shape[0] > 10:
            lag_corrs.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        else:
            lag_corrs.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].bar(list(lags), lag_corrs)
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_title("FIGURE 10A. OIL YoY AND CPI YoY LEAD-LAG CORRELATION", fontweight="bold")
    axes[0].set_xlabel("Lag in months (+ means oil leads)")
    axes[0].set_ylabel("Correlation")
    axes[0].grid(True, axis="y", alpha=0.3)

    season_df = df[["month", "average_house_price_monthly_change", "private_rental_price_monthly_change"]].copy()
    box_data_house = [
        season_df.loc[season_df["month"] == m, "average_house_price_monthly_change"].dropna().values
        for m in range(1, 13)
    ]
    box_data_rent = [
        season_df.loc[season_df["month"] == m, "private_rental_price_monthly_change"].dropna().values
        for m in range(1, 13)
    ]
    positions_house = np.arange(1, 13) - 0.18
    positions_rent = np.arange(1, 13) + 0.18
    axes[1].boxplot(box_data_house, positions=positions_house, widths=0.3, patch_artist=False)
    axes[1].boxplot(box_data_rent, positions=positions_rent, widths=0.3, patch_artist=False)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels([str(i) for i in range(1, 13)])
    axes[1].set_title("FIGURE 10B. MONTHLY SEASONALITY OF HOUSE AND RENT CHANGES", fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Monthly change (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(["House monthly change", "Rent monthly change"], loc="upper right")
    plt.tight_layout()
    plt.show()

    house_yearly = year_mean_by_lad(df, "average_house_price")
    rent_yearly = year_mean_by_lad(df, "average_private_rental_price")

    house_years_to_show = [2000, 2005, 2010, 2015, 2020, 2025]
    rent_years_to_show = [2015, 2017, 2019, 2021, 2023, 2025]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    house_box = [house_yearly.loc[house_yearly["year"] == y, "average_house_price"].dropna().values for y in house_years_to_show]
    rent_box = [rent_yearly.loc[rent_yearly["year"] == y, "average_private_rental_price"].dropna().values for y in rent_years_to_show]

    axes[0].boxplot(house_box, labels=[str(y) for y in house_years_to_show], showfliers=False)
    axes[0].set_title("FIGURE 11A. LAD DISTRIBUTION OF HOUSE PRICES", fontweight="bold")
    axes[0].set_ylabel("Average house price")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot(rent_box, labels=[str(y) for y in rent_years_to_show], showfliers=False)
    axes[1].set_title("FIGURE 11B. LAD DISTRIBUTION OF PRIVATE RENTS", fontweight="bold")
    axes[1].set_ylabel("Average private rental price")
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    latest_lad = df[df["date"] == latest_common_date].copy()
    latest_lad = latest_lad.dropna(subset=["average_house_price", "average_private_rental_price", "income"]).copy()
    latest_lad["rent_to_income_ratio"] = latest_lad["average_private_rental_price"] * 12 / latest_lad["income"]
    latest_lad["house_to_income_ratio"] = latest_lad["average_house_price"] / latest_lad["income"]

    if "population" in latest_lad.columns:
        latest_lad["population_for_size"] = latest_lad["population"].fillna(latest_lad["population"].median())
    else:
        latest_lad["population_for_size"] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    size_scale = np.sqrt(latest_lad["population_for_size"].clip(lower=1)) / 10

    axes[0].scatter(latest_lad["income"], latest_lad["average_private_rental_price"], s=size_scale, alpha=0.7)
    axes[0].set_title(f"FIGURE 12A. RENT VS INCOME BY LAD ({latest_common_date.strftime('%Y-%m')})", fontweight="bold")
    axes[0].set_xlabel("Income")
    axes[0].set_ylabel("Average private rent")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(latest_lad["income"], latest_lad["average_house_price"], s=size_scale, alpha=0.7)
    axes[1].set_title(f"FIGURE 12B. HOUSE PRICE VS INCOME BY LAD ({latest_common_date.strftime('%Y-%m')})", fontweight="bold")
    axes[1].set_xlabel("Income")
    axes[1].set_ylabel("Average house price")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    latest_lad["burden_score"] = (
        safe_zscore(latest_lad["rent_to_income_ratio"]) +
        safe_zscore(latest_lad["house_to_income_ratio"])
    )

    top_rent_burden = latest_lad.nlargest(20, "rent_to_income_ratio").sort_values("rent_to_income_ratio")
    top_house_burden = latest_lad.nlargest(20, "house_to_income_ratio").sort_values("house_to_income_ratio")
    afford_worst = latest_lad.nlargest(20, "burden_score").sort_values("burden_score")

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    axes[0].barh(top_rent_burden["lad_name"], top_rent_burden["rent_to_income_ratio"])
    axes[0].set_title("FIGURE 13A. TOP 20 RENT-TO-INCOME RATIOS", fontweight="bold")
    axes[0].set_xlabel("Ratio")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(top_house_burden["lad_name"], top_house_burden["house_to_income_ratio"])
    axes[1].set_title("FIGURE 13B. TOP 20 HOUSE-PRICE-TO-INCOME RATIOS", fontweight="bold")
    axes[1].set_xlabel("Ratio")
    axes[1].grid(True, axis="x", alpha=0.3)

    axes[2].barh(afford_worst["lad_name"], afford_worst["burden_score"])
    axes[2].set_title("FIGURE 13C. TOP 20 COMPOSITE COST PRESSURE LADs", fontweight="bold")
    axes[2].set_xlabel("Composite z-score")
    axes[2].grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    house_2015 = house_yearly.loc[house_yearly["year"] == 2015, ["lad_code", "lad_name", "average_house_price"]].rename(
        columns={"average_house_price": "house_2015"}
    )
    house_2025 = house_yearly.loc[house_yearly["year"] == 2025, ["lad_code", "lad_name", "average_house_price"]].rename(
        columns={"average_house_price": "house_2025"}
    )
    rent_2015 = rent_yearly.loc[rent_yearly["year"] == 2015, ["lad_code", "lad_name", "average_private_rental_price"]].rename(
        columns={"average_private_rental_price": "rent_2015"}
    )
    rent_2025 = rent_yearly.loc[rent_yearly["year"] == 2025, ["lad_code", "lad_name", "average_private_rental_price"]].rename(
        columns={"average_private_rental_price": "rent_2025"}
    )

    growth_house = house_2015.merge(house_2025, on=["lad_code", "lad_name"], how="inner")
    growth_house["house_growth_pct_2015_2025"] = (growth_house["house_2025"] / growth_house["house_2015"] - 1) * 100
    growth_house = growth_house.replace([np.inf, -np.inf], np.nan).dropna(subset=["house_growth_pct_2015_2025"])

    growth_rent = rent_2015.merge(rent_2025, on=["lad_code", "lad_name"], how="inner")
    growth_rent["rent_growth_pct_2015_2025"] = (growth_rent["rent_2025"] / growth_rent["rent_2015"] - 1) * 100
    growth_rent = growth_rent.replace([np.inf, -np.inf], np.nan).dropna(subset=["rent_growth_pct_2015_2025"])

    top_house_growth = growth_house.nlargest(20, "house_growth_pct_2015_2025").sort_values("house_growth_pct_2015_2025")
    top_rent_growth = growth_rent.nlargest(20, "rent_growth_pct_2015_2025").sort_values("rent_growth_pct_2015_2025")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].barh(top_house_growth["lad_name"], top_house_growth["house_growth_pct_2015_2025"])
    axes[0].set_title("FIGURE 14A. TOP 20 HOUSE PRICE GROWTH (2015-2025)", fontweight="bold")
    axes[0].set_xlabel("Growth (%)")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(top_rent_growth["lad_name"], top_rent_growth["rent_growth_pct_2015_2025"])
    axes[1].set_title("FIGURE 14B. TOP 20 RENT GROWTH (2015-2025)", fontweight="bold")
    axes[1].set_xlabel("Growth (%)")
    axes[1].grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    yearly_summary = england[[
        "cpi_yoy", "house_price_yoy", "rent_price_yoy", "income_yoy",
        "rent_to_income_ratio", "house_to_income_ratio", "living_cost_pressure_index"
    ]].copy()
    yearly_summary["year"] = yearly_summary.index.year
    yearly_mean = yearly_summary.groupby("year").mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_heatmap(
        ax,
        yearly_mean.T.values,
        "FIGURE 15. YEARLY MEAN HEATMAP OF COST-OF-LIVING INDICATORS",
        yearly_mean.index.astype(str).tolist(),
        yearly_mean.columns.tolist(),
        cmap="coolwarm",
        annotate=False
    )
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(england.index, england["living_cost_pressure_index"], linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-01"), alpha=0.15)
    ax.set_title("FIGURE 16. COMPOSITE LIVING-COST PRESSURE INDEX", fontweight="bold")
    ax.set_ylabel("Standardized pressure score")
    format_time_axis(ax, 5)
    plt.tight_layout()
    plt.show()

    print("\nAll analysis plots finished.")


def load_lad_boundaries():
    lad_map = gpd.read_file(BOUNDARY_URL)
    print("Boundary shape:", lad_map.shape)
    print("Boundary columns:")
    print(lad_map.columns.tolist())

    possible_code_cols = ["LAD23CD", "lad23cd", "LAD22CD", "lad22cd", "LAD24CD", "lad24cd"]
    possible_name_cols = ["LAD23NM", "lad23nm", "LAD22NM", "lad22nm", "LAD24NM", "lad24nm"]

    map_code_col = None
    map_name_col = None

    for c in possible_code_cols:
        if c in lad_map.columns:
            map_code_col = c
            break

    for c in possible_name_cols:
        if c in lad_map.columns:
            map_name_col = c
            break

    if map_code_col is None:
        raise ValueError("Could not find LAD code column in boundary file.")

    if map_name_col is None:
        print("Warning: LAD name column not found in boundary file.")
        map_name_col = map_code_col

    lad_map[map_code_col] = lad_map[map_code_col].astype(str)

    print("Using boundary LAD code column:", map_code_col)
    print("Using boundary LAD name column:", map_name_col)

    return lad_map, map_code_col, map_name_col


def get_latest_homelessness_snapshot(df):
    latest_total_date = df.loc[df["homelessness_total_assessments"].notna(), "date"].max()
    latest_rate_date = df.loc[df["homelessness_per_1000"].notna(), "date"].max()
    latest_date = max(latest_total_date, latest_rate_date)

    latest = df[df["date"] == latest_date].copy()
    latest = latest.sort_values(["lad_code"]).drop_duplicates(subset=["lad_code"], keep="first")

    print("Latest date with homelessness_total_assessments:", latest_total_date)
    print("Latest date with homelessness_per_1000:", latest_rate_date)
    print("Chosen latest homelessness date:", latest_date)
    print("Rows at latest date:", latest.shape[0])

    if "homelessness_total_assessments" in latest.columns:
        print("Non-null total assessments:", latest["homelessness_total_assessments"].notna().sum())
    if "homelessness_per_1000" in latest.columns:
        print("Non-null homelessness_per_1000:", latest["homelessness_per_1000"].notna().sum())

    return latest_date, latest


def plot_uk_homelessness_maps(df):
    latest_date, latest = get_latest_homelessness_snapshot(df)
    lad_map, map_code_col, map_name_col = load_lad_boundaries()

    keep_cols = ["lad_code", "lad_name", "homelessness_total_assessments", "homelessness_per_1000", "population"]
    keep_cols = [c for c in keep_cols if c in latest.columns]

    plot_gdf = lad_map.merge(
        latest[keep_cols],
        left_on=map_code_col,
        right_on="lad_code",
        how="left"
    )

    print("Matched LADs with homelessness_total_assessments:",
          plot_gdf["homelessness_total_assessments"].notna().sum())
    print("Matched LADs with homelessness_per_1000:",
          plot_gdf["homelessness_per_1000"].notna().sum())

    missing_rate_mask = (
        plot_gdf["homelessness_per_1000"].isna() &
        plot_gdf["homelessness_total_assessments"].notna() &
        plot_gdf["population"].notna() &
        (plot_gdf["population"] > 0)
    )

    plot_gdf.loc[missing_rate_mask, "homelessness_per_1000"] = (
        plot_gdf.loc[missing_rate_mask, "homelessness_total_assessments"]
        / plot_gdf.loc[missing_rate_mask, "population"] * 1000
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 12))

    for ax in axes:
        plot_gdf.boundary.plot(ax=ax, linewidth=0.2, color="gray")
        ax.set_axis_off()

    plot_gdf.plot(
        column="homelessness_total_assessments",
        ax=axes[0],
        cmap="OrRd",
        linewidth=0.2,
        edgecolor="white",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data"
        }
    )
    axes[0].set_title(
        f"HOMELESSNESS TOTAL ASSESSMENTS BY LAD\nLatest available date: {latest_date.strftime('%Y-%m')}",
        fontweight="bold"
    )

    plot_gdf.plot(
        column="homelessness_per_1000",
        ax=axes[1],
        cmap="YlOrRd",
        linewidth=0.2,
        edgecolor="white",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data"
        }
    )
    axes[1].set_title(
        f"HOMELESSNESS PER 1000 POPULATION BY LAD\nLatest available date: {latest_date.strftime('%Y-%m')}",
        fontweight="bold"
    )

    fig.suptitle(
        "LATEST HOMELESSNESS MAP ACROSS LOCAL AUTHORITY DISTRICTS (LADs)",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    fig.text(
        0.5, 0.02,
        "Note: Your homelessness dataset currently contains England LAD codes only. "
        "Other UK LADs may appear as no data.",
        ha="center",
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()

    print("\nTop 10 LADs by homelessness_total_assessments:")
    print(
        plot_gdf[[map_code_col, map_name_col, "homelessness_total_assessments"]]
        .dropna(subset=["homelessness_total_assessments"])
        .sort_values("homelessness_total_assessments", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 LADs by homelessness_per_1000:")
    print(
        plot_gdf[[map_code_col, map_name_col, "homelessness_per_1000"]]
        .dropna(subset=["homelessness_per_1000"])
        .sort_values("homelessness_per_1000", ascending=False)
        .head(10)
        .to_string(index=False)
    )


def plot_england_only_homelessness_map(df):
    latest_date = df.loc[df["homelessness_per_1000"].notna(), "date"].max()
    latest = df[df["date"] == latest_date].copy()
    latest = latest.sort_values("lad_code").drop_duplicates(subset="lad_code", keep="first")

    print("Latest date used for England-only map:", latest_date)
    print("Number of England LAD rows:", latest.shape[0])

    lad_map, map_code_col, map_name_col = load_lad_boundaries()
    lad_map = lad_map[lad_map[map_code_col].str.startswith("E", na=False)].copy()

    plot_gdf = lad_map.merge(
        latest[["lad_code", "lad_name", "homelessness_per_1000"]],
        left_on=map_code_col,
        right_on="lad_code",
        how="left"
    )

    print("Matched England LADs with data:", plot_gdf["homelessness_per_1000"].notna().sum())

    fig, ax = plt.subplots(figsize=(10, 12), facecolor="white")
    ax.set_facecolor("white")

    plot_gdf.plot(
        column="homelessness_per_1000",
        ax=ax,
        cmap="OrRd",
        scheme="Quantiles",
        k=6,
        linewidth=0.35,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "title": "Homelessness\nper 1000",
            "loc": "lower left",
            "bbox_to_anchor": (1.02, 0.15),
            "frameon": False,
            "fmt": "{:.2f}"
        },
        missing_kwds={
            "color": "#d9d9d9",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data"
        }
    )

    ax.set_title(
        f"Homelessness per 1000 population by LAD\nLatest available date: {latest_date.strftime('%Y-%m')}",
        fontsize=18,
        fontweight="bold",
        pad=16
    )

    ax.set_axis_off()

    minx, miny, maxx, maxy = plot_gdf.total_bounds
    x_pad = (maxx - minx) * 0.03
    y_pad = (maxy - miny) * 0.03
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)

    plt.tight_layout()
    plt.show()


def main():
    df = load_and_clean_data(FILE_PATH)
    england = build_england_monthly_series(df)

    print_quick_facts(england)
    plot_analysis_figures(df, england)
    plot_uk_homelessness_maps(df)
    plot_england_only_homelessness_map(df)

    print("\nFinished.")


if __name__ == "__main__":
    main()