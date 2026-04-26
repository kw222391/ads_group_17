
import os
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sm = None
    smf = None

warnings.filterwarnings("ignore")


from pathlib import Path


BASE_DIR = Path(r"D:\UOB\ads_group_17\ads_group_17\data_new")

MONTHLY_FILE_CANDIDATES = [
    BASE_DIR / "monthly_lad_panel_2000_2025_with_homelessness_2000_2025.csv",
    BASE_DIR / "all_data_for_ana.csv",
]

QUARTERLY_HOMELESS_FILE = BASE_DIR / "homelessness_total_assessments_quarterly_total_and_change.csv"

OUTPUT_DIR = BASE_DIR / "analysis_outputs_living_cost_homelessness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "analysis_outputs_living_cost_homelessness"

CPI_COL = "cpi_00_all_items"
HOMELESS_COL = "homelessness_total_assessments"
HRA_START = pd.Period("2018Q2", freq="Q")

QUARTER_LAGS = [0, 1, 2, 4, 8]

SHOW_FIGURES = os.getenv("SHOW_FIGURES", "1") == "1"
SAVE_FIGURES = os.getenv("SAVE_FIGURES", "1") == "1"
SAVE_TABLES = os.getenv("SAVE_TABLES", "1") == "1"

RUN_ENGLAND_MODELS = os.getenv("RUN_ENGLAND_MODELS", "1") == "1"
RUN_LAD_FE_MODELS = os.getenv("RUN_LAD_FE_MODELS", "0") == "1"

BASE_COLS = ["year", "month", "lad_code", "lad_name"]

LIVING_COST_RAW_FEATURES = [
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
]

HOMELESS_FEATURES = [
    "homelessness_total_owed",
    "homelessness_threatened",
    "homelessness_total_assessments",
    "homelessness_relief",
    "homelessness_per_1000",
]

REQUIRED_USECOLS = BASE_COLS + LIVING_COST_RAW_FEATURES + HOMELESS_FEATURES + [CPI_COL]

def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def choose_monthly_file():
    for path in MONTHLY_FILE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a monthly panel CSV. Expected one of:\n"
        + "\n".join(str(p) for p in MONTHLY_FILE_CANDIDATES)
    )


def read_header(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def available_usecols(path, desired_cols):
    header = read_header(path)
    header_set = {c for c in header if c}
    return [c for c in desired_cols if c in header_set]


def safe_to_csv(df, filename):
    if SAVE_TABLES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / filename, index=False)


def save_figure(fig, filename):
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")


def first_nonnull(x):
    x = x.dropna()
    return x.iloc[0] if len(x) else np.nan


def weighted_mean(values, weights):
    values = pd.Series(values).astype(float)
    weights = pd.Series(weights).astype(float)
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.any():
        return np.average(values[mask], weights=weights[mask])
    if values.notna().any():
        return values.mean()
    return np.nan


def zscore(s):
    s = pd.Series(s, dtype="float64")
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd


def clean_inf(df):
    return df.replace([np.inf, -np.inf], np.nan)


def is_lad_code(s):
    return pd.Series(s).astype(str).str.match(r"^E0[6789]")


def period_label(q):
    return np.where(q >= HRA_START, "post_2018_HRA", "pre_2018_HRA")


def pct_change_by_group(df, group_col, value_col, periods):
    return df.groupby(group_col, sort=False)[value_col].pct_change(periods) * 100


def diff_by_group(df, group_col, value_col, periods):
    return df.groupby(group_col, sort=False)[value_col].diff(periods)


def add_lags_by_group(df, group_col, sort_col, value_cols, lags, suffix="lag"):
    out = df.sort_values([group_col, sort_col]).copy()
    for col in value_cols:
        for lag in lags:
            out[f"{col}_{suffix}{lag}"] = out.groupby(group_col, sort=False)[col].shift(lag)
    return out


def add_calendar_quarter_lags(df, group_col, quarter_col, value_cols, lags):

    out = df.copy()
    base_cols = [group_col, quarter_col] + value_cols
    base = df[base_cols].copy()
    for lag in lags:
        shifted = base.copy()
        shifted[quarter_col] = shifted[quarter_col] + lag
        rename = {c: f"{c}_lag{lag}" for c in value_cols}
        shifted = shifted.rename(columns=rename)
        out = out.merge(shifted, on=[group_col, quarter_col], how="left")
    return out


def correlation_matrix(df, cols, method="pearson", min_non_missing=12):
    use_cols = []
    for c in cols:
        if c in df.columns and df[c].notna().sum() >= min_non_missing and df[c].nunique(dropna=True) > 1:
            use_cols.append(c)
    if len(use_cols) < 2:
        return pd.DataFrame()
    return df[use_cols].corr(method=method)


def top_correlation_pairs(corr, top_n=40):
    if corr.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "corr", "abs_corr"])
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                rows.append(
                    {
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "corr": val,
                        "abs_corr": abs(val),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_corr", ascending=False).head(top_n).reset_index(drop=True)


def normalize_to_100(s):
    s = pd.Series(s, dtype="float64")
    first = s.dropna().iloc[0] if s.notna().any() else np.nan
    if pd.isna(first) or first == 0:
        return s * np.nan
    return s / first * 100


def compact_label(name):
    replacements = {
        "average_private_rental_price": "rent_price",
        "private_rental_price_index": "rent_index",
        "average_house_price": "house_price",
        "seasonally_adjusted_average_house_price": "sa_house_price",
        "house_price_index": "house_index",
        "house_sales_volume": "sales_volume",
        "unemployment_per_1000": "unemp_per_1000",
        "internal_net_migration_per_1000": "internal_mig_per_1000",
        "international_net_migration_per_1000": "intl_mig_per_1000",
        "living_cost_pressure_index": "living_cost_level_index",
        "living_cost_growth_pressure_index": "living_cost_growth_index",
        "annual_rent_to_income": "annual_rent_income_ratio",
        "house_price_to_income": "house_income_ratio",
        "cpi_total": "CPI_total",
        "uk_bank_rate": "bank_rate",
        "brent_oil_price": "oil_price",
        "gbp_index": "GBP_index",
        "ftse_100": "FTSE_100",
        "real_income": "real_income",
        "real_house_price": "real_house_price",
        "real_private_rent": "real_rent",
    }
    return replacements.get(name, name)

def read_monthly_panel():
    print_section("Reading monthly panel")
    monthly_file = choose_monthly_file()
    usecols = available_usecols(monthly_file, REQUIRED_USECOLS)
    print(f"Using monthly file: {monthly_file.name}")
    print(f"Columns read: {len(usecols)}")
    print("CPI choice: using CPI total only -> cpi_00_all_items. Other CPI category columns are excluded.")

    missing_required = [c for c in BASE_COLS + [CPI_COL] if c not in usecols]
    if missing_required:
        raise ValueError(f"Missing required column(s) in monthly file: {missing_required}")

    df = pd.read_csv(monthly_file, usecols=usecols, low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df[df["year"].notna() & df["month"].notna()].copy()
    df["date"] = pd.to_datetime(dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1))
    df["quarter"] = df["date"].dt.to_period("Q")
    df["lad_code"] = df["lad_code"].astype(str)
    df["is_lad"] = is_lad_code(df["lad_code"])
    df["is_england_aggregate"] = df["lad_code"].eq("E92000001")
    df["is_london_aggregate"] = df["lad_code"].eq("E12000007")
    df["policy_period"] = period_label(df["quarter"])

    numeric_cols = [c for c in usecols if c not in BASE_COLS]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = clean_inf(df)

    print(f"Rows: {len(df):,}; LAD rows: {df['is_lad'].sum():,}; non-LAD aggregate rows: {(~df['is_lad']).sum():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"LAD codes: {df.loc[df['is_lad'], 'lad_code'].nunique():,}")
    return df


def build_feature_coverage(monthly):
    cols = [c for c in LIVING_COST_RAW_FEATURES + [CPI_COL] if c in monthly.columns]
    rows = []
    for sample_name, sub in [
        ("all_rows", monthly),
        ("lad_rows_only", monthly[monthly["is_lad"]]),
        ("england_aggregate_row", monthly[monthly["is_england_aggregate"]]),
    ]:
        for c in cols:
            n = sub[c].notna().sum()
            rows.append(
                {
                    "sample": sample_name,
                    "feature": c,
                    "non_missing": n,
                    "pct_non_missing": n / len(sub) * 100 if len(sub) else np.nan,
                    "min": sub[c].min(skipna=True),
                    "median": sub[c].median(skipna=True),
                    "max": sub[c].max(skipna=True),
                }
            )
    out = pd.DataFrame(rows)
    safe_to_csv(out, "01_monthly_living_cost_feature_coverage.csv")
    print_section("Living-cost feature coverage")
    print(out[out["sample"].eq("lad_rows_only")].sort_values("pct_non_missing", ascending=False).to_string(index=False))
    return out


def aggregate_england_living_cost(monthly):

    print_section("Building England-level living-cost trend")

    lad = monthly[monthly["is_lad"]].copy()
    eng_row = monthly[monthly["is_england_aggregate"]].copy()

    national_cols = [c for c in [CPI_COL, "gbp_index", "ftse_100", "uk_bank_rate", "brent_oil_price"] if c in monthly.columns]
    national = monthly.groupby("date", as_index=False).agg({c: first_nonnull for c in national_cols})
    national = national.rename(columns={CPI_COL: "cpi_total"})

    # LAD aggregates.
    grouped = lad.groupby("date", sort=True)
    agg = pd.DataFrame({"date": sorted(lad["date"].dropna().unique())})
    agg["population"] = grouped["population"].sum(min_count=1).reindex(agg["date"]).values if "population" in lad else np.nan
    agg["unemployment_count"] = grouped["unemployment_count"].sum(min_count=1).reindex(agg["date"]).values if "unemployment_count" in lad else np.nan
    agg["house_sales_volume_lad_sum"] = grouped["house_sales_volume"].sum(min_count=1).reindex(agg["date"]).values if "house_sales_volume" in lad else np.nan
    agg["internal_net_migration"] = grouped["internal_net_migration"].sum(min_count=1).reindex(agg["date"]).values if "internal_net_migration" in lad else np.nan
    agg["international_net_migration"] = grouped["international_net_migration"].sum(min_count=1).reindex(agg["date"]).values if "international_net_migration" in lad else np.nan

    weighted_mean_cols = [
        "average_house_price",
        "house_price_index",
        "private_rental_price_index",
        "average_private_rental_price",
        "income",
    ]
    for col in weighted_mean_cols:
        if col in lad.columns:
            vals = []
            for d, g in grouped:
                vals.append({"date": d, f"{col}_lad_weighted_mean": weighted_mean(g[col], g.get("population", pd.Series(index=g.index, dtype=float)))})
            vals = pd.DataFrame(vals)
            agg = agg.merge(vals, on="date", how="left")


    england_house_cols = [
        "average_house_price",
        "average_house_price_monthly_change",
        "average_house_price_annual_change",
        "seasonally_adjusted_average_house_price",
        "house_price_index",
        "house_sales_volume",
    ]
    available_house_cols = [c for c in england_house_cols if c in eng_row.columns]
    if len(eng_row) and available_house_cols:
        eng_house = eng_row[["date"] + available_house_cols].copy()
        rename_map = {
            "average_house_price": "average_house_price_england_official",
            "average_house_price_monthly_change": "average_house_price_monthly_change",
            "average_house_price_annual_change": "average_house_price_annual_change",
            "seasonally_adjusted_average_house_price": "seasonally_adjusted_average_house_price",
            "house_price_index": "house_price_index_england_official",
            "house_sales_volume": "house_sales_volume_england_official",
        }
        eng_house = eng_house.rename(columns=rename_map)
    else:
        eng_house = pd.DataFrame({"date": []})

    out = national.merge(agg, on="date", how="outer").merge(eng_house, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.to_period("Q")
    out["quarter_date"] = out["quarter"].dt.to_timestamp(how="start")
    out["policy_period"] = period_label(out["quarter"])

    out["average_house_price"] = out.get("average_house_price_england_official", pd.Series(index=out.index)).combine_first(
        out.get("average_house_price_lad_weighted_mean", pd.Series(index=out.index))
    )
    out["house_price_index"] = out.get("house_price_index_england_official", pd.Series(index=out.index)).combine_first(
        out.get("house_price_index_lad_weighted_mean", pd.Series(index=out.index))
    )
    out["house_sales_volume"] = out.get("house_sales_volume_england_official", pd.Series(index=out.index)).combine_first(
        out.get("house_sales_volume_lad_sum", pd.Series(index=out.index))
    )
    out["average_private_rental_price"] = out.get("average_private_rental_price_lad_weighted_mean", pd.Series(index=out.index))
    out["private_rental_price_index"] = out.get("private_rental_price_index_lad_weighted_mean", pd.Series(index=out.index))
    out["income"] = out.get("income_lad_weighted_mean", pd.Series(index=out.index))

    out["unemployment_per_1000"] = out["unemployment_count"] / out["population"] * 1000
    out["internal_net_migration_per_1000"] = out["internal_net_migration"] / out["population"] * 1000
    out["international_net_migration_per_1000"] = out["international_net_migration"] / out["population"] * 1000

    out["real_income"] = out["income"] / out["cpi_total"] * 100
    out["real_house_price"] = out["average_house_price"] / out["cpi_total"] * 100
    out["real_private_rent"] = out["average_private_rental_price"] / out["cpi_total"] * 100
    out["house_price_to_income"] = out["average_house_price"] / out["income"]
    out["annual_rent_to_income"] = out["average_private_rental_price"] * 12 / out["income"]
    out["rent_to_cpi_ratio"] = out["private_rental_price_index"] / out["cpi_total"]
    out["oil_to_cpi_ratio"] = out["brent_oil_price"] / out["cpi_total"]

    # Monthly and annual changes
    change_cols = [
        "cpi_total",
        "average_house_price",
        "house_price_index",
        "house_sales_volume",
        "average_private_rental_price",
        "private_rental_price_index",
        "income",
        "real_income",
        "real_house_price",
        "real_private_rent",
        "house_price_to_income",
        "annual_rent_to_income",
        "unemployment_count",
        "unemployment_per_1000",
        "gbp_index",
        "ftse_100",
        "uk_bank_rate",
        "brent_oil_price",
        "population",
        "internal_net_migration",
        "international_net_migration",
        "internal_net_migration_per_1000",
        "international_net_migration_per_1000",
    ]
    for c in change_cols:
        if c in out.columns:
            out[f"{c}_mom_pct"] = out[c].pct_change(1) * 100
            out[f"{c}_yoy_pct"] = out[c].pct_change(12) * 100
            out[f"{c}_yoy_diff"] = out[c].diff(12)

    out = clean_inf(out)

    # pressure index
    level_components = {
        "cpi_total": "positive",
        "average_house_price": "positive",
        "average_private_rental_price": "positive",
        "house_price_to_income": "positive",
        "annual_rent_to_income": "positive",
        "uk_bank_rate": "positive",
        "brent_oil_price": "positive",
        "unemployment_per_1000": "positive",
        "real_income": "negative",
    }
    level_z = pd.DataFrame(index=out.index)
    for col, direction in level_components.items():
        if col in out.columns:
            level_z[col] = zscore(out[col])
            if direction == "negative":
                level_z[col] = -level_z[col]
    out["living_cost_pressure_index"] = level_z.mean(axis=1, skipna=True)
    out["n_living_cost_level_components"] = level_z.notna().sum(axis=1)

    growth_components = {
        "cpi_total_yoy_pct": "positive",
        "average_house_price_yoy_pct": "positive",
        "average_private_rental_price_yoy_pct": "positive",
        "house_price_to_income_yoy_pct": "positive",
        "annual_rent_to_income_yoy_pct": "positive",
        "uk_bank_rate_yoy_diff": "positive",
        "brent_oil_price_yoy_pct": "positive",
        "unemployment_per_1000_yoy_diff": "positive",
        "real_income_yoy_pct": "negative",
    }
    growth_z = pd.DataFrame(index=out.index)
    for col, direction in growth_components.items():
        if col in out.columns:
            growth_z[col] = zscore(out[col])
            if direction == "negative":
                growth_z[col] = -growth_z[col]
    out["living_cost_growth_pressure_index"] = growth_z.mean(axis=1, skipna=True)
    out["n_living_cost_growth_components"] = growth_z.notna().sum(axis=1)

    # CPI total lags
    for lag in QUARTER_LAGS:
        # monthly lag equivalent for 1 quarter = 3 months.
        ml = lag * 3
        out[f"cpi_total_yoy_pct_lag{lag}q"] = out["cpi_total_yoy_pct"].shift(ml)
        out[f"living_cost_growth_pressure_index_lag{lag}q"] = out["living_cost_growth_pressure_index"].shift(ml)
        out[f"living_cost_pressure_index_lag{lag}q"] = out["living_cost_pressure_index"].shift(ml)

    safe_to_csv(out, "02_england_living_cost_monthly_trend.csv")

    summary_cols = [
        "date", "cpi_total", "cpi_total_yoy_pct", "average_house_price",
        "average_private_rental_price", "income", "real_income",
        "house_price_to_income", "annual_rent_to_income", "uk_bank_rate",
        "brent_oil_price", "unemployment_per_1000",
        "living_cost_pressure_index", "living_cost_growth_pressure_index",
    ]
    print(out[[c for c in summary_cols if c in out.columns]].tail(12).to_string(index=False))
    return out


def quarterly_england_living_cost(eng_monthly):
    agg_map = {}
    mean_cols = [
        "cpi_total", "cpi_total_yoy_pct",
        "average_house_price", "average_house_price_yoy_pct",
        "average_private_rental_price", "average_private_rental_price_yoy_pct",
        "income", "income_yoy_pct", "real_income", "real_income_yoy_pct",
        "house_price_to_income", "house_price_to_income_yoy_pct",
        "annual_rent_to_income", "annual_rent_to_income_yoy_pct",
        "unemployment_count", "unemployment_per_1000", "unemployment_per_1000_yoy_diff",
        "gbp_index", "ftse_100", "uk_bank_rate", "uk_bank_rate_yoy_diff",
        "brent_oil_price", "brent_oil_price_yoy_pct",
        "living_cost_pressure_index", "living_cost_growth_pressure_index",
        "n_living_cost_level_components", "n_living_cost_growth_components",
    ]
    for c in mean_cols:
        if c in eng_monthly.columns:
            agg_map[c] = "mean"
    q = eng_monthly.groupby("quarter", as_index=False).agg(agg_map)
    q["quarter_date"] = q["quarter"].dt.to_timestamp(how="start")
    q["policy_period"] = period_label(q["quarter"])
    for lag in QUARTER_LAGS:
        for c in [
            "cpi_total_yoy_pct",
            "living_cost_pressure_index",
            "living_cost_growth_pressure_index",
            "annual_rent_to_income",
            "house_price_to_income",
        ]:
            if c in q.columns:
                q[f"{c}_lag{lag}"] = q[c].shift(lag)
    safe_to_csv(q, "03_england_living_cost_quarterly_trend.csv")
    return q


def build_lad_living_cost_panel(monthly):
    print_section("Building LAD-level living-cost panel")
    lad = monthly[monthly["is_lad"]].copy().sort_values(["lad_code", "date"]).reset_index(drop=True)

    # Feature engineering at LAD level.
    lad["cpi_total"] = lad[CPI_COL]
    lad["unemployment_per_1000"] = lad["unemployment_count"] / lad["population"] * 1000
    lad["internal_net_migration_per_1000"] = lad["internal_net_migration"] / lad["population"] * 1000
    lad["international_net_migration_per_1000"] = lad["international_net_migration"] / lad["population"] * 1000
    lad["real_income"] = lad["income"] / lad["cpi_total"] * 100
    lad["real_house_price"] = lad["average_house_price"] / lad["cpi_total"] * 100
    lad["real_private_rent"] = lad["average_private_rental_price"] / lad["cpi_total"] * 100
    lad["house_price_to_income"] = lad["average_house_price"] / lad["income"]
    lad["annual_rent_to_income"] = lad["average_private_rental_price"] * 12 / lad["income"]
    lad["rent_to_cpi_ratio"] = lad["private_rental_price_index"] / lad["cpi_total"]
    lad["oil_to_cpi_ratio"] = lad["brent_oil_price"] / lad["cpi_total"]

    # LAD-level monthly YoY change variables.
    yoy_cols = [
        "average_house_price", "house_price_index", "house_sales_volume",
        "private_rental_price_index", "average_private_rental_price",
        "income", "real_income", "real_house_price", "real_private_rent",
        "house_price_to_income", "annual_rent_to_income",
        "unemployment_count", "unemployment_per_1000",
        "population", "internal_net_migration", "international_net_migration",
        "internal_net_migration_per_1000", "international_net_migration_per_1000",
    ]
    for c in yoy_cols:
        if c in lad.columns:
            lad[f"{c}_yoy_pct"] = pct_change_by_group(lad, "lad_code", c, 12)
            lad[f"{c}_yoy_diff"] = diff_by_group(lad, "lad_code", c, 12)

    # CPI
    for c in ["cpi_total", "gbp_index", "ftse_100", "uk_bank_rate", "brent_oil_price"]:
        if c in lad.columns:
            tmp = lad[["date", c]].drop_duplicates("date").sort_values("date").copy()
            tmp[f"{c}_yoy_pct"] = tmp[c].pct_change(12) * 100
            tmp[f"{c}_yoy_diff"] = tmp[c].diff(12)
            lad = lad.drop(columns=[f"{c}_yoy_pct", f"{c}_yoy_diff"], errors="ignore").merge(
                tmp[["date", f"{c}_yoy_pct", f"{c}_yoy_diff"]],
                on="date",
                how="left",
            )

    lad = clean_inf(lad)

    # Composite indexes at LAD level
    level_components = {
        "cpi_total": "positive",
        "average_house_price": "positive",
        "average_private_rental_price": "positive",
        "house_price_to_income": "positive",
        "annual_rent_to_income": "positive",
        "uk_bank_rate": "positive",
        "brent_oil_price": "positive",
        "unemployment_per_1000": "positive",
        "real_income": "negative",
    }
    zlev = pd.DataFrame(index=lad.index)
    for col, direction in level_components.items():
        if col in lad.columns:
            zlev[col] = zscore(lad[col])
            if direction == "negative":
                zlev[col] = -zlev[col]
    lad["living_cost_pressure_index"] = zlev.mean(axis=1, skipna=True)
    lad["n_living_cost_level_components"] = zlev.notna().sum(axis=1)

    growth_components = {
        "cpi_total_yoy_pct": "positive",
        "average_house_price_yoy_pct": "positive",
        "average_private_rental_price_yoy_pct": "positive",
        "house_price_to_income_yoy_pct": "positive",
        "annual_rent_to_income_yoy_pct": "positive",
        "uk_bank_rate_yoy_diff": "positive",
        "brent_oil_price_yoy_pct": "positive",
        "unemployment_per_1000_yoy_diff": "positive",
        "real_income_yoy_pct": "negative",
    }
    zg = pd.DataFrame(index=lad.index)
    for col, direction in growth_components.items():
        if col in lad.columns:
            zg[col] = zscore(lad[col])
            if direction == "negative":
                zg[col] = -zg[col]
    lad["living_cost_growth_pressure_index"] = zg.mean(axis=1, skipna=True)
    lad["n_living_cost_growth_components"] = zg.notna().sum(axis=1)

    safe_to_csv(lad.head(2000), "04_lad_living_cost_monthly_panel_sample_first_2000_rows.csv")
    print(f"LAD monthly rows: {len(lad):,}; LADs: {lad['lad_code'].nunique():,}")
    return lad


def quarterly_lad_living_cost(lad_monthly):
    print_section("Aggregating LAD living-cost data to quarter")
    keep_mean_cols = [
        "cpi_total", "cpi_total_yoy_pct",
        "average_house_price", "average_house_price_yoy_pct",
        "house_price_index", "house_price_index_yoy_pct",
        "house_sales_volume",
        "private_rental_price_index", "private_rental_price_index_yoy_pct",
        "average_private_rental_price", "average_private_rental_price_yoy_pct",
        "income", "income_yoy_pct", "real_income", "real_income_yoy_pct",
        "real_house_price", "real_private_rent",
        "house_price_to_income", "house_price_to_income_yoy_pct",
        "annual_rent_to_income", "annual_rent_to_income_yoy_pct",
        "unemployment_count", "unemployment_per_1000", "unemployment_per_1000_yoy_diff",
        "population",
        "gbp_index", "gbp_index_yoy_pct",
        "ftse_100", "ftse_100_yoy_pct",
        "uk_bank_rate", "uk_bank_rate_yoy_diff",
        "brent_oil_price", "brent_oil_price_yoy_pct",
        "internal_net_migration", "international_net_migration",
        "internal_net_migration_per_1000", "international_net_migration_per_1000",
        "living_cost_pressure_index", "living_cost_growth_pressure_index",
        "n_living_cost_level_components", "n_living_cost_growth_components",
    ]
    agg = {c: "mean" for c in keep_mean_cols if c in lad_monthly.columns}
    q = lad_monthly.groupby(["lad_code", "lad_name", "quarter"], as_index=False).agg(agg)
    q["quarter_date"] = q["quarter"].dt.to_timestamp(how="start")
    q["policy_period"] = period_label(q["quarter"])

    lag_cols = [
        "cpi_total_yoy_pct",
        "living_cost_pressure_index",
        "living_cost_growth_pressure_index",
        "annual_rent_to_income",
        "house_price_to_income",
        "average_private_rental_price_yoy_pct",
        "average_house_price_yoy_pct",
        "real_income_yoy_pct",
        "uk_bank_rate_yoy_diff",
        "unemployment_per_1000_yoy_diff",
    ]
    lag_cols = [c for c in lag_cols if c in q.columns]
    q = add_calendar_quarter_lags(q, "lad_code", "quarter", lag_cols, QUARTER_LAGS)
    safe_to_csv(q.head(2000), "05_lad_living_cost_quarterly_panel_sample_first_2000_rows.csv")
    print(f"LAD quarterly living-cost rows: {len(q):,}; LADs: {q['lad_code'].nunique():,}")
    return q


def run_pca_from_corr(df, cols, min_non_missing=80):

    use_cols = [
        c for c in cols
        if c in df.columns and df[c].notna().sum() >= min_non_missing and df[c].nunique(dropna=True) > 1
    ]
    if len(use_cols) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = df[use_cols].copy()
    X = X.dropna()
    if len(X) < 12:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    Z = (X - X.mean()) / X.std(ddof=1)
    Z = Z.dropna(axis=1)
    if Z.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    arr = Z.to_numpy()
    U, S, Vt = np.linalg.svd(arr, full_matrices=False)
    eigenvalues = (S ** 2) / (len(Z) - 1)
    explained = eigenvalues / eigenvalues.sum()

    loadings = pd.DataFrame(
        Vt.T,
        index=Z.columns,
        columns=[f"PC{i+1}" for i in range(Vt.shape[0])]
    ).reset_index().rename(columns={"index": "feature"})
    explained_df = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(explained))],
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": np.cumsum(explained),
        }
    )
    scores = pd.DataFrame(U * S, columns=[f"PC{i+1}" for i in range(Vt.shape[0])])
    scores["date"] = X.index if X.index.name == "date" else df.loc[X.index, "date"].values if "date" in df.columns else X.index
    return loadings, explained_df, scores


def analyze_living_cost_feature_relationships(eng_monthly, lad_monthly):
    print_section("Analyzing relationships among living-cost features")

    level_features = [
        "cpi_total",
        "average_house_price",
        "seasonally_adjusted_average_house_price",
        "house_price_index",
        "house_sales_volume",
        "average_private_rental_price",
        "private_rental_price_index",
        "income",
        "real_income",
        "real_house_price",
        "real_private_rent",
        "house_price_to_income",
        "annual_rent_to_income",
        "unemployment_count",
        "unemployment_per_1000",
        "gbp_index",
        "ftse_100",
        "uk_bank_rate",
        "brent_oil_price",
        "population",
        "internal_net_migration_per_1000",
        "international_net_migration_per_1000",
        "living_cost_pressure_index",
        "living_cost_growth_pressure_index",
    ]
    yoy_features = [
        "cpi_total_yoy_pct",
        "average_house_price_yoy_pct",
        "house_price_index_yoy_pct",
        "house_sales_volume_yoy_pct",
        "average_private_rental_price_yoy_pct",
        "private_rental_price_index_yoy_pct",
        "income_yoy_pct",
        "real_income_yoy_pct",
        "real_house_price_yoy_pct",
        "real_private_rent_yoy_pct",
        "house_price_to_income_yoy_pct",
        "annual_rent_to_income_yoy_pct",
        "unemployment_count_yoy_pct",
        "unemployment_per_1000_yoy_diff",
        "gbp_index_yoy_pct",
        "ftse_100_yoy_pct",
        "uk_bank_rate_yoy_diff",
        "brent_oil_price_yoy_pct",
        "population_yoy_pct",
        "internal_net_migration_per_1000_yoy_diff",
        "international_net_migration_per_1000_yoy_diff",
        "living_cost_growth_pressure_index",
    ]

    # England
    eng_level_corr = correlation_matrix(eng_monthly, level_features, method="pearson", min_non_missing=36)
    eng_yoy_corr = correlation_matrix(eng_monthly, yoy_features, method="pearson", min_non_missing=36)
    eng_level_top = top_correlation_pairs(eng_level_corr, top_n=50)
    eng_yoy_top = top_correlation_pairs(eng_yoy_corr, top_n=50)

    safe_to_csv(corr_to_long(eng_level_corr), "06_england_living_cost_level_correlation_matrix_long.csv")
    safe_to_csv(corr_to_long(eng_yoy_corr), "07_england_living_cost_yoy_correlation_matrix_long.csv")
    safe_to_csv(eng_level_top, "08_england_living_cost_top_level_correlation_pairs.csv")
    safe_to_csv(eng_yoy_top, "09_england_living_cost_top_yoy_correlation_pairs.csv")

    print("\nTop England-level correlations among living-cost LEVEL features:")
    print(eng_level_top.head(15).to_string(index=False))
    print("\nTop England-level correlations among living-cost YoY / change features:")
    print(eng_yoy_top.head(15).to_string(index=False))

    # LAD pooled
    lad_level_corr = correlation_matrix(lad_monthly, level_features, method="pearson", min_non_missing=1000)
    lad_yoy_corr = correlation_matrix(lad_monthly, yoy_features, method="pearson", min_non_missing=1000)

    within_cols = [c for c in level_features if c in lad_monthly.columns and lad_monthly[c].notna().sum() >= 1000]
    within = lad_monthly[["lad_code"] + within_cols].copy()
    for c in within_cols:
        within[c] = within[c] - within.groupby("lad_code")[c].transform("mean")
    lad_within_corr = correlation_matrix(within, within_cols, method="pearson", min_non_missing=1000)

    safe_to_csv(corr_to_long(lad_level_corr), "10_lad_living_cost_level_pooled_correlation_matrix_long.csv")
    safe_to_csv(corr_to_long(lad_yoy_corr), "11_lad_living_cost_yoy_pooled_correlation_matrix_long.csv")
    safe_to_csv(corr_to_long(lad_within_corr), "12_lad_living_cost_level_within_lad_correlation_matrix_long.csv")
    safe_to_csv(top_correlation_pairs(lad_level_corr, top_n=50), "13_lad_living_cost_top_level_correlation_pairs.csv")
    safe_to_csv(top_correlation_pairs(lad_yoy_corr, top_n=50), "14_lad_living_cost_top_yoy_correlation_pairs.csv")
    safe_to_csv(top_correlation_pairs(lad_within_corr, top_n=50), "15_lad_living_cost_top_within_lad_correlation_pairs.csv")

    # PCA
    pca_cols = [
        "cpi_total_yoy_pct",
        "average_house_price_yoy_pct",
        "average_private_rental_price_yoy_pct",
        "house_price_to_income_yoy_pct",
        "annual_rent_to_income_yoy_pct",
        "uk_bank_rate_yoy_diff",
        "brent_oil_price_yoy_pct",
        "unemployment_per_1000_yoy_diff",
        "real_income_yoy_pct",
    ]
    loadings, explained, scores = run_pca_from_corr(eng_monthly, pca_cols, min_non_missing=60)
    safe_to_csv(loadings, "16_england_living_cost_pca_loadings.csv")
    safe_to_csv(explained, "17_england_living_cost_pca_explained_variance.csv")
    safe_to_csv(scores, "18_england_living_cost_pca_scores.csv")

    if not explained.empty:
        print("\nPCA explained variance:")
        print(explained.head(5).to_string(index=False))
        print("\nPC1 loadings:")
        pc1 = loadings[["feature", "PC1"]].sort_values("PC1", key=lambda s: s.abs(), ascending=False)
        print(pc1.to_string(index=False))

    return {
        "eng_level_corr": eng_level_corr,
        "eng_yoy_corr": eng_yoy_corr,
        "eng_level_top": eng_level_top,
        "eng_yoy_top": eng_yoy_top,
        "lad_level_corr": lad_level_corr,
        "lad_yoy_corr": lad_yoy_corr,
        "lad_within_corr": lad_within_corr,
        "pca_loadings": loadings,
        "pca_explained": explained,
        "pca_scores": scores,
    }


def corr_to_long(corr):
    if corr is None or corr.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "corr"])
    tmp = corr.copy()
    tmp.index.name = "feature_1"
    return tmp.reset_index().melt(id_vars="feature_1", var_name="feature_2", value_name="corr")

def read_quarterly_homelessness():
    print_section("Reading quarterly homelessness")
    if not QUARTERLY_HOMELESS_FILE.exists():
        raise FileNotFoundError(f"Could not find: {QUARTERLY_HOMELESS_FILE}")

    q = pd.read_csv(QUARTERLY_HOMELESS_FILE, low_memory=False)
    q["quarter"] = pd.PeriodIndex(q["quarter"].astype(str), freq="Q")
    q["quarter_date"] = q["quarter"].dt.to_timestamp(how="start")
    q["lad_code"] = q["lad_code"].astype(str)
    q["is_lad"] = is_lad_code(q["lad_code"])
    q["policy_period"] = period_label(q["quarter"])
    for c in q.columns:
        if c not in ["lad_code", "lad_name", "quarter", "quarter_date", "policy_period"]:
            q[c] = pd.to_numeric(q[c], errors="coerce")
    q = clean_inf(q)
    print(f"Rows: {len(q):,}; LAD rows: {q['is_lad'].sum():,}; date range: {q['quarter'].min()} to {q['quarter'].max()}")
    return q


def build_england_homeless_from_lads(q_home):
    lad = q_home[q_home["is_lad"] & q_home[HOMELESS_COL].notna()].copy()
    eng = lad.groupby("quarter", as_index=False).agg(
        homelessness_total_assessments=(HOMELESS_COL, "sum"),
        n_lads=("lad_code", "nunique"),
    )
    eng = eng.sort_values("quarter")
    eng["quarter_date"] = eng["quarter"].dt.to_timestamp(how="start")
    eng["policy_period"] = period_label(eng["quarter"])
    eng["log_homeless"] = np.log1p(eng["homelessness_total_assessments"])
    eng["dlog_homeless"] = eng["log_homeless"].diff(1)
    eng["yoy_log_homeless"] = eng["log_homeless"].diff(4)
    eng["homeless_yoy_pct"] = eng["homelessness_total_assessments"].pct_change(4) * 100
    return clean_inf(eng)


def merge_lad_homeless_living(q_home, q_living):
    home_lad = q_home[q_home["is_lad"] & q_home[HOMELESS_COL].notna()].copy()
    home_lad = home_lad[["lad_code", "lad_name", "quarter", "quarter_date", HOMELESS_COL, "policy_period"]].copy()
    home_lad["log_homeless"] = np.log1p(home_lad[HOMELESS_COL].clip(lower=0))
    home_lad = add_calendar_quarter_lags(home_lad, "lad_code", "quarter", ["log_homeless", HOMELESS_COL], [1, 2, 4])
    home_lad["dlog_homeless"] = home_lad["log_homeless"] - home_lad["log_homeless_lag1"]
    home_lad["yoy_log_homeless"] = home_lad["log_homeless"] - home_lad["log_homeless_lag4"]
    home_lad["homeless_yoy_pct"] = (home_lad[HOMELESS_COL] / home_lad[f"{HOMELESS_COL}_lag4"] - 1) * 100

    merged = home_lad.merge(q_living, on=["lad_code", "lad_name", "quarter"], how="left", suffixes=("", "_living"))
    if "quarter_date_living" in merged.columns:
        merged["quarter_date"] = merged["quarter_date"].combine_first(merged["quarter_date_living"])
        merged = merged.drop(columns=["quarter_date_living"], errors="ignore")
    if "policy_period_living" in merged.columns:
        merged = merged.drop(columns=["policy_period_living"], errors="ignore")
    merged = clean_inf(merged)
    safe_to_csv(merged.head(3000), "19_lad_quarterly_homeless_living_cost_merged_sample_first_3000_rows.csv")
    return merged


def merge_england_homeless_living(eng_home, eng_living_q):
    eng = eng_home.merge(eng_living_q, on="quarter", how="left", suffixes=("", "_living"))
    if "quarter_date_living" in eng.columns:
        eng["quarter_date"] = eng["quarter_date"].combine_first(eng["quarter_date_living"])
    if "policy_period_living" in eng.columns:
        eng = eng.drop(columns=["policy_period_living"], errors="ignore")
    eng = clean_inf(eng)
    safe_to_csv(eng, "20_england_quarterly_homeless_living_cost_merged.csv")
    return eng


def lag_correlation_table(df, dep_cols, living_cols, by_period=True):
    rows = []
    periods = ["all_periods"] if not by_period else ["pre_2018_HRA", "post_2018_HRA"]
    for period in periods:
        sub = df.copy() if period == "all_periods" else df[df["policy_period"].eq(period)].copy()
        for dep in dep_cols:
            for base_col in living_cols:
                for lag in QUARTER_LAGS:
                    col = f"{base_col}_lag{lag}"
                    if lag == 0 and col not in sub.columns and base_col in sub.columns:
                        col = base_col
                    if dep in sub.columns and col in sub.columns:
                        tmp = sub[[dep, col]].dropna()
                        if len(tmp) >= 6 and tmp[dep].nunique() > 1 and tmp[col].nunique() > 1:
                            corr = tmp[dep].corr(tmp[col])
                        else:
                            corr = np.nan
                        rows.append(
                            {
                                "period": period,
                                "dependent_variable": dep,
                                "living_cost_variable": base_col,
                                "lag_quarters": lag,
                                "corr": corr,
                                "n": len(tmp),
                            }
                        )
    return pd.DataFrame(rows)


def run_homeless_living_relationships(eng, lad):
    print_section("Homelessness vs living-cost relationship")

    dep_cols = ["dlog_homeless", "yoy_log_homeless", "homeless_yoy_pct"]
    living_bases = [
        "cpi_total_yoy_pct",
        "living_cost_pressure_index",
        "living_cost_growth_pressure_index",
        "annual_rent_to_income",
        "house_price_to_income",
    ]
    living_bases = [c for c in living_bases if c in eng.columns or c in lad.columns]

    eng_corr = lag_correlation_table(eng, dep_cols, living_bases, by_period=True)
    lad_corr = lag_correlation_table(lad, dep_cols, living_bases, by_period=True)

    safe_to_csv(eng_corr, "21_england_homeless_living_cost_lag_correlations.csv")
    safe_to_csv(lad_corr, "22_lad_homeless_living_cost_lag_correlations.csv")

    print("\nEngland lag correlations: living cost vs homelessness YoY-log change")
    show = eng_corr[eng_corr["dependent_variable"].eq("yoy_log_homeless")].copy()
    print(show.sort_values(["period", "living_cost_variable", "lag_quarters"]).to_string(index=False))

    print("\nLAD pooled lag correlations: living cost vs homelessness YoY-log change")
    show2 = lad_corr[lad_corr["dependent_variable"].eq("yoy_log_homeless")].copy()
    print(show2.sort_values(["period", "living_cost_variable", "lag_quarters"]).to_string(index=False))

    # Simple models. These are association models, not causal estimates.
    eng_models = run_england_models(eng)
    lad_models = run_lad_models(lad)
    return eng_corr, lad_corr, eng_models, lad_models


def model_result_frame(model, model_name):
    if model is None:
        return pd.DataFrame()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": model.params.get(term, np.nan),
                "std_err": model.bse.get(term, np.nan),
                "t": model.tvalues.get(term, np.nan),
                "p_value": model.pvalues.get(term, np.nan),
                "nobs": getattr(model, "nobs", np.nan),
                "r2": getattr(model, "rsquared", np.nan),
            }
        )
    return pd.DataFrame(rows)


def run_england_models(eng):
    if not RUN_ENGLAND_MODELS:
        print("RUN_ENGLAND_MODELS=0, skipping England OLS models.")
        return pd.DataFrame()
    if smf is None:
        print("statsmodels is not installed, skipping OLS models.")
        return pd.DataFrame()

    rows = []
    model_specs = [
        ("eng_yoy_living_growth_lag0", "yoy_log_homeless ~ living_cost_growth_pressure_index"),
        ("eng_yoy_living_growth_lag1", "yoy_log_homeless ~ living_cost_growth_pressure_index_lag1"),
        ("eng_yoy_cpi_lag0", "yoy_log_homeless ~ cpi_total_yoy_pct"),
        ("eng_yoy_cpi_lag1", "yoy_log_homeless ~ cpi_total_yoy_pct_lag1"),
    ]
    for period in ["pre_2018_HRA", "post_2018_HRA"]:
        sub = eng[eng["policy_period"].eq(period)].copy()
        for name, formula in model_specs:
            needed = [v.strip() for v in formula.replace("~", "+").split("+")]
            needed = [v for v in needed if v and v != "1"]
            if all(v in sub.columns for v in needed):
                tmp = sub[needed].dropna()
                if len(tmp) >= 8:
                    try:
                        model = smf.ols(formula, data=sub).fit(cov_type="HC1")
                        rows.append(model_result_frame(model, f"{name}_{period}"))
                    except Exception as exc:
                        print(f"Skipped model {name} {period}: {exc}")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    safe_to_csv(out, "23_england_homeless_living_cost_models.csv")
    return out


def run_lad_models(lad):
    if not RUN_LAD_FE_MODELS:
        print("RUN_LAD_FE_MODELS=0, skipping LAD fixed-effect models. Set RUN_LAD_FE_MODELS=1 to run them.")
        return pd.DataFrame()
    if smf is None:
        print("statsmodels is not installed, skipping LAD FE models.")
        return pd.DataFrame()

    rows = []
    model_specs = [
        ("lad_fe_yoy_living_growth_lag0", "yoy_log_homeless ~ living_cost_growth_pressure_index + C(lad_code)"),
        ("lad_fe_yoy_living_growth_lag1", "yoy_log_homeless ~ living_cost_growth_pressure_index_lag1 + C(lad_code)"),
        ("lad_fe_yoy_cpi_lag0", "yoy_log_homeless ~ cpi_total_yoy_pct + C(lad_code)"),
        ("lad_fe_yoy_cpi_lag1", "yoy_log_homeless ~ cpi_total_yoy_pct_lag1 + C(lad_code)"),
    ]
    for period in ["pre_2018_HRA", "post_2018_HRA"]:
        sub = lad[lad["policy_period"].eq(period)].copy()
        sub["quarter_str"] = sub["quarter"].astype(str)
        for name, formula in model_specs:
            candidate_vars = [
                "yoy_log_homeless",
                "living_cost_growth_pressure_index",
                "living_cost_growth_pressure_index_lag1",
                "cpi_total_yoy_pct",
                "cpi_total_yoy_pct_lag1",
                "lad_code",
                "quarter_str",
            ]
            # Only run when the focal variable exists and enough observations.
            focal = formula.split("~")[1].split("+")[0].strip()
            if focal not in sub.columns:
                continue
            tmp = sub[["yoy_log_homeless", focal, "lad_code", "quarter_str"]].dropna()
            if len(tmp) < 100 or tmp["lad_code"].nunique() < 20 or tmp[focal].nunique() <= 1:
                continue
            try:
                model = smf.ols(formula, data=tmp).fit(cov_type="cluster", cov_kwds={"groups": tmp["quarter_str"]})
                rows.append(model_result_frame(model, f"{name}_{period}"))
            except Exception as exc:
                print(f"Skipped LAD model {name} {period}: {exc}")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    safe_to_csv(out, "24_lad_homeless_living_cost_fe_models.csv")
    return out



def plot_living_cost_index(eng_monthly):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eng_monthly["date"], eng_monthly["living_cost_pressure_index"], label="Level pressure index")
    ax.plot(eng_monthly["date"], eng_monthly["living_cost_growth_pressure_index"], label="Growth pressure index")
    ax.axvline(pd.Timestamp("2018-04-01"), linestyle="--", linewidth=1, label="2018Q2 HRA break")
    ax.axhline(0, linewidth=1)
    ax.set_title("England living-cost pressure trend")
    ax.set_ylabel("Standardized index; higher = more pressure")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "01_england_living_cost_pressure_index_trend.png")


def plot_normalized_core_trends(eng_monthly):
    cols = [
        "cpi_total",
        "average_house_price",
        "average_private_rental_price",
        "income",
        "uk_bank_rate",
        "brent_oil_price",
    ]
    cols = [c for c in cols if c in eng_monthly.columns and eng_monthly[c].notna().sum() > 12]
    fig, ax = plt.subplots(figsize=(13, 6))
    for c in cols:
        ax.plot(eng_monthly["date"], normalize_to_100(eng_monthly[c]), label=compact_label(c))
    ax.axvline(pd.Timestamp("2018-04-01"), linestyle="--", linewidth=1)
    ax.set_title("Core living-cost features normalized to 100 at first available observation")
    ax.set_ylabel("Index = 100 at first non-missing value")
    ax.set_xlabel("Date")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "02_core_living_cost_features_normalized_trends.png")


def plot_yoy_trends(eng_monthly):
    cols = [
        "cpi_total_yoy_pct",
        "average_house_price_yoy_pct",
        "average_private_rental_price_yoy_pct",
        "income_yoy_pct",
        "real_income_yoy_pct",
        "brent_oil_price_yoy_pct",
    ]
    cols = [c for c in cols if c in eng_monthly.columns and eng_monthly[c].notna().sum() > 12]
    fig, ax = plt.subplots(figsize=(13, 6))
    for c in cols:
        ax.plot(eng_monthly["date"], eng_monthly[c], label=compact_label(c.replace("_yoy_pct", "")) + " YoY %")
    ax.axvline(pd.Timestamp("2018-04-01"), linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_title("Annual growth rates of main living-cost features")
    ax.set_ylabel("YoY change, %")
    ax.set_xlabel("Date")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "03_living_cost_yoy_growth_trends.png")


def plot_affordability(eng_monthly):
    fig, ax = plt.subplots(figsize=(12, 6))
    if "house_price_to_income" in eng_monthly.columns:
        ax.plot(eng_monthly["date"], eng_monthly["house_price_to_income"], label="House price / annual income")
    if "annual_rent_to_income" in eng_monthly.columns:
        ax.plot(eng_monthly["date"], eng_monthly["annual_rent_to_income"], label="Annual rent / annual income")
    ax.axvline(pd.Timestamp("2018-04-01"), linestyle="--", linewidth=1)
    ax.set_title("England affordability pressure")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "04_affordability_pressure_trends.png")


def plot_corr_heatmap(corr, title, filename, max_features=24):
    if corr is None or corr.empty:
        return
    # Keep the most connected features if matrix is too large.
    c = corr.copy()
    if c.shape[0] > max_features:
        score = c.abs().sum().sort_values(ascending=False).head(max_features).index
        c = c.loc[score, score]
    labels = [compact_label(x) for x in c.columns]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(c.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_top_corr_pairs(top_pairs, title, filename, top_n=20):
    if top_pairs is None or top_pairs.empty:
        return
    sub = top_pairs.head(top_n).copy().sort_values("abs_corr")
    labels = [f"{compact_label(a)} | {compact_label(b)}" for a, b in zip(sub["feature_1"], sub["feature_2"])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(labels, sub["corr"])
    ax.axvline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Correlation")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_pca(scores, explained):
    if scores is None or scores.empty or "PC1" not in scores.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    date_col = "date"
    dates = pd.to_datetime(scores[date_col])
    ax.plot(dates, scores["PC1"], label="PC1 score")
    ax.axhline(0, linewidth=1)
    ax.axvline(pd.Timestamp("2018-04-01"), linestyle="--", linewidth=1)
    title = "First principal component of living-cost growth features"
    if explained is not None and not explained.empty:
        var = explained.loc[explained["component"].eq("PC1"), "explained_variance_ratio"]
        if len(var):
            title += f" ({var.iloc[0]*100:.1f}% variance)"
    ax.set_title(title)
    ax.set_ylabel("PC1 score")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "08_living_cost_growth_pca_pc1_trend.png")


def plot_homeless_living_england(eng):
    if eng.empty:
        return
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(eng["quarter_date"], eng["homeless_yoy_pct"], label="Homelessness YoY %")
    ax1.axhline(0, linewidth=1)
    ax1.set_ylabel("Homelessness YoY %")
    ax2 = ax1.twinx()
    if "living_cost_growth_pressure_index" in eng.columns:
        ax2.plot(eng["quarter_date"], eng["living_cost_growth_pressure_index"], linestyle="--", label="Living-cost growth pressure")
        ax2.set_ylabel("Living-cost growth pressure index")
    ax1.axvline(pd.Timestamp("2018-04-01"), linestyle="--", linewidth=1)
    ax1.set_title("England homelessness growth vs living-cost growth pressure")
    ax1.set_xlabel("Quarter")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "09_england_homelessness_vs_living_cost_growth.png")


def plot_homeless_living_scatter(eng):
    if "living_cost_growth_pressure_index" not in eng.columns:
        return
    for period in ["pre_2018_HRA", "post_2018_HRA"]:
        sub = eng[eng["policy_period"].eq(period)][["living_cost_growth_pressure_index", "yoy_log_homeless"]].dropna()
        if len(sub) < 6:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["living_cost_growth_pressure_index"], sub["yoy_log_homeless"])
        b = np.polyfit(sub["living_cost_growth_pressure_index"], sub["yoy_log_homeless"], 1)
        xs = np.linspace(sub["living_cost_growth_pressure_index"].min(), sub["living_cost_growth_pressure_index"].max(), 100)
        ax.plot(xs, b[0] * xs + b[1], linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
        ax.set_title(f"England: living-cost growth pressure vs homelessness YoY-log change, {period}")
        ax.set_xlabel("Living-cost growth pressure index")
        ax.set_ylabel("Homelessness YoY-log change")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_figure(fig, f"10_scatter_england_living_cost_growth_vs_homeless_yoy_{period}.png")


def plot_lag_correlations(eng_corr, lad_corr):
    if eng_corr is None or eng_corr.empty:
        return
    dep = "yoy_log_homeless"
    var = "living_cost_growth_pressure_index"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, corr, label in [
        (axes[0], eng_corr, "England"),
        (axes[1], lad_corr, "LAD pooled"),
    ]:
        for period in ["pre_2018_HRA", "post_2018_HRA"]:
            sub = corr[
                corr["dependent_variable"].eq(dep)
                & corr["living_cost_variable"].eq(var)
                & corr["period"].eq(period)
            ].sort_values("lag_quarters")
            if not sub.empty:
                ax.plot(sub["lag_quarters"], sub["corr"], marker="o", label=period)
        ax.axhline(0, linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Living-cost index lag, quarters")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Correlation with homelessness YoY-log change")
    fig.suptitle("Lag relationship: living-cost growth pressure vs homelessness")
    fig.tight_layout()
    save_figure(fig, "11_living_cost_homelessness_lag_correlations.png")


def plot_lad_distribution(lad_merged):
    if "living_cost_growth_pressure_index" not in lad_merged.columns:
        return
    sub = lad_merged[lad_merged["policy_period"].eq("post_2018_HRA")].copy()
    sub = sub[["quarter", "living_cost_growth_pressure_index", "homeless_yoy_pct"]].dropna()
    if sub.empty:
        return
    # Distribution by quarter across LADs.
    dist = sub.groupby("quarter").agg(
        p10=("living_cost_growth_pressure_index", lambda x: np.nanpercentile(x, 10)),
        median=("living_cost_growth_pressure_index", "median"),
        p90=("living_cost_growth_pressure_index", lambda x: np.nanpercentile(x, 90)),
    ).reset_index()
    dist["quarter_date"] = dist["quarter"].dt.to_timestamp(how="start")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dist["quarter_date"], dist["median"], label="median LAD")
    ax.plot(dist["quarter_date"], dist["p10"], linestyle="--", label="10th percentile")
    ax.plot(dist["quarter_date"], dist["p90"], linestyle="--", label="90th percentile")
    ax.axhline(0, linewidth=1)
    ax.set_title("Post-HRA LAD distribution of living-cost growth pressure")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Living-cost growth pressure index")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "12_lad_living_cost_growth_pressure_distribution_post_hra.png")


def make_plots(eng_monthly, relation_results, eng_merged, lad_merged, eng_corr, lad_corr):
    print_section("Creating visualizations")
    plot_living_cost_index(eng_monthly)
    plot_normalized_core_trends(eng_monthly)
    plot_yoy_trends(eng_monthly)
    plot_affordability(eng_monthly)

    plot_corr_heatmap(
        relation_results["eng_level_corr"],
        "England living-cost feature correlations: levels",
        "05_england_living_cost_level_correlation_heatmap.png",
    )
    plot_corr_heatmap(
        relation_results["eng_yoy_corr"],
        "England living-cost feature correlations: annual changes / YoY",
        "06_england_living_cost_yoy_correlation_heatmap.png",
    )
    plot_top_corr_pairs(
        relation_results["eng_level_top"],
        "Top England living-cost feature correlations: levels",
        "07_top_england_living_cost_level_correlation_pairs.png",
    )
    plot_pca(relation_results["pca_scores"], relation_results["pca_explained"])

    plot_homeless_living_england(eng_merged)
    plot_homeless_living_scatter(eng_merged)
    plot_lag_correlations(eng_corr, lad_corr)
    plot_lad_distribution(lad_merged)

    plot_corr_heatmap(
        relation_results["lad_yoy_corr"],
        "LAD pooled living-cost feature correlations: annual changes / YoY",
        "13_lad_living_cost_yoy_correlation_heatmap.png",
    )
    plot_corr_heatmap(
        relation_results["lad_within_corr"],
        "LAD within-area living-cost feature correlations: levels de-meaned by LAD",
        "14_lad_within_living_cost_level_correlation_heatmap.png",
    )

    if SAVE_FIGURES:
        print(f"Saved figures to: {OUTPUT_DIR}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


def trend_start_end_summary(eng_monthly):
    print_section("Living-cost trend start/end summary")
    cols = [
        "cpi_total",
        "average_house_price",
        "average_private_rental_price",
        "income",
        "real_income",
        "house_price_to_income",
        "annual_rent_to_income",
        "uk_bank_rate",
        "brent_oil_price",
        "unemployment_per_1000",
        "living_cost_pressure_index",
        "living_cost_growth_pressure_index",
    ]
    rows = []
    for c in cols:
        if c not in eng_monthly.columns:
            continue
        s = eng_monthly[["date", c]].dropna()
        if s.empty:
            continue
        first = s.iloc[0]
        last = s.iloc[-1]
        rows.append(
            {
                "feature": c,
                "first_date": first["date"],
                "first_value": first[c],
                "last_date": last["date"],
                "last_value": last[c],
                "absolute_change": last[c] - first[c],
                "pct_change": (last[c] / first[c] - 1) * 100 if first[c] != 0 else np.nan,
                "n_months_non_missing": len(s),
            }
        )
    out = pd.DataFrame(rows)
    safe_to_csv(out, "25_england_living_cost_start_end_trend_summary.csv")
    print(out.to_string(index=False))
    return out


def period_summary(eng_monthly):
    print_section("Living-cost period summary")
    cols = [
        "cpi_total_yoy_pct",
        "average_house_price_yoy_pct",
        "average_private_rental_price_yoy_pct",
        "income_yoy_pct",
        "real_income_yoy_pct",
        "annual_rent_to_income_yoy_pct",
        "house_price_to_income_yoy_pct",
        "uk_bank_rate",
        "brent_oil_price_yoy_pct",
        "unemployment_per_1000",
        "living_cost_pressure_index",
        "living_cost_growth_pressure_index",
    ]
    rows = []
    for period, sub in eng_monthly.groupby("policy_period"):
        for c in cols:
            if c in sub.columns:
                rows.append(
                    {
                        "period": period,
                        "feature": c,
                        "n": sub[c].notna().sum(),
                        "mean": sub[c].mean(skipna=True),
                        "median": sub[c].median(skipna=True),
                        "std": sub[c].std(skipna=True),
                        "min": sub[c].min(skipna=True),
                        "max": sub[c].max(skipna=True),
                    }
                )
    out = pd.DataFrame(rows)
    safe_to_csv(out, "26_england_living_cost_pre_post_2018_period_summary.csv")
    print(out.to_string(index=False))
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = read_monthly_panel()
    coverage = build_feature_coverage(monthly)

    eng_monthly = aggregate_england_living_cost(monthly)
    trend_summary = trend_start_end_summary(eng_monthly)
    pre_post_summary = period_summary(eng_monthly)
    eng_quarterly_living = quarterly_england_living_cost(eng_monthly)

    lad_monthly = build_lad_living_cost_panel(monthly)
    lad_quarterly_living = quarterly_lad_living_cost(lad_monthly)

    relation_results = analyze_living_cost_feature_relationships(eng_monthly, lad_monthly)

    q_home = read_quarterly_homelessness()
    eng_home = build_england_homeless_from_lads(q_home)
    eng_merged = merge_england_homeless_living(eng_home, eng_quarterly_living)
    lad_merged = merge_lad_homeless_living(q_home, lad_quarterly_living)

    eng_corr, lad_corr, eng_models, lad_models = run_homeless_living_relationships(eng_merged, lad_merged)

    make_plots(eng_monthly, relation_results, eng_merged, lad_merged, eng_corr, lad_corr)

    print_section("Done")
    print(f"All tables and figures have been saved to: {OUTPUT_DIR}")
    print("Reminder: correlations and regressions here are associations, not causal estimates.")
    print("CPI was restricted to CPI total only: cpi_00_all_items.")


if __name__ == "__main__":
    main()
