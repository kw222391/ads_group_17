from pathlib import Path
import json
import re

import pandas as pd
import requests


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data_new")
SOURCE_URL = (
    "https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/"
    "datasets/consumerpriceindices/current/mm23.csv"
)
SOURCE_PATH = DATA_DIR / "cpi_data.csv"
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "cpi_monthly_data.csv"
DICT_OUTPUT_PATH = OUTPUT_DIR / "cpi_dict.json"


def download_file(url: str, output_path: Path) -> None:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download CPI data from {url}. "
            f"Please check your network connection and try again."
        ) from exc
    output_path.write_bytes(response.content)


def load_source_data(source_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        download_file(SOURCE_URL, source_path)
    return pd.read_csv(source_path, low_memory=False)


def is_monthly_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.fullmatch(r"\d{4} [A-Z]{3}", value.strip()))


def clean_column_name(column_name: str) -> str:
    cleaned_name = column_name[10:-9]
    cleaned_name = cleaned_name.replace(":", " ").replace(",", ", ")
    cleaned_name = " ".join(cleaned_name.split())
    return cleaned_name


DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the CPI source file from data, or download it first if it is missing.
cpi_data = load_source_data(SOURCE_PATH)
if "Title" not in cpi_data.columns:
    raise ValueError("The CPI source file does not contain a Title column.")
cpi_data = cpi_data.rename(columns={"Title": "Date"})

# Keep the CPI index series and the source date column.
cpi_index_cols = cpi_data.columns[cpi_data.columns.str.contains("CPI INDEX ")]
cpi_index_sorted_cols = ["Date"]
for dot_count in range(4):
    matched_cols = sorted(
        name for name in cpi_index_cols if name[:21].count(".") == dot_count
    )
    cpi_index_sorted_cols.extend(matched_cols)
cpi_index_data = cpi_data[cpi_index_sorted_cols].copy()

# Keep only monthly rows before converting the dates.
cpi_index_data = cpi_index_data[cpi_index_data["Date"].map(is_monthly_date)].copy()
if cpi_index_data.empty:
    raise ValueError("No monthly CPI rows were found in the source file.")
cpi_index_data = cpi_index_data.dropna(subset=cpi_index_data.columns[1:], how="all")
cpi_index_data["Date"] = pd.to_datetime(cpi_index_data["Date"], format="%Y %b")
cpi_index_data = cpi_index_data.sort_values("Date").reset_index(drop=True)

# Clean the CPI series names and build a code-to-description dictionary.
cpi_index_data.columns = ["Date", *[clean_column_name(col) for col in cpi_index_data.columns[1:]]]
split_columns = cpi_index_data.columns[1:].str.split(" ", n=1, expand=True)
column_codes = []
cpi_dict = {}
for code, description in split_columns:
    column_codes.append(code)
    cpi_dict[code] = description.lower() if isinstance(description, str) else ""

cpi_index_data.columns = ["date", *column_codes]

# Save the monthly CPI dataset and the CPI series dictionary to data_new.
cpi_index_data.to_csv(MONTHLY_OUTPUT_PATH, index=False)
with open(DICT_OUTPUT_PATH, "w") as output_file:
    json.dump(cpi_dict, output_file, indent=2)

print(cpi_index_data.head())
print(cpi_index_data.tail())
print(f"Saved: {MONTHLY_OUTPUT_PATH}")
print(f"Saved: {DICT_OUTPUT_PATH}")
