from pathlib import Path
from typing import Tuple, List
import re
import pandas as pd

# =====================
# EDIT THESE 2 PATHS
# =====================
INPUT_FOLDER = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/homelessness"
OUTPUT_CSV = "data_raw/merged_homelessness.csv"

LA_CODE_PATTERN = re.compile(r"^E0[6-9]\d{6}$")
DATE_PATTERN = re.compile(r"(20\d{2})(03|06|09|12)")

QUARTER_MAP = {
    "03": "Q1",
    "06": "Q2",
    "09": "Q3",
    "12": "Q4",
}


def parse_period_from_filename(file_path: Path) -> Tuple[str, str]:
    match = DATE_PATTERN.search(file_path.stem)
    if not match:
        raise ValueError("Could not find YYYYMM in filename: {}".format(file_path.name))

    year, month = match.groups()
    quarter = QUARTER_MAP[month]
    return year, quarter


def read_a1_sheet(file_path: Path) -> pd.DataFrame:
    """
    Find and read the correct A1 sheet (robust to naming issues).
    """
    suffix = file_path.suffix.lower()

    if suffix == ".ods":
        excel_file = pd.ExcelFile(file_path, engine="odf")
    else:
        excel_file = pd.ExcelFile(file_path)

    # Get all sheet names
    sheet_names = excel_file.sheet_names

    # Try to find a sheet that STARTS with "A1"
    target_sheet = None
    for name in sheet_names:
        clean_name = name.strip().lower()
        if clean_name.startswith("a1"):
            target_sheet = name
            break

    if target_sheet is None:
        raise ValueError(
            f"{file_path.name}: No sheet starting with 'A1' found. Sheets = {sheet_names}"
        )

    return pd.read_excel(excel_file, sheet_name=target_sheet, header=None)


def extract_initial_assessments(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    if df.shape[1] < 5:
        raise ValueError("{}: expected at least 5 columns, found {}".format(file_path.name, df.shape[1]))

    out = df.iloc[:, [0, 1, 4]].copy()
    out.columns = ["local_authority_code", "local_authority", "households_assessed"]

    out["local_authority_code"] = out["local_authority_code"].astype(str).str.strip()
    out["local_authority"] = out["local_authority"].astype(str).str.strip()

    out = out[out["local_authority_code"].str.match(LA_CODE_PATTERN, na=False)].copy()

    out["households_assessed"] = pd.to_numeric(out["households_assessed"], errors="coerce")

    out = out.dropna(subset=["households_assessed"])
    out = out[out["local_authority"].notna()]
    out = out[out["local_authority"] != ""]
    out = out[out["local_authority"].str.lower() != "nan"]

    year, quarter = parse_period_from_filename(file_path)
    out["year"] = year
    out["quarter"] = quarter
    out["source_file"] = file_path.name

    return out[
        [
            "year",
            "quarter",
            "local_authority_code",
            "local_authority",
            "households_assessed",
            "source_file",
        ]
    ].reset_index(drop=True)


def find_files(folder: Path) -> List[Path]:
    files = []
    for pattern in ("*.xlsx", "*.xls", "*.ods"):
        files.extend(folder.glob(pattern))
    return sorted(files)


def main():
    input_folder = Path(INPUT_FOLDER)
    files = find_files(input_folder)

    if not files:
        raise FileNotFoundError("No .xlsx/.xls/.ods files found in {}".format(INPUT_FOLDER))

    merged = []
    failures = []

    for file_path in files:
        try:
            df = read_a1_sheet(file_path)
            clean = extract_initial_assessments(df, file_path)
            merged.append(clean)
            print("OK   {:<40} {:>4} rows".format(file_path.name, len(clean)))
        except Exception as e:
            failures.append((file_path.name, str(e)))
            print("FAIL {}: {}".format(file_path.name, e))

    if not merged:
        raise RuntimeError("No files were successfully processed.")

    final = pd.concat(merged, ignore_index=True)

    final = final.drop_duplicates(
        subset=["year", "quarter", "local_authority_code", "households_assessed", "source_file"]
    )

    final["year_num"] = pd.to_numeric(final["year"], errors="coerce")
    final["quarter_num"] = final["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})

    final = final.sort_values(
        ["year_num", "quarter_num", "local_authority"]
    ).drop(columns=["year_num", "quarter_num"])

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    print("\n====================")
    print("Saved merged CSV to:\n{}".format(output_path))
    print("Total rows: {:,}".format(len(final)))
    print("Files processed successfully: {}".format(len(merged)))
    print("Files failed: {}".format(len(failures)))

    if failures:
        print("\nFailed files:")
        for name, err in failures:
            print("- {}: {}".format(name, err))

    print("\nPreview:")
    print(final.head(12).to_string(index=False))


if __name__ == "__main__":
    main()