from pathlib import Path
import re
import pandas as pd

DATA_DIR_0916 = Path(r"D:\UOB\ads_group_17\ads_group_17\data\Homelessness 09-16")
FILE_1418 = Path(r"D:\UOB\ads_group_17\ads_group_17\data\Homelessness 14-18.xlsx")
FILE_1825 = Path(r"D:\UOB\ads_group_17\ads_group_17\data\Homelessness 18-25.xlsx")
OUTPUT = Path(r"D:\UOB\ads_group_17\ads_group_17\data\homelessness_integrated_09_25_zhou.csv")

KEEP = [
    "LAD_code",
    "LA_name",
    "Year",
    "Quarter",
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]

PERIOD_TO_Q = {
    "january to march": "Q1",
    "april to june": "Q2",
    "july to september": "Q3",
    "october to december": "Q4",
}

Q_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}

TOTAL_SHEET = "Section 1"
TOTAL_CODES = ["e16w", "e16g"]

RELIEF_SHEET = "Section 10"
RELIEF_CODES = ["e101b"]


def check_packages(paths):
    paths = [p for p in paths if p is not None and p.exists()]
    has_xlsx = any(p.suffix.lower() in {".xlsx", ".xlsm"} for p in paths)
    has_xls = any(p.suffix.lower() == ".xls" for p in paths)

    if has_xlsx:
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "Missing openpyxl. Please run:\n"
                r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\python.exe -m pip install openpyxl"
            )

    if has_xls:
        try:
            import xlrd
        except ImportError:
            raise ImportError(
                "Missing xlrd, cannot read .xls files. Please run:\n"
                r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\python.exe -m pip install xlrd"
            )


def cell(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def ncell(x):
    return cell(x).lower()


def clean_text(x):
    s = ncell(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def name_key(x):
    s = ncell(x)
    s = s.replace("&", "and")
    s = s.replace("_ua", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_lad_code(x):
    return bool(re.match(r"^E0[6789]\d{6}$", cell(x).upper()))


def read_book(path):
    if path.suffix.lower() == ".xls":
        engine = "xlrd"
    else:
        engine = "openpyxl"

    return pd.ExcelFile(path, engine=engine)


def read_raw(xls, sheet_name):
    return pd.read_excel(
        xls,
        sheet_name=sheet_name,
        header=None,
        dtype=object,
    )


def parse_period_from_filename(path):
    s = path.stem.lower()

    year_match = re.search(r"(?:19|20)\d{2}", s)

    if not year_match:
        raise ValueError(f"Cannot identify year from filename: {path.name}")

    year = int(year_match.group())

    for text, quarter in PERIOD_TO_Q.items():
        if text in s:
            return year, quarter

    raise ValueError(f"Cannot identify quarter from filename: {path.name}")


def parse_period_from_text(text):
    s = clean_text(text)

    match = re.search(
        r"(january to march|april to june|july to september|october to december)\s+((?:19|20)\d{2})",
        s,
    )

    if not match:
        return None

    quarter_text = match.group(1)
    year = int(match.group(2))

    return year, PERIOD_TO_Q[quarter_text]


def to_count(series):
    def fix_value(x):
        if pd.isna(x):
            return pd.NA

        if isinstance(x, str):
            y = x.strip()

            if y in {"", "-", "–", "—"}:
                return 0

            y = y.replace(",", "")

            return y

        return x

    return pd.to_numeric(series.map(fix_value), errors="coerce")


def first_matching_col(header, candidates):
    candidates = [c.lower() for c in candidates]

    for c in candidates:
        if c in header:
            return header.index(c)

    return None


def find_excel_files(data_dir):
    if not data_dir.exists():
        return []

    files = []

    for p in data_dir.iterdir():
        if not p.is_file():
            continue

        if p.name.startswith("~$"):
            continue

        if p.suffix.lower() not in {".xls", ".xlsx", ".xlsm"}:
            continue

        if p.name.lower().startswith("homelessness 14-18"):
            continue

        if p.name.lower().startswith("homelessness 18-25"):
            continue

        files.append(p)

    return sorted(files, key=lambda x: x.name.lower())


def find_header(raw, target_codes):
    target_codes = [x.lower() for x in target_codes]

    id_names = {
        "ons code",
        "onscode",
        "lad code",
        "lad_code",
        "la code",
        "lacode",
    }

    name_names = {
        "local authority",
        "local authority name",
        "la name",
        "la_name",
        "laname",
    }

    for i, row in raw.iterrows():
        vals = [ncell(v) for v in row.tolist()]

        has_id = any(v in id_names for v in vals)
        has_name = any(v in name_names for v in vals)
        has_target = any(code in vals for code in target_codes)

        if has_id and has_name and has_target:
            return i

    return None


def build_lad_lookup(files):
    dclg_to_lad = {}
    name_to_lad = {}

    for path in files:
        try:
            xls = read_book(path)
        except Exception as e:
            print(f"Read failed, skipping lookup: {path.name} | {e}")
            continue

        if "LA List" not in xls.sheet_names:
            continue

        try:
            raw = read_raw(xls, "LA List")
        except Exception as e:
            print(f"Failed to read LA List, skipping: {path.name} | {e}")
            continue

        for i, row in raw.iterrows():
            vals = [ncell(v) for v in row.tolist()]

            if "dclg code" in vals and "ons code" in vals and "la name" in vals:
                dclg_i = vals.index("dclg code")
                ons_i = vals.index("ons code")
                name_i = vals.index("la name")

                part = raw.iloc[i + 1:, [dclg_i, ons_i, name_i]].copy()
                part.columns = ["DCLG_code", "LAD_code", "LA_name"]

                for _, r in part.iterrows():
                    dclg_code = cell(r["DCLG_code"]).upper()
                    lad_code = cell(r["LAD_code"]).upper()
                    la_name = cell(r["LA_name"])

                    if is_lad_code(lad_code):
                        dclg_to_lad[dclg_code] = lad_code
                        name_to_lad[name_key(la_name)] = lad_code

                break

    return dclg_to_lad, name_to_lad


def extract_old_count(path, sheet_name, target_codes, output_col, dclg_to_lad, name_to_lad):
    xls = read_book(path)

    if sheet_name not in xls.sheet_names:
        return pd.DataFrame(columns=["LAD_code", "LA_name", output_col])

    raw = read_raw(xls, sheet_name)

    header_idx = find_header(raw, target_codes)

    if header_idx is None:
        return pd.DataFrame(columns=["LAD_code", "LA_name", output_col])

    header = [ncell(v) for v in raw.iloc[header_idx].tolist()]

    id_i = first_matching_col(
        header,
        [
            "ons code",
            "onscode",
            "lad code",
            "lad_code",
            "la code",
            "lacode",
        ],
    )

    name_i = first_matching_col(
        header,
        [
            "local authority",
            "local authority name",
            "la name",
            "la_name",
            "laname",
        ],
    )

    code_i = first_matching_col(header, target_codes)

    if id_i is None or name_i is None or code_i is None:
        return pd.DataFrame(columns=["LAD_code", "LA_name", output_col])

    out = raw.iloc[header_idx + 1:, [id_i, name_i, code_i]].copy()
    out.columns = ["source_code", "LA_name", output_col]

    out["source_code"] = out["source_code"].map(lambda x: cell(x).upper())
    out["LA_name"] = out["LA_name"].map(cell)

    source_is_lad = out["source_code"].where(out["source_code"].map(is_lad_code))
    mapped_from_source = out["source_code"].map(dclg_to_lad)
    mapped_from_name = out["LA_name"].map(lambda x: name_to_lad.get(name_key(x), pd.NA))

    out["LAD_code"] = (
        source_is_lad
        .combine_first(mapped_from_source)
        .combine_first(mapped_from_name)
    )

    out = out[out["LAD_code"].map(is_lad_code)].copy()

    out[output_col] = to_count(out[output_col])

    out = out[["LAD_code", "LA_name", output_col]]
    out = out.drop_duplicates(subset=["LAD_code"], keep="first")

    return out


def process_old_file(path, dclg_to_lad, name_to_lad):
    year, quarter = parse_period_from_filename(path)

    total = extract_old_count(
        path=path,
        sheet_name=TOTAL_SHEET,
        target_codes=TOTAL_CODES,
        output_col="Total_assessments",
        dclg_to_lad=dclg_to_lad,
        name_to_lad=name_to_lad,
    )

    relief = extract_old_count(
        path=path,
        sheet_name=RELIEF_SHEET,
        target_codes=RELIEF_CODES,
        output_col="Homeless_relief",
        dclg_to_lad=dclg_to_lad,
        name_to_lad=name_to_lad,
    )

    if total.empty and relief.empty:
        return pd.DataFrame(columns=KEEP)

    df = total.merge(
        relief,
        on="LAD_code",
        how="outer",
        suffixes=("_total", "_relief"),
    )

    if "LA_name_total" in df.columns and "LA_name_relief" in df.columns:
        df["LA_name"] = df["LA_name_total"].combine_first(df["LA_name_relief"])
    elif "LA_name_total" in df.columns:
        df["LA_name"] = df["LA_name_total"]
    elif "LA_name_relief" in df.columns:
        df["LA_name"] = df["LA_name_relief"]
    elif "LA_name" not in df.columns:
        df["LA_name"] = pd.NA

    df["Year"] = year
    df["Quarter"] = quarter

    df["Homeless_relief"] = pd.to_numeric(
        df.get("Homeless_relief"),
        errors="coerce",
    )

    df["Total_assessments"] = pd.to_numeric(
        df.get("Total_assessments"),
        errors="coerce",
    )

    df["Homeless_per_1000"] = (
        df["Homeless_relief"]
        .div(df["Total_assessments"])
        .mul(1000)
        .where(df["Total_assessments"].gt(0))
    )

    return df.reindex(columns=KEEP)


def find_1418_header(raw):
    for i, row in raw.iterrows():
        vals = [clean_text(v) for v in row.tolist()]

        has_id = "ons code" in vals
        has_name = any("local authority" in v for v in vals)
        has_total_decisions = "total decisions" in vals

        if has_id and has_name and has_total_decisions:
            return i

    return None


def read_1418_file(path):
    if not path.exists():
        print(f"14-18 file not found, skipping: {path}")
        return pd.DataFrame(columns=KEEP)

    xls = read_book(path)
    frames = []

    for sheet_name in xls.sheet_names:
        raw = read_raw(xls, sheet_name)

        period = None

        for r in range(min(5, len(raw))):
            for c in range(min(3, raw.shape[1])):
                period = parse_period_from_text(raw.iat[r, c])
                if period:
                    break

            if period:
                break

        if not period:
            continue

        header_idx = find_1418_header(raw)

        if header_idx is None:
            continue

        header = [clean_text(v) for v in raw.iloc[header_idx].tolist()]

        id_i = header.index("ons code")

        name_i = next(
            (i for i, h in enumerate(header) if "local authority" in h),
            None,
        )

        total_assess_i = header.index("total decisions") if "total decisions" in header else None

        relief_i = None

        for i, h in enumerate(header[:-1]):
            if h == "total" and "number per" in header[i + 1]:
                relief_i = i
                break

        if name_i is None or total_assess_i is None or relief_i is None:
            print(f"14-18 skip sheet, missing key columns: {sheet_name}")
            continue

        part = raw.iloc[header_idx + 1:, [id_i, name_i, relief_i, total_assess_i]].copy()
        part.columns = [
            "LAD_code",
            "LA_name",
            "Homeless_relief",
            "Total_assessments",
        ]

        part["LAD_code"] = part["LAD_code"].map(lambda x: cell(x).upper())
        part = part[part["LAD_code"].map(is_lad_code)].copy()

        if part.empty:
            continue

        part["LA_name"] = part["LA_name"].map(cell)
        part["Homeless_relief"] = to_count(part["Homeless_relief"])
        part["Total_assessments"] = to_count(part["Total_assessments"])

        year, quarter = period

        part["Year"] = year
        part["Quarter"] = quarter

        part["Homeless_per_1000"] = (
            part["Homeless_relief"]
            .div(part["Total_assessments"])
            .mul(1000)
            .where(part["Total_assessments"].gt(0))
        )

        part = part.reindex(columns=KEEP)

        frames.append(part)

        print(f"parsed 14-18: {sheet_name} | {year} {quarter} | rows={len(part)}")

    if not frames:
        return pd.DataFrame(columns=KEEP)

    return pd.concat(frames, ignore_index=True)


def read_1825_file(path):
    if not path.exists():
        print(f"18-25 file not found, skipping: {path}")
        return pd.DataFrame(columns=KEEP)

    xls = read_book(path)
    frames = []

    for sheet_name in xls.sheet_names:
        raw = read_raw(xls, sheet_name)

        period = None

        for r in range(min(5, len(raw))):
            for c in range(min(3, raw.shape[1])):
                period = parse_period_from_text(raw.iat[r, c])
                if period:
                    break

            if period:
                break

        if not period:
            continue

        if raw.shape[1] < 10:
            continue

        part = raw.iloc[:, [0, 1, 4, 9]].copy()
        part.columns = [
            "LAD_code",
            "LA_name",
            "Total_assessments",
            "Homeless_relief",
        ]

        part["LAD_code"] = part["LAD_code"].map(lambda x: cell(x).upper())
        part = part[part["LAD_code"].map(is_lad_code)].copy()

        if part.empty:
            continue

        part["LA_name"] = part["LA_name"].map(cell)
        part["Total_assessments"] = to_count(part["Total_assessments"])
        part["Homeless_relief"] = to_count(part["Homeless_relief"])

        year, quarter = period

        part["Year"] = year
        part["Quarter"] = quarter

        part["Homeless_per_1000"] = (
            part["Homeless_relief"]
            .div(part["Total_assessments"])
            .mul(1000)
            .where(part["Total_assessments"].gt(0))
        )

        part = part.reindex(columns=KEEP)

        frames.append(part)

        print(f"parsed 18-25: {sheet_name} | {year} {quarter} | rows={len(part)}")

    if not frames:
        return pd.DataFrame(columns=KEEP)

    return pd.concat(frames, ignore_index=True)


def main():
    old_files = find_excel_files(DATA_DIR_0916)

    paths_for_check = old_files.copy()

    if FILE_1418.exists():
        paths_for_check.append(FILE_1418)

    if FILE_1825.exists():
        paths_for_check.append(FILE_1825)

    check_packages(paths_for_check)

    frames = []
    skipped = []

    print(f"09-16 folder = {DATA_DIR_0916}")
    print(f"09-16 excel files found = {len(old_files)}")

    if old_files:
        dclg_to_lad, name_to_lad = build_lad_lookup(old_files)

        for path in old_files:
            try:
                df = process_old_file(path, dclg_to_lad, name_to_lad)

                if df.empty:
                    skipped.append(f"{path.name}: no usable data parsed")
                else:
                    df["_priority"] = 1
                    frames.append(df)
                    print(f"parsed 09-16: {path.name} | rows={len(df)}")

            except Exception as e:
                skipped.append(f"{path.name}: {e}")

    df_1418 = read_1418_file(FILE_1418)

    if not df_1418.empty:
        df_1418["_priority"] = 2
        frames.append(df_1418)

    df_1825 = read_1825_file(FILE_1825)

    if not df_1825.empty:
        df_1825["_priority"] = 3
        frames.append(df_1825)

    if not frames:
        raise RuntimeError("No usable data found.")

    out = pd.concat(frames, ignore_index=True)

    out = out.dropna(subset=["LAD_code", "Year", "Quarter"]).copy()

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["_q"] = out["Quarter"].map(Q_ORDER)

    before_dedup = len(out)

    out = (
        out.sort_values(["Year", "_q", "LAD_code", "_priority"])
        .drop_duplicates(subset=["LAD_code", "Year", "Quarter"], keep="first")
        .sort_values(["Year", "_q", "LAD_code"])
        .drop(columns=["_q", "_priority"], errors="ignore")
        .reset_index(drop=True)
    )

    after_dedup = len(out)

    out = out.reindex(columns=KEEP)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone")
    print(f"rows = {len(out)}")
    print(f"duplicates_removed = {before_dedup - after_dedup}")
    print(f"output = {OUTPUT}")

    if skipped:
        print("\nThe following files were skipped or failed to parse:")
        for item in skipped:
            print("  -", item)


if __name__ == "__main__":
    main()