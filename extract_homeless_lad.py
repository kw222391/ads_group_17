from pathlib import Path
import argparse
import csv
import re
import traceback

import pandas as pd


OUTPUT_COLUMNS = [
    "LAD_code",
    "LA_name",
    "Year",
    "Quarter",
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]

VALUE_COLUMNS = [
    "Homeless_relief",
    "Total_assessments",
    "Homeless_per_1000",
]

LAD_RE = re.compile(r"^E0[6-9]\d{6}$")
YEAR_RE = re.compile(r"\b(20\d{2})\b")

QUARTER_PATTERNS = [
    (
        re.compile(
            r"\bq\s*1\b|\bquarter\s*1\b|jan\w*\s*(?:-|to|–|—|_|\s)\s*mar\w*|january\s*(?:-|to|–|—|_|\s)\s*march",
            re.I,
        ),
        "Q1",
    ),
    (
        re.compile(
            r"\bq\s*2\b|\bquarter\s*2\b|apr\w*\s*(?:-|to|–|—|_|\s)\s*jun\w*|april\s*(?:-|to|–|—|_|\s)\s*june",
            re.I,
        ),
        "Q2",
    ),
    (
        re.compile(
            r"\bq\s*3\b|\bquarter\s*3\b|jul\w*\s*(?:-|to|–|—|_|\s)\s*sep\w*|july\s*(?:-|to|–|—|_|\s)\s*september",
            re.I,
        ),
        "Q3",
    ),
    (
        re.compile(
            r"\bq\s*4\b|\bquarter\s*4\b|oct\w*\s*(?:-|to|–|—|_|\s)\s*dec\w*|october\s*(?:-|to|–|—|_|\s)\s*december",
            re.I,
        ),
        "Q4",
    ),
]

SHEET_QUARTERS = {
    "mar": "Q1",
    "march": "Q1",
    "jun": "Q2",
    "june": "Q2",
    "sep": "Q3",
    "sept": "Q3",
    "september": "Q3",
    "dec": "Q4",
    "december": "Q4",
}

BAD_AREA_NAMES = {
    "", "nan", "none", "total", "england", "wales", "scotland", "great britain",
    "united kingdom", "local authority", "area name", "la name", "all authorities",
    "north east", "north west", "yorkshire and the humber", "east midlands",
    "west midlands", "east of england", "london", "south east", "south west",
}


def norm_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("–", "-").replace("—", "-").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def compact_text(x):
    return re.sub(r"[^a-z0-9]", "", norm_text(x))


def clean_code(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_value(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    if isinstance(x, str):
        s = x.strip()
        if s in {"", "-", "--", "—", "–", "..", ".", "*", "x", "X", "n/a", "N/A", "na", "NA"}:
            return ""

        s_num = s.replace(",", "").replace("%", "").strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", s_num):
            f = float(s_num)
            return int(f) if f.is_integer() else f

        return s

    if isinstance(x, float):
        return int(x) if x.is_integer() else x

    return x


def to_number_or_none(x):
    x = clean_value(x)
    if x == "":
        return None

    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)

    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def same_value(a, b):
    a_clean = clean_value(a)
    b_clean = clean_value(b)

    if a_clean == "" and b_clean == "":
        return True

    a_num = to_number_or_none(a_clean)
    b_num = to_number_or_none(b_clean)

    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) < 1e-9

    return norm_text(a_clean) == norm_text(b_clean)


def parse_period(*texts):
    text = " ".join(str(t) for t in texts if t is not None)
    text = text.replace("_", " ")
    text = text.replace(" - ", " to ").replace("-", " to ")

    year_match = YEAR_RE.search(text)
    year = int(year_match.group(1)) if year_match else None

    quarter = None

    for pattern, q in QUARTER_PATTERNS:
        if pattern.search(text):
            quarter = q
            break

    if quarter is None:
        m = re.search(
            r"(mar|march|jun|june|sep|sept|september|dec|december)[\s-]*(20\d{2})",
            text,
            re.I,
        )
        if m:
            quarter = SHEET_QUARTERS[m.group(1).lower()]
            year = year or int(m.group(2))

    if quarter is None:
        m = re.search(r"(20\d{2})\s*q\s*([1-4])", text, re.I)
        if m:
            year = year or int(m.group(1))
            quarter = f"Q{m.group(2)}"

    if quarter is None:
        m = re.search(r"q\s*([1-4])\s*(20\d{2})", text, re.I)
        if m:
            quarter = f"Q{m.group(1)}"
            year = year or int(m.group(2))

    if year is None or quarter is None:
        return None, None

    return year, quarter


def read_sheet(path, sheet_name):
    return pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        dtype=object,
    )


def get_sheets(path):
    return pd.ExcelFile(path).sheet_names


def combined_headers(df, start_row=0, end_row=8):
    headers = {}
    if df.empty:
        return headers

    end_row = min(end_row, df.shape[0])

    for c in range(df.shape[1]):
        parts = []
        for r in range(start_row, end_row):
            v = df.iat[r, c]
            if norm_text(v):
                parts.append(str(v))
        headers[c] = norm_text(" ".join(parts))

    return headers


def find_col(headers, must_have, must_not_have=None):
    must_not_have = must_not_have or []

    candidates = []
    for c, h in headers.items():
        if all(x in h for x in must_have) and not any(x in h for x in must_not_have):
            candidates.append(c)

    return min(candidates) if candidates else None


def find_lad_code_col(df, max_rows=120):
    max_rows = min(max_rows, df.shape[0])

    for c in range(df.shape[1]):
        hits = 0
        for r in range(max_rows):
            code = clean_code(df.iat[r, c])
            if LAD_RE.match(code):
                hits += 1

        if hits >= 2:
            return c

    return None


def find_code_name_header(df, max_rows=100):
    max_rows = min(max_rows, df.shape[0])

    for r in range(max_rows):
        row = [norm_text(df.iat[r, c]) for c in range(df.shape[1])]

        code_cols = []
        name_cols = []

        for c, h in enumerate(row):
            if (
                h in {"ons code", "lad code", "la code", "area code", "code"}
                or "ons code" in h
                or "gss code" in h
                or ("local authority" in h and "code" in h)
                or ("authority" in h and "code" in h)
                or ("area" in h and "code" in h)
                or ("district" in h and "code" in h)
            ):
                code_cols.append(c)

            if (
                h in {"local authority", "la name", "area name", "authority name", "name"}
                or "local authority name" in h
                or ("local authority" in h and "code" not in h)
                or ("authority" in h and "name" in h)
                or ("area" in h and "name" in h)
            ):
                name_cols.append(c)

        if code_cols:
            code_col = code_cols[0]
            name_col = name_cols[0] if name_cols else code_col + 1
            return r, code_col, name_col

    return None, None, None



def find_area_name_col(df, value_col=None, max_rows=350):
    """
    Fallback for legacy .xls files where there is no modern E06/E07/E08/E09 LAD code
    and the header does not clearly say "ONS code". We identify the column that looks
    most like local-authority names, preferably on rows that also have a non-empty value
    in the chosen total column.
    """
    max_rows = min(max_rows, df.shape[0])
    best_col = None
    best_score = -1

    bad_fragments = [
        "table", "section", "quarter", "quarterly", "homeless", "household",
        "decision", "total", "applications", "eligible", "intentionally",
        "priority", "need", "accepted", "not accepted", "code", "notes",
        "source", "contents", "return", "authority code",
    ]

    for c in range(df.shape[1]):
        score = 0
        unique_names = set()

        for r in range(max_rows):
            raw_name = df.iat[r, c]
            name = norm_text(raw_name)

            if not is_plausible_la_name(name):
                continue
            if any(fragment in name for fragment in bad_fragments):
                continue
            if len(name.split()) > 8:
                continue

            if value_col is not None and value_col < df.shape[1]:
                v = clean_value(df.iat[r, value_col])
                # Prefer rows that actually have a total value, but allow zero.
                if v == "":
                    continue

            unique_names.add(name)

        score = len(unique_names)

        # Local authority name columns usually have many unique names.
        if score > best_score:
            best_score = score
            best_col = c

    # Require enough rows to avoid accidentally picking a title/notes column.
    if best_score >= 20:
        return best_col

    return None


def is_plausible_la_name(name):
    n = norm_text(name)
    if n in BAD_AREA_NAMES:
        return False
    if len(n) < 3:
        return False
    if re.fullmatch(r"[0-9., -]+", n):
        return False
    return True


def iter_lad_rows(df, code_col, name_col):
    for r in range(df.shape[0]):
        code = clean_code(df.iat[r, code_col] if code_col < df.shape[1] else "")

        if not LAD_RE.match(code):
            continue

        name = ""
        if name_col is not None and name_col < df.shape[1]:
            name = clean_value(df.iat[r, name_col])

        yield r, code, name


def iter_old_area_rows(df, code_col, name_col, value_col=None, start_row=0):
    """
    Old 2012 .xls files sometimes use pre-GSS ONS codes rather than E06/E07 LAD codes.
    For those rows, keep the LA name and let merge_records map the name to a LAD_code
    using other years/files. Rows with no match are dropped later rather than invented.
    """
    for r in range(max(0, start_row), df.shape[0]):
        code = clean_code(df.iat[r, code_col] if code_col is not None and code_col < df.shape[1] else "")
        name = ""
        if name_col is not None and name_col < df.shape[1]:
            name = clean_value(df.iat[r, name_col])

        if LAD_RE.match(code):
            yield r, code, name
            continue

        # Name-only fallback for legacy-code rows. Only keep rows that have some value in the target column.
        if is_plausible_la_name(name):
            if value_col is None or value_col >= df.shape[1] or clean_value(df.iat[r, value_col]) != "":
                yield r, "", name


def make_record(
    code,
    name,
    year,
    quarter,
    source,
    homeless_relief="",
    total_assessments="",
    homeless_per_1000="",
):
    return {
        "LAD_code": clean_code(code),
        "LA_name": clean_value(name),
        "Year": int(year),
        "Quarter": quarter,
        "Homeless_relief": clean_value(homeless_relief),
        "Total_assessments": clean_value(total_assessments),
        "Homeless_per_1000": clean_value(homeless_per_1000),
        "_source": source,
    }


def find_old_total_col(df, header_row=None, code_col=None, name_col=None):
    # 1) Strongest signal: P1E code e16w / e16 in any of the upper part of the sheet.
    scan_rows = min(df.shape[0], 150)
    for r in range(scan_rows):
        for c in range(df.shape[1]):
            ct = compact_text(df.iat[r, c])
            if ct in {"e16w", "e16"} or ct.startswith("e16w"):
                return c

    # 2) Header text signals around the table.
    if header_row is None:
        header_row = 0

    windows = [
        (0, min(df.shape[0], 15)),
        (max(0, header_row - 5), min(df.shape[0], header_row + 12)),
        (0, min(df.shape[0], 80)),
    ]

    for start, end in windows:
        headers = combined_headers(df, start, end)

        for c, h in headers.items():
            hc = compact_text(h)
            if "e16w" in hc or hc == "e16":
                return c

        preferred = [
            ["total decisions"],
            ["total", "decisions"],
            ["total number", "households", "assessed"],
            ["total", "households", "assessed"],
            ["total", "applications"],
        ]

        for must in preferred:
            col = find_col(headers, must)
            if col is not None:
                return col

    # 3) Conservative fallback: numeric column to the right of name/code with a header containing total.
    if code_col is not None:
        headers = combined_headers(df, 0, min(df.shape[0], 100))
        candidates = []
        for c, h in headers.items():
            if c <= code_col:
                continue
            if name_col is not None and c == name_col:
                continue
            if "total" not in h:
                continue
            if any(bad in h for bad in ["per 1,000", "per 1000", "percentage", "rate"]):
                continue
            candidates.append(c)
        if candidates:
            return min(candidates)

    return None


def extract_old_09_16(path, sheets):
    records = []
    file_year, file_quarter = parse_period(path.stem)

    section_sheets = []
    other_sheets = []

    for s in sheets:
        ns = norm_text(s)
        if ns == "section 1":
            section_sheets.insert(0, s)
        elif "section 1" in ns:
            section_sheets.append(s)
        else:
            other_sheets.append(s)

    # Old single-quarter files should use the file name for Year/Quarter first.
    # This avoids tables whose internal title mentions March being incorrectly assigned to Q1.
    candidates = section_sheets + other_sheets

    for sheet in candidates:
        df = read_sheet(path, sheet)

        year = file_year
        quarter = file_quarter

        if not year or not quarter:
            top_text = " ".join(norm_text(x) for x in df.head(12).to_numpy().ravel())
            year, quarter = parse_period(sheet, top_text)

        if not year or not quarter:
            continue

        header_row, code_col, name_col = find_code_name_header(df)

        if code_col is None:
            code_col = find_lad_code_col(df)
            if code_col is not None:
                name_col = code_col + 1
                header_row = 0

        # Try to identify the total-assessments / total-decisions column even if
        # the 2012 .xls file has no usable modern LAD code column.
        total_col = find_old_total_col(df, header_row, code_col, name_col)

        if total_col is None:
            continue

        # 2012 Q2/Q3/Q4 legacy .xls files can have old non-E LAD/ONS codes or
        # unclear headers. If no code column was found, keep name-only rows and
        # let merge_records map LA_name -> LAD_code using unambiguous names from
        # the other files. Anything not uniquely mappable is dropped later.
        if name_col is None:
            name_col = find_area_name_col(df, value_col=total_col)

        if code_col is None and name_col is None:
            continue

        sheet_records = []
        start_row = (header_row + 1) if header_row is not None else 0

        for r, code, name in iter_old_area_rows(df, code_col, name_col, value_col=total_col, start_row=start_row):
            total = df.iat[r, total_col] if total_col < df.shape[1] else ""

            sheet_records.append(
                make_record(
                    code=code,
                    name=name,
                    year=year,
                    quarter=quarter,
                    source=f"{path.name}:{sheet}",
                    total_assessments=total,
                )
            )

        if sheet_records:
            records.extend(sheet_records)
            break

    return records


def extract_14_18(path, sheets):
    records = []

    for sheet in sheets:
        if not norm_text(sheet).endswith("_qtr"):
            continue

        df = read_sheet(path, sheet)

        top_text = " ".join(norm_text(x) for x in df.head(10).to_numpy().ravel())
        year, quarter = parse_period(path.stem, sheet, top_text)

        if not year or not quarter:
            continue

        headers = combined_headers(df, 0, 8)

        code_col = find_lad_code_col(df)
        if code_col is None:
            code_col = 0

        name_col = code_col + 1

        total_col = (
            find_col(headers, ["total decisions"])
            or find_col(headers, ["total", "decisions"])
        )

        per_candidates = []
        for c, h in headers.items():
            if ("number per" in h or "per 1,000" in h or "per 1000" in h) and (
                "1000" in h or "1,000" in h
            ):
                per_candidates.append(c)

        per_col = None
        if per_candidates:
            if total_col is not None:
                before_total = [c for c in per_candidates if c < total_col]
                per_col = min(before_total) if before_total else min(per_candidates)
            else:
                per_col = min(per_candidates)

        if total_col is None and per_col is None:
            continue

        for r, code, name in iter_lad_rows(df, code_col, name_col):
            total = df.iat[r, total_col] if total_col is not None and total_col < df.shape[1] else ""
            per1000 = df.iat[r, per_col] if per_col is not None and per_col < df.shape[1] else ""

            records.append(
                make_record(
                    code=code,
                    name=name,
                    year=year,
                    quarter=quarter,
                    source=f"{path.name}:{sheet}",
                    total_assessments=total,
                    homeless_per_1000=per1000,
                )
            )

    return records


def looks_like_18_25_a1(path, sheet, df):
    top_text = " ".join(norm_text(x) for x in df.head(10).to_numpy().ravel())
    sheet_text = norm_text(sheet)
    file_text = norm_text(path.name)

    if "table a1" in top_text or "table a1" in sheet_text:
        return True

    if "18-25" in file_text or "homelessness 18" in file_text:
        if "initial assessment" in top_text or "relief duty owed" in top_text:
            return True

    return False


def extract_18_25(path, sheets):
    records = []

    for sheet in sheets:
        df = read_sheet(path, sheet)

        if not looks_like_18_25_a1(path, sheet, df):
            continue

        top_text = " ".join(norm_text(x) for x in df.head(10).to_numpy().ravel())
        year, quarter = parse_period(path.stem, sheet, top_text)

        if not year or not quarter:
            continue

        headers = combined_headers(df, 0, 9)

        code_col = find_lad_code_col(df)
        if code_col is None:
            code_col = 0

        name_col = code_col + 1

        total_col = (
            find_col(headers, ["total initial assessments"])
            or find_col(headers, ["total", "initial", "assessment"])
            or find_col(headers, ["total number", "households", "assessed"])
            or find_col(headers, ["total", "households", "assessed"])
        )

        relief_col = None
        relief_candidates = []

        for c, h in headers.items():
            if (
                "relief duty owed" in h
                and "per" not in h
                and "section 21" not in h
                and "threatened" not in h
            ):
                relief_candidates.append(c)

        if relief_candidates:
            relief_col = min(relief_candidates)

        per_col = None
        per_candidates = []

        for c, h in headers.items():
            if (
                ("assessed as homeless" in h or "homeless per" in h)
                and "per" in h
                and ("000" in h or "1,000" in h or "1000" in h)
            ):
                per_candidates.append(c)

        if per_candidates:
            per_col = min(per_candidates)

        if total_col is None and relief_col is None and per_col is None:
            continue

        for r, code, name in iter_lad_rows(df, code_col, name_col):
            total = df.iat[r, total_col] if total_col is not None and total_col < df.shape[1] else ""
            relief = df.iat[r, relief_col] if relief_col is not None and relief_col < df.shape[1] else ""
            per1000 = df.iat[r, per_col] if per_col is not None and per_col < df.shape[1] else ""

            records.append(
                make_record(
                    code=code,
                    name=name,
                    year=year,
                    quarter=quarter,
                    source=f"{path.name}:{sheet}",
                    homeless_relief=relief,
                    total_assessments=total,
                    homeless_per_1000=per1000,
                )
            )

    return records


def choose_extractors(path, sheets):
    name = norm_text(path.name)
    extractors = []

    if "14-18" in name or "homelessness 14" in name:
        extractors.append(("14_18", extract_14_18))

    if "18-25" in name or "homelessness 18" in name:
        extractors.append(("18_25", extract_18_25))

    if any(norm_text(s).endswith("_qtr") for s in sheets):
        extractors.append(("14_18", extract_14_18))

    if not extractors:
        extractors.append(("09_16", extract_old_09_16))

    seen = set()
    final = []

    for label, func in extractors:
        if label not in seen:
            final.append((label, func))
            seen.add(label)

    return final


def build_name_to_lad(records):
    mapping = {}
    for rec in records:
        code = clean_code(rec.get("LAD_code", ""))
        name = clean_value(rec.get("LA_name", ""))
        if LAD_RE.match(code) and is_plausible_la_name(name):
            mapping.setdefault(norm_text(name), set()).add(code)

    # Only use unambiguous names. If a name maps to multiple codes, leave it blank.
    return {name: next(iter(codes)) for name, codes in mapping.items() if len(codes) == 1}


def merge_records(records):
    name_to_lad = build_name_to_lad(records)

    merged = {}
    conflicts = []
    dropped_name_only = 0

    for rec in records:
        rec = rec.copy()
        code = clean_code(rec.get("LAD_code", ""))

        if not LAD_RE.match(code):
            name_key = norm_text(rec.get("LA_name", ""))
            code = name_to_lad.get(name_key, "")
            rec["LAD_code"] = code

        if not LAD_RE.match(code):
            dropped_name_only += 1
            continue

        key = (code, int(rec["Year"]), rec["Quarter"])

        if key not in merged:
            merged[key] = {col: "" for col in OUTPUT_COLUMNS}
            merged[key]["LAD_code"] = code
            merged[key]["Year"] = int(rec["Year"])
            merged[key]["Quarter"] = rec["Quarter"]

        out = merged[key]

        if out["LA_name"] == "" and rec["LA_name"] != "":
            out["LA_name"] = rec["LA_name"]

        for col in VALUE_COLUMNS:
            old_val = clean_value(out.get(col, ""))
            new_val = clean_value(rec.get(col, ""))

            if old_val == "" and new_val != "":
                out[col] = new_val

            elif old_val != "" and new_val != "" and not same_value(old_val, new_val):
                conflicts.append(
                    {
                        "LAD_code": code,
                        "Year": int(rec["Year"]),
                        "Quarter": rec["Quarter"],
                        "Column": col,
                        "Kept_value": old_val,
                        "Other_value": new_val,
                        "Other_source": rec["_source"],
                    }
                )

    return list(merged.values()), conflicts, dropped_name_only


def sort_key(row):
    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    return int(row["Year"]), q_order.get(row["Quarter"], 99), row["LAD_code"]


def write_csv(rows, path, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def make_coverage(rows):
    coverage = {}

    for row in rows:
        key = (int(row["Year"]), row["Quarter"])

        if key not in coverage:
            coverage[key] = {
                "Year": int(row["Year"]),
                "Quarter": row["Quarter"],
                "LAD_rows": 0,
                "Homeless_relief_nonnull": 0,
                "Total_assessments_nonnull": 0,
                "Homeless_per_1000_nonnull": 0,
            }

        coverage[key]["LAD_rows"] += 1

        for col in VALUE_COLUMNS:
            if clean_value(row.get(col, "")) != "":
                coverage[key][f"{col}_nonnull"] += 1

    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

    return sorted(
        coverage.values(),
        key=lambda x: (x["Year"], q_order.get(x["Quarter"], 99)),
    )


def collect_records(data_dir):
    data_dir = Path(data_dir)

    excel_files = []
    for pattern in ["*.xls", "*.xlsx", "*.xlsm"]:
        excel_files.extend(data_dir.rglob(pattern))

    excel_files = sorted(
        p for p in excel_files
        if not p.name.startswith("~$")
    )

    all_records = []
    failed = []
    source_summary = []

    for path in excel_files:
        try:
            sheets = get_sheets(path)
        except Exception as e:
            failed.append(
                {
                    "file": str(path),
                    "stage": "open_workbook",
                    "error": repr(e),
                }
            )
            continue

        file_count = 0
        extractor_notes = []

        for label, func in choose_extractors(path, sheets):
            try:
                recs = func(path, sheets)
                all_records.extend(recs)
                file_count += len(recs)
                extractor_notes.append(f"{label}:{len(recs)}")
            except Exception as e:
                failed.append(
                    {
                        "file": str(path),
                        "stage": label,
                        "error": repr(e),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )

        source_summary.append(
            {
                "file": str(path),
                "records": file_count,
                "extractors": "; ".join(extractor_notes),
                "sheets": "; ".join(sheets),
            }
        )

    return all_records, failed, source_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/clean/homeless_lad_2009_2025_all_sources.csv")
    parser.add_argument("--conflicts", default="data/clean/homeless_lad_2009_2025_conflicts.csv")
    parser.add_argument("--failed-files", default="data/clean/homeless_lad_failed_files.csv")
    parser.add_argument("--coverage", default="data/clean/homeless_lad_coverage_by_quarter.csv")
    parser.add_argument("--source-summary", default="data/clean/homeless_lad_source_summary.csv")
    args = parser.parse_args()

    records, failed, source_summary = collect_records(args.data_dir)
    merged, conflicts, dropped_name_only = merge_records(records)
    merged = sorted(merged, key=sort_key)

    write_csv(merged, args.output, OUTPUT_COLUMNS)

    write_csv(
        conflicts,
        args.conflicts,
        [
            "LAD_code",
            "Year",
            "Quarter",
            "Column",
            "Kept_value",
            "Other_value",
            "Other_source",
        ],
    )

    write_csv(
        failed,
        args.failed_files,
        [
            "file",
            "stage",
            "error",
            "traceback",
        ],
    )

    write_csv(
        source_summary,
        args.source_summary,
        [
            "file",
            "records",
            "extractors",
            "sheets",
        ],
    )

    coverage = make_coverage(merged)
    write_csv(
        coverage,
        args.coverage,
        [
            "Year",
            "Quarter",
            "LAD_rows",
            "Homeless_relief_nonnull",
            "Total_assessments_nonnull",
            "Homeless_per_1000_nonnull",
        ],
    )

    print(f"Excel files scanned: {len(source_summary):,}")
    print(f"Raw records extracted: {len(records):,}")
    print(f"Merged LAD-quarter rows: {len(merged):,}")
    print(f"Conflicts written: {len(conflicts):,}")
    print(f"Dropped name-only rows without unique LAD mapping: {dropped_name_only:,}")
    print(f"Failed files/sheets written: {len(failed):,}")
    print(f"Saved output: {args.output}")
    print(f"Saved coverage: {args.coverage}")
    print(f"Saved source summary: {args.source_summary}")
    print(f"Saved conflicts: {args.conflicts}")
    print(f"Saved failed files: {args.failed_files}")


if __name__ == "__main__":
    main()
