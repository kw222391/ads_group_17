import pandas as pd

# load your already-wide file
df_wide = pd.read_csv("/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/hpi_wide.csv")

# load LAD lookup
lad = pd.read_csv("/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/lad_lookup.csv")

# build mapping
lad["region_key"] = (
    lad["LAD25NM"]
    .str.lower()
    .str.replace(" ", "-", regex=False)
    .str.strip()
)

mapping_dict = dict(zip(lad["region_key"], lad["LAD25CD"]))

# rename columns
new_cols = []
unmatched = []

for col in df_wide.columns:
    if col == "date":
        new_cols.append(col)
    else:
        key = str(col).lower().strip()
        if key in mapping_dict:
            new_cols.append(mapping_dict[key])
        else:
            new_cols.append(col)
            unmatched.append(col)

df_wide.columns = new_cols

# manual renaming for valid England LADs that do not match automatically
manual_map = {
    "city-of-bristol": "E06000023",
    "city-of-derby": "E06000015",
    "city-of-nottingham": "E06000018",
    "city-of-peterborough": "E06000031",
    "city-of-plymouth": "E06000026",
    "city-of-kingston-upon-hull": "E06000010",
    "city-of-westminster": "E09000033",
    "bournemouth-christchurch-and-poole": "E06000058",
}

df_wide = df_wide.rename(columns=manual_map)

# drop non-England / non-LAD columns
drop_cols = [
    "east-midlands",
    "east-of-england",
    "england",
    "cambridgeshire",
    "derbyshire",
    "devon",
    "east-sussex",
    "armagh-banbridge-and-craigavon",
    "derry-and-strabane",
    "city-of-aberdeen",
    "city-of-dundee",
    "city-of-glasgow",
]

df_wide = df_wide.drop(columns=[c for c in drop_cols if c in df_wide.columns])

# keep only date + England LAD codes
df_wide = df_wide[["date"] + [col for col in df_wide.columns if str(col).startswith("E")]]

# optional: make date unambiguous
df_wide["date"] = pd.to_datetime(df_wide["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

# save
out_file = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/hpi_wide_lad.csv"
df_wide.to_csv(out_file, index=False)

# final unmatched check after manual fixes and drops
still_unmatched = [
    c for c in df_wide.columns
    if c != "date" and not str(c).startswith("E")
]

eng_lad = lad[lad["LAD25CD"].str.startswith("E")]

print("England LADs in lookup:", eng_lad["LAD25CD"].nunique())

print("Saved to:", out_file)
print("Original unmatched columns:", unmatched[:20])
print("Still unmatched after cleaning:", still_unmatched[:20])
print("Number of LAD columns kept:", len(df_wide.columns) - 1)