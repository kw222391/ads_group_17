import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


clean_homeless = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/merged_homelessness.csv"
claimant_rate_quarterly = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/claimant_rate_quarterly.csv"
cpi_quarterly = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/cpi_quarterly.csv"
hpi_quarterly = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_clean/hpi_quarterly.csv"


df = pd.read_csv(clean_homeless)
claimant_df = pd.read_csv(claimant_rate_quarterly)
cpi_df = pd.read_csv(cpi_quarterly)
hpi_df = pd.read_csv(hpi_quarterly)


for temp_df in [df, claimant_df, cpi_df, hpi_df]:
    if "year" in temp_df.columns:
        temp_df["year"] = temp_df["year"].astype(int)

for temp_df in [df, claimant_df, cpi_df, hpi_df]:
    if "quarter" in temp_df.columns:
        temp_df["quarter"] = temp_df["quarter"].astype(str)

for temp_df in [df, claimant_df, hpi_df]:
    if "lad_code" in temp_df.columns:
        temp_df["lad_code"] = temp_df["lad_code"].astype(str)


def quarter_to_date(row):
    if row["quarter"] == "Q1":
        return f"{row['year']}-03-31"
    elif row["quarter"] == "Q2":
        return f"{row['year']}-06-30"
    elif row["quarter"] == "Q3":
        return f"{row['year']}-09-30"
    elif row["quarter"] == "Q4":
        return f"{row['year']}-12-31"
    return pd.NaT

df["date"] = pd.to_datetime(df.apply(quarter_to_date, axis=1))

df = df.sort_values(["lad_code", "date"]).reset_index(drop=True)


df["target_diff"] = df.groupby("lad_code")["homelessness_total_assessments"].diff()

df = df.merge(claimant_df, on=["year", "quarter", "lad_code"], how="left")

# HOMELESSNESS HISTORY FEATURES
df["lag1"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(1)
df["lag4"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(4)

df["rolling_mean_4"] = (
    df.groupby("lad_code")["homelessness_total_assessments"]
      .transform(lambda s: s.shift(1).rolling(4).mean())
)

df["trend_4q"] = df["lag1"] - df["lag4"]

# TIME FEATURES
df["quarter_num"] = df["quarter"].str.replace("Q", "", regex=False).astype(int)

df["covid"] = (
    (df["date"] >= "2020-01-01") &
    (df["date"] <= "2021-06-30")
).astype(int)


# CLAIMANT FEATURES
df["claimant_rate_lag1"] = df.groupby("lad_code")["claimant_rate"].shift(1)
df["claimant_rate_qoq"] = df.groupby("lad_code")["claimant_rate"].pct_change()
df["claimant_rate_yoy"] = df.groupby("lad_code")["claimant_rate"].pct_change(4)


# CPI FEATURES
quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
cpi_df["quarter_num"] = cpi_df["quarter"].map(quarter_order)
cpi_df = cpi_df.sort_values(["year", "quarter_num"]).reset_index(drop=True)

cpi_df["cpi_lag1"] = cpi_df["cpi"].shift(1)
cpi_df["cpi_qoq"] = cpi_df["cpi"].pct_change()
cpi_df["cpi_yoy"] = cpi_df["cpi"].pct_change(4)

cpi_df = cpi_df.drop(columns=["quarter_num"])

df = df.merge(cpi_df, on=["year", "quarter"], how="left")

# HPI FEATURES
df = df.merge(hpi_df, on=["year", "quarter", "lad_code"], how="left")
df = df.sort_values(["lad_code", "date"]).reset_index(drop=True)

df["hpi_lag1"] = df.groupby("lad_code")["hpi"].shift(1)
df["hpi_qoq"] = df.groupby("lad_code")["hpi"].pct_change()
df["hpi_yoy"] = df.groupby("lad_code")["hpi"].pct_change(4)


df["claimant_rate_qoq"] = df["claimant_rate_qoq"].clip(-1, 1)
df["claimant_rate_yoy"] = df["claimant_rate_yoy"].clip(-1, 1)
df["cpi_qoq"] = df["cpi_qoq"].clip(-1, 1)
df["cpi_yoy"] = df["cpi_yoy"].clip(-1, 1)
df["hpi_qoq"] = df["hpi_qoq"].clip(-1, 1)
df["hpi_yoy"] = df["hpi_yoy"].clip(-1, 1)

features = [
    "lag1",
    "rolling_mean_4",
    "trend_4q",
    "quarter_num",
    "covid",
    "claimant_rate",
    "claimant_rate_lag1",
    "claimant_rate_qoq",
    "claimant_rate_yoy",
    "cpi",
    "cpi_lag1",
    "cpi_qoq",
    "cpi_yoy",
    "hpi",
    "hpi_lag1",
    "hpi_qoq",
    "hpi_yoy"
]


model_df = df.dropna(subset=["target_diff"] + features).copy()


train = model_df[model_df["date"] < "2023-01-01"].copy()
test = model_df[model_df["date"] >= "2023-01-01"].copy()

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTarget diff summary:")
print(model_df["target_diff"].describe())



#linear regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train[features])
X_test_scaled = scaler.transform(test[features])

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, train["target_diff"])

# predict change
lin_pred_diff = lin_reg.predict(X_test_scaled)

# reconstruct level
lin_y_pred_level = test["lag1"] + lin_pred_diff

# true values
lin_y_true_level = test["homelessness_total_assessments"]
lin_y_true_diff = test["target_diff"]

print("\n=========================")
print("LINEAR REGRESSION BASELINE")
print("=========================")

print("\n-------------------------")
print("LEVEL RECONSTRUCTION")
print("-------------------------")
print("LR MAE:", mean_absolute_error(lin_y_true_level, lin_y_pred_level))
print("LR RMSE:", root_mean_squared_error(lin_y_true_level, lin_y_pred_level))
print("LR R2:", r2_score(lin_y_true_level, lin_y_pred_level))

print("\n-------------------------")
print("CHANGE TARGET")
print("-------------------------")
print("LR Change MAE:", mean_absolute_error(lin_y_true_diff, lin_pred_diff))
print("LR Change RMSE:", root_mean_squared_error(lin_y_true_diff, lin_pred_diff))
print("LR Change R2:", r2_score(lin_y_true_diff, lin_pred_diff))

print("\n-------------------------")
print("COEFFICIENTS")
print("-------------------------")
coef_df = pd.Series(
    lin_reg.coef_,
    index=features
).sort_values(key=np.abs, ascending=False)

print(coef_df)




#random forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(train[features], train["target_diff"])

pred_diff = rf.predict(test[features])


y_pred_level = test["lag1"] + pred_diff
y_true_level = test["homelessness_total_assessments"]

y_true_diff = test["target_diff"]

print("\n-------------------------")
print("RANDOM FOREST: LEVEL RECONSTRUCTION")
print("-------------------------")
print("RF MAE:", mean_absolute_error(y_true_level, y_pred_level))
print("RF RMSE:", root_mean_squared_error(y_true_level, y_pred_level))
print("RF R2:", r2_score(y_true_level, y_pred_level))

print("\n-------------------------")
print("RANDOM FOREST: CHANGE TARGET")
print("-------------------------")
print("RF Change MAE:", mean_absolute_error(y_true_diff, pred_diff))
print("RF Change RMSE:", root_mean_squared_error(y_true_diff, pred_diff))
print("RF Change R2:", r2_score(y_true_diff, pred_diff))

print("\n-------------------------")
print("FEATURE IMPORTANCES")
print("-------------------------")
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

results = test[[
    "lad_code",
    "year",
    "quarter",
    "date",
    "homelessness_total_assessments",
    "lag1",
    "target_diff"
]].copy()

results["pred_diff"] = pred_diff
results["pred_level"] = y_pred_level
results["error_level"] = results["homelessness_total_assessments"] - results["pred_level"]
results["abs_error_level"] = results["error_level"].abs()

print("\nSample predictions:")
print(results.head(10))

# results.to_csv("/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/change_model_predictions.csv", index=False)