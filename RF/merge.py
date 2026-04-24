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

for col in ["year"]:
    df[col] = df[col].astype(int)
    claimant_df[col] = claimant_df[col].astype(int)

for col in ["quarter", "lad_code"]:
    df[col] = df[col].astype(str)
    claimant_df[col] = claimant_df[col].astype(str)

def quarter_to_date(row):
    if row["quarter"] == "Q1":
        return f"{row['year']}-03-31"
    elif row["quarter"] == "Q2":
        return f"{row['year']}-06-30"
    elif row["quarter"] == "Q3":
        return f"{row['year']}-09-30"
    elif row["quarter"] == "Q4":
        return f"{row['year']}-12-31"
    else:
        return pd.NaT

df["date"] = pd.to_datetime(df.apply(quarter_to_date, axis=1))


#claimaint count
df = df.merge(claimant_df, on=["year", "quarter", "lad_code"], how="left")

df = df.sort_values(["lad_code", "date"]).reset_index(drop=True)

df["target_log"] = np.log1p(df["homelessness_total_assessments"])

df["lag1"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(1)
df["lag2"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(2)
df["lag3"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(3)
df["lag4"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(4)

df["quarter_num"] = df["quarter"].str.replace("Q", "", regex=False).astype(int)

df["covid"] = (
    (df["date"] >= "2020-04-01") &
    (df["date"] <= "2021-12-31")
).astype(int)

df["claimant_rate_lag1"] = df.groupby("lad_code")["claimant_rate"].shift(1)
df["claimant_rate_qoq"] = df.groupby("lad_code")["claimant_rate"].pct_change()
df["claimant_rate_yoy"] = df.groupby("lad_code")["claimant_rate"].pct_change(4)

#cpi
df = df.merge(cpi_df, on=["year","quarter"], how="left")
df["cpi_lag1"] = df["cpi"].shift(1)
df["cpi_qoq"] = df["cpi"].pct_change()
df["cpi_yoy"] = df["cpi"].pct_change(4)

#hpi
df = df.merge(hpi_df, on=["year","quarter","lad_code"], how="left")
df = df.sort_values(["lad_code", "date"])

df["hpi_lag1"] = df.groupby("lad_code")["hpi"].shift(1)
df["hpi_qoq"] = df.groupby("lad_code")["hpi"].pct_change()
df["hpi_yoy"] = df.groupby("lad_code")["hpi"].pct_change(4)

df = df.dropna().copy()

train = df[df["date"] < "2023-01-01"]
test = df[df["date"] >= "2023-01-01"]

y_true = test["homelessness_total_assessments"]

features = [
    #homeless lag
    "lag1", "lag2", "lag3", "lag4",
    "quarter_num", "covid",
    #cc
    "claimant_rate",
    "claimant_rate_lag1",
    "claimant_rate_qoq",
    "claimant_rate_yoy",
    #cpi
    "cpi", 
    "cpi_lag1", 
    "cpi_qoq", 
    "cpi_yoy",
    #hpi
    "hpi",
    "hpi_lag1",
    "hpi_qoq",
    "hpi_yoy"
]


rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(train[features], train["target_log"])

y_pred_log = rf.predict(test[features])
y_pred = np.expm1(y_pred_log)

print("-------------------------")
print("RANDOM FOREST")
print("-------------------------")

print("RF MAE:", mean_absolute_error(y_true, y_pred))
print("RF RMSE:", root_mean_squared_error(y_true, y_pred))
print("RF R2:", r2_score(y_true, y_pred))

importances = pd.Series(rf.feature_importances_, index=features)
print(importances.sort_values(ascending=False))


####linear regression
df["log_lag1"] = np.log1p(df["lag1"])

df["hpi_yoy"] = df["hpi_yoy"].clip(-1, 1)
df["cpi_yoy"] = df["cpi_yoy"].clip(-1, 1)

features_reg = [
    "log_lag1",
    "claimant_rate",
    "hpi_yoy",
    "cpi_yoy"
]

scaler = StandardScaler()

df["hpi_yoy"] = df["hpi_yoy"].clip(-1, 1)
df["cpi_yoy"] = df["cpi_yoy"].clip(-1, 1)

train = df[df["date"] < "2023-01-01"]
test = df[df["date"] >= "2023-01-01"]

X_train = train[features_reg]
y_train = train["target_log"]

X_test = test[features_reg]
y_test = test["target_log"]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_log = lr.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)


print("-------------------------")
print("LINEAR REGRESSION")
print("-------------------------")

print("Reg MAE:", mean_absolute_error(y_true, y_pred))
print("Reg RMSE:", root_mean_squared_error(y_true, y_pred))
print("Reg R2:", r2_score(y_true, y_pred))

coeffs = pd.Series(lr.coef_, index=features_reg)
print(coeffs)