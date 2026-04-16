import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


filename = "/Users/jagannathsankar/Desktop/Year3/Applied Data Science/RF/data_raw/merged_homelessness.csv"

df = pd.read_csv(filename)

def quarter_to_date(row):
    if row["quarter"] == "Q1":
        return f"{row['year']}-03-31"
    elif row["quarter"] == "Q2":
        return f"{row['year']}-06-30"
    elif row["quarter"] == "Q3":
        return f"{row['year']}-09-30"
    elif row["quarter"] == "Q4":
        return f"{row['year']}-12-31"

df["date"] = pd.to_datetime(df.apply(quarter_to_date, axis=1))

df = df.sort_values(["lad_code", "date"]).reset_index(drop=True)

df["lag1"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(1)
df["lag2"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(2)
df["lag3"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(3)
df["lag4"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(4)

df["target_log"] = np.log1p(df["homelessness_total_assessments"])

df["quarter_num"] = df["quarter"].str.replace("Q", "").astype(int)

df["covid"] = (
    (df["date"] >= "2020-04-01") & 
    (df["date"] <= "2021-12-31")
).astype(int)

df = df.dropna()


#training baseline
train = df[df["date"] < "2023-01-01"]
test = df[df["date"] >= "2023-01-01"]

y_pred_baseline = test["lag1"]

y_true = test["homelessness_total_assessments"]

print("MAE:", mean_absolute_error(y_true, y_pred_baseline))
print("RMSE:", mean_squared_error(y_true, y_pred_baseline, squared=False))
print("R2:", r2_score(y_true, y_pred_baseline))

#baseline RF

features = ["lag1", "lag2", "lag3", "lag4", "quarter_num", "covid"]

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(train[features], train["target_log"])

y_pred_log = rf.predict(test[features])
y_pred = np.expm1(y_pred_log)

print("RF MAE:", mean_absolute_error(y_true, y_pred))
print("RF RMSE:", mean_squared_error(y_true, y_pred, squared=False))
print("RF R2:", r2_score(y_true, y_pred))


# print(df.iloc[4])
# print(df.iloc[5])
# print(df.iloc[6])
# print(df.iloc[7])
# print(df.iloc[8])
# print(df.iloc[9])