import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from sklearn.frozen import FrozenEstimator

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

cutoff = pd.Timestamp("2023-01-01")
train_mask = df["date"] < cutoff

train_std_by_lad = (
    df.loc[train_mask]
      .groupby("lad_code")["target_diff"]
      .std()
)

df["spike_threshold"] = df["lad_code"].map(train_std_by_lad)
df["target_spike"] = (df["target_diff"] > df["spike_threshold"]).astype(int)

df = df.merge(claimant_df, on=["year", "quarter", "lad_code"], how="left")

df["lag1"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(1)
df["lag4"] = df.groupby("lad_code")["homelessness_total_assessments"].shift(4)

df["rolling_mean_4"] = (
    df.groupby("lad_code")["homelessness_total_assessments"]
      .transform(lambda s: s.shift(1).rolling(4).mean())
)

df["trend_4q"] = df["lag1"] - df["lag4"]

df["quarter_num"] = df["quarter"].str.replace("Q", "", regex=False).astype(int)

df["covid"] = (
    (df["date"] >= "2020-01-01") &
    (df["date"] <= "2021-06-30")
).astype(int)

df["claimant_rate_lag1"] = df.groupby("lad_code")["claimant_rate"].shift(1)
df["claimant_rate_qoq"] = df.groupby("lad_code")["claimant_rate"].pct_change()
df["claimant_rate_yoy"] = df.groupby("lad_code")["claimant_rate"].pct_change(4)

quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
cpi_df["quarter_num"] = cpi_df["quarter"].map(quarter_order)
cpi_df = cpi_df.sort_values(["year", "quarter_num"]).reset_index(drop=True)

cpi_df["cpi_lag1"] = cpi_df["cpi"].shift(1)
cpi_df["cpi_qoq"] = cpi_df["cpi"].pct_change()
cpi_df["cpi_yoy"] = cpi_df["cpi"].pct_change(4)

cpi_df = cpi_df.drop(columns=["quarter_num"])

df = df.merge(cpi_df, on=["year", "quarter"], how="left")

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

model_df = df.dropna(subset=["target_diff", "spike_threshold", "target_spike"] + features).copy()

train = model_df[model_df["date"] < cutoff].copy()
test = model_df[model_df["date"] >= cutoff].copy()

X_train = train[features]
y_train = train["target_spike"]

X_test = test[features]
y_test = test["target_spike"]

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain spike rate:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nTest spike rate:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

X_train = train[features]
y_train = train["target_spike"]

X_test = test[features]
y_test = test["target_spike"]

linear_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

linear_clf.fit(X_train, y_train)

y_prob_lin = linear_clf.predict_proba(X_test)[:, 1]

y_pred_lin = (y_prob_lin >= 0.30).astype(int)

print("\n-------------------------")
print("BASELINE LINEAR CLASSIFIER")
print("(Logistic Regression)")
print("-------------------------")

print(classification_report(y_test, y_pred_lin, digits=4))

print("ROC AUC:", roc_auc_score(y_test, y_prob_lin))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lin))

coef = pd.Series(
    linear_clf.named_steps["model"].coef_[0],
    index=features
).sort_values(key=abs, ascending=False)

print("\n-------------------------")
print("LINEAR MODEL COEFFICIENTS")
print("-------------------------")
print(coef)

# clf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1,
#     class_weight="balanced"
# )

# clf.fit(X_train, y_train)

# y_prob = clf.predict_proba(X_test)[:, 1]
# y_pred = (y_prob >= 0.30).astype(int)
# print(confusion_matrix(y_test, y_pred))

# # for t in [0.15,0.20,0.25,0.30]:
# #     pred = (y_prob >= t).astype(int)
# #     print("Threshold:", t)
# #     print(classification_report(y_test, pred))

# print("\n-------------------------")
# print("SPIKE CLASSIFIER RESULTS")
# print("-------------------------")
# print(classification_report(y_test, y_pred, digits=4))

# print("ROC AUC:", roc_auc_score(y_test, y_prob))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\n-------------------------")
# print("FEATURE IMPORTANCE")
# print("-------------------------")

# importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
# print(importances)

# Main split
train_full = model_df[model_df["date"] < cutoff].copy()
test = model_df[model_df["date"] >= cutoff].copy()

# Time-aware validation split inside train period
# Example: use 2022 as validation
val_start = pd.Timestamp("2022-01-01")

train = train_full[train_full["date"] < val_start].copy()
val = train_full[train_full["date"] >= val_start].copy()

X_train = train[features]
y_train = train["target_spike"]

X_val = val[features]
y_val = val["target_spike"]

X_test = test[features]
y_test = test["target_spike"]

print("Train:", train.shape, y_train.mean())
print("Val:", val.shape, y_val.mean())
print("Test:", test.shape, y_test.mean())

# --------------------------------------------------
# HYPERPARAMETER SEARCH
# --------------------------------------------------

param_grid = [
    {"n_estimators": 300, "max_depth": 6,  "min_samples_leaf": 5,  "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 8,  "min_samples_leaf": 5,  "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 800, "max_depth": 12, "min_samples_leaf": 10, "max_features": 0.5},
    {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 1000, "max_depth": None, "min_samples_leaf": 25, "max_features": 0.4},
]

results = []

for params in param_grid:
    rf = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)

    val_prob = rf.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, val_prob)
    pr = average_precision_score(y_val, val_prob)

    results.append({
        **params,
        "val_roc_auc": roc,
        "val_pr_auc": pr
    })

results_df = pd.DataFrame(results).sort_values(
    ["val_roc_auc", "val_pr_auc"], ascending=False
)

print("\nValidation tuning results:")
print(results_df)

best_params = results_df.iloc[0][["n_estimators", "max_depth", "min_samples_leaf", "max_features"]].to_dict()

best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

if pd.isna(best_params["max_depth"]):
    best_params["max_depth"] = None
else:
    best_params["max_depth"] = int(best_params["max_depth"])

print("\nBest params:", best_params)

# --------------------------------------------------
# REFIT BEST RF ON TRAIN ONLY
# --------------------------------------------------

best_rf = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

best_rf.fit(X_train, y_train)

val_prob_raw = best_rf.predict_proba(X_val)[:, 1]

print("\nRaw RF validation ROC AUC:", roc_auc_score(y_val, val_prob_raw))
print("Raw RF validation PR AUC:", average_precision_score(y_val, val_prob_raw))

# --------------------------------------------------
# CALIBRATION
# --------------------------------------------------

cal_rf = CalibratedClassifierCV(
    estimator=FrozenEstimator(best_rf),
    method="sigmoid"
)
cal_rf.fit(X_val, y_val)

val_prob_cal = cal_rf.predict_proba(X_val)[:, 1]

print("\nCalibrated RF validation ROC AUC:", roc_auc_score(y_val, val_prob_cal))
print("Calibrated RF validation PR AUC:", average_precision_score(y_val, val_prob_cal))

# --------------------------------------------------
# THRESHOLD SELECTION ON VALIDATION
# choose threshold that maximizes F1 for class 1
# --------------------------------------------------

thresholds = np.arange(0.10, 0.91, 0.05)

best_threshold = None
best_f1 = -1
threshold_rows = []

for t in thresholds:
    pred = (val_prob_cal >= t).astype(int)

    tp = ((pred == 1) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    threshold_rows.append({
        "threshold": t,
        "precision_1": precision,
        "recall_1": recall,
        "f1_1": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

threshold_df = pd.DataFrame(threshold_rows)
print("\nThreshold search on validation:")
print(threshold_df)

print("\nBest threshold:", best_threshold)
print("Best validation F1 for class 1:", best_f1)

# --------------------------------------------------
# FINAL MODEL:
# refit best RF on ALL pre-2023 data, then calibrate using validation-style holdout
# --------------------------------------------------

# Simpler practical option:
# fit on all train_full, use calibration on val already found above if you want quick comparison.
# Cleaner option is to preserve calibration split.
#
# Here we'll preserve the cleaner workflow:
final_rf = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
final_rf.fit(X_train, y_train)

final_cal_rf = CalibratedClassifierCV(
    estimator=FrozenEstimator(final_rf),
    method="sigmoid"
)
final_cal_rf.fit(X_val, y_val)

test_prob = final_cal_rf.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n-------------------------")
print("TUNED + CALIBRATED RF")
print("-------------------------")
print(classification_report(y_test, test_pred, digits=4))
print("Test ROC AUC:", roc_auc_score(y_test, test_prob))
print("Test PR AUC:", average_precision_score(y_test, test_prob))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# Feature importance from underlying fitted forest
importances = pd.Series(final_rf.feature_importances_, index=features).sort_values(ascending=False)

print("\n-------------------------")
print("FEATURE IMPORTANCE")
print("-------------------------")
print(importances)