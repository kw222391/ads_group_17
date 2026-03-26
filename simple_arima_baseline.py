import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

FILE_PATH = r"D:\UOB\ads_group_17\ads_group_17\data\merged_monthly_data_valid_with_homeless.csv"

df = pd.read_csv(FILE_PATH)

print("Original data shape:", df.shape)
print(df.head())

df["month"] = pd.to_datetime(df["month"], format="%d/%m/%Y")

df = df.sort_values("month").reset_index(drop=True)

print("\nFixed month column:")
print(df[["month"]].head(12))
print(df[["month"]].tail(12))

data = df[["month", "cpih_index", "homeless_households"]].copy()
data = data.dropna()

print("\nData used for ARIMAX:")
print(data.head())
print(data.shape)

# set index
data = data.set_index("month")

y = data["homeless_households"]
X = data[["cpih_index"]]

#train test split

train_size = int(len(data) * 0.8)

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

print("\nTrain size:", len(y_train))
print("Test size:", len(y_test))
print("Train period:", y_train.index.min(), "to", y_train.index.max())
print("Test period:", y_test.index.min(), "to", y_test.index.max())

#ARIMAX model
model = ARIMA(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1)
)

model_fit = model.fit()

print("\nModel summary:")
print(model_fit.summary())

# test set
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

forecast = pd.Series(forecast, index=y_test.index, name="forecast")
# result
mae = mean_absolute_error(y_test, forecast)
rmse = np.sqrt(mean_squared_error(y_test, forecast))

print("\nTest evaluation:")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")

# compare actual vs predicted
results_df = pd.DataFrame({
    "actual": y_test,
    "predicted": forecast
})

print("\nTest results:")
print(results_df)

# plot results

plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index, y_test, label="Test")
plt.plot(forecast.index, forecast, label="Forecast")
plt.title("ARIMAX Forecast: homeless_households using cpih_index")
plt.xlabel("Month")
plt.ylabel("homeless_households")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# # optional: plot actual vs predicted only on test
#
# plt.figure(figsize=(12, 5))
# plt.plot(y_test.index, y_test, marker="o", label="Actual")
# plt.plot(forecast.index, forecast, marker="o", label="Predicted")
# plt.title("Test Set: Actual vs Predicted")
# plt.xlabel("Month")
# plt.ylabel("homeless_households")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()