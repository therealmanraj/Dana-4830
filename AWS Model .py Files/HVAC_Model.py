import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(42)

hvac_data = pd.read_csv('hvac_model_zones.csv')
print('HVAC Data Shape:', hvac_data.shape)
hvac_data.head()

hvac_data.columns = hvac_data.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)

hvac_data = hvac_data.loc[hvac_data['Environment:Site Day Type Index'] != 0]
hvac_data["HVAC_kWh"] = hvac_data["Electricity:HVAC"] * 2.77778e-7

occupant_cols = [col for col in hvac_data.columns if 'Occupant' in col]
hvac_data["TotalOccupantCount"] = hvac_data[occupant_cols].sum(axis=1)

time_shifts = [0.5, 1, 1.5, 2]

for h in time_shifts:
    steps = int(h * 6)
    hvac_data[f"Occ_minus{int(h*60)}"] = hvac_data["TotalOccupantCount"].shift(steps)

for h in time_shifts:
    steps = int(h * 6)
    hvac_data[f"Occ_plus{int(h*60)}"] = hvac_data["TotalOccupantCount"].shift(-steps)

def map_day_type(x):
    if x in [0, 6, 7]:
        return 1
    else:
        return 0

hvac_data["WeekendOrHoliday"] = hvac_data["Environment:Site Day Type Index"].apply(map_day_type)

hvac_data.to_csv('hvac_data.csv')

hvac_data = hvac_data.dropna()

features = ['HVAC_kWh']
target = 'HVAC_kWh'

def compute_metrics(test_data,forecast_mean):
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)


    metrics = {
        "mae":mae,
        "mse":mse,
        "rmse":rmse
    }

    return metrics

train_size = int(len(hvac_data) * 0.8)
train_data = hvac_data['HVAC_kWh'][:train_size]
test_data = hvac_data['HVAC_kWh'][train_size:]

X_train = np.arange(len(train_data)).reshape(-1, 1) 
y_train = train_data.values

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_fit = linear_model.predict(X_train)

residuals = y_train - y_fit

lags = 5
X_resid = np.array([residuals[i - lags:i] for i in range(lags, len(residuals))])
y_resid = residuals[lags:]


xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
xgb_model.fit(X_resid, y_resid)

resid_predictions = xgb_model.predict(X_resid)

y_fit_boosted = y_fit[lags:] + resid_predictions

X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
y_pred = linear_model.predict(X_test)

X_test_resid = np.array([test_data.values[i - lags:i] - y_pred[i - lags:i] for i in range(lags, len(test_data))])
resid_test_predictions = xgb_model.predict(X_test_resid)

y_pred_boosted = y_pred[lags:] + resid_test_predictions

# train_set = pd.DataFrame({
#     'train': train_data[-len(y_fit_boosted):].values,
#     'train_pred': y_fit_boosted
# })
# train_set.to_csv('Animated_Graphs/hvac/train_set.csv', index=False)

# test_set = pd.DataFrame({
#     'test': test_data[-len(y_pred_boosted):].values,
#     'test_pred': y_pred_boosted
# })
# test_set.to_csv('Animated_Graphs/hvac/test_set.csv', index=False)


import matplotlib.pyplot as plt


plt.figure(figsize=(30, 10))

plt.plot(train_data.index, train_data, label='Training data', color='blue', linewidth=5)
plt.plot(train_data.index[-len(y_fit_boosted):], y_fit_boosted, label='Training predictions', color='red', linestyle='--', linewidth=2)

plt.plot(test_data.index, test_data, label='Test data', color='green', linewidth=5)
plt.plot(test_data.index[-len(y_pred_boosted):], y_pred_boosted, label='Test predictions', color='orange', linestyle='--', linewidth=2)

plt.legend()
plt.title('Predictions with hybrid model: LinearRegression + XGBoost')
plt.ylabel('Consumption')
plt.grid(True)
plt.tight_layout()

plt.show()
