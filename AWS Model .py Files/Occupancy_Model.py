import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

occupancy_data = pd.read_csv('occupancy_model_zones.csv')
occupancy_data.head()

occupancy_data.columns = occupancy_data.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)

occupancy_data = occupancy_data.loc[occupancy_data['Environment:Site Day Type Index'] != 0]

occupant_cols = [col for col in occupancy_data.columns if 'occupant' in str(col.lower())]
occupancy_data['total_occupants'] = occupancy_data[occupant_cols].sum(axis=1)
occupancy_data.drop(occupant_cols, axis=1, inplace=True)

start_datetime = pd.Timestamp(year=2004, month=1, day=1, hour=0, minute=0)

time_stamps = []
for i in range(len(occupancy_data)):
    current_time = start_datetime + pd.Timedelta(minutes=10 * i)
    time_stamps.append(current_time)
    
occupancy_data['timestamp'] = time_stamps

occupancy_data['timestamp'] = pd.to_datetime(occupancy_data['timestamp'])
occupancy_data['hour'] = occupancy_data['timestamp'].dt.hour
occupancy_data['day_of_week'] = occupancy_data['timestamp'].dt.dayofweek
occupancy_data['month'] = occupancy_data['timestamp'].dt.month

occupancy_data = occupancy_data.sort_values('timestamp')

time_shifts = [0.5, 1, 1.5, 2]

for h in time_shifts:
    steps = int(h * 6) 
    occupancy_data[f"Occ_minus{int(h*60)}"] = occupancy_data["total_occupants"].shift(steps)

for h in time_shifts:
    steps = int(h * 6)
    occupancy_data[f"Occ_plus{int(h*60)}"] = occupancy_data["total_occupants"].shift(-steps)

def map_day_type(x):
    if x in [0, 6, 7]:
        return 1
    else:
        return 0

occupancy_data["WeekendOrHoliday"] = occupancy_data["Environment:Site Day Type Index"].apply(map_day_type)

occupancy_data = occupancy_data.dropna()


features = [col for col in occupancy_data.columns if col not in ['total_occupants', 'timestamp']]
target = 'total_occupants'

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

train_size = int(len(occupancy_data) * 0.8)
train_data = occupancy_data[:train_size]
test_data = occupancy_data[train_size:]

X_train, X_test = train_data[features], test_data[features]
y_train, y_test = train_data[target], test_data[target]

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)
residuals_train = y_train.values - y_train_pred

lags = 5

X_train_resid = np.array([residuals_train[i - lags:i] for i in range(lags, len(residuals_train))])
y_train_resid = residuals_train[lags:]

xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
xgb_model.fit(X_train_resid, y_train_resid)

y_test_pred_linear = linear_model.predict(X_test)

residuals_test = y_test.values - y_test_pred_linear

X_test_resid = np.array([residuals_test[i - lags:i] for i in range(lags, len(residuals_test))])

resid_test_pred = xgb_model.predict(X_test_resid)

y_test_hybrid = y_test_pred_linear[lags:] + resid_test_pred


plt.figure(figsize=(20, 6))

X_train_resid_pred = np.array([residuals_train[i - lags:i] for i in range(lags, len(residuals_train))])
resid_train_pred = xgb_model.predict(X_train_resid_pred)
y_train_hybrid = y_train_pred[lags:] + resid_train_pred

plt.plot(y_train.index, y_train, label='Training Actual', color='blue', linewidth=5)
plt.plot(y_train.index[lags:], y_train_hybrid, label='Training Hybrid Predictions', color='red', linestyle='--', linewidth=2)

plt.plot(y_test.index, y_test, label='Test Actual', color='green', linewidth=5)
plt.plot(y_test.index[lags:], y_test_hybrid, label='Test Hybrid Predictions', color='orange', linestyle='--', linewidth=2)

plt.legend()
plt.title('Hybrid Model Predictions: Linear Regression + XGBoost Residual Correction')
plt.ylabel('Total Occupants')
plt.xlabel('Timestamp')
plt.grid(True)
plt.tight_layout()
plt.show()


metrics = compute_metrics(y_test.values[lags:], y_test_hybrid)
for key, value in metrics.items():
    print(f'{key}: {value}')



# train_set = pd.DataFrame({
#     'train': y_train[lags:],
#     'train_pred': y_train_hybrid
# })
# train_set.to_csv('Animated_Graphs/occupancy/train_set.csv', index=False)


# test_set = pd.DataFrame({
#     'test': y_test[lags:],
#     'test_pred': y_test_hybrid
# })
# test_set.to_csv('Animated_Graphs/occupancy/test_set.csv', index=False)