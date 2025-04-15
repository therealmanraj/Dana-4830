import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(test_data, forecast_mean):
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}

def add_lags(df):
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            target_map = df[col].to_dict()
            for lag in [7, 14, 21]:
                lag_col = f'{col}_lag{lag}'
                df[lag_col] = (df.index - pd.Timedelta(f'{lag} days')).map(target_map)
    return df.copy()

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
    if 'Environment:Site Day Type Index' in df.columns:
        df = df.loc[df['Environment:Site Day Type Index'] != 0]
    df['HVAC_kWh'] = df['Electricity:HVAC'] * 2.77778e-7
    df.drop(columns='Electricity:HVAC', inplace=True)

    occ_cols = [col for col in df.columns if 'Occupant' in col]
    df['TotalOccCount'] = df[occ_cols].sum(axis=1)
    df.drop(columns=occ_cols, inplace=True)

    df.index = pd.date_range(start="2004-01-01 00:00:00", periods=len(df), freq="10min")
    df = add_lags(df)
    df = create_features(df)
    return df

def plot_predictions(df, title):
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=df.index, y=df['HVAC_kWh'], label='Actual')
    sns.lineplot(x=df.index, y=df['Predicted_HVAC_kWh'], label='Predicted', dashes=(4, 2))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_and_predict(filepath, model_output_dir):
    df = prepare_data(filepath)
    target = 'HVAC_kWh'
    features = [col for col in df.columns if col != target]

    train_mask = df.index.month < 10
    val_mask = (df.index.month >= 10) & (df.index.month <= 11)
    test_mask = (df.index.month > 11) & (df.index.month <= 12)

    x_train, y_train = df[features][train_mask].bfill(), df[target][train_mask]
    x_val, y_val = df[features][val_mask], df[target][val_mask]
    x_test, y_test = df[features][test_mask], df[target][test_mask]

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_train_res = y_train - lin_reg.predict(x_train)

    xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=200,
                                 objective='reg:squarederror', max_depth=5, learning_rate=0.05)
    xgb_model.fit(x_train, y_train_res, verbose=100)

    y_pred_combined = lin_reg.predict(x_val) + xgb_model.predict(x_val)
    val_results = df[val_mask].copy()
    val_results['Predicted_HVAC_kWh'] = y_pred_combined
    plot_predictions(val_results, "Validation: HVAC Energy Consumption by Hour")
    print("Validation Metrics:", compute_metrics(y_val, y_pred_combined))

    top_features = pd.Series(xgb_model.feature_importances_, index=x_train.columns)
    top_15 = top_features.sort_values(ascending=False).head(15).index.tolist()

    lin_reg.fit(x_train[top_15], y_train)
    y_train_res = y_train - lin_reg.predict(x_train[top_15])
    xgb_model.fit(x_train[top_15], y_train_res, verbose=100)

    y_pred_combined = lin_reg.predict(x_val[top_15]) + xgb_model.predict(x_val[top_15])
    val_results['Predicted_HVAC_kWh'] = y_pred_combined
    plot_predictions(val_results, "Validation (Top 15 Features): HVAC Energy Consumption by Hour")
    print("Top 15 Feature Validation Metrics:", compute_metrics(y_val, y_pred_combined))

    if len(x_test) == 0:
        print("Warning: No future test data available for forecasting.")
        return

    forecast_index = pd.date_range(x_test.index.min(), x_test.index.max(), freq='10min')
    forecast_df = pd.DataFrame(index=forecast_index)
    forecast_df['HVAC_kWh'] = np.nan

    future_source = df.drop(columns=['HVAC_kWh'])
    combined = pd.concat([future_source, forecast_df], axis=0)
    combined = add_lags(combined)
    combined = create_features(combined)

    future_only = combined.loc[forecast_index].copy()
    future_only = future_only.dropna(subset=[col for col in top_15 if col in future_only.columns])

    # Fill any missing top_15 columns with 0 to avoid KeyError
    for col in top_15:
        if col not in future_only.columns:
            future_only[col] = 0

    y_pred_combined = lin_reg.predict(future_only[top_15]) + xgb_model.predict(future_only[top_15])
    future_result = pd.DataFrame(index=future_only.index)
    future_result['HVAC_kWh'] = np.nan
    future_result['Predicted_HVAC_KWh'] = y_pred_combined
    plot_predictions(future_result, "Forecasted HVAC Energy Consumption by Hour")
    print("Forecast Metrics:", compute_metrics(y_test[:len(future_result)], y_pred_combined))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_output_dir', type=str, required=True)
    args = parser.parse_args()
    train_and_predict(args.data_path, args.model_output_dir)