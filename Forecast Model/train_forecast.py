#!/usr/bin/env python
import io
import os
import pickle
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}

def transform_data(df):
    """
    Applies preprocessing steps:
      - Remove rows where 'Environment:Site Day Type Index' is 0.
      - Convert 'Electricity:HVAC' to kWh.
      - Sum occupant columns into 'TotalOccupantCount'.
      - Create time-shifted occupant features.
      - Create a 'WeekendOrHoliday' indicator.
      - Drop rows with missing values.
    """
    df = df.loc[df['Environment:Site Day Type Index'] != 0]
    df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
    occupant_cols = [col for col in df.columns if 'Occupant' in col]
    df["TotalOccupantCount"] = df[occupant_cols].sum(axis=1)
    
    time_shifts = [0.5, 1, 1.5, 2]
    for h in time_shifts:
        steps = int(h * 6)
        df[f"Occ_minus{int(h*60)}"] = df["TotalOccupantCount"].shift(steps)
    for h in time_shifts:
        steps = int(h * 6)
        df[f"Occ_plus{int(h*60)}"] = df["TotalOccupantCount"].shift(-steps)
    
    df["WeekendOrHoliday"] = df["Environment:Site Day Type Index"].apply(
        lambda x: 1 if x in [0, 6, 7] else 0
    )
    df = df.dropna()
    return df

def main():
    # --- AWS S3 and Region Configuration ---
    region = 'ca-central-1'
    s3_bucket = 'dana-minicapstone-ca'
    # Use hvac_test.csv as your single source of historical data
    s3_key_hist = 'data/hvac_test.csv'
    # Save forecast predictions to hvac_forecast.csv
    s3_key_forecast = 'data/hvac_forecast.csv'
    
    # Create S3 client
    s3 = boto3.client('s3', region_name=region)
    
    # --- Load Historical Data from S3 ---
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key_hist)
    hist_df = pd.read_csv(response['Body'])
    hist_df = transform_data(hist_df)
    
    # --- Use All Available Historical Data for Training ---
    # (Instead of using only the last 30 days, we use the complete dataset.)
    train_df = hist_df.copy()
    
    # --- Train the Hybrid Model ---
    # Use a sequential index as the feature.
    X_train = np.arange(len(train_df)).reshape(-1, 1)
    y_train = train_df["HVAC_kWh"].values

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Compute residuals for XGBoost residual correction.
    y_fit = linear_model.predict(X_train)
    residuals = y_train - y_fit

    lags = 5
    if len(residuals) <= lags:
        xgb_model = None
    else:
        X_resid = np.array([residuals[i - lags:i] for i in range(lags, len(residuals))])
        y_resid = residuals[lags:]
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            max_depth=3,
            seed=42
        )
        xgb_model.fit(X_resid, y_resid)

    # --- Forecasting the Next 7 Days ---
    # Each day consists of 144 rows. For a 7-day forecast, forecast 144 * 7 = 1008 rows.
    rows_per_day = 144
    forecast_days = 1
    forecast_horizon = rows_per_day * forecast_days  # 1008 rows
    last_index = len(train_df)  # The forecast starts after the last row of historical data

    X_forecast = np.arange(last_index, last_index + forecast_horizon).reshape(-1, 1)
    base_forecast = linear_model.predict(X_forecast)

    if xgb_model is not None and forecast_horizon > lags:
        forecast_predictions = base_forecast.copy()
        # Apply recursive residual correction.
        # For each new forecast row starting at index 'lags', use the previously corrected forecast values.
        for i in range(lags, forecast_horizon):
            window = forecast_predictions[i - lags:i] - base_forecast[i - lags:i]
            window = window.reshape(1, -1)
            resid_corr = xgb_model.predict(window)[0]
            forecast_predictions[i] += resid_corr
        final_forecast = forecast_predictions
    else:
        final_forecast = base_forecast

    # --- Save Forecast Predictions to S3 ---
    forecast_indices = np.arange(last_index, last_index + forecast_horizon)
    forecast_df = pd.DataFrame({
        'Index': forecast_indices,
        'Predicted_HVAC_kWh': final_forecast
    })
    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=s3_bucket, Key=s3_key_forecast, Body=csv_buffer.getvalue())
    print("Forecast predictions saved to S3 at key:", s3_key_forecast)
    
    # --- (Optional) Compute Metrics on Training Data ---
    # Here you can compute in-sample metrics using the training data if desired.
    train_metrics = compute_metrics(y_train, linear_model.predict(X_train))
    print("In-sample training metrics:", train_metrics)
    
    # --- Plot the Forecast Predictions ---
    plt.figure(figsize=(14, 7))
    plt.plot(forecast_df['Index'], forecast_df['Predicted_HVAC_kWh'],
             label="7-Day Forecast", linestyle="--", marker="o", color="blue")
    plt.xlabel("Time Index (each row represents a time step; 144 rows = 1 day)")
    plt.ylabel("Predicted HVAC kWh")
    plt.title("7-Day Forecast of HVAC Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Save Model Artifacts for Deployment ---
    model_dir = os.environ.get("SM_MODEL_DIR", "./model_artifacts")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
        pickle.dump(linear_model, f)
    with open(os.path.join(model_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

if __name__ == "__main__":
    main()
