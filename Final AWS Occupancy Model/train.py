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
    Preprocess occupancy data:
      - Clean column names.
      - Remove rows where 'Environment:Site Day Type Index' is 0.
      - Create a 'total_occupants' column by summing any column that includes 'occupant'.
      - Optionally drop the individual occupant columns.
      - Create a timestamp (if not provided) at a 10-minute frequency.
      - Engineer additional time features: hour, day_of_week, month.
      - Create time-shifted features for 'total_occupants' (both lag and lead).
      - Create a 'WeekendOrHoliday' indicator.
      - Drop any rows with missing values.
    """
    # Clean column names
    df.columns = df.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
    
    # Remove rows where the day type is zero (if needed)
    df = df.loc[df['Environment:Site Day Type Index'] != 0]
    
    # Create 'total_occupants' from any column with 'occupant' (case-insensitive)
    occupant_cols = [col for col in df.columns if 'occupant' in col.lower()]
    if not occupant_cols:
        raise ValueError("No occupant columns found in the data.")
    df['total_occupants'] = df[occupant_cols].sum(axis=1)
    # Optionally, drop the individual occupant columns
    df.drop(occupant_cols, axis=1, inplace=True)
    
    # Create timestamp if missing – assume data arrives in 10-minute intervals
    if 'timestamp' not in df.columns:
        start_datetime = pd.Timestamp(year=2004, month=1, day=1, hour=0, minute=0)
        df['timestamp'] = [start_datetime + pd.Timedelta(minutes=10 * i) for i in range(len(df))]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    df = df.sort_values('timestamp')
    
    # Create time-shifted features for total_occupants
    time_shifts = [0.5, 1, 1.5, 2]  # in hours
    for h in time_shifts:
        steps = int(h * 6)  # 6 intervals per hour
        df[f"Occ_minus{int(h*60)}"] = df['total_occupants'].shift(steps)
    for h in time_shifts:
        steps = int(h * 6)
        df[f"Occ_plus{int(h*60)}"] = df['total_occupants'].shift(-steps)
    
    # Create a WeekendOrHoliday indicator based on Environment:Site Day Type Index
    df["WeekendOrHoliday"] = df["Environment:Site Day Type Index"].apply(lambda x: 1 if x in [0,6,7] else 0)
    
    df = df.dropna()
    return df

def main():
    # --- AWS S3 configuration ---
    region = 'ca-central-1'
    s3_bucket = 'dana-minicapstone-ca'
    # Occupancy data is stored at:
    s3_key_data = 'data/occupancy_train.csv'
    
    # Create S3 client
    s3 = boto3.client('s3', region_name=region)
    
    # --- Load Data from S3 ---
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key_data)
    df = pd.read_csv(response['Body'])
    df = transform_data(df)
    
    # --- Split Data into Training (80%) and Testing (20%) ---
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df  = df.iloc[train_size:].copy()
    
    # Define features and target: use all columns except target and timestamp
    features = [col for col in df.columns if col not in ['total_occupants', 'timestamp']]
    target = 'total_occupants'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test  = test_df[features]
    y_test  = test_df[target]
    
    # --- Train Hybrid Model ---
    # Train a linear regression model:
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_train_pred = linear_model.predict(X_train)
    
    # Train an XGBoost regressor on the residuals:
    residuals_train = y_train.values - y_train_pred
    lags = 5
    if len(residuals_train) > lags:
        X_train_resid = np.array([residuals_train[i - lags:i] for i in range(lags, len(residuals_train))])
        y_train_resid = residuals_train[lags:]
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_estimators=100,
                                     max_depth=3,
                                     random_state=42)
        xgb_model.fit(X_train_resid, y_train_resid)
    else:
        xgb_model = None
    
    # --- Evaluate on Test Data ---
    y_test_pred_linear = linear_model.predict(X_test)
    if xgb_model is not None and len(y_test) > lags:
        residuals_test = y_test.values - y_test_pred_linear
        X_test_resid = np.array([residuals_test[i - lags:i] for i in range(lags, len(residuals_test))])
        resid_test_pred = xgb_model.predict(X_test_resid)
        y_test_hybrid = y_test_pred_linear.copy()
        y_test_hybrid[lags:] += resid_test_pred
    else:
        y_test_hybrid = y_test_pred_linear
    metrics = compute_metrics(y_test.values[lags:], y_test_hybrid[lags:])
    print("Test Metrics:", metrics)
    
    # --- Save Model Artifact ---
    # Save a dictionary that includes the linear model, xgboost model, and feature list.
    artifact = {
        "linear_model": linear_model,
        "xgb_model": xgb_model,
        "features": features
    }
    model_dir = os.environ.get("SM_MODEL_DIR", "./model_artifacts")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "occupancy_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print("Occupancy model saved to:", model_path)

if __name__ == "__main__":
    main()
