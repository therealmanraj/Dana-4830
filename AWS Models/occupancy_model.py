#!/usr/bin/env python
import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sagemaker.s3 import S3Downloader
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
import traceback

import subprocess
subprocess.check_call(["pip", "install", "xgboost"])


def compute_metrics(test_data, forecast_mean):
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}

def safe_makedirs(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except PermissionError:
        fallback_dir = "/tmp/ml_model"
        print(f"Permission denied for {dir_path}. Falling back to {fallback_dir}")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir
    return dir_path

def train_model(data_path, model_output_dir):
    try:
        # Download data if S3
        if data_path.startswith("s3://"):
            local_data_path = os.path.join("/tmp", os.path.basename(data_path))
            print(f"Downloading {data_path} to {local_data_path}")
            S3Downloader.download(data_path, "/tmp")
        else:
            local_data_path = data_path

        print("Local data path:", local_data_path)
        occupancy_data = pd.read_csv(local_data_path)
        print("Data read successfully.")
        print("Columns:", occupancy_data.columns.tolist())
        print("Shape:", occupancy_data.shape)

        # Data preprocessing steps
        occupancy_data.columns = occupancy_data.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
        occupancy_data = occupancy_data.loc[occupancy_data['Environment:Site Day Type Index'] != 0]
        occupant_cols = [col for col in occupancy_data.columns if 'occupant' in str(col.lower())]
        print("Found occupant columns:", occupant_cols)
        occupancy_data['total_occupants'] = occupancy_data[occupant_cols].sum(axis=1)
        occupancy_data.drop(occupant_cols, axis=1, inplace=True)

        start_datetime = pd.Timestamp(year=2004, month=1, day=1, hour=0, minute=0)
        occupancy_data['timestamp'] = [start_datetime + pd.Timedelta(minutes=10 * i) for i in range(len(occupancy_data))]
        occupancy_data['timestamp'] = pd.to_datetime(occupancy_data['timestamp'])
        occupancy_data['hour'] = occupancy_data['timestamp'].dt.hour
        occupancy_data['day_of_week'] = occupancy_data['timestamp'].dt.dayofweek
        occupancy_data['month'] = occupancy_data['timestamp'].dt.month
        occupancy_data = occupancy_data.sort_values('timestamp')

        time_shifts = [0.5, 1, 1.5, 2]
        for h in time_shifts:
            steps = int(h * 6)
            occupancy_data[f"Occ_minus{int(h*60)}"] = occupancy_data["total_occupants"].shift(steps)
            occupancy_data[f"Occ_plus{int(h*60)}"] = occupancy_data["total_occupants"].shift(-steps)
        
        occupancy_data["WeekendOrHoliday"] = occupancy_data["Environment:Site Day Type Index"].apply(lambda x: 1 if x in [0,6,7] else 0)
        occupancy_data = occupancy_data.dropna()

        features = [col for col in occupancy_data.columns if col not in ['total_occupants', 'timestamp']]
        target = 'total_occupants'
        print("Final dataset shape:", occupancy_data.shape)

        total_rows = len(occupancy_data)
        training_rows = 30 * 144  # 4320 rows for 30 days
        if total_rows < training_rows:
            raise ValueError("Not enough data for 30-day training window.")
        train_data = occupancy_data.iloc[-training_rows:]
        print("Using last", training_rows, "rows for training.")

        X_train = train_data[features]
        y_train = train_data[target]

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_train_pred = linear_model.predict(X_train)
        residuals_train = y_train.values - y_train_pred

        lags = 5
        X_train_resid = np.array([residuals_train[i - lags:i] for i in range(lags, len(residuals_train))])
        y_train_resid = residuals_train[lags:]

        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
        xgb_model.fit(X_train_resid, y_train_resid)

        # Save models
        model_output_dir = safe_makedirs(model_output_dir)
        with open(os.path.join(model_output_dir, "linear_model.pkl"), "wb") as f:
            pickle.dump(linear_model, f)
        with open(os.path.join(model_output_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)

        print("Training complete and models saved.")
    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import traceback
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, default="s3://dana-minicapstone/data/occupancy_model_zones.csv")
        parser.add_argument("--model-output-dir", type=str, default="/opt/ml/model")
        args = parser.parse_args()
        train_model(args.data_path, args.model_output_dir)
    except Exception as e:
        print("Error in script:")
        traceback.print_exc()
        raise
