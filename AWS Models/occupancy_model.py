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
import matplotlib.pyplot as plt
import sys
import traceback

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
        print("Permission denied for {}. Falling back to {}".format(dir_path, fallback_dir))
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir
    return dir_path

def train_model(data_path, model_output_dir):
    try:
        # Download CSV if provided as an S3 URI
        if data_path.startswith("s3://"):
            local_data_path = os.path.join("/tmp", os.path.basename(data_path))
            print("Downloading {} to {}".format(data_path, local_data_path))
            S3Downloader.download(data_path, "/tmp")
        else:
            local_data_path = data_path

        print("Local data path: {}".format(local_data_path))
        occupancy_data = pd.read_csv(local_data_path)
        print("Data read successfully. Shape: {}".format(occupancy_data.shape))
        print("Columns: {}".format(occupancy_data.columns.tolist()))

        # Data preprocessing
        occupancy_data.columns = occupancy_data.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
        occupancy_data = occupancy_data.loc[occupancy_data['Environment:Site Day Type Index'] != 0]
        occupant_cols = [col for col in occupancy_data.columns if 'occupant' in str(col.lower())]
        print("Found occupant columns: {}".format(occupant_cols))
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
            occupancy_data["Occ_minus{}".format(int(h*60))] = occupancy_data["total_occupants"].shift(steps)
            occupancy_data["Occ_plus{}".format(int(h*60))] = occupancy_data["total_occupants"].shift(-steps)
        
        occupancy_data["WeekendOrHoliday"] = occupancy_data["Environment:Site Day Type Index"].apply(lambda x: 1 if x in [0,6,7] else 0)
        occupancy_data = occupancy_data.dropna()
        print("Final dataset shape after preprocessing: {}".format(occupancy_data.shape))

        features = [col for col in occupancy_data.columns if col not in ['total_occupants', 'timestamp']]
        target = 'total_occupants'
        total_rows = len(occupancy_data)
        training_rows = 30 * 144  # 4320 rows for 30 days
        if total_rows < training_rows:
            raise ValueError("Not enough data for 30-day training window.")
        train_data = occupancy_data.iloc[-training_rows:]
        print("Using last {} rows for training.".format(training_rows))

        X_train = train_data[features]
        y_train = train_data[target]

        # Train Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_train_pred = linear_model.predict(X_train)
        residuals_train = y_train.values - y_train_pred

        lags = 5
        X_train_resid = np.array([residuals_train[i - lags:i] for i in range(lags, len(residuals_train))])
        y_train_resid = residuals_train[lags:]
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
        xgb_model.fit(X_train_resid, y_train_resid)

        # (Optional) Generate and save a plot of predictions
        y_test_pred_linear = linear_model.predict(train_data[features])
        residuals_test = y_train.values - y_test_pred_linear
        X_test_resid = np.array([residuals_test[i - lags:i] for i in range(lags, len(residuals_test))])
        resid_test_pred = xgb_model.predict(X_test_resid)
        # In this simple code, we reuse y_test_pred_linear as a placeholder
        y_test_hybrid = y_test_pred_linear[lags:] + resid_test_pred

        plt.figure(figsize=(20, 6))
        X_train_resid_pred = np.array([residuals_train[i - lags:i] for i in range(lags, len(residuals_train))])
        resid_train_pred = xgb_model.predict(X_train_resid_pred)
        y_train_hybrid = y_train_pred[lags:] + resid_train_pred
        plt.plot(train_data.index, y_train, label='Training Actual', color='blue', linewidth=5)
        plt.plot(train_data.index[lags:], y_train_hybrid, label='Training Hybrid Predictions', color='red', linestyle='--', linewidth=2)
        plt.legend()
        plt.title('Hybrid Model Predictions')
        plt.xlabel('Index')
        plt.ylabel('Total Occupants')
        plt.grid(True)
        plt.tight_layout()
        plot_file = os.path.join(model_output_dir, "training_plot.png")
        plt.savefig(plot_file)
        print("Plot saved to {}".format(plot_file))

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
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, default="s3://dana-minicapstone/data/occupancy_model_zones.csv",
                            help="Path to the occupancy CSV data (S3 or local)")
        parser.add_argument("--model-output-dir", type=str, default="/opt/ml/model",
                            help="Directory to save model artifacts")
        args = parser.parse_args()
        train_model(args.data_path, args.model_output_dir)
    except Exception as e:
        print("Error in script:")
        traceback.print_exc()
        raise
