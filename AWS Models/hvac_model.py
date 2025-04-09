#!/usr/bin/env python
import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sagemaker.s3 import S3Downloader
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt  # Optional: for local debugging plots
import sys
import traceback

def compute_metrics(test_data, forecast_mean):
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}

def safe_makedirs(dir_path):
    # In SageMaker training the /opt/ml/model should be writable.
    # For local testing if permission issues occur, fall back to /tmp/ml_model.
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
        # If the data path is an S3 URI, download the CSV file to /tmp
        if data_path.startswith("s3://"):
            local_data_path = os.path.join("/tmp", os.path.basename(data_path))
            print("Downloading {} to {}".format(data_path, local_data_path))
            S3Downloader.download(data_path, "/tmp")
        else:
            local_data_path = data_path

        print("Local data path: {}".format(local_data_path))
        hvac_data = pd.read_csv(local_data_path)
        print("HVAC Data Shape:", hvac_data.shape)
        print("Columns:", hvac_data.columns.tolist())

        # Preprocess the data
        hvac_data.columns = hvac_data.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
        hvac_data = hvac_data.loc[hvac_data['Environment:Site Day Type Index'] != 0]
        # Create a new column for HVAC energy consumption in kWh
        hvac_data["HVAC_kWh"] = hvac_data["Electricity:HVAC"] * 2.77778e-7

        # Sum any occupant-related columns into TotalOccupantCount (if needed)
        occupant_cols = [col for col in hvac_data.columns if 'Occupant' in col]
        if occupant_cols:
            hvac_data["TotalOccupantCount"] = hvac_data[occupant_cols].sum(axis=1)
        else:
            print("No occupant columns found.")
        
        # (Optional) Create time-shifted features from TotalOccupantCount
        time_shifts = [0.5, 1, 1.5, 2]
        for h in time_shifts:
            steps = int(h * 6)
            hvac_data["Occ_minus{}".format(int(h*60))] = hvac_data["TotalOccupantCount"].shift(steps)
        for h in time_shifts:
            steps = int(h * 6)
            hvac_data["Occ_plus{}".format(int(h*60))] = hvac_data["TotalOccupantCount"].shift(-steps)
        
        # Create a binary indicator for weekends/holidays
        hvac_data["WeekendOrHoliday"] = hvac_data["Environment:Site Day Type Index"].apply(
            lambda x: 1 if x in [0, 6, 7] else 0
        )
        hvac_data = hvac_data.dropna()
        print("Final dataset shape after preprocessing:", hvac_data.shape)

        # For simplicity, use HVAC_kWh as the sole feature and target.
        features = ['HVAC_kWh']
        target = 'HVAC_kWh'

        # Split data: use 80% for training and 20% for testing
        train_size = int(len(hvac_data) * 0.8)
        train_data = hvac_data.iloc[:train_size]
        test_data  = hvac_data.iloc[train_size:]

        # Use sequential indices as the feature (i.e. time index)
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data[target].values

        # Train a simple Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_train_pred = linear_model.predict(X_train)
        residuals = y_train - y_train_pred

        # Set up a lag of 5 for the XGBoost model (for residual correction)
        lags = 5
        if len(residuals) <= lags:
            raise ValueError("Not enough residual data for lag={}".format(lags))
        X_resid = np.array([residuals[i-lags:i] for i in range(lags, len(residuals))])
        y_resid = residuals[lags:]
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                                      max_depth=3, random_state=42)
        xgb_model.fit(X_resid, y_resid)

        # (Optional) Plotting code has been commented out because it’s not needed during SageMaker training.
        """
        # Optionally, produce a plot for visual inspection
        y_fit_boosted = y_train_pred[lags:] + xgb_model.predict(X_resid)
        plt.figure(figsize=(30, 10))
        plt.plot(np.arange(len(train_data))[lags:], train_data[target].values[lags:], label='Training Data', color='blue', linewidth=5)
        plt.plot(np.arange(len(train_data))[lags:], y_fit_boosted, label='Hybrid Predictions', color='red', linestyle='--', linewidth=2)
        plt.legend()
        plt.title('Hybrid Model Predictions: Linear Regression + XGBoost Residual Correction')
        plt.xlabel('Index')
        plt.ylabel('HVAC Consumption (kWh)')
        plt.grid(True)
        plt.tight_layout()
        plot_file = os.path.join(model_output_dir, "training_plot.png")
        plt.savefig(plot_file)
        print("Plot saved to", plot_file)
        """

        # Save trained models to the output directory.
        # The directory for model artifacts must be under /opt/ml/model for SageMaker.
        model_output_dir = safe_makedirs(model_output_dir)
        with open(os.path.join(model_output_dir, "linear_model.pkl"), "wb") as f:
            pickle.dump(linear_model, f)
        with open(os.path.join(model_output_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)

        print("Training complete and models saved.")
    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="s3://dana-minicapstone/data/hvac_model_zones.csv",
                        help="Path to the HVAC CSV data (S3 or local)")
    parser.add_argument("--model-output-dir", type=str, default="/opt/ml/model",
                        help="Directory to save model artifacts")
    args = parser.parse_args()
    train_model(args.data_path, args.model_output_dir)
