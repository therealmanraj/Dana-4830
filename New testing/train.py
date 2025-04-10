#!/usr/bin/env python
import io
import os
import pickle
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
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
    # S3 configuration
    region = 'ca-central-1'
    s3_bucket = 'dana-minicapstone-ca'
    
    # S3 keys for train, test, and output predictions
    s3_key_train = 'data/hvac_train.csv'
    s3_key_test  = 'data/hvac_test.csv'
    s3_key_pred  = 'data/hvac_pred.csv'
    
    # Create S3 client
    s3 = boto3.client('s3', region_name=region)
    
    # --- Training Phase ---
    # 1. Read TRAIN data from S3 and preprocess
    train_response = s3.get_object(Bucket=s3_bucket, Key=s3_key_train)
    train_df = pd.read_csv(train_response['Body'])
    train_df = transform_data(train_df)
    
    # 2. Train the hybrid model (Linear Regression + XGBoost residual)
    X_train = np.arange(len(train_df)).reshape(-1, 1)
    y_train = train_df["HVAC_kWh"].values
    
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Compute residuals for XGBoost
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
    
    # --- Testing/Prediction Phase ---
    # 3. Read TEST data from S3 and transform
    test_response = s3.get_object(Bucket=s3_bucket, Key=s3_key_test)
    test_df = pd.read_csv(test_response['Body'])
    test_df = transform_data(test_df)
    
    if len(test_df) == 0:
        print("No test data after filtering; skipping predictions.")
        return
    
    # 4. Generate predictions on the test set
    X_test = np.arange(len(train_df), len(train_df) + len(test_df)).reshape(-1, 1)
    y_pred_linear = linear_model.predict(X_test)
    
    if xgb_model and len(test_df) >= lags:
        y_test = test_df["HVAC_kWh"].values
        X_test_resid = [y_test[i - lags:i] - y_pred_linear[i - lags:i] for i in range(lags, len(y_test))]
        X_test_resid = np.array(X_test_resid)
        resid_test_predictions = xgb_model.predict(X_test_resid)
        y_pred_boosted = y_pred_linear.copy()
        y_pred_boosted[lags:] += resid_test_predictions
        final_predictions = y_pred_boosted
    else:
        final_predictions = y_pred_linear
    
    # 5. Compute metrics on the test set
    if len(test_df) >= lags and xgb_model:
        actual_test = test_df["HVAC_kWh"].values[lags:]
        pred_test   = final_predictions[lags:]
        metrics = compute_metrics(actual_test, pred_test)
    else:
        actual_test = test_df["HVAC_kWh"].values
        pred_test   = final_predictions
        metrics = compute_metrics(actual_test, pred_test)
    
    print("Test Metrics:", metrics)
    
    # 6. Prepare predictions DataFrame and upload to S3
    if len(test_df) >= lags and xgb_model:
        test_index = test_df.index[lags:]
        out_actual = test_df["HVAC_kWh"].iloc[lags:].values
        out_pred   = final_predictions[lags:]
    else:
        test_index = test_df.index
        out_actual = test_df["HVAC_kWh"].values
        out_pred   = final_predictions
    
    prediction_df = pd.DataFrame({
        'Index': test_index,
        'Actual': out_actual,
        'Predicted': out_pred
    })
    
    csv_buffer = io.StringIO()
    prediction_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=s3_bucket, Key=s3_key_pred, Body=csv_buffer.getvalue())
    
    # --- Save Model Artifacts for Deployment ---
    # Write the trained models to the directory that SageMaker uses: /opt/ml/model
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # model_dir = os.environ.get("SM_MODEL_DIR", "./model_artifacts")
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    
    # Save the Linear Regression model
    with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
        pickle.dump(linear_model, f)
    
    # Save the XGBoost model if it was trained
    if xgb_model is not None:
        with open(os.path.join(model_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)

if __name__ == "__main__":
    main()
