#!/usr/bin/env python
import io
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
    Applies the same preprocessing steps to both train and test data:
      - Remove rows where 'Environment:Site Day Type Index' is 0.
      - Convert 'Electricity:HVAC' to kWh.
      - Sum occupant columns into 'TotalOccupantCount'.
      - Create time-shifted occupant features (Occ_minusXX, Occ_plusXX).
      - Create a 'WeekendOrHoliday' indicator.
      - Drop rows with missing values.
    """
    # Filter out rows with day index == 0
    df = df.loc[df['Environment:Site Day Type Index'] != 0]
    
    # Convert HVAC from J to kWh
    df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
    
    # Sum occupant columns (if any)
    occupant_cols = [col for col in df.columns if 'Occupant' in col]
    df["TotalOccupantCount"] = df[occupant_cols].sum(axis=1)
    
    # Create time-shifted occupant features (though they are not used in this training script)
    time_shifts = [0.5, 1, 1.5, 2]
    for h in time_shifts:
        steps = int(h * 6)
        df[f"Occ_minus{int(h*60)}"] = df["TotalOccupantCount"].shift(steps)
    for h in time_shifts:
        steps = int(h * 6)
        df[f"Occ_plus{int(h*60)}"] = df["TotalOccupantCount"].shift(-steps)
    
    # Flag for weekend or holiday
    df["WeekendOrHoliday"] = df["Environment:Site Day Type Index"].apply(
        lambda x: 1 if x in [0, 6, 7] else 0
    )
    
    # Drop any rows that became NaN due to shifting
    df = df.dropna()
    
    return df

def main():
    # -- S3 configuration --
    region = 'us-east-2'
    s3_bucket = 'dana-minicapstone'
    
    # Paths in S3
    s3_key_train = 'data/hvac_train.csv'       # training data
    s3_key_test  = 'data/hvac_test.csv'        # testing data
    s3_key_pred  = 'data/hvac_pred.csv'        # output predictions
    
    # Create S3 client
    s3 = boto3.client('s3', region_name=region)
    
    # 1. Read TRAIN data from S3 directly into memory
    train_response = s3.get_object(Bucket=s3_bucket, Key=s3_key_train)
    train_df = pd.read_csv(train_response['Body'])
    train_df = transform_data(train_df)
    
    # 2. Train the hybrid model (Linear + XGBoost residual)
    X_train = np.arange(len(train_df)).reshape(-1, 1)
    y_train = train_df["HVAC_kWh"].values
    
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Residuals for XGBoost
    y_fit = linear_model.predict(X_train)
    residuals = y_train - y_fit
    
    # Train XGBoost on the residuals
    lags = 5
    if len(residuals) <= lags:
        # If there's not enough data for the lag, skip XGBoost training
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
    
    # 3. Read TEST data from S3 and transform
    test_response = s3.get_object(Bucket=s3_bucket, Key=s3_key_test)
    test_df = pd.read_csv(test_response['Body'])
    test_df = transform_data(test_df)
    
    if len(test_df) == 0:
        # No test data, so nothing to predict
        print("No test data after filtering; skipping predictions.")
        return
    
    # 4. Predict on test set using linear model + optional XGBoost residual correction
    #    We preserve the "timeline" by continuing where train_data left off
    X_test = np.arange(len(train_df), len(train_df) + len(test_df)).reshape(-1, 1)
    y_pred_linear = linear_model.predict(X_test)
    
    if xgb_model and len(test_df) >= lags:
        # Prepare test residual windows for XGBoost
        y_test = test_df["HVAC_kWh"].values
        X_test_resid = [
            y_test[i - lags:i] - y_pred_linear[i - lags:i]
            for i in range(lags, len(y_test))
        ]
        X_test_resid = np.array(X_test_resid)
        
        resid_test_predictions = xgb_model.predict(X_test_resid)
        # Combine linear + XGBoost predictions
        y_pred_boosted = y_pred_linear.copy()
        y_pred_boosted[lags:] += resid_test_predictions
        final_predictions = y_pred_boosted
    else:
        # Fall back to linear model if XGBoost model wasn't trained or not enough data
        final_predictions = y_pred_linear
    
    # 5. Compute metrics on the test set (only where predictions exist)
    #    The earliest `lags` points might not have a "boosted" residual if using XGBoost,
    #    so we compute metrics on the region where final predictions are valid.
    if len(test_df) >= lags and xgb_model:
        actual_test = test_df["HVAC_kWh"].values[lags:]
        pred_test   = final_predictions[lags:]
        metrics = compute_metrics(actual_test, pred_test)
    else:
        actual_test = test_df["HVAC_kWh"].values
        pred_test   = final_predictions
        metrics = compute_metrics(actual_test, pred_test)
    
    print("Test Metrics:", metrics)
    
    # 6. Prepare predictions DataFrame
    #    Align indices to the portion where final_predictions are valid
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
    
    # 7. Upload predictions to S3 from an in-memory CSV
    csv_buffer = io.StringIO()
    prediction_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=s3_bucket, Key=s3_key_pred, Body=csv_buffer.getvalue())

if __name__ == "__main__":
    main()
