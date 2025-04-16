# train.py
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import boto3

import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

def transform_data(df):
    df.columns = df.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
    df = df.loc[df['Environment:Site Day Type Index'] != 0]
    df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
    df.drop(columns='Electricity:HVAC', inplace=True)
    occupant_cols = [col for col in df.columns if 'Occupant' in col]
    df["TotalOccCount"] = df[occupant_cols].sum(axis=1)
    df.drop(columns=occupant_cols, inplace=True)
    df.index = pd.date_range(start="2004-01-01 00:00:00", periods=len(df), freq="10min")
    return df

def add_lags(df):
    for col in df.columns:
        target_map = df[col].to_dict()
        df[f'{col}_lag1'] = (df.index - pd.Timedelta('1 days')).map(target_map)
        df[f'{col}_lag2'] = (df.index - pd.Timedelta('3 days')).map(target_map)
        df[f'{col}_lag3'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    return df

def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofmonth'] = df.index.day
    # df['weekofyear'] = df.index.isocalendar().week
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    return df

def compute_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted))
    }

def train_model(X_train, y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    residuals = y_train - lin_reg.predict(X_train)
    xgb_reg = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        n_estimators=200,
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.05
    )
    xgb_reg.fit(X_train, residuals, verbose=100)
    return lin_reg, xgb_reg

def main():
    print("Loading data...")
    region = 'ca-central-1'
    s3_bucket = 'dana-minicapstone-ca'
    s3_key = 'data/hvac_model_zones.csv'
    s3 = boto3.client('s3', region_name=region)
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(response['Body'])
        
    print("Transform data...")
    df = transform_data(df)
    
    print("Adding Lags to data...")
    df = add_lags(df)
    
    print("Creating Features data...")
    df = create_features(df)
    
    df = df.bfill()

    features = ['weekofyear','hour','dayofmonth','month',
                'HVAC_kWh_lag3','TotalOccCount_lag3','TotalOccCount_lag2',
                'TotalOccCount_lag1','Environment:Site Day Type Index_lag1',
                'Environment:Site Outdoor Air Drybulb Temperature_lag2',
                'Environment:Site Outdoor Air Drybulb Temperature_lag1',
                'Environment:Site Outdoor Air Wetbulb Temperature_lag1',
                'Environment:Site Outdoor Air Wetbulb Temperature_lag3',
                'HVAC_kWh_lag1','HVAC_kWh_lag2']

    target = 'HVAC_kWh'

    X = df[features]
    y = df[target]

    print("Splitting data...")
    X_train = X[df.index.month <= 11]
    y_train = y[df.index.month <= 11]
    
    print("Training model...")
    lin_reg_model, xgb_model = train_model(X_train, y_train)

    print("Saving models...")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lin_reg_model, os.path.join(model_dir, 'lin_reg_model.pkl'))
    joblib.dump(xgb_model, os.path.join(model_dir, 'xgb_model.pkl'))

    print("Model training completed and saved.")

if __name__ == "__main__":
    main()
