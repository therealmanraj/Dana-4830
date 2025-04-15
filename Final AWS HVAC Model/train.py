import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

def compute_metrics(test_data, forecast_mean):
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}

def add_lags(df):
    for col in df.columns:
        target_map = df[col].to_dict()
        df[f'{col}_lag1'] = (df.index - pd.Timedelta('7 days')).map(target_map)
        df[f'{col}_lag2'] = (df.index - pd.Timedelta('14 days')).map(target_map)
        df[f'{col}_lag3'] = (df.index - pd.Timedelta('21 days')).map(target_map)
    return df

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def train_model(data_path, model_output_dir):
    # Load data
    hvac_data = pd.read_csv(data_path)
    hvac_data.columns = hvac_data.columns.str.replace(r"^b'|'$|\\[.*?\\]", "", regex=True)
    hvac_data = hvac_data.loc[hvac_data['Environment:Site Day Type Index'] != 0]
    hvac_data["HVAC_kWh"] = hvac_data["Electricity:HVAC"] * 2.77778e-7
    hvac_data.drop(columns='Electricity:HVAC', axis=1, inplace=True)

    occupant_cols = [col for col in hvac_data.columns if 'Occupant' in col]
    hvac_data["TotalOccCount"] = hvac_data[occupant_cols].sum(axis=1)
    hvac_data.drop(columns=occupant_cols, axis=1, inplace=True)

    lenOfData = len(hvac_data)
    hvac_data.index = pd.date_range(start="2004-01-01 00:00:00", periods=lenOfData, freq="10min")

    hvac_data = add_lags(hvac_data)
    hvac_data = create_features(hvac_data)

    features = [col for col in hvac_data.columns if '_lag' in col]
    features += ['hour', 'dayofweek', 'month', 'year', 'dayofmonth', 'weekofyear']
    target = 'HVAC_kWh'

    hvac_data = hvac_data[features + [target]]

    x_train = hvac_data[features][hvac_data.index.month < 10].bfill()
    y_train = hvac_data[target][hvac_data.index.month < 10]

    # Linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    y_train_residuals = y_train - pd.Series(lin_reg.predict(x_train), index=x_train.index)

    xgb_model = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        n_estimators=200,
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.05
    )
    xgb_model.fit(x_train, y_train_residuals)

    os.makedirs(model_output_dir, exist_ok=True)
    joblib.dump(lin_reg, os.path.join(model_output_dir, "lin_reg_model.joblib"))
    joblib.dump(xgb_model, os.path.join(model_output_dir, "xgb_model.joblib"))
    joblib.dump(features, os.path.join(model_output_dir, "features.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_output_dir', type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path, args.model_output_dir)
