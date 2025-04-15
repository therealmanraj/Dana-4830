# train_local.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
pd.set_option('display.max_columns', None)

# -------------------- Functions --------------------
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
    df['weekofyear'] = df.index.isocalendar().week
    return df

def compute_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted))
    }

def plot_results(y_true, y_pred):
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=y_true.index, y=y_true, label="Actual")
    sns.lineplot(x=y_pred.index, y=y_pred, label="Predicted", dashes=(4,2))
    plt.title("HVAC Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------- Main --------------------
def main():
    df = pd.read_csv("hvac_model_zones.csv")
    df = transform_data(df)
    df = add_lags(df)
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

    X_train = X[df.index.month <= 11]
    y_train = y[df.index.month <= 11]
    X_test = X[df.index.month == 12]
    y_test = y[df.index.month == 12]

    # Train hybrid model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    residuals = y_train - lin_reg.predict(X_train)

    xgb_model = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        n_estimators=200,
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.05
    )
    xgb_model.fit(X_train, residuals)

    # Predict
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_combined = y_pred_lin + y_pred_xgb
    y_pred_combined = pd.Series(y_pred_combined, index=X_test.index)

    # Metrics
    metrics = compute_metrics(y_test, y_pred_combined)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    # Plot
    plot_results(y_test, y_pred_combined)

if __name__ == "__main__":
    main()
