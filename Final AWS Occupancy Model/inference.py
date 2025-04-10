#!/usr/bin/env python
import os
import pickle
import io
import pandas as pd
import numpy as np

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

def model_fn(model_dir):
    """
    Loads the occupancy model artifact from SM_MODEL_DIR.
    The artifact is expected to be a dictionary with:
      - 'linear_model': a trained LinearRegression model
      - 'xgb_model': a trained XGBRegressor (or None)
      - 'features': list of feature columns used for training
    """
    model_path = os.path.join(model_dir, "occupancy_model.pkl")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact

def input_fn(input_data, content_type):
    """
    Deserializes incoming CSV input data into a pandas DataFrame.
    """
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(input_data))
        return df
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(data, model_artifact):
    """
    Uses the loaded occupancy model to predict Total Occupants.
    The input DataFrame is expected to include the same feature columns as during training.
    """
    linear_model = model_artifact["linear_model"]
    xgb_model = model_artifact["xgb_model"]
    features = model_artifact["features"]
    
    data = transform_data(data)
    
    # Select features from data
    X = data[features]
    
    # Get base predictions from linear model
    base_pred = linear_model.predict(X)
    
    # If available and desired, apply residual correction using xgb_model.
    # Here, we use a simple recursive procedure if there are enough rows.
    lags = 5
    if xgb_model is not None and len(data) > lags:
        hybrid_pred = base_pred.copy()
        for i in range(lags, len(base_pred)):
            window = hybrid_pred[i - lags:i] - base_pred[i - lags:i]
            window = window.reshape(1, -1)
            resid_corr = xgb_model.predict(window)[0]
            hybrid_pred[i] += resid_corr
        return hybrid_pred
    else:
        return base_pred

def output_fn(prediction, accept):
    """
    Serializes the prediction (a NumPy array) into CSV format.
    """
    if accept == "text/csv":
        out_df = pd.DataFrame(prediction, columns=["Predicted_TotalOccupants"])
        buffer = io.StringIO()
        out_df.to_csv(buffer, index=False)
        return buffer.getvalue()
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))

if __name__ == "__main__":
    # For local testing, run: python3 inference.py input.csv output.csv
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: {} <input_csv> <output_csv>".format(sys.argv[0]))
    model_dir = os.environ.get("SM_MODEL_DIR", "./model_artifacts")
    artifact = model_fn(model_dir)
    with open(sys.argv[1], "r") as f:
        input_data = f.read()
    data = input_fn(input_data, "text/csv")
    preds = predict_fn(data, artifact)
    output_data = output_fn(preds, "text/csv")
    with open(sys.argv[2], "w") as f:
        f.write(output_data)
