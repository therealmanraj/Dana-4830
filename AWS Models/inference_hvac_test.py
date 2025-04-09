#!/usr/bin/env python
import os
import pickle
import io
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

def model_fn(model_dir):
    """
    Load model artifacts from the model_dir.  
    This function is called by SageMaker on endpoint initialization.
    It loads the linear_model.pkl and, if available, xgb_model.pkl files.
    """
    # Load the linear regression model
    linear_model_path = os.path.join(model_dir, "linear_model.pkl")
    with open(linear_model_path, "rb") as f:
        linear_model = pickle.load(f)

    # Load the XGBoost model if it exists
    xgb_model_path = os.path.join(model_dir, "xgb_model.pkl")
    if os.path.exists(xgb_model_path):
        with open(xgb_model_path, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        xgb_model = None

    return {"linear": linear_model, "xgb": xgb_model}

def input_fn(input_data, content_type):
    """
    Deserialize the incoming request data.
    We expect CSV format.
    """
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(input_data))
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(data, model):
    """
    Apply the model to the incoming request.
    Here we use a sequential index as our feature for the linear model.
    We also perform an optional XGBoost residual correction if the model exists.
    
    The DataFrame `data` is expected to include a column named 'HVAC_kWh' containing observed values.
    """
    linear_model = model["linear"]
    xgb_model = model["xgb"]

    # Number of records in the input payload
    n = data.shape[0]
    # Create a sequential feature array
    X_seq = np.arange(n).reshape(-1, 1)
    # Get base predictions from the linear model
    predictions = linear_model.predict(X_seq)

    # Set the lag value (must match the training)
    lags = 5
    # If the XGBoost model is available and we have enough data, do residual corrections.
    if xgb_model is not None and n > lags and "HVAC_kWh" in data.columns:
        # Extract observed values from the data
        observed = data["HVAC_kWh"].values
        # Build residual windows for indices [lags, n)
        X_resid = []
        for i in range(lags, n):
            window = observed[i - lags:i] - predictions[i - lags:i]
            X_resid.append(window)
        X_resid = np.array(X_resid)
        # Predict residual corrections using the XGBoost model
        resid_corrections = xgb_model.predict(X_resid)
        # Add residual corrections to the base predictions (starting at index lags)
        predictions[lags:] += resid_corrections

    return predictions

def output_fn(prediction, accept):
    """
    Serialize the prediction to CSV format.
    """
    if accept == "text/csv":
        # Wrap predictions in a DataFrame so they can be returned as CSV
        output_df = pd.DataFrame(prediction, columns=["Predicted_HVAC_kWh"])
        buffer = io.StringIO()
        output_df.to_csv(buffer, index=False)
        return buffer.getvalue()
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))

# For local testing of the inference script
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: {} <input_csv> <output_csv>".format(sys.argv[0]))
    # Use SM_MODEL_DIR if defined, otherwise default to current directory
    model_dir = os.environ.get("SM_MODEL_DIR", ".")
    # Load the model using model_fn
    model = model_fn(model_dir)
    # Read the input CSV from file
    with open(sys.argv[1], 'r') as f:
        input_data = f.read()
    # Deserialize input
    data = input_fn(input_data, "text/csv")
    # Generate predictions
    predictions = predict_fn(data, model)
    # Serialize the output to CSV and write to file
    output_data = output_fn(predictions, "text/csv")
    with open(sys.argv[2], 'w') as f:
        f.write(output_data)
