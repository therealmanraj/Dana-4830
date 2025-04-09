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
    This function loads the model artifacts from the model directory.
    It loads linear_model.pkl and, if available, xgb_model.pkl.
    """
    linear_model_path = os.path.join(model_dir, "linear_model.pkl")
    with open(linear_model_path, "rb") as f:
        linear_model = pickle.load(f)
    
    xgb_model_path = os.path.join(model_dir, "xgb_model.pkl")
    if os.path.exists(xgb_model_path):
        with open(xgb_model_path, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        xgb_model = None

    return {"linear": linear_model, "xgb": xgb_model}

# def input_fn(input_data, content_type):
#     """
#     Deserializes the incoming request data (expected to be CSV) into a DataFrame.
#     """
#     if content_type == "text/csv":
#         return pd.read_csv(io.StringIO(input_data))
#     else:
#         raise ValueError("Unsupported content type: {}".format(content_type))

def input_fn(input_data, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(input_data))
        # If HVAC_kWh is missing, compute it:
        if "HVAC_kWh" not in df.columns and "Electricity:HVAC" in df.columns:
            df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
        return df
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))


def predict_fn(data, model):
    """
    Generates predictions from the input data using the loaded models.
    This example uses a sequential index as a feature.
    If available, applies XGBoost residual correction.
    """
    linear_model = model["linear"]
    xgb_model = model["xgb"]
    
    n = data.shape[0]
    # Create sequential feature for prediction
    X_seq = np.arange(n).reshape(-1, 1)
    predictions = linear_model.predict(X_seq)
    
    lags = 5
    # If using XGBoost residual correction and if sufficient data is available:
    if xgb_model is not None and n > lags and "HVAC_kWh" in data.columns:
        observed = data["HVAC_kWh"].values
        X_resid = []
        for i in range(lags, n):
            window = observed[i - lags:i] - predictions[i - lags:i]
            X_resid.append(window)
        X_resid = np.array(X_resid)
        resid_corrections = xgb_model.predict(X_resid)
        predictions[lags:] += resid_corrections

    return predictions

def output_fn(prediction, accept):
    """
    Serializes the prediction (as CSV) to return to the client.
    """
    if accept == "text/csv":
        out_df = pd.DataFrame(prediction, columns=["Predicted_HVAC_kWh"])
        buffer = io.StringIO()
        out_df.to_csv(buffer, index=False)
        return buffer.getvalue()
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))

# For local testing of the inference script:
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: {} <input_csv> <output_csv>".format(sys.argv[0]))
    model_dir = os.environ.get("SM_MODEL_DIR", ".")
    model = model_fn(model_dir)
    with open(sys.argv[1], 'r') as f:
        input_data = f.read()
    data = input_fn(input_data, "text/csv")
    predictions = predict_fn(data, model)
    output_data = output_fn(predictions, "text/csv")
    with open(sys.argv[2], 'w') as f:
        f.write(output_data)
