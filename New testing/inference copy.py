# #!/usr/bin/env python
# import os
# import pickle
# import io
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.linear_model import LinearRegression

# def transform_data(df):
#     df = df.loc[df['Environment:Site Day Type Index'] != 0]
#     df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
#     occupant_cols = [col for col in df.columns if 'Occupant' in col]
#     df["TotalOccupantCount"] = df[occupant_cols].sum(axis=1)
    
#     time_shifts = [0.5, 1, 1.5, 2]
#     for h in time_shifts:
#         steps = int(h * 6)
#         df[f"Occ_minus{int(h*60)}"] = df["TotalOccupantCount"].shift(steps)
#     for h in time_shifts:
#         steps = int(h * 6)
#         df[f"Occ_plus{int(h*60)}"] = df["TotalOccupantCount"].shift(-steps)
    
#     df["WeekendOrHoliday"] = df["Environment:Site Day Type Index"].apply(
#         lambda x: 1 if x in [0, 6, 7] else 0
#     )
#     df = df.dropna()
#     return df


# def model_fn(model_dir):
#     """
#     This function loads the model artifacts from the model directory.
#     It loads linear_model.pkl and, if available, xgb_model.pkl.
#     """
#     try:
#         linear_model_path = os.path.join(model_dir, "linear_model.pkl")
#         with open(linear_model_path, "rb") as f:
#             linear_model = pickle.load(f)
#     except Exception as e:
#         print("Error loading linear_model:", e)
#         raise
    
#     try:
#         xgb_model_path = os.path.join(model_dir, "xgb_model.pkl")
#         if os.path.exists(xgb_model_path):
#             with open(xgb_model_path, "rb") as f:
#                 xgb_model = pickle.load(f)
#         else:
#             xgb_model = None
#     except Exception as e:
#         print("Error loading linear_model:", e)
#         raise

#     return {"linear": linear_model, "xgb": xgb_model}

# def input_fn(input_data, content_type):
#     if content_type == "text/csv":
#         df = pd.read_csv(io.StringIO(input_data))
#         # Ensure that the input has the computed field by applying the same transformation.
#         # (Make sure to import transform_data from a common utility if needed.)
#         # from train import transform_data  # if train.py and inference.py are in the same directory
#         df = transform_data(df)
#         return df
#     else:
#         raise ValueError("Unsupported content type: {}".format(content_type))


# # def input_fn(input_data, content_type):
# #     if content_type == "text/csv":
# #         df = pd.read_csv(io.StringIO(input_data))
# #         # If HVAC_kWh is missing, compute it:
# #         if "HVAC_kWh" not in df.columns and "Electricity:HVAC" in df.columns:
# #             df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
# #         return df
# #     else:
# #         raise ValueError("Unsupported content type: {}".format(content_type))


# def predict_fn(data, model):
#     """
#     Generates predictions from the input data using the loaded models.
#     This example uses a sequential index as a feature.
#     If available, applies XGBoost residual correction.
#     """
#     linear_model = model["linear"]
#     xgb_model = model["xgb"]
    
#     n = data.shape[0]
#     # Create sequential feature for prediction
#     X_seq = np.arange(n).reshape(-1, 1)
#     predictions = linear_model.predict(X_seq)
    
#     lags = 5
#     # If using XGBoost residual correction and if sufficient data is available:
#     if "HVAC_kWh" not in data.columns and "Electricity:HVAC" in data.columns:
#             data["HVAC_kWh"] = data["Electricity:HVAC"] * 2.77778e-7
            
#     if xgb_model is not None and n > lags and "HVAC_kWh" in data.columns:
#         observed = data["HVAC_kWh"].values
#         X_resid = []
#         for i in range(lags, n):
#             window = observed[i - lags:i] - predictions[i - lags:i]
#             X_resid.append(window)
#         X_resid = np.array(X_resid)
#         resid_corrections = xgb_model.predict(X_resid)
#         predictions[lags:] += resid_corrections

#     return predictions

# def output_fn(prediction, accept):
#     """
#     Serializes the prediction (as CSV) to return to the client.
#     """
#     if accept == "text/csv":
#         out_df = pd.DataFrame(prediction, columns=["Predicted_HVAC_kWh"])
#         buffer = io.StringIO()
#         out_df.to_csv(buffer, index=False)
#         return buffer.getvalue()
#     else:
#         raise ValueError("Unsupported accept type: {}".format(accept))

# # For local testing of the inference script:
# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) != 3:
#         sys.exit("Usage: {} <input_csv> <output_csv>".format(sys.argv[0]))
#     # model_dir = os.environ.get("SM_MODEL_DIR", "./model_artifacts")
#     model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
#     model = model_fn(model_dir)
#     with open(sys.argv[1], 'r') as f:
#         input_data = f.read()
#     data = input_fn(input_data, "text/csv")
#     predictions = predict_fn(data, model)
#     output_data = output_fn(predictions, "text/csv")
#     with open(sys.argv[2], 'w') as f:
#         f.write(output_data)


#!/usr/bin/env python
import os
import pickle
import io
import logging
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

# --- Setup Logging ---
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler: logs to stdout, captured by CloudWatch if deployed
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler: logs to a local file "inference_log.txt"
file_handler = logging.FileHandler("inference_log.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# S3 configuration for uploading logs (update as needed)
LOG_S3_BUCKET = 'dana-minicapstone'
LOG_S3_PREFIX = 'logs/inference_logs/'

def upload_log_to_s3():
    """Uploads the local log file to S3."""
    try:
        s3_client = boto3.client('s3', region_name=os.environ.get("AWS_DEFAULT_REGION", "ca-central-1"))
        with open("inference_log.txt", "r") as f:
            log_contents = f.read()
        import time
        log_key = LOG_S3_PREFIX + "inference_log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        s3_client.put_object(Bucket=LOG_S3_BUCKET, Key=log_key, Body=log_contents)
        logger.info("Uploaded log file to s3://%s/%s", LOG_S3_BUCKET, log_key)
    except Exception as e:
        logger.error("Failed to upload log file to S3: %s", e)

# --- Common Transformation Function ---
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

# --- Inference Functions ---
def model_fn(model_dir):
    """
    Loads the model artifacts from the model directory.
    It loads linear_model.pkl and, if available, xgb_model.pkl.
    """
    try:
        linear_model_path = os.path.join(model_dir, "linear_model.pkl")
        with open(linear_model_path, "rb") as f:
            linear_model = pickle.load(f)
        logger.info("Loaded Linear Regression model.")
    except Exception as e:
        logger.error("Error loading linear_model.pkl: %s", e)
        raise

    try:
        xgb_model_path = os.path.join(model_dir, "xgb_model.pkl")
        if os.path.exists(xgb_model_path):
            with open(xgb_model_path, "rb") as f:
                xgb_model = pickle.load(f)
            logger.info("Loaded XGBoost model.")
        else:
            logger.info("No XGBoost model found; xgb_model set to None.")
            xgb_model = None
    except Exception as e:
        logger.error("Error loading xgb_model.pkl: %s", e)
        raise

    return {"linear": linear_model, "xgb": xgb_model}

def input_fn(input_data, content_type):
    """
    Deserializes the incoming request data (expected to be CSV) into a DataFrame.
    Applies the same data transformation as during training.
    """
    try:
        if content_type == "text/csv":
            df = pd.read_csv(io.StringIO(input_data))
            logger.info("Input CSV read successfully; applying transformation.")
            df = transform_data(df)
            logger.info("Transformation complete. DataFrame shape: %s", df.shape)
            return df
        else:
            raise ValueError("Unsupported content type: {}".format(content_type))
    except Exception as e:
        logger.error("Error in input_fn: %s", e)
        raise

def predict_fn(data, model):
    """
    Generates predictions from the input data using the loaded models.
    Uses a sequential index as a feature and applies XGBoost residual correction if available.
    """
    try:
        linear_model = model["linear"]
        xgb_model = model["xgb"]
        
        n = data.shape[0]
        X_seq = np.arange(n).reshape(-1, 1)
        predictions = linear_model.predict(X_seq)
        lags = 5
        
        # Ensure HVAC_kWh column exists (should after transform_data)
        if "HVAC_kWh" not in data.columns and "Electricity:HVAC" in data.columns:
            data["HVAC_kWh"] = data["Electricity:HVAC"] * 2.77778e-7
            logger.info("Computed HVAC_kWh in predict_fn.")
            
        if xgb_model is not None and n > lags and "HVAC_kWh" in data.columns:
            observed = data["HVAC_kWh"].values
            X_resid = []
            for i in range(lags, n):
                window = observed[i-lags:i] - predictions[i-lags:i]
                X_resid.append(window)
            X_resid = np.array(X_resid)
            resid_corrections = xgb_model.predict(X_resid)
            predictions[lags:] += resid_corrections
            logger.info("Applied XGBoost residual corrections.")
        return predictions
    except Exception as e:
        logger.error("Error in predict_fn: %s", e)
        raise

def output_fn(prediction, accept):
    """
    Serializes the prediction (as CSV) to return to the client.
    """
    try:
        if accept == "text/csv":
            out_df = pd.DataFrame(prediction, columns=["Predicted_HVAC_kWh"])
            buffer = io.StringIO()
            out_df.to_csv(buffer, index=False)
            logger.info("Output serialization complete.")
            return buffer.getvalue()
        else:
            raise ValueError("Unsupported accept type: {}".format(accept))
    except Exception as e:
        logger.error("Error in output_fn: %s", e)
        raise

# --- For Local Testing ---
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: {} <input_csv> <output_csv>".format(sys.argv[0]))
    # Use the SM_MODEL_DIR environment variable if set, else default to /opt/ml/model
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    logger.info("Using model directory: %s", model_dir)
    model = model_fn(model_dir)
    with open(sys.argv[1], 'r') as f:
        input_data = f.read()
    data = input_fn(input_data, "text/csv")
    predictions = predict_fn(data, model)
    output_data = output_fn(predictions, "text/csv")
    with open(sys.argv[2], 'w') as f:
        f.write(output_data)
    # Upload the log file to S3 for debugging purposes
    upload_log_to_s3()
