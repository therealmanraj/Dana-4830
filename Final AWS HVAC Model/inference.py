#!/usr/bin/env python
import os
import pickle
import io
import logging
import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# --- Setup Logging ---
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("inference_log.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# S3 log config
LOG_S3_BUCKET = 'dana-minicapstone-ca'
LOG_S3_PREFIX = 'logs/inference_logs/'

def upload_log_to_s3():
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

# --- Model Functions ---
def model_fn(model_dir):
    try:
        with open(os.path.join(model_dir, "lin_reg_model.pkl"), "rb") as f:
            lin_model = pickle.load(f)
        logger.info("Loaded Linear Regression model.")
    except Exception as e:
        logger.error("Error loading lin_reg_model.pkl: %s", e)
        raise

    try:
        with open(os.path.join(model_dir, "xgb_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)
        logger.info("Loaded XGBoost model.")
    except Exception as e:
        logger.error("Error loading xgb_model.pkl: %s", e)
        raise

    return {"linear": lin_model, "xgb": xgb_model}

def input_fn(input_data, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(input_data))
        df = df.bfill()
        return df
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(data, model):
    try:
        lin_model = model["linear"]
        xgb_model = model["xgb"]

        lin_preds = lin_model.predict(data)
        xgb_preds = xgb_model.predict(data)
        final_preds = lin_preds + xgb_preds

        logger.info("Inference completed successfully.")
        return final_preds
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise

def output_fn(prediction, accept):
    if accept == "text/csv":
        out_df = pd.DataFrame(prediction, columns=["Predicted_HVAC_kWh"])
        buffer = io.StringIO()
        out_df.to_csv(buffer, index=False)
        return buffer.getvalue()
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))

# --- Local Testing ---
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: {} <input_csv> <output_csv>".format(sys.argv[0]))

    model_dir = os.environ.get("SM_MODEL_DIR", "./model")
    logger.info("Using model directory: %s", model_dir)

    model = model_fn(model_dir)

    with open(sys.argv[1], 'r') as f:
        input_data = f.read()

    df = input_fn(input_data, "text/csv")
    predictions = predict_fn(df, model)
    output_data = output_fn(predictions, "text/csv")

    with open(sys.argv[2], 'w') as f:
        f.write(output_data)

    upload_log_to_s3()