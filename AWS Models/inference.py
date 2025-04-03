import pickle
import json
import pandas as pd

def model_fn(model_dir):
    """Load and return the model. This function is called once when the container is created."""
    # Load the linear and XGBoost models from the model directory
    with open(f"{model_dir}/linear_model.pkl", "rb") as f:
        linear_model = pickle.load(f)
    with open(f"{model_dir}/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    # You can also return additional parameters (like lags) if needed
    return {"linear_model": linear_model, "xgb_model": xgb_model, "lags": 5}

def input_fn(request_body, content_type='application/json'):
    """Deserialize the request body into a DataFrame."""
    if content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    """Apply model to the incoming request."""
    # Here, we simply use the linear model prediction.
    # For a hybrid prediction, you’d need to calculate residuals with lags,
    # which typically requires previous observations.
    linear_model = model["linear_model"]
    predictions = linear_model.predict(input_data)
    return predictions

def output_fn(prediction, content_type='application/json'):
    """Serialize predictions into JSON."""
    result = prediction.tolist()
    return json.dumps(result), content_type
