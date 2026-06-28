import joblib
import json
import os
from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
ERROR_COUNT = Counter("api_errors_total", "Total Errors")
PREDICTION_COUNT = Counter("api_predictions_total", "Total Predictions")

LATENCY_HISTOGRAM = Histogram("api_latency_seconds", "Response Latency")

RISK_DISTRIBUTION = Counter("api_risk_distribution", "Distribution of Risk levels", ["level"])

# وضعیت سراسری مدل
app_state = {
    "model": None,
    "metadata": {},
    "optimal_threshold": 0.5,
    "model_name_version": "HistGradientBoosting-v1.0"
}

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "late_delivery_model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

def load_artifacts():
    try:
        app_state["model"] = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            app_state["metadata"] = metadata
            app_state["optimal_threshold"] = metadata.get("threshold", 0.5)
            app_state["model_name_version"] = f"{metadata.get('model_type', 'Model')}-{metadata.get('trained_at_utc', 'v1')[:10]}"
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")