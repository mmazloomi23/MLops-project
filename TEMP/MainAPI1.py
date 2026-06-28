from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import json
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

# ==========================================
# 1. Pydantic Schemas (Input & Output Models)
# ==========================================

class OrderDeliveryInput(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the order")
    
    # Time based features
    purchase_hour: int
    purchase_dayofweek: int
    purchase_month: int
    is_weekend_purchase: int
    estimated_delivery_days: float
    has_approval_timestamp: int
    approval_delay_hours: float

    # Product based features
    num_items: int
    total_price: float
    avg_price: float
    max_price: float
    total_freight: float
    avg_freight: float
    max_freight: float
    num_sellers: int
    num_products: int
    num_product_categories: int
    avg_product_weight_g: float
    max_product_weight_g: float
    avg_product_volume_cm3: float
    max_product_volume_cm3: float
    main_product_category: str

    # Financial features
    freight_price_ratio: float
    payment_value: float
    payment_installments: int
    num_payment_types: int
    main_payment_type: str

    # Geographical features
    seller_zip_code_prefix: float
    num_seller_states: int
    main_seller_city: str
    main_seller_state: str
    customer_zip_code_prefix: float
    customer_city: str
    customer_state: str
    same_state: int
    same_city: int
    zip_prefix_diff: float


class PredictionOutput(BaseModel):
    order_id: str
    late_probability: float
    predicted_is_late: bool
    risk_level: str
    model_version: str
    recommended_action: str

# ==========================================
# 2. Global State & Helper Functions
# ==========================================

# Global parameters
model = None
metadata = {}
optimal_threshold = 0.5
model_name_version = "HistGradientBoosting-v1.0"

# Summary Metrics
app_metrics = {
    "total_requests": 0,
    "total_orders_processed": 0,
    "predicted_late_count": 0,
    "sum_probabilities": 0.0,
    "start_time": datetime.now().isoformat()
}

def get_risk_and_action(probability: float, threshold: float) -> tuple[str, str]:
    """ Risk analysis and delay exception"""
    if probability >= 0.8:
        return "Critical", "Urgent Review" 
    elif probability >= threshold:
        return "High", "Prioritize"
    elif probability >= (threshold * 0.6):
        return "Medium", "Standard"
    else:
        return "Low", "None"

# ==========================================
# 3. Application Startup
# ==========================================

def load_artifacts():
    global model, metadata, optimal_threshold, model_name_version
    try:
        model = joblib.load("late_delivery_model.joblib")
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
            optimal_threshold = metadata.get("threshold", 0.5)
            model_name_version = f"{metadata.get('model_type', 'UnknownModel')}-{metadata.get('trained_at_utc', 'v1')[:10]}"
        print("✅ Model and Metadata loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        # If model not loaded Error 500
        raise RuntimeError(f"Failed to load model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts() 
    yield
    print("✅ App is shutting down...")

app = FastAPI(
    title="Olist Late Delivery Predictor",
    description="API for predicting late deliveries with Batch support and Metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

# ==========================================
# 4. API Endpoints
# ==========================================

@app.get("/health", tags=["System"])
def health_check():
    """health check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model-info", tags=["System"])
def get_model_info():
    """Model and metadata"""
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")
    return {
        "model_version": model_name_version,
        "optimal_threshold": optimal_threshold,
        "features": metadata.get("feature_columns", []),
        "training_date": metadata.get("trained_at_utc")
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_single(order: OrderDeliveryInput):


    global app_metrics
    app_metrics["total_requests"] += 1
    
    df = pd.DataFrame([order.model_dump()])
    
    order_id = df["order_id"].iloc[0]
    X = df.drop(columns=["order_id"])
    
    probability = float(model.predict_proba(X)[0, 1])
    is_late = bool(probability >= optimal_threshold)
    risk_level, action = get_risk_and_action(probability, optimal_threshold)
    
    app_metrics["total_orders_processed"] += 1
    app_metrics["sum_probabilities"] += probability
    if is_late:
        app_metrics["predicted_late_count"] += 1

    return {
        "order_id": order_id,
        "late_probability": round(probability, 4),
        "predicted_is_late": is_late,
        "risk_level": risk_level,
        "model_version": model_name_version,
        "recommended_action": action
    }

@app.post("/predict-batch", response_model=List[PredictionOutput], tags=["Prediction"])
def predict_batch(orders: List[OrderDeliveryInput]):
    """Batch predition"""
    global app_metrics
    app_metrics["total_requests"] += 1
    
    if not orders:
        raise HTTPException(status_code=400, detail="Empty list provided")

    df = pd.DataFrame([order.model_dump() for order in orders])
    order_ids = df["order_id"].tolist()
    X = df.drop(columns=["order_id"])
    
    probabilities = model.predict_proba(X)[:, 1]
    
    results = []
    for order_id, prob in zip(order_ids, probabilities):
        prob = float(prob)
        is_late = bool(prob >= optimal_threshold)
        risk, action = get_risk_and_action(prob, optimal_threshold)
        
        app_metrics["total_orders_processed"] += 1
        app_metrics["sum_probabilities"] += prob
        if is_late:
            app_metrics["predicted_late_count"] += 1
            
        results.append({
            "order_id": order_id,
            "late_probability": round(prob, 4),
            "predicted_is_late": is_late,
            "risk_level": risk,
            "model_version": model_name_version,
            "recommended_action": action
        })
        
    return results

@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Summary Metrics"""
    total_processed = app_metrics["total_orders_processed"]
    avg_prob = (app_metrics["sum_probabilities"] / total_processed) if total_processed > 0 else 0
    late_ratio = (app_metrics["predicted_late_count"] / total_processed) if total_processed > 0 else 0
    
    return {
        "uptime_since": app_metrics["start_time"],
        "total_api_requests": app_metrics["total_requests"],
        "total_orders_processed": total_processed,
        "predicted_late_count": app_metrics["predicted_late_count"],
        "predicted_late_ratio": round(late_ratio, 4),
        "average_late_probability": round(avg_prob, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)