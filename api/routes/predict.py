from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import time
from models import OrderDeliveryInput, PredictionOutput
from dependencies import app_state, REQUEST_COUNT, ERROR_COUNT, PREDICTION_COUNT, LATENCY_HISTOGRAM, RISK_DISTRIBUTION

router = APIRouter(prefix="/predict", tags=["Prediction"])

def get_risk_and_action(prob: float, threshold: float) -> tuple[str, str]:
    if prob >= 0.8: return "Critical", "Urgent Review"
    if prob >= threshold: return "High", "Prioritize"
    if prob >= (threshold * 0.6): return "Medium", "Standard"
    return "Low", "None"

@router.post("/", response_model=PredictionOutput)
def predict_single(order: OrderDeliveryInput):
    start_time = time.time()
    try:    
        REQUEST_COUNT.inc() 
        df = pd.DataFrame([order.model_dump()])
        X = df.drop(columns=["order_id"])
        
        prob = float(app_state["model"].predict_proba(X)[0, 1])
        is_late = bool(prob >= app_state["optimal_threshold"])
        risk, action = get_risk_and_action(prob, app_state["optimal_threshold"])
        
        PREDICTION_COUNT.inc() 
        RISK_DISTRIBUTION.labels(level=risk).inc() 
            
        latency = time.time() - start_time
        LATENCY_HISTOGRAM.observe(latency) 
        
        return {
            "order_id": order.order_id, "late_probability": round(prob, 4),
            "predicted_is_late": is_late, "risk_level": risk,
            "model_version": app_state["model_name_version"], "recommended_action": action
        }
    except Exception as e:
        ERROR_COUNT.inc() 
        raise HTTPException(status_code=500, detail=f"single prediction error: {str(e)}")

@router.post("/batch", response_model=List[PredictionOutput])
def predict_batch(orders: List[OrderDeliveryInput]):
    start_time = time.time()
    if not orders:
        raise HTTPException(status_code=400, detail="Empty list provided")
    try:    
        REQUEST_COUNT.inc()
        df = pd.DataFrame([o.model_dump() for o in orders])
        order_ids = df["order_id"].tolist()
        X = df.drop(columns=["order_id"])
        
        probabilities = app_state["model"].predict_proba(X)[:, 1]
        
        results = []
        for order_id, prob in zip(order_ids, probabilities):
            prob = float(prob)
            is_late = bool(prob >= app_state["optimal_threshold"])
            risk, action = get_risk_and_action(prob, app_state["optimal_threshold"])
            
            PREDICTION_COUNT.inc()
            RISK_DISTRIBUTION.labels(level=risk).inc()
                
            results.append({
                    "order_id": order_id,
                    "late_probability": round(prob, 4),
                    "predicted_is_late": is_late,
                    "risk_level": risk,
                    "model_version": app_state["model_name_version"],
                    "recommended_action": action
                })

        latency = time.time() - start_time
        LATENCY_HISTOGRAM.observe(latency)
        return results
    
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")