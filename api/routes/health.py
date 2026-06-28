from fastapi import APIRouter, HTTPException
from dependencies import app_state
from datetime import datetime

router = APIRouter(tags=["System"])

@router.get("/health")
def health_check():
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}