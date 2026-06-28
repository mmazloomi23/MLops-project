from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=["Monitoring"])

@router.get("/metrics")
def get_metrics():

    return Response(
        content=generate_latest(), 
        media_type=CONTENT_TYPE_LATEST
    )