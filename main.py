from fastapi import FastAPI
from contextlib import asynccontextmanager
from dependencies import load_artifacts
from api.routes import predict, health
from api import metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(title="Olist Prediction API", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(metrics.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)