"""
Dashboard Mahi — FastAPI ML Backend
=====================================
Wraps WindPredictor for 5 PLTB locations via REST API.
""" 

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predictor import WindPredictor

# ── Shared predictor instance (set during lifespan) ───────────────────────────
predictor: WindPredictor | None = None

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "/app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print(f"🚀  Loading WindPredictor from {ARTIFACT_DIR} ...")
    predictor = WindPredictor(artifact_dir=ARTIFACT_DIR)
    # Eager-load all models at startup so first request isn't slow
    for loc_id in predictor.location_ids():
        try:
            predictor._model(loc_id)
            print(f"  ✓  {loc_id}")
        except FileNotFoundError as e:
            print(f"  ⚠️  {loc_id}: {e}")
    print("✅  All models ready.")
    yield
    print("🛑  Shutting down.")


app = FastAPI(
    title="Dashboard Mahi — ML API",
    description="Wind speed prediction API for PLTB feasibility analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from routers import health, predict, ranking, chat  # noqa: E402

app.include_router(health.router,   prefix="/ml/api/v1")
app.include_router(predict.router,  prefix="/ml/api/v1")
app.include_router(ranking.router,  prefix="/ml/api/v1")
app.include_router(chat.router,     prefix="/api/v1")
