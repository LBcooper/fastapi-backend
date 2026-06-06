"""GET /api/v1/health"""

from datetime import datetime, timezone
import os

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: str
    models_loaded: list[str]


@router.get("/health", response_model=HealthResponse)
def health_check():
    from main import predictor  # imported here to avoid circular import at module load

    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=os.getenv("ENV", "production"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=predictor.location_ids() if predictor else [],
    )
