"""
POST /api/v1/predict
GET  /api/v1/predict/locations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional

router = APIRouter(tags=["Prediction"])

# ── Feasibility metadata (from your research, not in model_registry.json) ─────
_SITE_META: dict[str, dict] = {
    "baron":      {"rank": 3, "category": "Kelas II (Rendah)",        "feasibility_score": 0.689, "coordinates": {"lat": -8.0572,  "lng": 110.5378}},
    "pandeglang": {"rank": 1, "category": "Kelas III (Sedang)",       "feasibility_score": 0.875, "coordinates": {"lat": -6.3083,  "lng": 105.8783}},
    "bawean":     {"rank": 2, "category": "Kelas III (Sedang)",       "feasibility_score": 0.823, "coordinates": {"lat": -5.7983,  "lng": 112.6767}},
    "situbondo":  {"rank": 5, "category": "Kelas I (Sangat Rendah)",  "feasibility_score": 0.487, "coordinates": {"lat": -7.7067,  "lng": 114.0017}},
    "sukabumi":   {"rank": 4, "category": "Kelas II (Rendah)",        "feasibility_score": 0.642, "coordinates": {"lat": -7.0667,  "lng": 106.9333}},
}

_WIND_STATS: dict[str, dict] = {
    "baron":      {"meanWindSpeed": 3.42, "windPowerDensity": 28.5,  "operationalHoursPct": 61.2, "windStabilityCV": 0.38},
    "pandeglang": {"meanWindSpeed": 5.21, "windPowerDensity": 87.3,  "operationalHoursPct": 82.4, "windStabilityCV": 0.22},
    "bawean":     {"meanWindSpeed": 4.87, "windPowerDensity": 71.2,  "operationalHoursPct": 76.8, "windStabilityCV": 0.26},
    "situbondo":  {"meanWindSpeed": 2.87, "windPowerDensity": 16.2,  "operationalHoursPct": 44.8, "windStabilityCV": 0.51},
    "sukabumi":   {"meanWindSpeed": 3.15, "windPowerDensity": 22.1,  "operationalHoursPct": 55.6, "windStabilityCV": 0.44},
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    location: str = Field(..., example="baron")
    recent_ws10m: list[float] = Field(
        ...,
        description=(
            "Hourly WS10M readings in chronological order "
            "(index 0 = oldest, last = most recent). "
            "Minimum 24 values recommended."
        ),
        example=[3.2, 3.5, 4.1, 3.8, 3.6, 3.9,
                 4.2, 4.0, 3.7, 3.5, 3.3, 3.1,
                 3.0, 2.9, 3.2, 3.4, 3.6, 3.8,
                 4.0, 4.1, 3.9, 3.7, 3.5, 3.4],
    )
    target_time: str = Field(
        ...,
        description="ISO-8601 datetime for the forecast target.",
        example="2026-06-01T10:00:00",
    )

    @field_validator("recent_ws10m")
    @classmethod
    def no_negative_wind(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("Wind speed values must be >= 0.")
        return v


class PredictResponse(BaseModel):
    location: str
    target_time: str
    predicted_ws10m: float
    unit: str
    scenario: str
    model_confidence_r2: float
    model_test_mae: float


class LocationMetrics(BaseModel):
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    r2: Optional[float]


class LocationInfo(BaseModel):
    id: str
    name: str
    scenario: str
    status: str
    metrics: LocationMetrics
    feature_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
def predict_wind_speed(req: PredictRequest):
    from main import predictor

    location_id = req.location.lower().strip()

    try:
        result = predictor.predict(
            location=location_id,
            recent_ws10m=req.recent_ws10m,
            target_time=req.target_time,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictResponse(**result)


@router.get("/predict/locations", response_model=list[LocationInfo])
def get_locations():
    from main import predictor

    result = []
    for loc_id in predictor.location_ids():
        meta = predictor.location_meta(loc_id)
        m = meta["metrics"]
        result.append(LocationInfo(
            id=loc_id,
            name=meta["name"],
            scenario=meta["scenario"],
            status=meta.get("status", "layak"),
            metrics=LocationMetrics(
                mae=round(m["mae"],  4),
                rmse=round(m["rmse"], 4),
                mape=round(m["mape"], 4),
                r2=round(m["r2"],   4),
            ),
            feature_count=len(meta["feature_order"]),
        ))

    # Sort by feasibility rank (best first)
    result.sort(key=lambda x: _SITE_META.get(x.id, {}).get("rank", 99))
    return result
