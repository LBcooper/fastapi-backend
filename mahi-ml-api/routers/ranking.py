"""GET /api/v1/ranking"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Ranking"])

# Imported inside functions to avoid circular import at module load
_SITE_META: dict[str, dict] = {
    "baron":      {"rank": 3, "category": "Kelas II (Rendah)",       "feasibility_score": 0.689, "coordinates": {"lat": -8.0572,  "lng": 110.5378}},
    "pandeglang": {"rank": 1, "category": "Kelas III (Sedang)",      "feasibility_score": 0.875, "coordinates": {"lat": -6.3083,  "lng": 105.8783}},
    "bawean":     {"rank": 2, "category": "Kelas III (Sedang)",      "feasibility_score": 0.823, "coordinates": {"lat": -5.7983,  "lng": 112.6767}},
    "situbondo":  {"rank": 5, "category": "Kelas I (Sangat Rendah)", "feasibility_score": 0.487, "coordinates": {"lat": -7.7067,  "lng": 114.0017}},
    "sukabumi":   {"rank": 4, "category": "Kelas II (Rendah)",       "feasibility_score": 0.642, "coordinates": {"lat": -7.0667,  "lng": 106.9333}},
}

_WIND_STATS: dict[str, dict] = {
    "baron":      {"meanWindSpeed": 3.42, "windPowerDensity": 28.5,  "operationalHoursPct": 61.2, "windStabilityCV": 0.38},
    "pandeglang": {"meanWindSpeed": 5.21, "windPowerDensity": 87.3,  "operationalHoursPct": 82.4, "windStabilityCV": 0.22},
    "bawean":     {"meanWindSpeed": 4.87, "windPowerDensity": 71.2,  "operationalHoursPct": 76.8, "windStabilityCV": 0.26},
    "situbondo":  {"meanWindSpeed": 2.87, "windPowerDensity": 16.2,  "operationalHoursPct": 44.8, "windStabilityCV": 0.51},
    "sukabumi":   {"meanWindSpeed": 3.15, "windPowerDensity": 22.1,  "operationalHoursPct": 55.6, "windStabilityCV": 0.44},
}


class RankingMetrics(BaseModel):
    meanWindSpeed: float
    windPowerDensity: float
    operationalHoursPct: float
    windStabilityCV: float
    modelR2: float


class RankingCoordinates(BaseModel):
    lat: float
    lng: float


class RankingSite(BaseModel):
    id: str
    name: str
    rank: int
    coordinates: RankingCoordinates
    feasibilityScore: float
    status: str
    category: str
    bestScenario: str
    metrics: RankingMetrics


@router.get("/ranking", response_model=list[RankingSite])
def get_ranking():
    from main import predictor

    result = []
    for loc_id in predictor.location_ids():
        meta = predictor.location_meta(loc_id)
        site = _SITE_META.get(loc_id)
        wind = _WIND_STATS.get(loc_id)
        if not site or not wind:
            continue

        result.append(RankingSite(
            id=loc_id,
            name=meta["name"],
            rank=site["rank"],
            coordinates=RankingCoordinates(**site["coordinates"]),
            feasibilityScore=site["feasibility_score"],
            status=meta.get("status", "layak"),
            category=site["category"],
            bestScenario=meta["scenario"],
            metrics=RankingMetrics(
                **wind,
                modelR2=round(meta["metrics"]["r2"], 4),
            ),
        ))

    result.sort(key=lambda x: x.rank)
    return result
