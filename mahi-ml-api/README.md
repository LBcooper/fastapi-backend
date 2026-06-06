# Mahi ML API — Deployment Guide

## Directory structure on your VPS

```
fastapi-backend/
├── main.py
├── predictor.py
├── routers/
│   ├── __init__.py
│   ├── health.py
│   ├── predict.py
│   ├── ranking.py
│   └── chat.py
├── model_registry.json        ← your artifact
├── preprocessing_recipe.json  ← your artifact
├── models/                    ← place .joblib files here
│   ├── model_baron.joblib
│   ├── model_pandeglang.joblib
│   ├── model_bawean.joblib
│   ├── model_situbondo.joblib
│   └── model_sukabumi.joblib
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Deploy

```bash
# 1. Copy files to VPS
scp -r fastapi-backend/ user@your-vps:/opt/mahi-ml/

# 2. SSH in
ssh user@your-vps
cd /opt/mahi-ml/fastapi-backend

# 3. Place your .joblib files in ./models/

# 4. Build and start
docker compose up -d --build

# 5. Check logs
docker compose logs -f

# 6. Test
curl http://localhost:8002/api/v1/health
```

## Test a prediction

```bash
curl -X POST http://localhost:8002/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "baron",
    "recent_ws10m": [3.2,3.5,4.1,3.8,3.6,3.9,4.2,4.0,3.7,3.5,3.3,3.1,
                     3.0,2.9,3.2,3.4,3.6,3.8,4.0,4.1,3.9,3.7,3.5,3.4],
    "target_time": "2026-06-07T10:00:00"
  }'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/v1/health` | Health + loaded models |
| POST | `/api/v1/predict` | Run wind speed prediction |
| GET  | `/api/v1/predict/locations` | All locations + metrics |
| GET  | `/api/v1/ranking` | Feasibility ranking |
| POST | `/api/v1/chat` | Chat stub (wire your LLM) |

## Frontend env

Set in your Next.js `.env.local`:

```
NEXT_PUBLIC_API_URL=http://your-vps-ip:8002
```

## Notes

- `scikit-learn==1.6.1` and `numpy==2.0.2` are pinned to match your training env.
- Models are loaded **eagerly** at startup — cold start takes ~5s, then all predictions are fast.
- Pandeglang and Bawean use **S4** (4 features); the other 3 use **S7** (10 features).
  The `WindPredictor` handles this automatically per-location via `feature_order` in `model_registry.json`.
