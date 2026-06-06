# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Installed packages from builder
COPY --from=builder /install /usr/local

# Application source
COPY main.py        .
COPY predictor.py   .
COPY routers/       ./routers/

# ── Runtime mounts (via docker-compose volumes) ───────────────────────────────
# /app/models/                  ← .joblib files
# /app/model_registry.json      ← location + feature metadata
# /app/preprocessing_recipe.json ← feature engineering config

ENV ARTIFACT_DIR=/app
ENV ENV=production
# Comma-separated allowed origins, e.g. "https://dashboard.mahi.id,https://api.mahi.id"
ENV CORS_ORIGINS=*

EXPOSE 8002

# 2 workers is safe for inference; increase if your VPS has more cores
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
