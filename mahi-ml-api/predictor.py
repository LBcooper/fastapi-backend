"""Inference module — PLTB wind-speed forecasting (Random Forest)."""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn


class WindPredictor:
    """
    Registry for 5 RF models per location.
    Lazy-load + cache. Single predict() interface.
    """

    def __init__(self, artifact_dir: str):
        self.dir = artifact_dir

        with open(os.path.join(artifact_dir, "model_registry.json")) as f:
            reg = json.load(f)
        with open(os.path.join(artifact_dir, "preprocessing_recipe.json")) as f:
            self.recipe = json.load(f)

        self.locations: dict = reg["locations"]
        self.horizon: int = reg.get("horizon_hours", 1)
        self._check_versions(reg.get("library_versions", {}))
        self._cache: dict = {}

        r = self.recipe
        self.min_history = max(
            r["lags"] + r["rolling_mean_windows"] + r["rolling_std_windows"]
        )

    # ── Version check ─────────────────────────────────────────────────────────
    def _check_versions(self, saved: dict):
        cur = sklearn.__version__
        if saved.get("scikit_learn") and saved["scikit_learn"] != cur:
            warnings.warn(
                f"sklearn mismatch: model trained on {saved['scikit_learn']}, "
                f"backend running {cur}. Pin versions to avoid silent errors."
            )

    # ── Model loader (lazy + cached) ──────────────────────────────────────────
    def _model(self, loc_id: str):
        if loc_id not in self._cache:
            model_filename = os.path.basename(self.locations[loc_id]["model_file"])
            local_path = os.path.join(self.dir, "models", model_filename)
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Model file not found: {local_path}\n"
                    f"Mount your .joblib files into the container at /app/models/"
                )
            self._cache[loc_id] = joblib.load(local_path)
        return self._cache[loc_id]

    # ── Feature engineering ───────────────────────────────────────────────────
    def _build_row(
        self,
        ws_hist: list[float],
        target_dt: datetime,
        feature_order: list[str],
    ) -> pd.DataFrame:
        ws = np.asarray(ws_hist, dtype=float)

        if np.isnan(ws).any():
            raise ValueError("recent_ws10m contains NaN values.")
        if len(ws) < self.min_history:
            raise ValueError(
                f"Insufficient history: need >= {self.min_history} hours, got {len(ws)}."
            )

        r = self.recipe

        # S0 dummy scenario (no real features)
        if feature_order == [r["s0_dummy_col"]]:
            return pd.DataFrame([[1.0]], columns=feature_order)

        feat: dict = {}

        # Lag features
        for k in r["lags"]:
            feat[f"WS10M_lag{k}"] = ws[-k]

        # Rolling mean features
        for w in r["rolling_mean_windows"]:
            feat[f"WS10M_roll{w}"] = ws[-w:].mean()

        # Rolling std features (ddof=1 matches pandas default)
        for w in r["rolling_std_windows"]:
            feat[f"WS10M_std{w}"] = ws[-w:].std(ddof=1)

        # Cyclical hour encoding
        ch = r["cyclical"]["hour"]
        feat["target_hour_sin"] = np.sin(2 * np.pi * target_dt.hour / ch["period"])
        feat["target_hour_cos"] = np.cos(2 * np.pi * target_dt.hour / ch["period"])

        # Cyclical month encoding
        if r.get("use_month_cyclical"):
            cm = r["cyclical"]["month"]
            sub = 1 if cm["subtract_one"] else 0
            feat["target_month_sin"] = np.sin(
                2 * np.pi * (target_dt.month - sub) / cm["period"]
            )
            feat["target_month_cos"] = np.cos(
                2 * np.pi * (target_dt.month - sub) / cm["period"]
            )

        missing = [c for c in feature_order if c not in feat]
        if missing:
            raise ValueError(f"Features could not be built: {missing}")

        return pd.DataFrame(
            [[feat[c] for c in feature_order]], columns=feature_order
        )

    # ── Public predict ────────────────────────────────────────────────────────
    def predict(
        self,
        location: str,
        recent_ws10m: list[float],
        target_time: str,
    ) -> dict:
        if location not in self.locations:
            raise KeyError(
                f"Location '{location}' not found. "
                f"Available: {list(self.locations.keys())}"
            )

        meta = self.locations[location]
        target_dt = datetime.fromisoformat(target_time)
        X = self._build_row(recent_ws10m, target_dt, meta["feature_order"])
        pred = float(self._model(location).predict(X)[0])

        return {
            "location": location,
            "target_time": target_time,
            "predicted_ws10m": round(max(pred, 0.0), 4),
            "unit": "m/s",
            "scenario": meta["scenario"],
            "model_confidence_r2": meta["metrics"]["r2"],
            "model_test_mae": meta["metrics"]["mae"],
        }

    # ── Utility helpers ───────────────────────────────────────────────────────
    def location_ids(self) -> list[str]:
        return list(self.locations.keys())

    def location_meta(self, loc_id: str) -> dict:
        return self.locations[loc_id]
