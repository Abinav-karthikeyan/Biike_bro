"""
Prediction Service — Real XGBoost model trained on Glasgow synthetic data.

Trains on startup using 725k+ zone_snapshots joined with zone metadata.
Target: binary classification of zone_full (occupancy_pct > 85%).
Features: hour, day_of_week, weather, temperature, events, venue type,
          transit score, capacity.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from backend.models.schemas import AlternativeZone, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# ── Venue type encoding ───────────────────────────────────────────────────
VENUE_ENCODE = {
    "transit": 0, "retail": 1, "park": 2,
    "residential": 3, "university": 4, "mixed": 5,
}

# ── WMO weather grouping ─────────────────────────────────────────────────
WMO_RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 95, 96}
WMO_DRIZZLE_CODES = {51, 53, 55}


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class PredictionService:
    """
    XGBoost-based zone fill prediction trained on synthetic data.
    Call predict() to get fill probability for a zone at a future time.
    """

    def __init__(self, db_store=None):
        self.db = db_store
        self.model: Optional[XGBClassifier] = None
        self.model_version = "xgb-v1-synthetic"
        self.accuracy = 0.0
        self.feature_columns = [
            "hour_of_day", "day_of_week", "weather_code", "temp_celsius",
            "is_event_nearby", "venue_type_enc", "transit_score", "capacity",
            "is_rain", "is_rush_hour", "is_weekend",
        ]

        if db_store:
            self._train_model()
        else:
            logger.warning("PredictionService: no DB store — running in stub mode")

    # ── Training ──────────────────────────────────────────────────────────

    def _train_model(self) -> None:
        """Train XGBoost on zone_snapshots data."""
        logger.info("Training XGBoost model on synthetic data...")
        df = self.db.get_snapshots_for_training()
        logger.info(f"Training data: {len(df):,} rows")

        # Feature engineering
        df["venue_type_enc"] = df["venue_type"].map(VENUE_ENCODE).fillna(5).astype(int)
        df["is_event_nearby"] = df["is_event_nearby"].astype(int)
        df["is_rain"] = df["weather_code"].isin(WMO_RAIN_CODES).astype(int)
        df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Target: zone is "full" (>85% occupancy)
        df["zone_full"] = (df["occupancy_pct"] > 85).astype(int)

        X = df[self.feature_columns].copy()
        y = df["zone_full"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train XGBoost
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=float(len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logger.info(
            f"XGBoost trained — Accuracy: {self.accuracy:.3f}, "
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
        )
        full_pct = 100 * y.mean()
        logger.info(f"Class balance: {full_pct:.1f}% zone-full samples")

    # ── Feature Building ──────────────────────────────────────────────────

    def _build_features(
        self,
        zone_id: str,
        ts: datetime,
        zone_meta: dict,
        weather: Optional[dict] = None,
    ) -> np.ndarray:
        """Build feature vector matching training schema."""
        hour = ts.hour
        dow = ts.weekday()
        wcode = weather.get("wmo_code", 0) if weather else 0
        temp = weather.get("temp_celsius", 12.0) if weather else 12.0
        is_event = 0  # default; could query events table

        venue_enc = VENUE_ENCODE.get(zone_meta.get("venue_type", "mixed"), 5)
        transit = zone_meta.get("transit_score", 0.5)
        capacity = zone_meta.get("capacity", 15)
        is_rain = 1 if wcode in WMO_RAIN_CODES else 0
        is_rush = 1 if hour in (7, 8, 9, 17, 18, 19) else 0
        is_weekend = 1 if dow >= 5 else 0

        return np.array([
            hour, dow, wcode, temp,
            is_event, venue_enc, transit, capacity,
            is_rain, is_rush, is_weekend,
        ], dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────

    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Predict zone fill probability for a future timestamp."""
        ts = request.current_timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        target_ts = ts + timedelta(minutes=request.lookahead_mins)

        # Look up zone metadata
        zone_meta = self.db.get_zone(request.zone_id) if self.db else {}
        if not zone_meta:
            zone_meta = {"venue_type": "mixed", "transit_score": 0.5, "capacity": 15}

        # Get weather for target hour
        weather = None
        if self.db:
            wx = self.db.get_weather_recent(hours=72)
            target_hour_str = target_ts.strftime("%Y-%m-%dT%H")
            for w in wx:
                w_ts = str(w.get("timestamp", ""))
                if w_ts.startswith(target_hour_str[:13]):
                    weather = w
                    break

        # Build features and predict
        features = self._build_features(request.zone_id, target_ts, zone_meta, weather)

        if self.model is not None:
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            fill_prob = float(proba[1])  # probability of zone_full=1
            confidence = float(max(proba))
        else:
            # Stub fallback
            fill_prob = 0.5
            confidence = 0.6

        # Alternative zones if fill is high
        alternatives = []
        if fill_prob > 0.65 and self.db:
            alternatives = await self._find_alternatives(
                request.zone_id, target_ts, zone_meta, weather
            )

        reason = self._generate_reason(target_ts, fill_prob, zone_meta, weather)

        return PredictResponse(
            zone_id=request.zone_id,
            fill_probability=round(fill_prob, 4),
            confidence=round(confidence, 4),
            alternative_zones=alternatives,
            reason=reason,
            lookahead_mins=request.lookahead_mins,
            model_version=self.model_version,
        )

    async def _find_alternatives(
        self,
        zone_id: str,
        target_ts: datetime,
        zone_meta: dict,
        weather: Optional[dict],
    ) -> List[AlternativeZone]:
        """Find zones with lower predicted fill probability."""
        if not self.db or not self.model:
            return []

        zones = self.db.get_zones()
        scored = []
        src_lat = zone_meta.get("lat", 0)
        src_lon = zone_meta.get("lon", 0)

        for z in zones:
            if z["zone_id"] == zone_id:
                continue
            feats = self._build_features(z["zone_id"], target_ts, z, weather)
            proba = self.model.predict_proba(feats.reshape(1, -1))[0]
            fill = float(proba[1])
            dist = _haversine_m(src_lat, src_lon, z["lat"], z["lon"])
            if fill < 0.65 and dist < 3000:  # within 3km
                scored.append((z["zone_id"], fill, dist))

        scored.sort(key=lambda x: (x[1], x[2]))  # lowest fill first
        return [
            AlternativeZone(
                zone_id=s[0],
                fill_probability=round(s[1], 4),
                distance_m=round(s[2], 0),
            )
            for s in scored[:3]
        ]

    def _generate_reason(
        self,
        ts: datetime,
        fill_prob: float,
        zone_meta: dict,
        weather: Optional[dict],
    ) -> str:
        """Generate human-readable prediction reason."""
        parts = []
        hour = ts.hour

        # Time context
        if 7 <= hour <= 9:
            parts.append("morning commute rush")
        elif 17 <= hour <= 19:
            parts.append("evening rush hour")
        elif 12 <= hour <= 14:
            parts.append("lunchtime")
        elif 22 <= hour or hour <= 6:
            parts.append("overnight low-activity period")
        else:
            parts.append("off-peak period")

        # Venue context
        vtype = zone_meta.get("venue_type", "")
        if vtype == "transit":
            parts.append("high-traffic transit hub")
        elif vtype == "university":
            parts.append("near university campus")

        # Weather
        if weather:
            wmo = weather.get("wmo_code", 0)
            if wmo in WMO_RAIN_CODES:
                parts.append("rainy weather reduces cycling demand")
            elif wmo in {0, 1}:
                parts.append("clear weather increases demand")

        # Transit score
        ts_score = zone_meta.get("transit_score", 0)
        if ts_score and ts_score > 0.8:
            parts.append("zone has excellent transit connectivity")

        level = "High" if fill_prob > 0.7 else "Moderate" if fill_prob > 0.4 else "Low"
        context = "; ".join(parts[:3])
        return f"{level} fill probability — {context}"

    def get_model_metrics(self) -> dict:
        """Return model performance metrics."""
        if not self.model:
            return {"status": "no model", "accuracy": 0}
        return {
            "model_version": self.model_version,
            "accuracy": round(self.accuracy, 4),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "feature_columns": self.feature_columns,
        }
