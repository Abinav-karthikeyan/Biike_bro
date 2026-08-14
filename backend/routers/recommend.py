"""
Recommend Router — retrieval-first fast path with no LLM in the loop.

GET /recommend?zone_id=GLW_Z008&k=5&max_distance_m=2000&venue_type=transit

Pipeline (all in-process or single-shot DB reads)
──────────────────────────────────────────────────
1. HNSW semantic search — top-(k+1) nearest zones in 256-D feature space.
2. GeofenceRuleEngine   — flag ride_end_rules violations per candidate.
3. XGBoost batch score  — single predict_proba call across all candidates.
4. Sort + annotate      — lowest fill first; rule violations pushed to back.

Design notes
────────────
• DB calls: two (get_latest_occupancy, get_weather_recent) regardless of k.
  No per-zone snapshot queries; no Ollama round-trip.
• XGBoost: one batch predict_proba((n_candidates, 11)) call — not n loops.
• Target: <50 ms P95 on local hardware for k ≤ 10.  Actual latency is
  reported in the response as latency_ms so it can be monitored.
"""

import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from backend.services.hnsw_search import _haversine_m

router = APIRouter()


# ── Response schemas ──────────────────────────────────────────────────────────

class RecommendZone(BaseModel):
    zone_id: str
    name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    venue_type: Optional[str] = None
    distance_m: Optional[float] = None
    fill_probability: float
    confidence: float
    similarity: float
    current_occupancy_pct: Optional[float] = None
    bikes_available: Optional[int] = None
    station_parking: bool = True
    rule_violations: List[str] = []
    reason: str


class RecommendResponse(BaseModel):
    queried_zone_id: str
    recommended_at: datetime
    candidates: List[RecommendZone]
    source_zone_fill_probability: float
    latency_ms: float
    slm_available: bool = False


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "/",
    summary="Retrieval-first zone recommendation — HNSW + rules + XGBoost, no LLM",
    response_model=RecommendResponse,
)
async def recommend(
    request: Request,
    zone_id: str = Query(..., description="Anchor zone — find parking alternatives to this zone"),
    k: int = Query(default=5, ge=1, le=20, description="Number of candidate zones to return"),
    max_distance_m: Optional[float] = Query(
        default=3000.0,
        description="Restrict candidates to within this radius in metres (Haversine)",
    ),
    venue_type: Optional[str] = Query(
        default=None,
        description="Only return zones with this venue_type (transit / retail / park / …)",
    ),
):
    """
    Fast zone recommendation without an LLM in the critical path.

    Candidates are ranked by XGBoost fill probability (lowest first).
    Zones with geofence rule violations (e.g. station_parking required) are
    surfaced at the bottom of the list so the caller can warn the rider.

    The SLM narration layer (POST /slm/query) can be called on top of these
    results when natural-language guidance is needed — it is optional.
    """
    t0 = time.perf_counter()
    now = datetime.now(timezone.utc)

    hnsw = request.app.state.hnsw_service
    db = request.app.state.db
    prediction = request.app.state.prediction_service
    geofence = request.app.state.geofence_engine
    slm = getattr(request.app.state, "slm_service", None)

    # ── 1. HNSW semantic search ───────────────────────────────────────────
    emb = hnsw.get_zone_embedding(zone_id) if hnsw else None
    if emb is None:
        return RecommendResponse(
            queried_zone_id=zone_id,
            recommended_at=now,
            candidates=[],
            source_zone_fill_probability=0.0,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            slm_available=bool(slm and getattr(slm, "_ok", False)),
        )

    # Fetch k+1 so we have spare candidates after excluding the anchor zone
    raw_candidates = hnsw.zone_semantic_search(
        emb,
        k=k + 1,
        max_distance_m=max_distance_m,
        venue_type=venue_type,
        query_zone_id=zone_id,
    )
    # Exclude the queried zone itself, cap at k
    candidates = [c for c in raw_candidates if c.zone_id != zone_id][:k]

    # ── 2. Single-shot DB reads ───────────────────────────────────────────
    # get_latest_occupancy() returns zone meta + live occupancy in one query.
    all_zones = {z["zone_id"]: z for z in db.get_latest_occupancy()}
    weather_records = db.get_weather_recent(hours=1)
    weather = weather_records[0] if weather_records else None

    anchor_meta = all_zones.get(zone_id, {})

    # ── 3. Batch XGBoost scoring ──────────────────────────────────────────
    candidate_ids = [c.zone_id for c in candidates]
    feature_rows = []
    for cid in candidate_ids:
        zmeta = all_zones.get(cid, {})
        feats = prediction._build_features(cid, now, zmeta, weather)
        feature_rows.append(feats)

    if feature_rows and prediction.model is not None:
        X = np.array(feature_rows, dtype=np.float32)
        probas = prediction.model.predict_proba(X)   # (n, 2)
        fill_probs = probas[:, 1].tolist()
        confidences = probas.max(axis=1).tolist()
    else:
        fill_probs = [0.5] * len(candidate_ids)
        confidences = [0.6] * len(candidate_ids)

    # Score the anchor zone for comparison
    anchor_feats = prediction._build_features(zone_id, now, anchor_meta, weather)
    if prediction.model is not None:
        a_proba = prediction.model.predict_proba(anchor_feats.reshape(1, -1))[0]
        source_fill = float(a_proba[1])
    else:
        source_fill = 0.5

    # ── 4. Geofence evaluation + response assembly ────────────────────────
    anchor_lat = anchor_meta.get("lat") or 0.0
    anchor_lon = anchor_meta.get("lon") or 0.0

    result_zones: List[RecommendZone] = []
    for c, fill, conf in zip(candidates, fill_probs, confidences):
        zmeta = all_zones.get(c.zone_id, {})
        violations = geofence.evaluate(c.zone_id)
        rule_summary = geofence.get_rule_summary(c.zone_id) or {}

        dist = None
        if anchor_lat and anchor_lon and zmeta.get("lat") and zmeta.get("lon"):
            dist = _haversine_m(anchor_lat, anchor_lon, zmeta["lat"], zmeta["lon"])

        occ = zmeta.get("occupancy_pct")
        bikes = zmeta.get("bikes_available")

        reason = prediction._generate_reason(now, float(fill), zmeta, weather)

        result_zones.append(
            RecommendZone(
                zone_id=c.zone_id,
                name=zmeta.get("name"),
                lat=zmeta.get("lat"),
                lon=zmeta.get("lon"),
                venue_type=zmeta.get("venue_type"),
                distance_m=round(dist, 0) if dist is not None else None,
                fill_probability=round(float(fill), 4),
                confidence=round(float(conf), 4),
                similarity=round(c.similarity, 4),
                current_occupancy_pct=round(float(occ), 1) if occ is not None else None,
                bikes_available=int(bikes) if bikes is not None else None,
                station_parking=rule_summary.get("station_parking", True),
                rule_violations=[v.description for v in violations],
                reason=reason,
            )
        )

    # Sort: no-violations + lowest fill first; violations pushed to back
    result_zones.sort(key=lambda z: (len(z.rule_violations) > 0, z.fill_probability))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return RecommendResponse(
        queried_zone_id=zone_id,
        recommended_at=now,
        candidates=result_zones,
        source_zone_fill_probability=round(source_fill, 4),
        latency_ms=latency_ms,
        slm_available=bool(slm and getattr(slm, "_ok", False)),
    )
