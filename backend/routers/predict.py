"""Prediction router — POST /predict."""

import time

import structlog
from fastapi import APIRouter, HTTPException, Request

from backend.models.schemas import PredictRequest, PredictResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/", response_model=PredictResponse, summary="Predict zone fill probability")
async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """
    **POST /predict** — Core prediction endpoint (PRD §8).

    Returns zone fill probability for the requested lookahead window,
    along with alternative zones and a human-readable reason.

    - `fill_probability`: 0.0 (empty) → 1.0 (full)
    - `confidence`: model confidence in prediction
    - `alternative_zones`: ranked list if fill_probability > 0.65
    - `reason`: natural-language explanation (SLM-generated in production)
    """
    t0 = time.perf_counter()
    prediction_service = request.app.state.prediction_service

    try:
        result = await prediction_service.predict(payload)
    except Exception as exc:
        logger.error("Prediction failed", zone=payload.zone_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction service error") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Prediction served",
        zone=payload.zone_id,
        fill_prob=result.fill_probability,
        latency_ms=round(latency_ms, 1),
    )

    if latency_ms > 200:
        logger.warning("Latency exceeded 200ms SLO", latency_ms=round(latency_ms, 1))

    return result
