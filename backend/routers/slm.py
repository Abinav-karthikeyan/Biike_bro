"""
SLM Router — natural language + tool-calling endpoints.

Endpoints
---------
POST /slm/query          NL query → Ollama/Qwen2.5 with tool dispatch
POST /slm/predict        XGBoost + SLM narrative hybrid
GET  /slm/status         Ollama connectivity & model info
GET  /slm/tools          OpenAI-compatible tool definitions
"""

from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request / response schemas ────────────────────────────────────────────────

class SLMQueryRequest(BaseModel):
    message: str = Field(..., description="Natural-language rider question")
    zone_context: Optional[str] = Field(
        default=None, description="Current zone_id the rider is viewing"
    )


class SLMPredictRequest(BaseModel):
    zone_id: str = Field(..., description="Target zone to predict fill for")
    lookahead_mins: int = Field(default=30, ge=10, le=120)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", summary="Natural-language query via Qwen2.5 + tool-calling")
async def slm_query(request: Request, payload: SLMQueryRequest):
    """
    Send a natural-language question to Qwen2.5 running in Ollama.

    The SLM may invoke any of the three PRD tool functions:
    - `zone_semantic_search` — find similar zones
    - `get_zone_forecast`    — get fill probability from XGBoost
    - `log_outcome`          — record telemetry

    Returns the assistant reply, list of tool calls made, and latency.
    """
    slm = request.app.state.slm_service
    result = await slm.query(payload.message, zone_context=payload.zone_context)
    return result


@router.post("/predict", summary="XGBoost prediction + SLM narrative")
async def slm_predict(request: Request, payload: SLMPredictRequest):
    """
    Hybrid prediction: deterministic XGBoost fill probability +
    Qwen2.5 natural-language narrative recommendation.

    - `fill_probability`  : XGBoost score (always present, fast, reliable)
    - `slm_narrative`     : 2-3 sentence riding recommendation from Qwen2.5
    - `slm_tool_calls`    : tools the SLM invoked while composing the narrative
    """
    slm = request.app.state.slm_service
    return await slm.predict_with_slm(payload.zone_id, payload.lookahead_mins)


@router.get("/status", summary="Ollama connectivity and model status")
async def slm_status(request: Request):
    """Returns Ollama health, loaded model name, and tool count."""
    slm = request.app.state.slm_service
    return slm.get_status()


@router.get("/tools", summary="OpenAI-compatible tool definitions")
async def slm_tools():
    """Returns the three tool schemas in OpenAI function_calling format."""
    from backend.services.slm_tools import TOOL_DEFINITIONS
    return TOOL_DEFINITIONS
