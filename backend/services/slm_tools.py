"""
SLM Tool-call Definitions — PRD §8 SLM Tool Calls.

These functions are registered as callable tools that the on-device SLM
(Qwen2.5-0.5B or Phi-3.5-mini) can invoke via function_calling.

Three tools defined per PRD §8:
  1. zone_semantic_search(current_zone_id, k, filter) → [{zone_id, similarity, occupancy_profile}]
  2. get_zone_forecast(zone_id, time_horizon_mins)     → {zone_id, arrival_time, expected_fill, ...}
  3. log_outcome(zone_id, timestamp, actual_availability, rider_satisfaction)  [telemetry]

The TOOL_DEFINITIONS list contains OpenAI-compatible function_calling schemas
that can be passed directly to any compatible SLM inference runtime.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from backend.models.schemas import (
    ZoneForecastResponse,
    ZoneSemanticSearchResult,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible tool schemas (pass to SLM runtime)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "zone_semantic_search",
            "description": (
                "Find parking zones with similar characteristics (transit connectivity, "
                "venue type, occupancy dynamics) to the current zone. "
                "Use when the user asks for nearby alternatives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_zone_id": {
                        "type": "string",
                        "description": "The zone the rider is currently targeting.",
                    },
                    "k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of similar zones to return (max 20).",
                    },
                    "max_distance_m": {
                        "type": "number",
                        "description": "Maximum walking distance in metres.",
                    },
                    "venue_type": {
                        "type": "string",
                        "description": "Filter by venue type (e.g. 'restaurant', 'park', 'transit_hub').",
                    },
                },
                "required": ["current_zone_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_zone_forecast",
            "description": (
                "Get a fill-probability forecast for a specific zone "
                "at the rider's estimated arrival time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "string",
                        "description": "Target parking zone identifier.",
                    },
                    "time_horizon_mins": {
                        "type": "integer",
                        "default": 30,
                        "description": "Minutes from now until estimated arrival.",
                    },
                },
                "required": ["zone_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_outcome",
            "description": (
                "Record the actual parking outcome for a rider's session. "
                "Opt-in telemetry used for model retraining."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "string",
                        "description": "The zone where the rider attempted to park.",
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "ISO-8601 timestamp of arrival.",
                    },
                    "actual_availability": {
                        "type": "number",
                        "description": "0.0 = zone full, 1.0 = zone empty.",
                    },
                    "rider_satisfaction": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Optional 1–5 satisfaction rating.",
                    },
                },
                "required": ["zone_id", "timestamp", "actual_availability"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations (server-side; stub — wire in real logic)
# ─────────────────────────────────────────────────────────────────────────────


async def zone_semantic_search(
    current_zone_id: str,
    k: int = 3,
    max_distance_m: Optional[float] = None,
    venue_type: Optional[str] = None,
) -> List[ZoneSemanticSearchResult]:
    """
    Tool implementation for zone_semantic_search.

    TODO:
      1. Fetch current zone embedding from zone_embeddings_hnsw table
      2. Call HNSWSearchService.zone_semantic_search(embedding, k, max_distance_m, venue_type)
      3. Attach real occupancy_profile from zone_snapshots
    """
    logger.info(
        "zone_semantic_search called",
        zone=current_zone_id,
        k=k,
        max_distance_m=max_distance_m,
    )
    # Stub response
    return [
        ZoneSemanticSearchResult(zone_id=f"{current_zone_id}-alt-{i+1}", similarity=0.9 - i * 0.1)
        for i in range(k)
    ]


async def get_zone_forecast(
    zone_id: str,
    time_horizon_mins: int = 30,
) -> ZoneForecastResponse:
    """
    Tool implementation for get_zone_forecast.

    TODO:
      1. Call PredictionService.predict(PredictRequest(zone_id, lookahead_mins))
      2. Map to ZoneForecastResponse including recommended_action logic
    """
    logger.info("get_zone_forecast called", zone=zone_id, horizon=time_horizon_mins)
    arrival = datetime.now(timezone.utc) + timedelta(minutes=time_horizon_mins)

    # Stub: ~60% fill, recommend park if < 70%
    expected_fill = 0.60
    action = "park here" if expected_fill < 0.70 else f"reroute — zone is {expected_fill:.0%} full"

    return ZoneForecastResponse(
        zone_id=zone_id,
        arrival_time=arrival,
        expected_fill=expected_fill,
        confidence=0.82,
        recommended_action=action,
    )


async def log_outcome(
    zone_id: str,
    timestamp: str,
    actual_availability: float,
    rider_satisfaction: Optional[int] = None,
) -> Dict[str, str]:
    """
    Tool implementation for log_outcome (telemetry).

    TODO: Persist to rider_outcomes table via async DB session.
    """
    logger.info(
        "log_outcome called",
        zone=zone_id,
        availability=actual_availability,
        satisfaction=rider_satisfaction,
    )
    # TODO: write to DB
    return {"status": "ok", "message": "Outcome logged (stub)"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher — maps SLM function_call.name → implementation
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "zone_semantic_search": zone_semantic_search,
    "get_zone_forecast": get_zone_forecast,
    "log_outcome": log_outcome,
}


async def dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> Any:
    """
    Dispatch an SLM function_call to its implementation.
    Called by your SLM inference loop when it emits a tool call.
    """
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return await TOOL_REGISTRY[name](**arguments)
