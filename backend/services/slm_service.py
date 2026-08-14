"""
SLM Service — Groq/Llama inference with tool-calling and RAG context.

Architecture
────────────
• Two inference tiers, both served by the Groq API:
    cloud  — llama-3.1-8b-instant (8B); full RAG context; ~500-token budget.
    edge   — llama-3.2-1b-preview (1B); compressed RAG context; ~200-token budget.
  Tier is selected per-request via the `inference_tier` field in the
  query body.  Both tiers share the same API key; no separate pull step needed.

• RAG context (Phase 4) is injected into the system prompt before the
  rider's query.  The assembler fetches: (a) current occupancy,
  (b) 7-day same-hour history, (c) similar zones via HNSW (cloud only),
  (d) current weather, (e) nearby events (cloud only),
  (f) geofence rule violations.
  Pass `include_context=False` to compare tool-calling-only answers.

Tool-calling flow
──────────────────
 Rider query (NL)
       │
       ▼
 [RAG assembler → context block injected into system prompt]
       │
       ▼
 Groq.chat.completions.create(model=<tier>, tools=TOOL_DEFINITIONS)
       │
       ├─ tool_call → zone_semantic_search   → HNSWSearchService
       ├─ tool_call → get_zone_forecast      → PredictionService.predict()
       └─ tool_call → log_outcome            → DuckDBStore
       │
       ▼
 Final response assembled → SLMQueryResponse
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from groq import AsyncGroq

from backend.config import get_settings
from backend.models.schemas import PredictRequest, PredictResponse
from backend.services.rag_context import RAGContextAssembler
from backend.services.slm_tools import TOOL_DEFINITIONS, dispatch_tool_call

logger = logging.getLogger(__name__)
settings = get_settings()


class SLMService:
    """
    Wraps Groq/Llama for natural-language parking queries with tool-calling.

    Usage
    -----
    svc = SLMService(prediction_service=ps, db_store=db)
    result = await svc.query("Will GLW_Z008 be full when I arrive in 20 minutes?")

    The service is registered on app.state.slm_service in main.py lifespan.
    """

    def __init__(
        self,
        model: str = "",
        prediction_service=None,
        db_store=None,
        hnsw_service=None,
        geofence_engine=None,
    ) -> None:
        self.model = model or settings.GROQ_CLOUD_MODEL
        self.edge_model = settings.GROQ_EDGE_MODEL
        self._client = AsyncGroq(
            api_key=settings.GROQ_API_KEY,
            timeout=float(settings.GROQ_TIMEOUT_SECONDS),
        )
        self.prediction_service = prediction_service
        self.db = db_store
        self.hnsw_service = hnsw_service
        self._rag = RAGContextAssembler(
            db_store=db_store,
            hnsw_service=hnsw_service,
            geofence_engine=geofence_engine,
        )

        # Availability flags — no network call needed; verified at first real request
        self._groq_ok: bool = bool(settings.GROQ_API_KEY)
        self._edge_ok: bool = bool(settings.GROQ_API_KEY)

        if self._groq_ok:
            logger.info(
                "Groq ready — cloud='%s' edge='%s'", self.model, self.edge_model
            )
        else:
            logger.warning("GROQ_API_KEY not set; SLM will stub responses")

    # ── Public API ────────────────────────────────────────────────────────

    async def query(
        self,
        user_message: str,
        zone_context: Optional[str] = None,
        include_context: bool = True,
        inference_tier: str = "cloud",
    ) -> Dict[str, Any]:
        """
        Send a natural-language query to Groq/Llama with parking tools available.

        Parameters
        ----------
        user_message : str
            Rider's natural language question.
        zone_context : str, optional
            The zone_id the rider is currently viewing.
        include_context : bool
            When True (default), RAG context is assembled and injected into the
            system prompt.  Set False for tool-calling-only (useful for A/B tests).
        inference_tier : str
            "cloud" (default) — llama-3.1-8b-instant, full RAG (~500 tokens).
            "edge"            — llama-3.2-1b-preview, compressed context (~200 tokens).

        Returns
        -------
        dict with keys:
            content          : str
            tool_calls       : list[str]
            tool_results     : list[dict]
            latency_ms       : float
            model            : str
            llm_used         : bool
            rag_context_used : bool
            inference_tier   : str
        """
        t0 = time.perf_counter()

        model, effective_tier = self._resolve_tier(inference_tier)

        if not (self._groq_ok or self._edge_ok):
            return self._fallback_response(user_message, zone_context, t0, tier=effective_tier)

        # ── RAG context assembly ──────────────────────────────────────────
        rag_context = ""
        if include_context and zone_context and self.db:
            try:
                rag_context = self._rag.assemble(
                    zone_id=zone_context,
                    query_time=datetime.now(timezone.utc),
                    tier=effective_tier,
                    cloud_token_budget=settings.RAG_CLOUD_TOKEN_BUDGET,
                    edge_token_budget=settings.RAG_EDGE_TOKEN_BUDGET,
                )
            except Exception as exc:
                logger.warning("RAG assembly failed, continuing without context: %s", exc)

        system_prompt = self._build_system_prompt(zone_context, rag_context)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as exc:
            logger.error("Groq chat error (tier=%s): %s", effective_tier, exc)
            self._groq_ok = False
            self._edge_ok = False
            return self._fallback_response(
                user_message, zone_context, t0, error=str(exc), tier=effective_tier
            )

        # ── Handle tool calls ─────────────────────────────────────────────
        tool_results: List[Dict[str, Any]] = []
        msg = response.choices[0].message
        final_content = msg.content or ""

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args: Dict[str, Any] = {}
                if tc.function.arguments:
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                logger.info("SLM tool call: %s(%s)", tool_name, tool_args)
                try:
                    result = await self._dispatch(tool_name, tool_args)
                    tool_results.append({
                        "id": tc.id,
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                    })
                except Exception as e:
                    tool_results.append({
                        "id": tc.id,
                        "tool": tool_name,
                        "args": tool_args,
                        "error": str(e),
                    })

            # Second pass — send tool results back for natural-language summary
            if tool_results:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": r["id"],
                            "type": "function",
                            "function": {
                                "name": r["tool"],
                                "arguments": json.dumps(r["args"]),
                            },
                        }
                        for r in tool_results
                    ],
                })
                for r in tool_results:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(
                            r.get("result", r.get("error")), default=str
                        ),
                        "tool_call_id": r["id"],
                    })

                try:
                    final_resp = await self._client.chat.completions.create(
                        model=model, messages=messages
                    )
                    final_content = final_resp.choices[0].message.content or final_content
                except Exception as exc:
                    logger.warning("Groq second pass failed (tier=%s): %s", effective_tier, exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "SLM query done in %.1fms tier=%s tools=%d rag=%s",
            latency_ms, effective_tier, len(tool_results), bool(rag_context),
        )

        return {
            "content": final_content,
            "tool_calls": [tc.function.name for tc in (msg.tool_calls or [])],
            "tool_results": [{k: v for k, v in r.items() if k != "id"} for r in tool_results],
            "latency_ms": round(latency_ms, 1),
            "model": model,
            "llm_used": True,
            "rag_context_used": bool(rag_context),
            "inference_tier": effective_tier,
        }

    async def predict_with_slm(self, zone_id: str, lookahead_mins: int = 30) -> Dict[str, Any]:
        """
        Hybrid: XGBoost prediction + SLM natural-language explanation.
        Used by /predict/slm endpoint.
        """
        xgb_result: Optional[PredictResponse] = None
        if self.prediction_service:
            try:
                xgb_result = await self.prediction_service.predict(
                    PredictRequest(zone_id=zone_id, lookahead_mins=lookahead_mins)
                )
            except Exception as exc:
                logger.warning("XGBoost predict failed: %s", exc)

        fill_pct = round((xgb_result.fill_probability if xgb_result else 0.5) * 100)
        query = (
            f"Zone {zone_id}: XGBoost predicts {fill_pct}% fill probability in "
            f"{lookahead_mins} minutes. "
            "In 2 sentences, give a parking recommendation to the rider."
        )
        slm_out = await self.query(query, zone_context=zone_id)

        return {
            "zone_id": zone_id,
            "lookahead_mins": lookahead_mins,
            "fill_probability": xgb_result.fill_probability if xgb_result else None,
            "confidence": xgb_result.confidence if xgb_result else None,
            "alternative_zones": [az.model_dump() for az in (xgb_result.alternative_zones if xgb_result else [])],
            "xgb_reason": xgb_result.reason if xgb_result else None,
            "slm_narrative": slm_out["content"],
            "slm_tool_calls": slm_out["tool_calls"],
            "slm_latency_ms": slm_out["latency_ms"],
            "model_version": xgb_result.model_version if xgb_result else "stub",
            "slm_model": slm_out["model"],
            "llm_used": slm_out["llm_used"],
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_tier(self, requested_tier: str):
        """Return (model_name, effective_tier), falling back gracefully."""
        if requested_tier == "edge":
            if self._edge_ok:
                return self.edge_model, "edge"
            if self._groq_ok:
                return self.model, "cloud"
        else:
            if self._groq_ok:
                return self.model, "cloud"
            if self._edge_ok:
                return self.edge_model, "edge"
        return self.model, "cloud"  # stub fallback — _groq_ok gates the call

    def _build_system_prompt(
        self, zone_context: Optional[str], rag_context: str = ""
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%A %H:%M UTC")
        zone_hint = f" The rider is at zone {zone_context}." if zone_context else ""
        ctx_block = f"\n\nZone context:\n{rag_context}" if rag_context else ""
        return (
            f"You are Parking Buddy, an AI assistant for dockless bike-share parking in Glasgow. "
            f"Current time: {now}.{zone_hint}"
            f"{ctx_block}\n\n"
            "Use the provided tools to get live zone forecasts and alternatives. "
            "Be concise — riders are on the move. Max 3 sentences in your final reply."
        )

    async def _dispatch(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Route tool call — first tries real services, falls back to stub."""

        if tool_name == "zone_semantic_search" and self.hnsw_service:
            current_zone_id = tool_args.get("current_zone_id", "")
            k = int(tool_args.get("k", 3))
            max_dist = tool_args.get("max_distance_m")
            venue_filter = tool_args.get("venue_type")

            query_emb = self.hnsw_service.get_zone_embedding(current_zone_id)
            if query_emb is not None:
                raw = self.hnsw_service.zone_semantic_search(
                    query_emb, k=k + 1, max_distance_m=max_dist,
                    venue_type=venue_filter, db_store=self.db,
                    query_zone_id=current_zone_id,
                )
                results = [r for r in raw if r.zone_id != current_zone_id][:k]
                return [r.model_dump() for r in results]

        if tool_name == "get_zone_forecast" and self.prediction_service:
            zone_id = tool_args.get("zone_id", "GLW_Z001")
            horizon = tool_args.get("time_horizon_mins", 30)
            req = PredictRequest(zone_id=zone_id, lookahead_mins=horizon)
            resp = await self.prediction_service.predict(req)
            return {
                "zone_id": resp.zone_id,
                "fill_probability": resp.fill_probability,
                "confidence": resp.confidence,
                "reason": resp.reason,
                "alternatives": [a.model_dump() for a in resp.alternative_zones],
            }

        if tool_name == "log_outcome" and self.db:
            try:
                self.db.log_rider_outcome(
                    zone_id=tool_args.get("zone_id", ""),
                    timestamp=tool_args.get("timestamp", ""),
                    actual_availability=float(tool_args.get("actual_availability", 0.0)),
                    rider_satisfaction=tool_args.get("rider_satisfaction"),
                )
                return {"status": "ok", "message": "Outcome logged"}
            except Exception as exc:
                logger.warning("log_outcome DB write failed: %s", exc)
                return {"status": "error", "message": str(exc)}

        result = await dispatch_tool_call(tool_name, tool_args)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, list):
            return [r.model_dump() if hasattr(r, "model_dump") else r for r in result]
        return result

    def _fallback_response(
        self,
        user_message: str,
        zone_context: Optional[str],
        t0: float,
        error: str = "",
        tier: str = "cloud",
    ) -> Dict[str, Any]:
        """Return a structured stub when Groq is unavailable."""
        latency_ms = (time.perf_counter() - t0) * 1000
        note = f" (Groq error: {error})" if error else " (Groq unavailable — set GROQ_API_KEY)"
        return {
            "content": f"Parking Buddy is running in offline mode.{note}",
            "tool_calls": [],
            "tool_results": [],
            "latency_ms": round(latency_ms, 1),
            "model": self.model,
            "llm_used": False,
            "rag_context_used": False,
            "inference_tier": tier,
        }

    def get_status(self) -> Dict[str, Any]:
        """Health/status dict for /slm/status endpoint."""
        return {
            "provider": "groq",
            "groq_available": self._groq_ok,
            "cloud_model": self.model,
            "edge_model": self.edge_model,
            "edge_available": self._edge_ok,
            "tool_count": len(TOOL_DEFINITIONS),
            "rag_enabled": True,
            "tiers_available": (
                ["cloud", "edge"] if (self._groq_ok and self._edge_ok)
                else ["cloud"] if self._groq_ok
                else ["edge"] if self._edge_ok
                else []
            ),
        }
