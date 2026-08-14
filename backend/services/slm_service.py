"""
SLM Service — Ollama/Qwen2.5 inference with tool-calling and RAG context.

Architecture
────────────
• Two inference tiers, both served by the local Ollama daemon:
    cloud  — qwen2.5 (7B Q4_K_M); full RAG context; ~500-token budget.
    edge   — qwen2.5:0.5b; compressed RAG context; ~200-token budget.
             Pull with: ollama pull qwen2.5:0.5b
  Tier is selected per-request via the `inference_tier` field in the
  query body.  If the requested model is not pulled, the service falls
  back to whichever tier IS available (cloud → edge → stub).

• RAG context (Phase 4) is injected into the system prompt before the
  rider's query.  The assembler fetches: (a) current occupancy,
  (b) 7-day same-hour history, (c) similar zones via HNSW (cloud only),
  (d) current weather, (e) nearby events (cloud only).
  Pass `include_context=False` to compare tool-calling-only answers.

Tool-calling flow
──────────────────
 Rider query (NL)
       │
       ▼
 [RAG assembler → context block injected into system prompt]
       │
       ▼
 OllamaClient.chat(model=<tier>, tools=TOOL_DEFINITIONS)
       │
       ├─ tool_call → zone_semantic_search   → HNSWSearchService
       ├─ tool_call → get_zone_forecast      → PredictionService.predict()
       └─ tool_call → log_outcome            → DuckDBStore
       │
       ▼
 Final response assembled → SLMQueryResponse

On-device migration path
────────────────────────
  In a true on-device deployment the phone runs a GGUF / CoreML quantised
  model and consumes context from a /context endpoint (same RAGContextAssembler,
  no inference server-side).  The two-tier implementation here is the
  server-side precedent for that split — same code, different model weights
  and token budgets.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import ollama  # ollama-python SDK

from backend.config import get_settings
from backend.models.schemas import PredictRequest, PredictResponse
from backend.services.rag_context import RAGContextAssembler
from backend.services.slm_tools import TOOL_DEFINITIONS, dispatch_tool_call

logger = logging.getLogger(__name__)
settings = get_settings()


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon cannot be reached."""


class SLMService:
    """
    Wraps Ollama/Qwen2.5 for natural-language parking queries with tool-calling.

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
        self.model = model or settings.OLLAMA_MODEL_NAME
        self.edge_model = settings.OLLAMA_EDGE_MODEL_NAME
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self._client = ollama.Client(host=self.ollama_base_url)
        self.prediction_service = prediction_service
        self.db = db_store
        self.hnsw_service = hnsw_service
        self._ollama_ok: Optional[bool] = None
        self._edge_ok: Optional[bool] = None
        self._rag = RAGContextAssembler(
            db_store=db_store,
            hnsw_service=hnsw_service,
            geofence_engine=geofence_engine,
        )
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Ping Ollama; record which tiers (cloud, edge) are available."""
        try:
            tags = self._client.list()
            names = [m.model for m in tags.models]
            self._ollama_ok = any(self.model in n for n in names)
            self._edge_ok = any(self.edge_model in n for n in names)
            if self._ollama_ok:
                logger.info("Ollama ready — cloud model '%s' confirmed", self.model)
            else:
                logger.warning(
                    "Cloud model '%s' not pulled. Available: %s", self.model, names
                )
            if self._edge_ok:
                logger.info("Ollama edge tier — model '%s' confirmed", self.edge_model)
            else:
                logger.info(
                    "Edge model '%s' not pulled; edge tier will fall back to cloud. "
                    "Run: ollama pull %s", self.edge_model, self.edge_model
                )
        except Exception as exc:
            logger.warning("Ollama not reachable (%s); SLM will stub responses", exc)
            self._ollama_ok = False
            self._edge_ok = False

    # ── Public API ────────────────────────────────────────────────────────

    async def query(
        self,
        user_message: str,
        zone_context: Optional[str] = None,
        include_context: bool = True,
        inference_tier: str = "cloud",
    ) -> Dict[str, Any]:
        """
        Send a natural-language query to Qwen2.5 with parking tools available.

        Parameters
        ----------
        user_message : str
            Rider's natural language question.
        zone_context : str, optional
            The zone_id the rider is currently viewing.
        include_context : bool
            When True (default), RAG context is assembled and injected into the
            system prompt before the query.  Set False to run tool-calling only
            (useful for A/B comparison of RAG vs no-RAG answers).
        inference_tier : str
            "cloud" (default) — qwen2.5 7B, full RAG context (~500 tokens).
            "edge"            — qwen2.5:0.5b, compressed context (~200 tokens).
            Falls back to the available tier if the requested model is not pulled.

        Returns
        -------
        dict with keys:
            content          : str
            tool_calls       : list[str]
            tool_results     : list[dict]
            latency_ms       : float
            model            : str
            ollama_used      : bool
            rag_context_used : bool
            inference_tier   : str
        """
        t0 = time.perf_counter()

        # ── Tier selection ────────────────────────────────────────────────
        model, effective_tier = self._resolve_tier(inference_tier)

        if not (self._ollama_ok or self._edge_ok):
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self._client.chat(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as exc:
            logger.error("Ollama chat error (tier=%s): %s", effective_tier, exc)
            return self._fallback_response(
                user_message, zone_context, t0, error=str(exc), tier=effective_tier
            )

        # ── Handle tool calls ─────────────────────────────────────────────
        tool_results = []
        final_content = response.message.content or ""

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments or {}
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                logger.info(f"SLM tool call: {tool_name}({tool_args})")
                try:
                    result = await self._dispatch(tool_name, tool_args)
                    tool_results.append({"tool": tool_name, "args": tool_args, "result": result})
                except Exception as e:
                    tool_results.append({"tool": tool_name, "args": tool_args, "error": str(e)})

            # Second pass — send tool results back for natural language summary
            if tool_results:
                tool_content = json.dumps([r.get("result", r.get("error")) for r in tool_results], default=str)
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": r["tool"], "arguments": r["args"]}}
                    for r in tool_results
                ]})
                messages.append({"role": "tool", "content": tool_content})

                try:
                    final_resp = self._client.chat(model=model, messages=messages)
                    final_content = final_resp.message.content or final_content
                except Exception as exc:
                    logger.warning("Second Ollama pass failed (tier=%s): %s", effective_tier, exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "SLM query done in %.1fms tier=%s tools=%d rag=%s",
            latency_ms, effective_tier, len(tool_results), bool(rag_context),
        )

        return {
            "content": final_content,
            "tool_calls": [tc.function.name for tc in (response.message.tool_calls or [])],
            "tool_results": tool_results,
            "latency_ms": round(latency_ms, 1),
            "model": model,
            "ollama_used": True,
            "rag_context_used": bool(rag_context),
            "inference_tier": effective_tier,
        }

    async def predict_with_slm(self, zone_id: str, lookahead_mins: int = 30) -> Dict[str, Any]:
        """
        Hybrid: XGBoost prediction + SLM natural-language explanation.
        Used by /predict/slm endpoint.
        """
        # Run deterministic XGBoost first (always fast, always available)
        xgb_result: Optional[PredictResponse] = None
        if self.prediction_service:
            try:
                xgb_result = await self.prediction_service.predict(
                    PredictRequest(zone_id=zone_id, lookahead_mins=lookahead_mins)
                )
            except Exception as exc:
                logger.warning(f"XGBoost predict failed: {exc}")

        # Build a concise query for the SLM to narrate
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
            "ollama_used": slm_out["ollama_used"],
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_tier(self, requested_tier: str):
        """
        Return (model_name, effective_tier) for the request.

        Falls back gracefully:
          requested=cloud, cloud ok  → (cloud_model, "cloud")
          requested=cloud, cloud down, edge ok → (edge_model, "edge")
          requested=edge, edge ok    → (edge_model, "edge")
          requested=edge, edge down  → (cloud_model, "cloud")  if cloud ok
        """
        if requested_tier == "edge":
            if self._edge_ok:
                return self.edge_model, "edge"
            if self._ollama_ok:
                logger.info(
                    "Edge model not available; falling back to cloud tier. "
                    "Pull with: ollama pull %s", self.edge_model
                )
                return self.model, "cloud"
        else:
            if self._ollama_ok:
                return self.model, "cloud"
            if self._edge_ok:
                logger.info("Cloud model unavailable; falling back to edge tier")
                return self.edge_model, "edge"
        return self.model, "cloud"  # stub fallback — _ollama_ok will gate the call

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
                # Search for k+1 so we can exclude the query zone itself
                raw = self.hnsw_service.zone_semantic_search(
                    query_emb, k=k + 1, max_distance_m=max_dist,
                    venue_type=venue_filter, db_store=self.db,
                    query_zone_id=current_zone_id,
                )
                results = [r for r in raw if r.zone_id != current_zone_id][:k]
                return [r.model_dump() for r in results]
            # Zone not in index yet — fall through to stub

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
                logger.warning(f"log_outcome DB write failed: {exc}")
                return {"status": "error", "message": str(exc)}

        # Falls back to the stub dispatcher in slm_tools.py
        result = await dispatch_tool_call(tool_name, tool_args)
        # Serialise pydantic models so they JSON-encode cleanly
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
        """Return a structured stub when Ollama is unavailable."""
        latency_ms = (time.perf_counter() - t0) * 1000
        note = f" (Ollama error: {error})" if error else " (Ollama unavailable — stub response)"
        return {
            "content": f"Parking Buddy is running in offline mode.{note}",
            "tool_calls": [],
            "tool_results": [],
            "latency_ms": round(latency_ms, 1),
            "model": self.model,
            "ollama_used": False,
            "rag_context_used": False,
            "inference_tier": tier,
        }

    def get_status(self) -> Dict[str, Any]:
        """Health/status dict for /slm/status endpoint."""
        return {
            "ollama_reachable": self._ollama_ok,
            "model": self.model,
            "edge_model": self.edge_model,
            "edge_model_available": self._edge_ok,
            "base_url": self.ollama_base_url,
            "tool_count": len(TOOL_DEFINITIONS),
            "rag_enabled": True,
            "tiers_available": (
                ["cloud", "edge"] if (self._ollama_ok and self._edge_ok)
                else ["cloud"] if self._ollama_ok
                else ["edge"] if self._edge_ok
                else []
            ),
        }
