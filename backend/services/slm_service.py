"""
SLM Service — Ollama/Qwen2.5 inference with tool-calling.

Architecture
────────────
• Uses the local Ollama daemon (http://localhost:11434) via the
  `ollama` Python SDK — same pattern as qwen/ollama_qwen.py.
• Model: qwen2.5 (7.6B Q4_K_M, already pulled) — swappable via config.
• Exposes 3 PRD tool functions as Ollama-native tool_calls so the SLM
  can decide which tool to invoke based on the rider's natural language query.
• Falls back to XGBoost PredictionService if Ollama is unreachable.

Tool-calling flow
──────────────────
 Rider query (NL)
       │
       ▼
 OllamaClient.chat(tools=TOOL_DEFINITIONS)
       │
       ├─ tool_call → zone_semantic_search   → HNSWSearchService (or fallback)
       ├─ tool_call → get_zone_forecast      → PredictionService.predict()
       └─ tool_call → log_outcome            → DB write (stub → real when DuckDB→prod)
       │
       ▼
 Final response assembled → SLMQueryResponse

Fine-tuning hook
────────────────
  See qwen/finetune/ for the training data schema and Unsloth/PEFT scripts.
  The trained adapter is loaded by setting OLLAMA_MODEL_NAME to the modelfile
  tag, or by pointing SLM_ADAPTER_PATH to the GGUF export.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import ollama  # ollama-python SDK

from backend.config import get_settings
from backend.models.schemas import PredictRequest, PredictResponse
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
    ) -> None:
        self.model = model or settings.OLLAMA_MODEL_NAME
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self._client = ollama.Client(host=self.ollama_base_url)
        self.prediction_service = prediction_service
        self.db = db_store
        self._ollama_ok: Optional[bool] = None
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Ping Ollama to confirm it's reachable and the model is pulled."""
        try:
            tags = self._client.list()
            names = [m.model for m in tags.models]
            if not any(self.model in n for n in names):
                logger.warning(
                    f"Ollama model '{self.model}' not found in pulled models: {names}. "
                    "Falling back to XGBoost stub."
                )
                self._ollama_ok = False
            else:
                logger.info(f"Ollama ready — model '{self.model}' confirmed")
                self._ollama_ok = True
        except Exception as exc:
            logger.warning(f"Ollama not reachable ({exc}); SLM will stub responses")
            self._ollama_ok = False

    # ── Public API ────────────────────────────────────────────────────────

    async def query(self, user_message: str, zone_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a natural-language query to Qwen2.5 with parking tools available.

        Parameters
        ----------
        user_message : str
            Rider's natural language question.
        zone_context : str, optional
            The zone_id the rider is currently viewing (injected into system prompt).

        Returns
        -------
        dict with keys:
            content       : str  — final assistant reply
            tool_calls    : list — raw tool_calls from model (may be empty)
            tool_results  : list — results of each dispatched tool call
            latency_ms    : float
            model         : str
            ollama_used   : bool
        """
        t0 = time.perf_counter()

        if not self._ollama_ok:
            return self._fallback_response(user_message, zone_context, t0)

        system_prompt = self._build_system_prompt(zone_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as exc:
            logger.error(f"Ollama chat error: {exc}")
            return self._fallback_response(user_message, zone_context, t0, error=str(exc))

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
                    final_resp = self._client.chat(model=self.model, messages=messages)
                    final_content = final_resp.message.content or final_content
                except Exception as exc:
                    logger.warning(f"Second Ollama pass failed: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"SLM query completed in {latency_ms:.1f}ms, tools={len(tool_results)}")

        return {
            "content": final_content,
            "tool_calls": [tc.function.name for tc in (response.message.tool_calls or [])],
            "tool_results": tool_results,
            "latency_ms": round(latency_ms, 1),
            "model": self.model,
            "ollama_used": True,
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

    def _build_system_prompt(self, zone_context: Optional[str]) -> str:
        now = datetime.now(timezone.utc).strftime("%A %H:%M UTC")
        zone_hint = f" The rider is currently looking at zone {zone_context}." if zone_context else ""
        return (
            f"You are Parking Buddy, an AI assistant for dockless bike-share parking in Glasgow. "
            f"Current time: {now}.{zone_hint} "
            "Use the provided tools to get live zone forecasts and alternatives. "
            "Be concise — riders are on the move. Max 3 sentences in your final reply."
        )

    async def _dispatch(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Route tool call — first tries real services, falls back to stub."""
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
        # Falls back to the stub dispatcher in slm_tools.py
        result = await dispatch_tool_call(tool_name, tool_args)
        # Serialise pydantic models so they JSON-encode cleanly
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, list):
            return [r.model_dump() if hasattr(r, "model_dump") else r for r in result]
        return result

    def _fallback_response(
        self, user_message: str, zone_context: Optional[str], t0: float, error: str = ""
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
        }

    def get_status(self) -> Dict[str, Any]:
        """Health/status dict for /slm/status endpoint."""
        return {
            "ollama_reachable": self._ollama_ok,
            "model": self.model,
            "base_url": self.ollama_base_url,
            "tool_count": len(TOOL_DEFINITIONS),
        }
