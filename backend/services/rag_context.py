"""
RAG Context Assembler — builds structured context blocks for SLM prompts.

Phase 4 of the Bike Parking Buddy roadmap.

Design
──────
• Assembles up to five context signals into a compact text block injected
  into the SLM system prompt before the rider's query.
• Token budget is tier-aware:
    cloud  (~500 tok): all five signals (occupancy + history + similar zones
                       + weather + nearby events)
    edge   (~200 tok): three signals only (occupancy + history + weather)
                       — similar-zone HNSW search and event filtering are
                       dropped to hit the tighter latency budget required
                       for mobile / on-device scenarios.
• Budgets are enforced by a simple char-count proxy (~4 chars / token for
  Qwen2.5's BPE vocab) — avoids a tiktoken dependency for a different
  tokeniser.

Context signals
───────────────
(a) Latest occupancy  — live bikes_available + occupancy_pct from the DB.
(b) Same-hour history — average occupancy at this hour over the past 7 days.
(c) Similar zones     — top-3 HNSW neighbours + their current occupancy.
                        Cloud tier only.
(d) Current weather   — temperature + WMO code from the weather table.
(e) Nearby events     — events starting within 2 hours within 1 km.
                        Cloud tier only.

On-device migration path
────────────────────────
In a true on-device deployment the server would expose a /context endpoint
that returns the assembled text block; the device runs its own quantised
model (GGUF / CoreML) and never sends inference traffic to this server.
This implementation is the server-side half of that future split: the same
assembler can be called from the context endpoint without any restructuring.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough proxy for Qwen2.5 BPE


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two (lat, lon) points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class RAGContextAssembler:
    """
    Assembles a compact context string for injection into SLM system prompts.

    Usage
    -----
    assembler = RAGContextAssembler(db_store, hnsw_service, geofence_engine)
    ctx = assembler.assemble("GLW_Z008", datetime.now(timezone.utc), tier="cloud")
    # ctx → "Current: 67% full, 8 bikes. Typical at 14:00: 61% (7-day avg). ..."
    """

    def __init__(self, db_store, hnsw_service, geofence_engine=None) -> None:
        self.db = db_store
        self.hnsw = hnsw_service
        self.geofence = geofence_engine

    def assemble(
        self,
        zone_id: str,
        query_time: Optional[datetime] = None,
        tier: str = "cloud",
        cloud_token_budget: int = 500,
        edge_token_budget: int = 200,
    ) -> str:
        """
        Build and return a context string for the given zone and inference tier.

        Parameters
        ----------
        zone_id : str
            The parking zone the rider is querying.
        query_time : datetime, optional
            Defaults to now (UTC).  Used for same-hour lookback and event filtering.
        tier : str
            "cloud" — full five-signal context, larger token budget.
            "edge"  — three-signal context, compressed for mobile inference.
        cloud_token_budget : int
            Max tokens allowed for cloud tier (enforced via char proxy).
        edge_token_budget : int
            Max tokens allowed for edge tier.
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)

        token_budget = cloud_token_budget if tier == "cloud" else edge_token_budget
        max_chars = token_budget * _CHARS_PER_TOKEN

        parts: List[str] = []

        # (a) Latest occupancy
        zone = self.db.get_zone(zone_id)
        if zone:
            occ = zone.get("occupancy_pct")
            bikes = zone.get("bikes_available")
            if occ is not None:
                bikes_str = str(int(bikes)) if bikes is not None else "?"
                parts.append(f"Current: {occ:.0f}% full, {bikes_str} bikes available.")
        else:
            parts.append(f"Zone {zone_id} not found in DB.")

        # (b) Same-hour historical average (7-day window)
        same_hour_avg = self._same_hour_avg(zone_id, query_time.hour)
        if same_hour_avg is not None:
            parts.append(
                f"Typical at {query_time.hour:02d}:00: {same_hour_avg:.0f}% (7-day avg)."
            )

        # (d) Current weather — fetched before similar zones so we can truncate
        #     gracefully if budget is tight on edge tier
        weather_str = self._weather_line()
        if weather_str:
            parts.append(weather_str)

        # (f) Geofence rule violations — both tiers
        if self.geofence and zone:
            rule_line = self.geofence.get_context_line(zone_id)
            if rule_line:
                parts.append(rule_line)

        # Cloud-only signals ─────────────────────────────────────────────────
        if tier == "cloud":
            # (c) HNSW similar zones
            if self.hnsw and zone:
                similar_str = self._similar_zones_line(zone_id, zone)
                if similar_str:
                    parts.append(similar_str)

            # (e) Nearby events within 2 hours and 1 km
            if zone:
                events_str = self._nearby_events_line(zone, query_time)
                if events_str:
                    parts.append(events_str)

        context = " ".join(parts)

        # Hard truncate to budget, respecting word boundaries
        if len(context) > max_chars:
            context = context[:max_chars].rsplit(" ", 1)[0] + "…"
            logger.debug(
                "RAG context truncated to %d chars (tier=%s, budget=%d tok)",
                len(context), tier, token_budget,
            )

        logger.debug(
            "RAG context assembled: tier=%s, ~%d tokens, zone=%s",
            tier, len(context) // _CHARS_PER_TOKEN, zone_id,
        )
        return context

    # ── Private helpers ───────────────────────────────────────────────────────

    def _same_hour_avg(self, zone_id: str, hour: int) -> Optional[float]:
        """Average occupancy at `hour` across the past 7 days of snapshots."""
        try:
            snapshots: List[Dict[str, Any]] = self.db.get_zone_snapshots(zone_id, hours=168)
            same_hour = [
                s["occupancy_pct"]
                for s in snapshots
                if s.get("hour_of_day") == hour and s.get("occupancy_pct") is not None
            ]
            if not same_hour:
                return None
            return sum(same_hour) / len(same_hour)
        except Exception as exc:
            logger.warning("same_hour_avg failed for %s: %s", zone_id, exc)
            return None

    def _weather_line(self) -> Optional[str]:
        """One-liner with current temperature and WMO weather code."""
        try:
            records = self.db.get_weather_recent(hours=1)
            if not records:
                return None
            w = records[0]
            temp = w.get("temp_celsius")
            wmo = w.get("wmo_code")
            if temp is None:
                return None
            desc = _wmo_description(int(wmo)) if wmo is not None else "unknown"
            return f"Weather: {temp:.1f}°C, {desc}."
        except Exception as exc:
            logger.warning("weather_line failed: %s", exc)
            return None

    def _similar_zones_line(self, zone_id: str, zone: Dict[str, Any]) -> Optional[str]:
        """Top-3 HNSW neighbours with their current occupancy."""
        try:
            emb = self.hnsw.get_zone_embedding(zone_id)
            if emb is None:
                return None
            results = self.hnsw.zone_semantic_search(
                emb, k=4, db_store=self.db, query_zone_id=zone_id
            )
            neighbours = [r for r in results if r.zone_id != zone_id][:3]
            if not neighbours:
                return None
            parts = []
            for r in neighbours:
                occ_str = "?%"
                if r.occupancy_profile:
                    occ_str = f"{r.occupancy_profile[0].avg_occupancy_pct:.0f}%"
                parts.append(f"{r.zone_id} ({occ_str})")
            return f"Similar zones: {', '.join(parts)}."
        except Exception as exc:
            logger.warning("similar_zones_line failed: %s", exc)
            return None

    def _nearby_events_line(
        self, zone: Dict[str, Any], query_time: datetime
    ) -> Optional[str]:
        """Events starting within 2 hours and within 1 km of the zone."""
        try:
            zone_lat = zone.get("lat")
            zone_lon = zone.get("lon")
            if zone_lat is None or zone_lon is None:
                return None

            events = self.db.get_events()
            window_end = query_time + timedelta(hours=2)

            # Normalise event datetimes to UTC-aware for comparison
            nearby = []
            for ev in events:
                st = ev.get("start_time")
                et = ev.get("end_time")
                vlat = ev.get("venue_lat")
                vlon = ev.get("venue_lon")
                if st is None or vlat is None or vlon is None:
                    continue
                if isinstance(st, str):
                    st = datetime.fromisoformat(st)
                if isinstance(et, str):
                    et = datetime.fromisoformat(et)

                # Make tz-aware if naive
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                if et is not None and et.tzinfo is None:
                    et = et.replace(tzinfo=timezone.utc)

                ends_after_now = (et is None) or (et >= query_time)
                starts_before_window = st <= window_end
                close_enough = _haversine_m(zone_lat, zone_lon, vlat, vlon) < 1000

                if ends_after_now and starts_before_window and close_enough:
                    nearby.append(ev)

            if not nearby:
                return None

            ev_strs = [
                f"{ev.get('category', 'event')} at {ev['start_time'].strftime('%H:%M') if hasattr(ev.get('start_time'), 'strftime') else ev.get('start_time', '?')}"
                for ev in nearby[:2]
            ]
            return f"Nearby events: {', '.join(ev_strs)}."
        except Exception as exc:
            logger.warning("nearby_events_line failed: %s", exc)
            return None


# ── WMO weather code descriptions (subset) ────────────────────────────────────

def _wmo_description(code: int) -> str:
    """Human-readable label for WMO weather interpretation codes."""
    _MAP = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "icy fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow",
        80: "showers", 81: "rain showers", 82: "heavy showers",
        95: "thunderstorm",
    }
    return _MAP.get(code, f"code {code}")
