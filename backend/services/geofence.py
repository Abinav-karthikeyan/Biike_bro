"""
GeofenceRuleEngine — evaluates GBFS v3 ride_end_rules per zone at runtime.

Loaded from synthetic_seed/geofencing_zones.json at startup.
Geometry in the seed is Point (centroid) + radius_m from the zones table;
point-in-zone checks are handled upstream using Haversine against radius_m.

Rules that vary across the 60 synthetic zones
──────────────────────────────────────────────
  station_parking  — ~half the zones require docked-station drop-off only.
                     When False, dockless drop-off is not permitted.

Rules that are uniform (no violations fire for these)
──────────────────────────────────────────────────────
  ride_through_allowed = True   (all 60 zones)
  maximum_speed_kph    = 15     (all 60 zones)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_GEOFENCING_SEED = (
    Path(__file__).resolve().parent.parent.parent
    / "synthetic_seed"
    / "geofencing_zones.json"
)


class RuleViolation:
    """A single rule violation at a zone."""

    __slots__ = ("rule", "description")

    def __init__(self, rule: str, description: str) -> None:
        self.rule = rule
        self.description = description

    def __repr__(self) -> str:
        return f"RuleViolation({self.rule!r})"


class GeofenceRuleEngine:
    """
    In-process geofence rule evaluator backed by a JSON rule table.

    Evaluation is pure dict lookup — O(1) per zone, no network calls.
    The engine is safe to instantiate multiple times (tests, startup guard).
    """

    def __init__(self, geofencing_path: Optional[Path] = None) -> None:
        self._rules: Dict[str, dict] = {}
        self._load(geofencing_path or _GEOFENCING_SEED)

    # ── Load ─────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
            features = data["data"]["geofencing_zones"]["features"]
            for feat in features:
                props = feat["properties"]
                zone_id = props["zone_id"]
                raw = props.get("rules", [{}])[0].get("ride_end_rules", {})
                self._rules[zone_id] = {
                    "station_parking": raw.get("station_parking", True),
                    "max_speed_kph": raw.get("maximum_speed_kph", 15),
                    "ride_through_allowed": raw.get("ride_through_allowed", True),
                }
            logger.info("GeofenceRuleEngine: loaded rules for %d zones", len(self._rules))
        except Exception as exc:
            logger.warning("GeofenceRuleEngine: failed to load %s — %s", path, exc)

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(self, zone_id: str) -> List[RuleViolation]:
        """Return any rule violations for ending a ride at zone_id."""
        rule = self._rules.get(zone_id)
        if rule is None:
            return []
        violations: List[RuleViolation] = []
        if not rule["station_parking"]:
            violations.append(
                RuleViolation(
                    rule="station_parking",
                    description=(
                        f"Zone {zone_id}: dockless drop-off not permitted — "
                        "station parking required."
                    ),
                )
            )
        return violations

    def get_rule_summary(self, zone_id: str) -> Optional[dict]:
        """Raw rule dict for a zone, or None if zone is unknown."""
        return self._rules.get(zone_id)

    def get_context_line(self, zone_id: str) -> Optional[str]:
        """
        One-liner for RAG context injection.
        Returns None when there are no violations (the common case).
        """
        violations = self.evaluate(zone_id)
        if not violations:
            return None
        return "Geofence: " + "; ".join(v.description for v in violations)

    def zone_count(self) -> int:
        return len(self._rules)
