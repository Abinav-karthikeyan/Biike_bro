"""
Data Validation — Schema validation and data quality checks.

Provides validators for zone metadata, snapshots, rides, and other entities
to ensure data consistency and catch issues early.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Validation Errors
# ─────────────────────────────────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when data validation fails."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Zone Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_zone(zone: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate zone metadata record.

    Required fields:
    - zone_id: string identifier
    - name: display name
    - lat, lon: numeric coordinates in [-90, 90] and [-180, 180]
    - capacity: positive integer or None
    - venue_type: string
    - transit_score: float in [0, 1]

    Args:
        zone: Zone metadata dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Required fields
    if not zone.get("zone_id"):
        errors.append("Missing zone_id")
    if not zone.get("name"):
        errors.append("Missing name")

    # Coordinates
    if "lat" not in zone or "lon" not in zone:
        errors.append("Missing lat/lon coordinates")
    else:
        lat, lon = zone["lat"], zone["lon"]
        if not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
            errors.append(f"Invalid latitude: {lat}")
        if not isinstance(lon, (int, float)) or not (-180 <= lon <= 180):
            errors.append(f"Invalid longitude: {lon}")

    # Optional numeric fields
    if "capacity" in zone and zone["capacity"] is not None:
        if not isinstance(zone["capacity"], int) or zone["capacity"] < 0:
            errors.append(f"Invalid capacity: {zone.get('capacity')}")

    if "transit_score" in zone and zone["transit_score"] is not None:
        score = zone["transit_score"]
        if not isinstance(score, (int, float)) or not (0 <= score <= 1):
            errors.append(f"Transit score out of range [0,1]: {score}")

    return len(errors) == 0, errors


def validate_zones(zones: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """
    Validate list of zone records and return summary.

    Args:
        zones: List of zone dictionaries

    Returns:
        Tuple of (valid_count, error_list)
    """
    valid_count = 0
    all_errors = []

    for i, zone in enumerate(zones):
        is_valid, errors = validate_zone(zone)
        if is_valid:
            valid_count += 1
        else:
            for error in errors:
                all_errors.append(f"Zone #{i}: {error}")

    return valid_count, all_errors


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_snapshot(snapshot: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate zone snapshot (occupancy) record.

    Required fields:
    - zone_id: string identifier
    - timestamp: datetime-compatible value
    - occupancy_pct: float in [0, 1]
    - available_bikes: non-negative integer
    - docks_used: non-negative integer

    Args:
        snapshot: Snapshot dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not snapshot.get("zone_id"):
        errors.append("Missing zone_id")

    if "timestamp" not in snapshot:
        errors.append("Missing timestamp")

    if "occupancy_pct" in snapshot and snapshot["occupancy_pct"] is not None:
        occ = snapshot["occupancy_pct"]
        if not isinstance(occ, (int, float)) or not (0 <= occ <= 1):
            errors.append(f"Occupancy out of range [0,1]: {occ}")

    for field in ["available_bikes", "docks_used"]:
        if field in snapshot and snapshot[field] is not None:
            val = snapshot[field]
            if not isinstance(val, int) or val < 0:
                errors.append(f"Invalid {field}: {val}")

    return len(errors) == 0, errors


def validate_snapshots(
    snapshots: List[Dict[str, Any]], max_errors: int = 10
) -> Tuple[int, List[str]]:
    """
    Validate list of snapshot records with early exit on too many errors.

    Args:
        snapshots: List of snapshot dictionaries
        max_errors: Stop validation after this many errors

    Returns:
        Tuple of (valid_count, error_list)
    """
    valid_count = 0
    all_errors = []

    for i, snapshot in enumerate(snapshots):
        if len(all_errors) >= max_errors:
            all_errors.append(
                f"... and {len(snapshots) - i} more records not validated"
            )
            break

        is_valid, errors = validate_snapshot(snapshot)
        if is_valid:
            valid_count += 1
        else:
            for error in errors:
                all_errors.append(f"Snapshot #{i}: {error}")

    return valid_count, all_errors


# ─────────────────────────────────────────────────────────────────────────────
# Ride Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_ride(ride: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate ride record.

    Required fields:
    - ride_id: string identifier
    - bike_id: string identifier
    - start_zone_id, end_zone_id: zone identifiers
    - start_time, end_time: timestamps
    - duration_secs: non-negative integer
    - distance_m: non-negative numeric

    Args:
        ride: Ride dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    required = ["ride_id", "bike_id", "start_zone_id", "end_zone_id"]
    for field in required:
        if not ride.get(field):
            errors.append(f"Missing {field}")

    for field in ["start_time", "end_time"]:
        if field not in ride:
            errors.append(f"Missing {field}")

    if "duration_secs" in ride and ride["duration_secs"] is not None:
        dur = ride["duration_secs"]
        if not isinstance(dur, int) or dur < 0:
            errors.append(f"Invalid duration_secs: {dur}")

    if "distance_m" in ride and ride["distance_m"] is not None:
        dist = ride["distance_m"]
        if not isinstance(dist, (int, float)) or dist < 0:
            errors.append(f"Invalid distance_m: {dist}")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_rider_outcome(outcome: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate rider outcome telemetry record.

    Required fields:
    - zone_id: string identifier
    - timestamp: arrival timestamp
    - actual_availability: float in [0, 1]

    Optional fields:
    - rider_satisfaction: integer, typically [1, 5]

    Args:
        outcome: Outcome dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not outcome.get("zone_id"):
        errors.append("Missing zone_id")

    if "timestamp" not in outcome:
        errors.append("Missing timestamp")

    if "actual_availability" in outcome and outcome["actual_availability"] is not None:
        avail = outcome["actual_availability"]
        if not isinstance(avail, (int, float)) or not (0 <= avail <= 1):
            errors.append(f"Availability out of range [0,1]: {avail}")

    if (
        "rider_satisfaction" in outcome
        and outcome["rider_satisfaction"] is not None
    ):
        sat = outcome["rider_satisfaction"]
        if not isinstance(sat, int) or sat < 1 or sat > 5:
            logger.warning(f"Unusual satisfaction rating: {sat} (expected [1,5])")

    return len(errors) == 0, errors
