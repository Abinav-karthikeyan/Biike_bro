"""
Bike Parking Buddy — Synthetic Data Generator
=============================================
Schema is GBFS-aligned. Real data swap = column remap only.
Runs: Python 3.9+ with numpy + pandas (stdlib otherwise).
Outputs: SQLite DB (prototype) + CSV exports (DuckDB-ready).

DuckDB DDL is included in each section — copy-paste when you
have real data and a DuckDB instance.

City modelled: Glasgow (Nextbike / Lime footprint approximation)
"""

import sqlite3
import json
import math
import random
import csv
import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ── Output paths ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
DB_PATH = OUT_DIR / "parking_buddy.db"

# ── City seed — Glasgow approximation ────────────────────────────
CITY_CENTER_LAT = 55.8617
CITY_CENTER_LON = -4.2583
NUM_ZONES       = 60
NUM_BIKES       = 400
HISTORY_DAYS    = 42   # 6 weeks
SNAPSHOT_FREQ   = 5    # minutes between occupancy snapshots


# ═══════════════════════════════════════════════════════════════
# SCHEMA REFERENCE
# ═══════════════════════════════════════════════════════════════
SCHEMA_NOTES = """
GBFS field mapping (dockless, GBFS v3):
──────────────────────────────────────────────────────────────
zones             ← geofencing_zones.features[]
  zone_id         ← feature.properties.zone_id (GBFS v3) / synthesised for v2
  name            ← feature.properties.name
  lat/lon         ← feature.geometry centroid
  radius_m        ← derived from geometry bbox
  zone_type       ← feature.properties.rules[].ride_end_rules type
  capacity        ← NOT in GBFS — operator-specific extension or estimated

zone_snapshots    ← derived: aggregate free_bike_status by zone + timestamp
  zone_id         ← spatial join of vehicle lat/lon to zone polygon
  bikes_available ← count of vehicles in zone polygon
  occupancy_pct   ← bikes_available / capacity * 100
  timestamp       ← polling timestamp (your Airflow DAG, 5-min cadence)

bikes             ← free_bike_status.bikes[]  (GBFS §8)
  bike_id         ← bike_id
  lat/lon         ← lat/lon
  vehicle_type_id ← vehicle_type_id
  is_reserved     ← is_reserved
  is_disabled     ← is_disabled
  current_range_m ← current_range_meters

rides             ← NOT in GBFS — operator proprietary API / MDS (Mobility Data Specification)
  ride_id         ← trip_id (MDS)
  start_zone_id   ← origin_zone (spatial)
  end_zone_id     ← destination_zone (spatial)
  start_time      ← event_time where event_type = trip_start
  end_time        ← event_time where event_type = trip_end
  duration_secs   ← derived
  cost_pence      ← cost.amount (MDS)
  was_redirected  ← synthesised / inferred from GPS trace

weather           ← OpenMeteo API or Met Office — join on (city, hour)
local_events      ← Ticketmaster / Songkick API — join on (city, date)

DuckDB DDL — copy this when you have real data:
──────────────────────────────────────────────────────────────
CREATE TABLE zones (
    zone_id          VARCHAR PRIMARY KEY,
    name             VARCHAR,
    lat              DOUBLE,
    lon              DOUBLE,
    radius_m         DOUBLE,
    zone_type        VARCHAR,        -- 'recommended','mandatory','no_parking'
    venue_type       VARCHAR,        -- 'transit','retail','park','residential','university'
    capacity         INTEGER,
    transit_score    DOUBLE,         -- 0-1, proximity to PT
    neighborhood     VARCHAR,
    gbfs_region_id   VARCHAR,        -- maps to regions.json region_id
    created_at       TIMESTAMPTZ
);

CREATE TABLE zone_snapshots (
    snapshot_id      BIGINT,
    zone_id          VARCHAR,
    timestamp        TIMESTAMPTZ,
    bikes_available  INTEGER,
    capacity         INTEGER,
    occupancy_pct    DOUBLE,
    weather_code     INTEGER,        -- WMO code
    temp_celsius     DOUBLE,
    is_event_nearby  BOOLEAN,
    day_of_week      INTEGER,        -- 0=Mon
    hour_of_day      INTEGER,
    PRIMARY KEY (zone_id, timestamp)
);

CREATE TABLE bikes (
    bike_id          VARCHAR PRIMARY KEY,
    vehicle_type_id  VARCHAR,        -- maps to vehicle_types.json
    lat              DOUBLE,
    lon              DOUBLE,
    is_reserved      BOOLEAN,
    is_disabled      BOOLEAN,
    current_range_m  INTEGER,        -- NULL for non-electric
    last_reported    TIMESTAMPTZ,
    current_zone_id  VARCHAR         -- spatial join result
);

CREATE TABLE rides (
    ride_id          VARCHAR PRIMARY KEY,
    bike_id          VARCHAR,
    rider_hash       VARCHAR,        -- pseudonymised rider id
    start_zone_id    VARCHAR,
    end_zone_id      VARCHAR,
    start_lat        DOUBLE,
    start_lon        DOUBLE,
    end_lat          DOUBLE,
    end_lon          DOUBLE,
    start_time       TIMESTAMPTZ,
    end_time         TIMESTAMPTZ,
    duration_secs    INTEGER,
    distance_m       INTEGER,
    cost_pence       INTEGER,
    was_redirected   BOOLEAN,        -- key label for training
    redirect_reason  VARCHAR,        -- 'zone_full','no_bikes_nearby','geofence'
    minutes_wasted   DOUBLE          -- extra billing time due to redirect
);

CREATE TABLE weather (
    city_id          VARCHAR,
    timestamp        TIMESTAMPTZ,
    temp_celsius     DOUBLE,
    precip_mm        DOUBLE,
    wind_kmh         DOUBLE,
    wmo_code         INTEGER,
    is_daylight      BOOLEAN,
    PRIMARY KEY (city_id, timestamp)
);

CREATE TABLE local_events (
    event_id         VARCHAR PRIMARY KEY,
    city_id          VARCHAR,
    event_date       DATE,
    start_time       TIMESTAMPTZ,
    end_time         TIMESTAMPTZ,
    venue_lat        DOUBLE,
    venue_lon        DOUBLE,
    expected_attendance INTEGER,
    category         VARCHAR         -- 'football','concert','market','festival'
);

CREATE TABLE zone_embeddings (
    zone_id          VARCHAR PRIMARY KEY,
    embedding        DOUBLE[256],    -- DuckDB native array
    model_version    VARCHAR,
    computed_at      TIMESTAMPTZ
);
"""


# ═══════════════════════════════════════════════════════════════
# 1. ZONES
# ═══════════════════════════════════════════════════════════════

VENUE_TYPES    = ["transit","retail","park","residential","university","mixed"]
ZONE_TYPES     = ["recommended","mandatory"]
NEIGHBORHOODS  = [
    "City Centre","West End","Merchant City","Finnieston",
    "Partick","Hillhead","Govan","Shawlands","Dennistoun","Byres Road"
]

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def make_zones(n):
    zones = []
    for i in range(n):
        angle = random.uniform(0, 2 * math.pi)
        dist  = random.gauss(0, 0.025)          # ~2.5km std from centre
        lat   = CITY_CENTER_LAT + dist * math.cos(angle)
        lon   = CITY_CENTER_LON + dist * math.sin(angle) * 1.5  # elongate E-W
        vtype = random.choices(
            VENUE_TYPES,
            weights=[0.25, 0.25, 0.10, 0.20, 0.10, 0.10]
        )[0]
        capacity = {
            "transit": random.randint(15, 30),
            "retail":  random.randint(8,  20),
            "park":    random.randint(10, 25),
            "residential": random.randint(5, 12),
            "university": random.randint(20, 40),
            "mixed":   random.randint(10, 20),
        }[vtype]
        dist_to_centre = haversine_m(lat, lon, CITY_CENTER_LAT, CITY_CENTER_LON)
        transit_score  = max(0.0, 1.0 - dist_to_centre / 4000)
        zones.append({
            "zone_id":       f"GLW_Z{str(i+1).zfill(3)}",
            "name":          f"{random.choice(NEIGHBORHOODS)} Zone {i+1}",
            "lat":           round(lat, 6),
            "lon":           round(lon, 6),
            "radius_m":      random.randint(30, 80),
            "zone_type":     random.choice(ZONE_TYPES),
            "venue_type":    vtype,
            "capacity":      capacity,
            "transit_score": round(transit_score, 3),
            "neighborhood":  random.choice(NEIGHBORHOODS),
            "gbfs_region_id":"GLW_R01",
            "created_at":    "2024-01-01T00:00:00Z",
        })
    return zones

zones = make_zones(NUM_ZONES)
print(f"✓ {len(zones)} zones generated")


# ═══════════════════════════════════════════════════════════════
# 2. BIKES
# ═══════════════════════════════════════════════════════════════

VEHICLE_TYPES = ["standard_v1","ebike_v2","cargo_v1"]

def make_bikes(n, zones_list):
    bikes = []
    for i in range(n):
        zone  = random.choice(zones_list)
        angle = random.uniform(0, 2*math.pi)
        r_deg = (zone["radius_m"] / 111_000) * random.uniform(0, 1)
        lat   = zone["lat"] + r_deg * math.cos(angle)
        lon   = zone["lon"] + r_deg * math.sin(angle)
        vtype = random.choices(VEHICLE_TYPES, weights=[0.55, 0.35, 0.10])[0]
        bikes.append({
            "bike_id":         f"GLW_B{str(i+1).zfill(4)}",
            "vehicle_type_id": vtype,
            "lat":             round(lat, 6),
            "lon":             round(lon, 6),
            "is_reserved":     False,
            "is_disabled":     random.random() < 0.04,
            "current_range_m": random.randint(8000, 35000) if "ebike" in vtype else None,
            "last_reported":   datetime.utcnow().isoformat() + "Z",
            "current_zone_id": zone["zone_id"],
        })
    return bikes

bikes = make_bikes(NUM_BIKES, zones)
print(f"✓ {len(bikes)} bikes generated")


# ═══════════════════════════════════════════════════════════════
# 3. WEATHER  (synthetic WMO codes, realistic Scottish patterns)
# ═══════════════════════════════════════════════════════════════

WMO_CLEAR  = [0, 1]
WMO_CLOUD  = [2, 3]
WMO_DRIZZLE= [51, 53, 55]
WMO_RAIN   = [61, 63, 65, 80, 81]
WMO_STORM  = [95, 96]

def weather_for_hour(dt):
    hour  = dt.hour
    month = dt.month
    rain_prob = 0.55 if month in [10,11,12,1,2,3] else 0.30  # Scotland is Scotland
    rain_prob += 0.10 if 7 <= hour <= 9 or 16 <= hour <= 18 else 0

    if random.random() < rain_prob:
        wmo = random.choice(WMO_RAIN + WMO_DRIZZLE)
        temp_base = 7 if month in [12,1,2] else 12
    else:
        wmo = random.choice(WMO_CLEAR + WMO_CLOUD)
        temp_base = 10 if month in [12,1,2] else 16

    temp  = round(temp_base + random.gauss(0, 2), 1)
    precip = round(random.uniform(0.2, 4.5), 1) if wmo in WMO_RAIN+WMO_DRIZZLE else 0.0
    is_day = 7 <= hour <= 21

    return {
        "city_id":      "GLW",
        "timestamp":    dt.isoformat() + "Z",
        "temp_celsius": temp,
        "precip_mm":    precip,
        "wind_kmh":     round(random.uniform(5, 35), 1),
        "wmo_code":     wmo,
        "is_daylight":  is_day,
    }

start_dt = datetime.utcnow() - timedelta(days=HISTORY_DAYS)
hourly_weather = []
cursor = start_dt.replace(minute=0, second=0, microsecond=0)
end_dt = datetime.utcnow() + timedelta(hours=48)   # +48h forecast
while cursor <= end_dt:
    hourly_weather.append(weather_for_hour(cursor))
    cursor += timedelta(hours=1)

print(f"✓ {len(hourly_weather)} hourly weather records")


# ═══════════════════════════════════════════════════════════════
# 4. LOCAL EVENTS
# ═══════════════════════════════════════════════════════════════

EVENT_CATS  = ["football","concert","market","festival","conference"]
VENUES = [
    (55.8477, -4.3123, "Ibrox"),          # football
    (55.8499, -4.2527, "Hydro"),          # concerts
    (55.8617, -4.2583, "George Square"),  # market/festival
    (55.8724, -4.2908, "Kelvingrove"),
    (55.8576, -4.2485, "SECC"),
]

def make_events():
    events = []
    eid = 0
    d = start_dt.date()
    end_d = (datetime.utcnow() + timedelta(days=14)).date()
    while d <= end_d:
        # 2-4 events per week roughly
        if random.random() < 0.40:
            venue_lat, venue_lon, venue_name = random.choice(VENUES)
            cat = random.choices(
                EVENT_CATS,
                weights=[0.25, 0.30, 0.20, 0.15, 0.10]
            )[0]
            start_hr = random.choice([12, 14, 17, 19, 20])
            dur_hrs  = random.choice([2, 3, 4])
            start_ts = datetime(d.year, d.month, d.day, start_hr)
            events.append({
                "event_id":              f"GLW_E{str(eid).zfill(4)}",
                "city_id":               "GLW",
                "event_date":            d.isoformat(),
                "start_time":            start_ts.isoformat() + "Z",
                "end_time":              (start_ts + timedelta(hours=dur_hrs)).isoformat() + "Z",
                "venue_lat":             venue_lat,
                "venue_lon":             venue_lon,
                "expected_attendance":   random.randint(500, 50000),
                "category":              cat,
            })
            eid += 1
        d += timedelta(days=1)
    return events

local_events = make_events()
print(f"✓ {len(local_events)} local events")


# ═══════════════════════════════════════════════════════════════
# 5. ZONE SNAPSHOTS  (the core training table)
# ═══════════════════════════════════════════════════════════════

def demand_multiplier(dt, zone, weather_row, events_nearby):
    """
    Realistic demand model:
    - Peak hours: 8-9am, 5-7pm
    - Weekend mid-day spike
    - Transit zones deplete faster in peaks
    - Rain reduces demand by 30-50%
    - Events nearby cause spikes
    """
    hour = dt.hour
    dow  = dt.weekday()   # 0=Mon
    mult = 1.0

    # Time-of-day pattern
    if 7 <= hour <= 9:
        mult *= 1.6 + (0.4 if zone["venue_type"] == "transit" else 0)
    elif 17 <= hour <= 19:
        mult *= 1.8 + (0.5 if zone["venue_type"] == "transit" else 0)
    elif 12 <= hour <= 14 and dow >= 5:
        mult *= 1.4   # weekend lunch
    elif 22 <= hour or hour <= 6:
        mult *= 0.15  # overnight quiet
    elif 10 <= hour <= 16:
        mult *= 1.0

    # Weather
    if weather_row and weather_row["wmo_code"] in WMO_RAIN + WMO_STORM:
        mult *= 0.45
    elif weather_row and weather_row["wmo_code"] in WMO_DRIZZLE:
        mult *= 0.65

    # Events
    if events_nearby:
        mult *= 1.5 + random.uniform(0, 0.5)

    # Venue affinity
    venue_base = {
        "transit":     1.3,
        "retail":      0.9,
        "park":        0.7 if dow < 5 else 1.1,
        "residential": 0.5,
        "university":  1.1 if dow < 5 else 0.4,
        "mixed":       1.0,
    }.get(zone["venue_type"], 1.0)

    return mult * venue_base

def nearby_event(dt, events):
    """True if any event overlaps this hour."""
    for e in events:
        s = datetime.fromisoformat(e["start_time"].replace("Z",""))
        end = datetime.fromisoformat(e["end_time"].replace("Z",""))
        if s <= dt <= end:
            return True
    return False

def make_snapshots(zones_list, hourly_wx, events_list):
    """
    Generate a snapshot every SNAPSHOT_FREQ minutes for each zone.
    Uses a simple state machine: each zone has a 'current fill'
    that drifts based on demand_multiplier.
    """
    snapshots = []
    # Build weather lookup: hour → row
    wx_lookup = {row["timestamp"][:13]: row for row in hourly_wx}

    sid = 0
    for zone in zones_list:
        cap   = zone["capacity"]
        fill  = random.randint(int(cap*0.2), int(cap*0.6))  # random start

        t = start_dt
        end = datetime.utcnow()
        while t <= end:
            wx_key = t.strftime("%Y-%m-%dT%H")
            wx     = wx_lookup.get(wx_key)
            ev     = nearby_event(t, events_list)

            mult   = demand_multiplier(t, zone, wx, ev)
            # Stochastic fill change: higher mult → more bikes used → lower fill
            delta  = int(np.random.normal(0, 1.5) - (mult - 1.0) * 2)
            fill   = max(0, min(cap, fill + delta))

            occ    = round(fill / cap * 100, 1)

            snapshots.append({
                "snapshot_id":    sid,
                "zone_id":        zone["zone_id"],
                "timestamp":      t.isoformat() + "Z",
                "bikes_available":fill,
                "capacity":       cap,
                "occupancy_pct":  occ,
                "weather_code":   wx["wmo_code"] if wx else 0,
                "temp_celsius":   wx["temp_celsius"] if wx else 12.0,
                "is_event_nearby":ev,
                "day_of_week":    t.weekday(),
                "hour_of_day":    t.hour,
            })
            sid += 1
            t += timedelta(minutes=SNAPSHOT_FREQ)

    return snapshots

print("Generating zone snapshots (this takes ~20s)...")
snapshots = make_snapshots(zones, hourly_weather, local_events)
print(f"✓ {len(snapshots):,} zone snapshots")


# ═══════════════════════════════════════════════════════════════
# 6. RIDES  (synthetic with redirect labels — key training signal)
# ═══════════════════════════════════════════════════════════════

REDIRECT_REASONS = ["zone_full","no_bikes_nearby","geofence_violation"]

def make_rides(zones_list, snap_df, n_rides=8000):
    """
    Generate rides from snapshot data.
    If destination zone occupancy was high at arrival → was_redirected=True.
    """
    rides = []
    snap_by_zone = snap_df.groupby("zone_id")

    for i in range(n_rides):
        start_zone = random.choice(zones_list)
        end_zone   = random.choice(zones_list)

        # Pick a random historical time
        days_back = random.randint(0, HISTORY_DAYS - 1)
        hour      = random.choices(
            range(24),
            weights=[0.01,0.01,0.01,0.01,0.01,0.02,
                     0.04,0.08,0.09,0.06,0.05,0.06,
                     0.07,0.06,0.05,0.05,0.06,0.08,
                     0.07,0.05,0.04,0.03,0.02,0.01]
        )[0]
        start_ts = (datetime.utcnow() - timedelta(days=days_back)).replace(
            hour=hour, minute=random.randint(0,59), second=0, microsecond=0
        )
        dist_m   = haversine_m(
            start_zone["lat"], start_zone["lon"],
            end_zone["lat"],   end_zone["lon"]
        )
        speed_ms = random.uniform(3.5, 5.5)
        dur_s    = max(60, int(dist_m / speed_ms) + random.randint(-30, 120))
        end_ts   = start_ts + timedelta(seconds=dur_s)

        # Check destination fill at arrival
        key = end_ts.strftime("%Y-%m-%dT%H")
        try:
            zone_snaps = snap_by_zone.get_group(end_zone["zone_id"])
            zone_snaps = zone_snaps[zone_snaps["timestamp"].str.startswith(key[:13])]
            occ = zone_snaps["occupancy_pct"].mean() if len(zone_snaps) else 50.0
        except Exception:
            occ = 50.0

        # Redirect if zone is >85% full
        redirected = occ > 85.0
        wasted     = round(random.uniform(2, 8), 1) if redirected else 0.0
        cost_pence = int((dur_s + wasted * 60) / 60 * 15)  # 15p/min synthetic

        rides.append({
            "ride_id":        f"GLW_R{str(i).zfill(6)}",
            "bike_id":        random.choice(bikes)["bike_id"],
            "rider_hash":     hashlib.sha256(
                                f"rider_{random.randint(0, 2000)}".encode()
                              ).hexdigest()[:16],
            "start_zone_id":  start_zone["zone_id"],
            "end_zone_id":    end_zone["zone_id"],
            "start_lat":      round(start_zone["lat"] + random.gauss(0, 0.0005), 6),
            "start_lon":      round(start_zone["lon"] + random.gauss(0, 0.0005), 6),
            "end_lat":        round(end_zone["lat"]   + random.gauss(0, 0.0005), 6),
            "end_lon":        round(end_zone["lon"]   + random.gauss(0, 0.0005), 6),
            "start_time":     start_ts.isoformat() + "Z",
            "end_time":       end_ts.isoformat() + "Z",
            "duration_secs":  dur_s,
            "distance_m":     int(dist_m),
            "cost_pence":     cost_pence,
            "was_redirected": redirected,
            "redirect_reason":random.choice(REDIRECT_REASONS) if redirected else None,
            "minutes_wasted": wasted,
        })

    return rides

print("Generating rides...")
snap_df = pd.DataFrame(snapshots)
rides   = make_rides(zones, snap_df, n_rides=8000)
redirected_count = sum(1 for r in rides if r["was_redirected"])
print(f"✓ {len(rides):,} rides | {redirected_count} redirects ({100*redirected_count//len(rides)}%)")


# ═══════════════════════════════════════════════════════════════
# 7. WRITE TO SQLITE (prototype DB)
# ═══════════════════════════════════════════════════════════════

print("\nWriting to SQLite...")
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS zones (
    zone_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL,
    radius_m REAL, zone_type TEXT, venue_type TEXT, capacity INTEGER,
    transit_score REAL, neighborhood TEXT, gbfs_region_id TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS bikes (
    bike_id TEXT PRIMARY KEY, vehicle_type_id TEXT, lat REAL, lon REAL,
    is_reserved INTEGER, is_disabled INTEGER, current_range_m INTEGER,
    last_reported TEXT, current_zone_id TEXT
);
CREATE TABLE IF NOT EXISTS weather (
    city_id TEXT, timestamp TEXT, temp_celsius REAL, precip_mm REAL,
    wind_kmh REAL, wmo_code INTEGER, is_daylight INTEGER,
    PRIMARY KEY (city_id, timestamp)
);
CREATE TABLE IF NOT EXISTS local_events (
    event_id TEXT PRIMARY KEY, city_id TEXT, event_date TEXT,
    start_time TEXT, end_time TEXT, venue_lat REAL, venue_lon REAL,
    expected_attendance INTEGER, category TEXT
);
CREATE TABLE IF NOT EXISTS zone_snapshots (
    snapshot_id INTEGER, zone_id TEXT, timestamp TEXT,
    bikes_available INTEGER, capacity INTEGER, occupancy_pct REAL,
    weather_code INTEGER, temp_celsius REAL, is_event_nearby INTEGER,
    day_of_week INTEGER, hour_of_day INTEGER,
    PRIMARY KEY (zone_id, timestamp)
);
CREATE TABLE IF NOT EXISTS rides (
    ride_id TEXT PRIMARY KEY, bike_id TEXT, rider_hash TEXT,
    start_zone_id TEXT, end_zone_id TEXT,
    start_lat REAL, start_lon REAL, end_lat REAL, end_lon REAL,
    start_time TEXT, end_time TEXT,
    duration_secs INTEGER, distance_m INTEGER, cost_pence INTEGER,
    was_redirected INTEGER, redirect_reason TEXT, minutes_wasted REAL
);
""")

def insert_many(cur, table, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    cur.executemany(sql, [
        [int(v) if isinstance(v, bool) else v for v in row.values()]
        for row in rows
    ])

insert_many(cur, "zones",        zones)
insert_many(cur, "bikes",        bikes)
insert_many(cur, "weather",      hourly_weather)
insert_many(cur, "local_events", local_events)
insert_many(cur, "zone_snapshots", snapshots)
insert_many(cur, "rides",        rides)
conn.commit()
conn.close()
print(f"✓ SQLite DB: {DB_PATH}")


# ═══════════════════════════════════════════════════════════════
# 8. EXPORT CSV  (DuckDB can read these directly)
# ═══════════════════════════════════════════════════════════════

def write_csv(name, rows):
    if not rows:
        return
    path = OUT_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.name} ({len(rows):,} rows)")

print("\nExporting CSVs for DuckDB...")
write_csv("zones",         zones)
write_csv("bikes",         bikes)
write_csv("weather",       hourly_weather)
write_csv("local_events",  local_events)
write_csv("zone_snapshots",snapshots)
write_csv("rides",         rides)


# ═══════════════════════════════════════════════════════════════
# 9. GBFS-FORMATTED JSON SNAPSHOTS  (drop-in mock API responses)
# ═══════════════════════════════════════════════════════════════

# free_bike_status.json
gbfs_bikes = {
    "last_updated": int(datetime.utcnow().timestamp()),
    "ttl": 10,
    "version": "3.0",
    "data": {
        "bikes": [
            {
                "bike_id":         b["bike_id"],
                "lat":             b["lat"],
                "lon":             b["lon"],
                "is_reserved":     b["is_reserved"],
                "is_disabled":     b["is_disabled"],
                "vehicle_type_id": b["vehicle_type_id"],
                "current_range_meters": b["current_range_m"],
                "last_reported":   b["last_reported"],
            }
            for b in bikes
        ]
    }
}

# geofencing_zones.json
gbfs_zones = {
    "last_updated": int(datetime.utcnow().timestamp()),
    "ttl": 3600,
    "version": "3.0",
    "data": {
        "geofencing_zones": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",  # simplified — real is Polygon
                        "coordinates": [z["lon"], z["lat"]]
                    },
                    "properties": {
                        "zone_id":   z["zone_id"],
                        "name":      z["name"],
                        "rules": [{
                            "vehicle_type_id": ["standard_v1","ebike_v2","cargo_v1"],
                            "ride_end_rules": {
                                "ride_through_allowed":   True,
                                "station_parking":        z["zone_type"] == "mandatory",
                                "maximum_speed_kph":      15
                            }
                        }]
                    }
                }
                for z in zones
            ]
        }
    }
}

with open(OUT_DIR / "free_bike_status.json", "w") as f:
    json.dump(gbfs_bikes, f, indent=2)
with open(OUT_DIR / "geofencing_zones.json", "w") as f:
    json.dump(gbfs_zones, f, indent=2)
print("\n✓ GBFS JSON mock feeds written")


# ═══════════════════════════════════════════════════════════════
# 10. SUMMARY + DUCKDB QUICKSTART
# ═══════════════════════════════════════════════════════════════

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          SYNTHETIC DATA GENERATION COMPLETE                  ║
╠══════════════════════════════════════════════════════════════╣
║  zones:          {len(zones):>6,}                                   ║
║  bikes:          {len(bikes):>6,}                                   ║
║  weather rows:   {len(hourly_weather):>6,}  (hourly, {HISTORY_DAYS}d + 48h fcast)    ║
║  local events:   {len(local_events):>6,}                                   ║
║  zone snapshots: {len(snapshots):>6,}  ({SNAPSHOT_FREQ}-min cadence, {HISTORY_DAYS}d)       ║
║  rides:          {len(rides):>6,}  ({redirected_count} redirected)              ║
╠══════════════════════════════════════════════════════════════╣
║  Output dir: {str(OUT_DIR):<48}║
╚══════════════════════════════════════════════════════════════╝

── DuckDB quickstart (run locally) ──────────────────────────────

  pip install duckdb
  python3 << 'EOF'
  import duckdb
  con = duckdb.connect("parking_buddy.duckdb")

  # Ingest CSVs
  for tbl in ["zones","bikes","weather","local_events","zone_snapshots","rides"]:
      con.execute(
          f"CREATE TABLE IF NOT EXISTS " + tbl + " AS "
          f"SELECT * FROM read_csv_auto('" + str(OUT_DIR) + "/" + "' + tbl + '.csv')"
      )

  # Quick validation
  con.sql("SELECT zone_id, AVG(occupancy_pct), MAX(occupancy_pct) FROM zone_snapshots GROUP BY zone_id LIMIT 5").show()
  con.sql("SELECT was_redirected, COUNT(*) FROM rides GROUP BY 1").show()
  EOF

── GKE swap checklist ───────────────────────────────────────────
  1. Replace zone_snapshots CSV with Airflow DAG polling real GBFS
  2. Replace geofencing_zones.json with operator GBFS endpoint
  3. Replace free_bike_status.json with real operator feed
  4. Rides table: MDS (Mobility Data Specification) API from operator
  5. weather: OpenMeteo free API (no key required)
  6. local_events: Predicthq or Ticketmaster API
  7. DuckDB → Cloud SQL (Postgres) for zone_snapshots writes
     DuckDB stays for analytics queries on GCS Parquet exports

── Real data column remap ───────────────────────────────────────
  synthetic zone_id          → GBFS feature.properties.zone_id
  synthetic bikes_available  → COUNT(vehicles WHERE zone spatial join)
  synthetic was_redirected   → MDS trip.event_type = 'trip_cancel' or
                               GPS trace deviation at destination
  synthetic occupancy_pct    → bikes_available / zone.capacity * 100
""")
