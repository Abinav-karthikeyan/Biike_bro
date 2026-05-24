"""
GBFS Feed Ingest — async polling stub for dockless bike-share zone occupancy.

PRD §9 — Deployment:
  - GBFS ingest: Airflow DAG, 5-min poll, upsert zone snapshots
  - Target operators: Lime, Voi, Tier (any operator with published GBFS feeds)

This module provides:
  1. GBFSClient — async HTTP client for GBFS auto-discovery + station_status
  2. GBFSIngestJob — scheduler-ready ingest job (APScheduler compatible)

TODO: Configure GBFS_FEED_URLS in .env and replace stub parsing with
      operator-specific field mapping if feeds diverge from GBFS v2.3 spec.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# GBFS v2.3 standard field names (https://github.com/MobilityData/gbfs)
GBFS_STATION_STATUS_FEED = "station_status"
GBFS_FREE_BIKE_STATUS_FEED = "free_bike_status"
GBFS_GEOFENCING_ZONES_FEED = "geofencing_zones"


# ─────────────────────────────────────────────────────────────────────────────
# GBFS HTTP Client
# ─────────────────────────────────────────────────────────────────────────────


class GBFSClient:
    """
    Async GBFS v2.x client.

    Usage:
        async with GBFSClient("https://data.lime.bike/.../gbfs.json") as client:
            statuses = await client.get_zone_statuses()
    """

    def __init__(self, gbfs_url: str, timeout_s: int = 10):
        self.gbfs_url = gbfs_url
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: Optional[aiohttp.ClientSession] = None
        self._feed_urls: Dict[str, str] = {}

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        await self._discover_feeds()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def _discover_feeds(self) -> None:
        """Parse GBFS auto-discovery manifest to find feed URLs."""
        try:
            async with self._session.get(self.gbfs_url) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            # GBFS v2: data.{lang}.feeds[]
            for lang_data in data.get("data", {}).values():
                for feed in lang_data.get("feeds", []):
                    self._feed_urls[feed["name"]] = feed["url"]

            logger.info(
                "GBFS feeds discovered",
                url=self.gbfs_url,
                feeds=list(self._feed_urls.keys()),
            )
        except Exception as exc:
            logger.error("GBFS discovery failed", url=self.gbfs_url, error=str(exc))

    async def get_zone_statuses(self) -> List[Dict[str, Any]]:
        """
        Fetch station_status feed and return normalised zone snapshot dicts.

        Returns list of:
            {
              "zone_id": str,
              "timestamp": datetime,
              "available_bikes": int,
              "docks_used": int,
              "occupancy_pct": float,
            }

        TODO: Add weather_code and local_events_mask enrichment here.
        """
        url = self._feed_urls.get(GBFS_STATION_STATUS_FEED)
        if not url:
            logger.warning("station_status feed not found in GBFS manifest")
            return []

        try:
            async with self._session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            last_updated = datetime.fromtimestamp(
                data.get("last_updated", 0), tz=timezone.utc
            )
            stations = data.get("data", {}).get("stations", [])

            snapshots = []
            for s in stations:
                num_bikes = s.get("num_bikes_available", 0)
                capacity = s.get("capacity") or s.get("num_docks_available", 0) + num_bikes
                occupancy = (capacity - num_bikes) / capacity if capacity > 0 else 0.0

                snapshots.append(
                    {
                        "zone_id": str(s["station_id"]),
                        "timestamp": last_updated,
                        "available_bikes": num_bikes,
                        "docks_used": capacity - num_bikes,
                        "occupancy_pct": round(occupancy, 4),
                        # TODO: enrich with weather_code, local_events_mask
                        "weather_code": None,
                        "local_events_mask": None,
                    }
                )
            return snapshots

        except Exception as exc:
            logger.error("Failed to fetch zone statuses", url=url, error=str(exc))
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Ingest Job (APScheduler / Airflow compatible)
# ─────────────────────────────────────────────────────────────────────────────


class GBFSIngestJob:
    """
    Polls all configured GBFS feeds and upserts zone snapshots into PostgreSQL.

    Schedule with APScheduler:
        scheduler.add_job(ingest_job.run, "interval", seconds=300)

    Or wrap in an Airflow PythonOperator / DAG task.
    """

    def __init__(self, feed_urls: List[str], db_session_factory=None):
        self.feed_urls = feed_urls
        self.db_session_factory = db_session_factory  # TODO: inject AsyncSession factory

    async def run(self) -> None:
        """Main poll cycle — fetch all feeds and persist snapshots."""
        if not self.feed_urls:
            logger.warning("No GBFS_FEED_URLS configured — skipping ingest")
            return

        all_snapshots = []
        tasks = [self._fetch_feed(url) for url in self.feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(self.feed_urls, results):
            if isinstance(result, Exception):
                logger.error("Feed fetch error", url=url, error=str(result))
            else:
                all_snapshots.extend(result)

        if all_snapshots:
            await self._upsert_snapshots(all_snapshots)
            logger.info("Ingest complete", snapshot_count=len(all_snapshots))

    async def _fetch_feed(self, url: str) -> List[Dict[str, Any]]:
        async with GBFSClient(url) as client:
            return await client.get_zone_statuses()

    async def _upsert_snapshots(self, snapshots: List[Dict[str, Any]]) -> None:
        """
        TODO: Implement DB upsert using SQLAlchemy async session.

        Example:
            async with self.db_session_factory() as session:
                for snap in snapshots:
                    obj = ZoneSnapshot(**snap)
                    await session.merge(obj)
                await session.commit()
        """
        logger.debug(
            "Upsert stub — snapshots not persisted",
            count=len(snapshots),
        )
