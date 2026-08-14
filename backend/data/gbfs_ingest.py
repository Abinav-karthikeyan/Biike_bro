"""
GBFS Feed Ingest — async polling for dockless bike-share zone occupancy.

Proof-of-concept feed: Nextbike Banja Luka (nextbike_bj).
Glasgow OVO Bikes / Nextbike Glasgow does not expose a public GBFS endpoint
(absent from the MobilityData systems.csv registry as of 2026-08-14).
The Nextbike feed is used to demonstrate the full ingest pipeline; real
Glasgow data would replace it once an operator publishes a feed.

Set GBFS_INGEST_ENABLED=true and provide GBFS_FEED_URLS in .env to activate.

Two classes:
  GBFSClient        — async HTTP client for GBFS auto-discovery + station feeds
  GBFSIngestJob     — polling loop; tracks success/error state for /health/ingest
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from backend.data.loaders import normalize_zone_snapshot

logger = structlog.get_logger(__name__)

GBFS_STATION_STATUS_FEED = "station_status"
GBFS_STATION_INFORMATION_FEED = "station_information"


# ─────────────────────────────────────────────────────────────────────────────
# GBFS HTTP Client
# ─────────────────────────────────────────────────────────────────────────────


class GBFSClient:
    """
    Async GBFS v2.x client.

    Usage:
        async with GBFSClient("https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_bj/gbfs.json") as c:
            zone_info = await c.get_station_information()
            statuses  = await c.get_zone_statuses()
    """

    def __init__(self, gbfs_url: str, timeout_s: int = 10):
        self.gbfs_url = gbfs_url
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: Optional[aiohttp.ClientSession] = None
        self._feed_urls: Dict[str, str] = {}
        self._discovery_error: Optional[str] = None

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
            for lang_data in data.get("data", {}).values():
                for feed in lang_data.get("feeds", []):
                    self._feed_urls[feed["name"]] = feed["url"]
            logger.info(
                "GBFS feeds discovered: %s feeds from %s",
                len(self._feed_urls), self.gbfs_url,
            )
        except Exception as exc:
            self._discovery_error = str(exc)
            logger.error("GBFS discovery failed for %s: %s", self.gbfs_url, exc)

    async def get_station_information(self) -> List[Dict[str, Any]]:
        """
        Fetch station_information feed.

        Returns list of:
            {"station_id": str, "name": str, "lat": float, "lon": float,
             "capacity": int|None, "region_id": str|None}
        """
        url = self._feed_urls.get(GBFS_STATION_INFORMATION_FEED)
        if not url:
            return []
        try:
            async with self._session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            return data.get("data", {}).get("stations", [])
        except Exception as exc:
            logger.error("Failed to fetch station_information from %s: %s", url, exc)
            return []

    async def get_zone_statuses(self) -> tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Fetch station_status feed.

        Returns (snapshots, last_updated_epoch) where snapshots are normalised
        to the internal ZoneSnapshot field names via normalize_zone_snapshot().
        last_updated_epoch is None if the feed is unreachable or malformed.
        """
        url = self._feed_urls.get(GBFS_STATION_STATUS_FEED)
        if not url:
            logger.warning("station_status feed not found in GBFS manifest for %s", self.gbfs_url)
            return [], None

        try:
            async with self._session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            last_updated_epoch: Optional[int] = data.get("last_updated")
            if last_updated_epoch:
                last_updated = datetime.fromtimestamp(last_updated_epoch, tz=timezone.utc)
            else:
                last_updated = datetime.now(timezone.utc)
                last_updated_epoch = None

            stations = data.get("data", {}).get("stations", [])
            snapshots = [normalize_zone_snapshot(s, last_updated) for s in stations]
            return snapshots, last_updated_epoch

        except Exception as exc:
            logger.error("Failed to fetch station_status from %s: %s", url, exc)
            return [], None


# ─────────────────────────────────────────────────────────────────────────────
# Ingest Job
# ─────────────────────────────────────────────────────────────────────────────


class GBFSIngestJob:
    """
    Polls all configured GBFS feeds and persists zone snapshots.

    State is stored on this object so /health/ingest can read it:
      last_success_at   — datetime of last successful poll
      last_error        — string description of most recent error (None = clean)
      error_count       — cumulative error count since startup
      snapshots_total   — cumulative snapshots written since startup
    """

    def __init__(self, feed_urls: List[str], db_store=None):
        self.feed_urls = feed_urls
        self.db_store = db_store
        self.last_success_at: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.error_count: int = 0
        self.snapshots_total: int = 0
        # Per-feed last_updated epoch: skip write when the feed hasn't advanced
        self._feed_last_updated: Dict[str, Optional[int]] = {u: None for u in feed_urls}
        self._running = False

    async def start_polling(self, interval_seconds: int = 300) -> None:
        """Continuous polling loop.  Call as an asyncio task from lifespan."""
        self._running = True
        logger.info("GBFS ingest poller started — %d feed(s), interval=%ds",
                    len(self.feed_urls), interval_seconds)
        while self._running:
            await self.run_once()
            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        """Signal the polling loop to exit on its next sleep boundary."""
        self._running = False

    async def run_once(self) -> None:
        """Single poll cycle — fetch all feeds and persist new snapshots."""
        if not self.feed_urls:
            logger.warning("No GBFS_FEED_URLS configured — skipping poll")
            return

        tasks = [self._poll_feed(url) for url in self.feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        written = 0
        for url, result in zip(self.feed_urls, results):
            if isinstance(result, Exception):
                self.error_count += 1
                self.last_error = f"{url}: {result}"
                logger.error("Feed poll error for %s: %s", url, result)
            else:
                written += result

        if written > 0:
            self.snapshots_total += written
            self.last_success_at = datetime.now(timezone.utc)
            logger.info("GBFS ingest cycle complete — %d new snapshots written", written)
        elif not any(isinstance(r, Exception) for r in results):
            # All feeds succeeded but nothing was new (last_updated didn't advance)
            self.last_success_at = datetime.now(timezone.utc)
            logger.info("GBFS ingest cycle complete — feeds up-to-date, no new snapshots")

    async def _poll_feed(self, url: str) -> int:
        """Fetch one feed; return number of snapshot rows written."""
        async with GBFSClient(url) as client:
            if client._discovery_error:
                raise RuntimeError(f"Discovery failed: {client._discovery_error}")

            # Upsert zone reference data first (FK parent before child)
            station_info = await client.get_station_information()
            if station_info and self.db_store:
                await asyncio.to_thread(self._upsert_stations_sync, station_info)

            snapshots, last_updated_epoch = await client.get_zone_statuses()

        if not snapshots:
            return 0

        # Skip write if this feed's last_updated hasn't advanced (prevents duplicates)
        prev_epoch = self._feed_last_updated.get(url)
        if last_updated_epoch is not None and last_updated_epoch == prev_epoch:
            logger.info("Feed %s unchanged (last_updated=%s) — skipping write", url, last_updated_epoch)
            return 0

        self._feed_last_updated[url] = last_updated_epoch

        if self.db_store:
            written = await asyncio.to_thread(self.db_store.insert_zone_snapshots, snapshots)
        else:
            written = len(snapshots)
            logger.debug("No db_store — %d snapshots not persisted (dry-run)", written)

        return written

    def _upsert_stations_sync(self, stations: List[Dict[str, Any]]) -> None:
        """Sync helper: upsert zone rows for real GBFS stations. Called via to_thread."""
        for s in stations:
            try:
                self.db_store.upsert_real_zone(
                    zone_id=str(s["station_id"]),
                    name=s.get("name", str(s["station_id"])),
                    lat=float(s["lat"]),
                    lon=float(s["lon"]),
                    capacity=s.get("capacity"),
                    gbfs_region_id=str(s["region_id"]) if s.get("region_id") else None,
                )
            except Exception as exc:
                logger.error("Failed to upsert zone %s: %s", s.get("station_id"), exc)

    def status_dict(self) -> Dict[str, Any]:
        """Return a serialisable summary for /health/ingest."""
        return {
            "status": "ok" if self.last_success_at else "never_polled",
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "snapshots_total": self.snapshots_total,
            "feed_count": len(self.feed_urls),
        }
