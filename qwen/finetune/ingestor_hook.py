"""
Ingestor Hook — pipes live GBFS/ride data into the fine-tuning pipeline.
═══════════════════════════════════════════════════════════════════════════

When real GBFS data starts flowing (via DuckDB migration), this script
hooks into the ingestor and appends new conversation examples to the
training JSONL in real time — so the model continuously improves from
real parking outcomes.

Usage:
  python qwen/finetune/ingestor_hook.py                  # one-shot export
  python qwen/finetune/ingestor_hook.py --watch 300      # poll every 5min
  python qwen/finetune/ingestor_hook.py --since 2026-06-01  # delta export

Hook contract
─────────────
Ingestors call: IngestorHook.on_new_snapshot(zone_id, occupancy_pct, timestamp, weather)
Log outcomes:   IngestorHook.on_ride_outcome(ride_id, zone_id, was_redirected, minutes_wasted)

These are auto-converted to training examples and appended to the JSONL.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.data.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

TRAIN_DATA = Path(__file__).resolve().parent.parent.parent / "data" / "finetune" / "train.jsonl"
SEED_DIR   = Path(__file__).resolve().parent.parent.parent / "synthetic_seed"

SYSTEM_PROMPT = (
    "You are Parking Buddy, an AI assistant embedded in a dockless bike-share app in Glasgow. "
    "Help riders find available parking zones. Be concise — 2-3 sentences maximum."
)


class IngestorHook:
    """
    Bridges the GBFS ingestor → fine-tuning data pipeline.

    Drop this into backend/data/gbfs_ingest.py once real feeds are live:

        hook = IngestorHook()
        # After each 5-min GBFS poll:
        for snap in new_snapshots:
            hook.on_new_snapshot(snap["zone_id"], snap["occupancy_pct"], ...)
    """

    def __init__(self, out_path: Path = TRAIN_DATA):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_new_snapshot(
        self,
        zone_id: str,
        zone_name: str,
        occupancy_pct: float,
        timestamp: datetime,
        weather_code: int = 0,
        is_event_nearby: bool = False,
    ) -> None:
        """
        Called by the GBFS ingestor after each zone snapshot is written.
        Generates a Q&A pair and appends to training JSONL.
        """
        fill_word = "full" if occupancy_pct > 80 else "half-full" if occupancy_pct > 40 else "available"
        hour = timestamp.hour
        mins = 30  # default lookahead

        user_msg = f"Is {zone_name} available right now at {hour:02d}:00?"
        assistant_msg = (
            f"{zone_name} is currently {round(occupancy_pct)}% occupied — "
            f"it's {fill_word}."
        )
        if occupancy_pct > 80:
            assistant_msg += " I recommend finding an alternative zone nearby."
        elif is_event_nearby:
            assistant_msg += " There's an event nearby which may increase demand soon."

        self._append_example(user_msg, assistant_msg, {
            "zone_id": zone_id,
            "occupancy_pct": occupancy_pct,
            "source": "live_gbfs",
            "timestamp": timestamp.isoformat(),
        })

    def on_ride_outcome(
        self,
        ride_id: str,
        zone_id: str,
        zone_name: str,
        was_redirected: bool,
        minutes_wasted: float,
        predicted_fill: Optional[float] = None,
    ) -> None:
        """
        Called after a ride completes with the real outcome.
        Creates a reinforcement signal: was prediction correct?
        """
        if predicted_fill is None:
            return  # can't build a useful training pair without prediction context

        pred_pct = round(predicted_fill * 100)
        user_msg = f"How did parking at {zone_name} actually go for the last rider?"

        if was_redirected:
            assistant_msg = (
                f"The rider was redirected from {zone_name} — "
                f"the zone was too full to park (we predicted {pred_pct}% fill). "
                f"This cost them {minutes_wasted:.1f} extra minutes. "
                "Next time I'll suggest alternatives earlier."
            )
        else:
            assistant_msg = (
                f"The rider successfully parked at {zone_name}. "
                f"Our {pred_pct}% fill prediction was accurate — "
                "they found a spot without issues."
            )

        self._append_example(user_msg, assistant_msg, {
            "zone_id": zone_id,
            "was_redirected": was_redirected,
            "minutes_wasted": minutes_wasted,
            "predicted_fill": predicted_fill,
            "source": "ride_outcome",
        })

    def _append_example(self, user_msg: str, assistant_msg: str, metadata: dict) -> None:
        example = {
            "conversations": [
                {"from": "system",    "value": SYSTEM_PROMPT},
                {"from": "human",     "value": user_msg},
                {"from": "assistant", "value": assistant_msg},
            ],
            "metadata": metadata,
        }
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # ── Bulk export from DuckDB (for back-filling) ────────────────────────

    def export_from_duckdb(self, since: Optional[str] = None) -> int:
        """
        Export historical rides + snapshots from DuckDB into training JSONL.
        Use `since` (ISO date string) for incremental delta exports.
        """
        db = DuckDBStore(seed_dir=str(SEED_DIR))
        rides = db.get_rides(limit=8000)
        zones_map = {z["zone_id"]: z for z in db.get_zones()}
        count = 0

        for ride in rides:
            zone = zones_map.get(ride.get("end_zone_id", ""), {})
            if not zone:
                continue
            self.on_ride_outcome(
                ride_id=ride.get("ride_id", ""),
                zone_id=zone.get("zone_id", ""),
                zone_name=zone.get("name", zone.get("zone_id", "")),
                was_redirected=bool(ride.get("was_redirected")),
                minutes_wasted=float(ride.get("minutes_wasted", 0)),
                predicted_fill=None,  # no retroactive predictions
            )
            count += 1

        return count


def one_shot_export():
    hook = IngestorHook()
    print("Exporting historical rides from DuckDB to training JSONL...")
    count = hook.export_from_duckdb()
    print(f"✓ {count} ride outcome examples appended to {TRAIN_DATA}")


def watch_mode(interval_seconds: int):
    hook = IngestorHook()
    print(f"Watch mode: polling DuckDB every {interval_seconds}s for new data...")
    seen = set()
    while True:
        db = DuckDBStore(seed_dir=str(SEED_DIR))
        rides = db.get_rides(limit=8000)
        zones_map = {z["zone_id"]: z for z in db.get_zones()}
        new = 0
        for ride in rides:
            rid = ride.get("ride_id", "")
            if rid in seen:
                continue
            seen.add(rid)
            zone = zones_map.get(ride.get("end_zone_id", ""), {})
            if zone:
                hook.on_ride_outcome(
                    ride_id=rid,
                    zone_id=zone.get("zone_id", ""),
                    zone_name=zone.get("name", ""),
                    was_redirected=bool(ride.get("was_redirected")),
                    minutes_wasted=float(ride.get("minutes_wasted", 0)),
                    predicted_fill=None,
                )
                new += 1
        if new:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Appended {new} new ride examples")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Ingestor hook: pipe ride data to fine-tuning JSONL")
    parser.add_argument("--watch", type=int, default=0,
                        help="Poll interval in seconds (0 = one-shot)")
    parser.add_argument("--since", default=None,
                        help="Only export rides after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.watch > 0:
        watch_mode(args.watch)
    else:
        one_shot_export()


if __name__ == "__main__":
    main()
