"""
Training data builder for Qwen2.5 fine-tuning on Bike Parking Buddy.
═══════════════════════════════════════════════════════════════════════

Converts zone_snapshots + rides history from DuckDB into a JSONL
dataset suitable for instruction fine-tuning (supervised SFT).

Output format  — ShareGPT / Unsloth compatible:
  { "conversations": [
      {"from": "system",    "value": "<system_prompt>"},
      {"from": "human",     "value": "<user turn>"},
      {"from": "assistant", "value": "<ideal response with or without tool call>"}
  ]}

Usage:
  python qwen/finetune/build_training_data.py
  python qwen/finetune/build_training_data.py --out data/finetune/train.jsonl --limit 2000
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.data.duckdb_store import DuckDBStore

SEED_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic_seed"
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "data" / "finetune" / "train.jsonl"

SYSTEM_PROMPT = (
    "You are Parking Buddy, an AI assistant embedded in a dockless bike-share app in Glasgow. "
    "You help riders find available parking zones in real time. "
    "Use the tools available to you to fetch zone forecasts and alternatives. "
    "Be concise — riders are on the move. Reply in 2-3 sentences maximum."
)

# ── Templates for synthetic conversation pairs ────────────────────────────────

QUERY_TEMPLATES = [
    "Will {zone_name} be full when I get there in {mins} minutes?",
    "Is there space at {zone_name} right now?",
    "I'm heading to {zone_name} — should I worry about parking?",
    "What's the parking situation at {zone_name} around {hour}:00?",
    "Is {zone_name} likely to be crowded during {period}?",
    "Find me an alternative to {zone_name} — it looks busy.",
    "Quick question: {zone_name} at {hour}pm — good idea?",
    "Predict parking at {zone_name} in {mins} min please.",
]

PERIODS = {
    (7, 9): "morning rush",
    (12, 14): "lunchtime",
    (17, 19): "evening rush",
    (22, 6): "overnight",
}

def period_for_hour(hour: int) -> str:
    for (lo, hi), label in PERIODS.items():
        if lo <= hour <= hi or (lo > hi and (hour >= lo or hour <= hi)):
            return label
    return "off-peak"


def ideal_response(zone: dict, fill_prob: float, mins: int, alternatives: list) -> str:
    """Generate the ground-truth assistant response."""
    pct = round(fill_prob * 100)
    name = zone.get("name", zone["zone_id"])
    vtype = zone.get("venue_type", "zone")

    if pct > 80:
        action = (
            f"{name} is predicted to be {pct}% full in {mins} minutes — I'd skip it. "
        )
        if alternatives:
            alt = alternatives[0]
            action += f"Try {alt['zone_id']} instead, which looks {round(alt['fill_probability']*100)}% full."
        else:
            action += "Look for nearby alternatives on the map."
    elif pct > 50:
        action = (
            f"{name} ({vtype}) should be about {pct}% full in {mins} minutes — "
            "it might be tight but worth a try. Head there now before it fills up."
        )
    else:
        action = (
            f"Great news — {name} is predicted to be only {pct}% full in {mins} minutes. "
            "You should have no trouble parking there."
        )
    return action


def build_dataset(limit: int = 3000) -> list:
    """Pull data from DuckDB and generate conversation pairs."""
    db = DuckDBStore(seed_dir=str(SEED_DIR))
    zones = db.get_zones()
    rides = db.get_rides(limit=limit)

    random.seed(42)
    examples = []

    for zone in zones:
        # Generate several scenarios per zone
        for _ in range(limit // len(zones) + 1):
            mins = random.choice([10, 15, 20, 30, 45, 60])
            hour = random.randint(0, 23)
            period = period_for_hour(hour)

            # Synthetic fill probability based on zone features
            base = 0.4
            if hour in range(7, 10):
                base += 0.35 if zone.get("venue_type") == "transit" else 0.2
            elif hour in range(17, 20):
                base += 0.4 if zone.get("venue_type") == "transit" else 0.25
            elif hour in range(22, 24) or hour in range(0, 6):
                base += 0.5 if zone.get("venue_type") == "residential" else 0.1
            fill_prob = max(0.0, min(1.0, base + random.gauss(0, 0.1)))

            # Sample some alternatives (lower fill zones)
            other_zones = [z for z in zones if z["zone_id"] != zone["zone_id"]]
            alt_sample = random.sample(other_zones, min(3, len(other_zones)))
            alternatives = [
                {"zone_id": z["zone_id"], "fill_probability": max(0.0, fill_prob - random.uniform(0.1, 0.4))}
                for z in alt_sample
                if fill_prob > 0.65
            ]

            # Pick a query template
            template = random.choice(QUERY_TEMPLATES)
            user_msg = template.format(
                zone_name=zone.get("name", zone["zone_id"]),
                mins=mins,
                hour=hour,
                period=period,
            )

            response = ideal_response(zone, fill_prob, mins, alternatives)

            examples.append({
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": user_msg},
                    {"from": "assistant", "value": response},
                ],
                "metadata": {
                    "zone_id": zone["zone_id"],
                    "fill_probability": round(fill_prob, 3),
                    "lookahead_mins": mins,
                    "hour_of_day": hour,
                    "venue_type": zone.get("venue_type"),
                },
            })

    random.shuffle(examples)
    return examples[:limit]


def main():
    parser = argparse.ArgumentParser(description="Build SFT training data from synthetic DuckDB store")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=3000, help="Max examples to generate")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building {args.limit} training examples...")
    dataset = build_dataset(limit=args.limit)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✓ {len(dataset)} examples written to {out_path}")
    print(f"  To launch fine-tuning: python qwen/finetune/finetune_unsloth.py")


if __name__ == "__main__":
    main()
