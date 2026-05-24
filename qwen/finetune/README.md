# Qwen Fine-tuning Scripts

Scripts to adapt Qwen2.5 for Bike Parking Buddy.

## Workflow

```
1. Build training data
   python qwen/finetune/build_training_data.py [--limit 3000]
   → data/finetune/train.jsonl

2. Fine-tune (GPU required — 8GB+ VRAM)
   pip install unsloth trl datasets accelerate bitsandbytes
   python qwen/finetune/finetune_unsloth.py [--max-steps 200]
   → models/parking_buddy_qwen/lora_adapter/

3. Export to GGUF + create Ollama model
   python qwen/finetune/finetune_unsloth.py --export-gguf
   ollama create parking-buddy -f Modelfile
   # Then set OLLAMA_MODEL_NAME=parking-buddy in .env

4. Hook live ingestor (once real GBFS is live)
   python qwen/finetune/ingestor_hook.py --watch 300
```

## Ingestor Integration

```python
from qwen.finetune.ingestor_hook import IngestorHook
hook = IngestorHook()

# Call after each GBFS snapshot poll:
hook.on_new_snapshot(zone_id, zone_name, occupancy_pct, timestamp, weather_code)

# Call after each ride completes:
hook.on_ride_outcome(ride_id, zone_id, zone_name, was_redirected, minutes_wasted, predicted_fill)
```
