# Paths used by data_collection, pipeline, and visualize (relative to repo root)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

for p in (DATA_RAW, DATA_PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
