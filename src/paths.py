# All pipeline outputs live under output/ (relative to repo root)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DATA_RAW = OUTPUT / "raw"
DATA_PROCESSED = OUTPUT / "processed"
RESULTS = OUTPUT / "figures"

for p in (DATA_RAW, DATA_PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
