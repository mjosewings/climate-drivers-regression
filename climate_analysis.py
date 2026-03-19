# Run data collection, then pipeline, then visualization (from repo root: python climate_analysis.py)

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("src.data_collection", "Collecting data from NOAA, NASA, OWID…"),
    ("src.pipeline", "Running pipeline (features, models, importance)…"),
    ("src.visualize", "Generating plots…"),
]


def run() -> int:
    for module, msg in STEPS:
        print(f"\n--- {msg} ---")
        code = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(ROOT),
        )
        if code.returncode != 0:
            return code.returncode
    print("\nDone. Outputs: output/raw, output/processed, output/figures")
    return 0


if __name__ == "__main__":
    sys.exit(run())
