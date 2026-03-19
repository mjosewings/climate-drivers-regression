# Convenience runner that executes the full pipeline in order:
#   1) src.data_collection (creates output/raw/raw_merged.csv)
#   2) src.pipeline      (creates output/processed/*.csv)
#   3) src.visualize     (creates output/figures/*.png)
#
# This file is intended for coursework submission so you can run one command
# and get the full set of artifacts.

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("src.data_collection", "Collecting data from NOAA, NASA, OWID…"),
    ("src.pipeline", "Running pipeline (features, models, importance)…"),
    ("src.visualize", "Generating plots…"),
]

if __name__ == "__main__":
    exit_code = 0
    for module, msg in STEPS:
        print(f"\n--- {msg} ---")
        code = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(ROOT),
        )
        if code.returncode != 0:
            exit_code = code.returncode
            break
    if exit_code == 0:
        print("\nDone. Outputs: output/raw, output/processed, output/figures")
    sys.exit(exit_code)
