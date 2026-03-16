"""
climate_analysis.py – Run the full climate-drivers pipeline.

Executes in order:
  1. data_collection.py – fetch and merge data from NOAA GML, NASA GISS, NCEI, OWID
  2. pipeline.py       – feature engineering, models, feature importance
  3. visualize.py      – generate all required plots

Usage
-----
    python climate_analysis.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = ["data_collection.py", "pipeline.py", "visualize.py"]


def main() -> int:
    root = Path(__file__).resolve().parent
    for name in SCRIPTS:
        path = root / name
        if not path.exists():
            print(f"Missing {name}", file=sys.stderr)
            return 1
        print(f"\n--- Running {name} ---")
        code = subprocess.run([sys.executable, str(path)], cwd=str(root))
        if code.returncode != 0:
            return code.returncode
    print("\nDone. Check outputs/ for CSVs and plots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
