#!/usr/bin/env python
"""Run this file in PyCharm to open the simulation GUI (standard-library Tkinter)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.simulation.gui import main


if __name__ == "__main__":
    raise SystemExit(main())
