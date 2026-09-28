#!/usr/bin/env python3
"""Small command-line entrance for the handoff.

Keeping this file tiny is intentional: the scientific logic lives in importable
modules, where it can be tested and reused in a notebook or future paper.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lbf_handoff.runner import main


if __name__ == "__main__":
    main(ROOT)

