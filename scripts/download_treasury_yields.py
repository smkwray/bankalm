#!/usr/bin/env python3
"""CLI wrapper for the Treasury yield curve downloader."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bankfragility.downloads.treasury_yields import main

if __name__ == "__main__":
    main()
