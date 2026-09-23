#!/usr/bin/env python3
"""CLI entry: nested OK tuner v2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MVP = ROOT.parent / "aniso_idw_mvp"
if MVP.exists() and str(MVP) not in sys.path:
    sys.path.insert(0, str(MVP))

from aniso_ok_tuner_v2.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
