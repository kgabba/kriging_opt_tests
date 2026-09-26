#!/usr/bin/env python3
"""CLI: domain-wise anisotropic OK tuner with Optuna trial logs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# sibling MVP for variography
MVP = ROOT.parent / "aniso_idw_mvp"
if MVP.exists() and str(MVP) not in sys.path:
    sys.path.insert(0, str(MVP))

from aniso_ok_tuner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
