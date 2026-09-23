"""CLI for nested OK tuner v2."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nested Optuna OK tuner v2 (outer×inner)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(args.data, args.out_dir, args.config)
    print(f"Done. See {args.out_dir / 'summary.json'}", flush=True)
    return 0
