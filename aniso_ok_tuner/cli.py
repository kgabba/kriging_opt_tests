"""CLI entry for aniso_ok_tuner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Tune anisotropic Ordinary Kriging per Domain "
            "(MOI priors + Optuna delete-d; trials.csv like aniso_idw_tuner)."
        )
    )
    p.add_argument("--data", type=Path, required=True, help="CSV with X,Y,Grade[,Domain]")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--trials", type=int, default=None, help="Override optuna.n_trials")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # optional trials override via rewriting config path is awkward; pass through env-less:
    # load happens inside pipeline — monkey-patch by writing temp not needed if we inject
    cfg_path = args.config
    if args.trials is not None:
        # light override: pipeline reads yaml; we set via monkey by editing after load
        # simplest: set environment-like by wrapping run_pipeline
        from .pipeline import load_config

        cfg = load_config(cfg_path)
        cfg.setdefault("optuna", {})
        cfg["optuna"]["n_trials"] = int(args.trials)
        # write ephemeral config next to out
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ephemeral = args.out_dir / "_run_config.yaml"
        import yaml

        ephemeral.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        cfg_path = ephemeral

    run_pipeline(args.data, args.out_dir, cfg_path)
    print(f"Done. See {args.out_dir / 'summary.json'}", flush=True)
    return 0
