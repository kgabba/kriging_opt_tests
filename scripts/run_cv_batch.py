#!/usr/bin/env python3
"""Tune iso OK for 4 CV methods (800 trials each) and write named predicts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("loo", "kfold5", "spatial_block", "buffer", "buffered_delete_d")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch iso OK CV tune + predict")
    p.add_argument("--data", type=Path, default=ROOT.parent / "data" / "data_algo.csv")
    p.add_argument("--val", type=Path, default=ROOT.parent / "data" / "jura_val_loc.csv")
    p.add_argument("--pred-dir", type=Path, default=ROOT.parent / "data")
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
    )
    p.add_argument("--python", type=Path, default=ROOT.parent / ".venv" / "bin" / "python")
    args = p.parse_args(argv)

    py = str(args.python)
    tune_py = str(ROOT / "tune.py")
    pred_py = str(ROOT / "scripts" / "predict_val.py")
    failed: list[str] = []

    for method in args.methods:
        cfg = ROOT / "configs" / f"iso_cv_{method}.yaml"
        out_dir = ROOT / "results" / f"jura_iso_800_{method}"
        log_path = ROOT / "results" / f"jura_iso_800_{method}_run.log"
        pred_out = args.pred_dir / f"jura_val_predict_cv_{method}.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n======== CV={method} tune → {out_dir} ========", flush=True)

        with open(log_path, "w", encoding="utf-8") as log:
            tune = subprocess.run(
                [py, tune_py, "--data", str(args.data), "--config", str(cfg), "--out-dir", str(out_dir)],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if tune.returncode != 0:
            print(f"[FAIL] tune {method} rc={tune.returncode} see {log_path}", flush=True)
            failed.append(method)
            continue
        print(f"[OK] tune {method}; predicting → {pred_out}", flush=True)

        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n--- predict ---\n")
            pred = subprocess.run(
                [
                    py, pred_py,
                    "--train", str(args.data),
                    "--val", str(args.val),
                    "--results-dir", str(out_dir),
                    "--out", str(pred_out),
                ],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if pred.returncode != 0:
            print(f"[FAIL] predict {method} rc={pred.returncode}", flush=True)
            failed.append(method)
            continue
        print(f"[OK] {pred_out}", flush=True)

    if failed:
        print(f"Failed methods: {failed}", flush=True)
        return 1
    print("All CV methods done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
