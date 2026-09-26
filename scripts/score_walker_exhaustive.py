#!/usr/bin/env python3
"""Score Walker exhaustive grid with best OK params from a tune out-dir."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aniso_ok_tuner.kriging import (  # noqa: E402
    ordinary_kriging_elliptical,
    ordinary_kriging_isotropic,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--exhaustive", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--domain", type=str, default="Walker")
    p.add_argument(
        "--exclude-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude train sample X,Y from exhaustive RMSE (default: True)",
    )
    args = p.parse_args()

    payload = json.loads((args.results_dir / args.domain / "best_params.json").read_text())
    opt = payload["optimization"]
    fixed = opt["fixed"]
    theta = opt["best_theta"]
    sill_total = float(fixed["sill_total"])
    nug = float(np.clip(float(theta["nugget"]), 0.0, 0.95 * sill_total))
    sil = max(sill_total - nug, 1e-12)

    train = pd.read_csv(args.train)
    if "Grade" in train.columns:
        xy = train[["X", "Y"]].to_numpy(float)
        z = train["Grade"].to_numpy(float)
    else:
        xy = train[["X", "Y"]].to_numpy(float)
        z = train["V"].to_numpy(float)

    ex = pd.read_csv(args.exhaustive)
    ex_xy = ex[["X", "Y"]].to_numpy(float)
    ex_true = ex["V"].to_numpy(float)

    sample_keys = set(zip(np.round(xy[:, 0], 6), np.round(xy[:, 1], 6)))
    not_sample = np.array(
        [(round(float(x), 6), round(float(y), 6)) not in sample_keys for x, y in ex_xy]
    )

    isotropic = bool(fixed.get("isotropic", opt.get("isotropic", False)))
    print(
        f"Predict exhaustive n={len(ex)} iso={isotropic} "
        f"exclude_sample={args.exclude_sample} "
        f"theta={theta} alpha={fixed.get('alpha_deg')} "
        f"a={fixed.get('a_major')}/{fixed.get('a_minor')}",
        flush=True,
    )

    if isotropic:
        pred = ordinary_kriging_isotropic(
            xy, z, ex_xy,
            range_=float(fixed["a_major"]),
            nugget=nug, sill=sil,
            range_scale=float(theta["range_scale"]),
            radius=float(theta.get("R", theta["R_major"])),
            n_max=int(theta["N_max"]),
        )
    else:
        pred = ordinary_kriging_elliptical(
            xy, z, ex_xy,
            alpha_deg=float(fixed["alpha_deg"]),
            a_major=float(fixed["a_major"]),
            a_minor=float(fixed["a_minor"]),
            nugget=nug, sill=sil,
            range_scale=float(theta["range_scale"]),
            r_major=float(theta["R_major"]),
            r_minor=float(theta["R_minor"]),
            n_max=int(theta["N_max"]),
        )

    mask = np.isfinite(pred)
    if args.exclude_sample:
        mask = mask & not_sample
    rmse = float(np.sqrt(np.mean((ex_true[mask] - pred[mask]) ** 2)))
    mae = float(np.mean(np.abs(ex_true[mask] - pred[mask])))
    out = {
        "domain": args.domain,
        "n_exhaustive": int(len(ex)),
        "n_sample": int(len(xy)),
        "n_overlap": int((~not_sample).sum()),
        "exclude_sample": bool(args.exclude_sample),
        "n_scored": int(mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "best_theta": theta,
        "fixed": fixed,
        "cv_best_rmse": opt.get("best_delete_d_rmse"),
        "cv_details": opt.get("cv_details"),
    }
    out_path = args.results_dir / "exhaustive_score.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  n_scored={mask.sum()}")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
