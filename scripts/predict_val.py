#!/usr/bin/env python3
"""Predict jura_val_loc with best OK params from a tune out-dir → V1,V2 CSV."""

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
MVP = ROOT.parent / "aniso_idw_mvp"
if MVP.exists() and str(MVP) not in sys.path:
    sys.path.insert(0, str(MVP))

from aniso_idw_mvp.io import load_points  # noqa: E402
from aniso_ok_tuner.kriging import (  # noqa: E402
    ordinary_kriging_elliptical,
    ordinary_kriging_isotropic,
)


def _load_val(path: Path) -> pd.DataFrame:
    val = pd.read_csv(path)
    val = val.rename(columns={c: str(c).strip('"') for c in val.columns})
    if "seq(1:100)" in val.columns:
        val = val.rename(columns={"seq(1:100)": "id"})
    elif val.columns[0] not in ("id", "V1"):
        val = val.rename(columns={val.columns[0]: "id"})
    return val


def _domain_payload(results_dir: Path, domain: str) -> dict:
    p = results_dir / domain / "best_params.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def predict_domain(
    train_df: pd.DataFrame,
    pred_xy: np.ndarray,
    payload: dict,
) -> np.ndarray:
    opt = payload.get("optimization") or {}
    if opt.get("status") == "SKIPPED":
        z = train_df["Grade"].to_numpy(float)
        return np.full(pred_xy.shape[0], float(np.mean(z)))

    fixed = opt.get("fixed") or payload.get("fixed_model") or {}
    theta = opt.get("best_theta") or {}
    if not theta:
        raise ValueError(f"missing best_theta in domain={payload.get('domain')}")

    sill_total = float(fixed["sill_total"])
    nug = float(np.clip(float(theta["nugget"]), 0.0, 0.95 * sill_total))
    sil = max(sill_total - nug, 1e-12)
    xy = train_df[["X", "Y"]].to_numpy(float)
    z = train_df["Grade"].to_numpy(float)
    isotropic = bool(fixed.get("isotropic", opt.get("isotropic", False)))

    if isotropic:
        pred = ordinary_kriging_isotropic(
            xy, z, pred_xy,
            range_=float(fixed["a_major"]),
            nugget=nug,
            sill=sil,
            range_scale=float(theta["range_scale"]),
            radius=float(theta.get("R", theta["R_major"])),
            n_max=int(theta["N_max"]),
        )
    else:
        pred = ordinary_kriging_elliptical(
            xy, z, pred_xy,
            alpha_deg=float(fixed["alpha_deg"]),
            a_major=float(fixed["a_major"]),
            a_minor=float(fixed["a_minor"]),
            nugget=nug,
            sill=sil,
            range_scale=float(theta["range_scale"]),
            r_major=float(theta["R_major"]),
            r_minor=float(theta["R_minor"]),
            n_max=int(theta["N_max"]),
        )
    mean_z = float(np.mean(z))
    return np.where(np.isfinite(pred), pred, mean_z)


def run_predict(
    *,
    train_csv: Path,
    val_csv: Path,
    results_dir: Path,
    out_csv: Path,
) -> pd.DataFrame:
    train = load_points(train_csv)
    val = _load_val(val_csv)
    preds = np.empty(len(val), dtype=float)
    is_arg = val["Rock"].astype(int) == 1

    for domain, mask in [("Argovian", is_arg), ("Other", ~is_arg)]:
        payload = _domain_payload(results_dir, domain)
        sub = train.loc[train["Domain"] == domain]
        xy_pred = val.loc[mask, ["Xloc", "Yloc"]].to_numpy(float)
        if xy_pred.size == 0:
            continue
        preds[mask.to_numpy()] = predict_domain(sub, xy_pred, payload)

    out = pd.DataFrame({"V1": val["id"].astype(int), "V2": preds})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OK predict val → V1,V2 CSV")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    out = run_predict(
        train_csv=args.train,
        val_csv=args.val,
        results_dir=args.results_dir,
        out_csv=args.out,
    )
    print(
        f"Wrote {args.out} n={len(out)} mean={out.V2.mean():.4f} "
        f"nan={int(out.V2.isna().sum())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
