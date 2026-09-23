"""Outer Optuna: variography / geometry; logs best-per-outer only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .inner_search import run_inner
from .models import MODEL_NAMES
from .search_space import build_inner_space
from .variography import OuterParams, rebuild_variography

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_outer(
    *,
    xy: np.ndarray,
    z: np.ndarray,
    splits: Sequence,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run nested search for one domain; write outer_trials.csv + best_params.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    oc = cfg.get("outer", {})
    ic = cfg.get("inner", {})
    opt = cfg.get("optuna", {})

    n_outer = int(oc.get("n_trials", 30))
    n_inner = int(ic.get("n_trials", 200))
    seed = int(oc.get("seed", 42))
    n_startup = int(oc.get("n_startup_trials", max(5, n_outer // 3)))
    penalty = float(opt.get("invalid_penalty", 1e6))
    n_ei = int(opt.get("n_ei_candidates", 64))
    min_pairs = int(oc.get("min_pairs", 8))

    rows: list[dict[str, Any]] = []
    global_best = float("inf")
    global_payload: dict[str, Any] | None = None
    t0 = time.perf_counter()

    def _flush() -> None:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "outer_trials.csv", index=False)
        if global_payload is not None:
            (out_dir / "best_params.json").write_text(
                json.dumps(global_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _objective(trial: optuna.Trial) -> float:
        nonlocal global_best, global_payload
        t_outer0 = time.perf_counter()
        outer = OuterParams(
            geometry=trial.suggest_categorical("geometry", ["isotropic", "anisotropic"]),
            variogram_model=trial.suggest_categorical("variogram_model", list(MODEL_NAMES)),
            n_lags=int(
                trial.suggest_int("n_lags", int(oc.get("n_lags_min", 10)), int(oc.get("n_lags_max", 14)))
            ),
            max_dist_percentile=float(
                trial.suggest_float(
                    "max_dist_percentile",
                    float(oc.get("max_dist_percentile_min", 40)),
                    float(oc.get("max_dist_percentile_max", 60)),
                )
            ),
            tolerance_deg=float(
                trial.suggest_float(
                    "tolerance_deg",
                    float(oc.get("tolerance_deg_min", 16)),
                    float(oc.get("tolerance_deg_max", 24)),
                )
            ),
            radius_prior_scale=float(
                trial.suggest_float(
                    "radius_prior_scale",
                    float(oc.get("radius_prior_scale_min", 1.0)),
                    float(oc.get("radius_prior_scale_max", 1.5)),
                )
            ),
        )

        vg = rebuild_variography(xy, z, outer, min_pairs=min_pairs)
        space = build_inner_space(
            vg.fixed,
            radius_margin=float(ic.get("radius_margin", 0.20)),
            nugget_margin=float(ic.get("nugget_margin", 0.20)),
            range_scale_min=float(ic.get("range_scale_min", 0.8)),
            range_scale_max=float(ic.get("range_scale_max", 1.2)),
            nmax_min=int(ic.get("nmax_min", 5)),
            nmax_max=int(ic.get("nmax_max", 40)),
            radius_prior_scale=outer.radius_prior_scale,
        )

        # unique inner seed per outer for diversity but reproducible
        inner_seed = int(ic.get("seed", 42)) + trial.number * 1009
        inner = run_inner(
            xy=xy, z=z, fixed=vg.fixed, space=space, splits=splits,
            n_trials=n_inner, seed=inner_seed, invalid_penalty=penalty,
            n_startup_trials=ic.get("n_startup_trials"),
            n_ei_candidates=n_ei,
            progress_every=max(50, n_inner // 4) if n_inner >= 50 else 0,
        )

        best_inner = inner.get("best_inner_rmse")
        if best_inner is None:
            best_inner = penalty

        row = {
            "outer_trial": trial.number,
            **outer.as_dict(),
            "alpha_deg": vg.fixed.alpha_deg,
            "a_major": vg.fixed.a_major,
            "a_minor": vg.fixed.a_minor,
            "nugget_fit": vg.fixed.nugget_fit,
            "sill_total": vg.fixed.sill_total,
            "best_inner_rmse": best_inner,
            "best_inner_mae": inner.get("best_inner_mae"),
            "best_inner_theta": json.dumps(inner.get("best_inner_theta")),
            "inner_status": inner.get("status"),
            "outer_runtime_seconds": time.perf_counter() - t_outer0,
        }
        rows.append(row)

        print(
            f"  [outer {trial.number + 1}/{n_outer}] geom={outer.geometry} "
            f"model={outer.variogram_model} lags={outer.n_lags} "
            f"best_inner={best_inner:.6g} global_best={min(global_best, best_inner):.6g}",
            flush=True,
        )

        if best_inner < global_best and inner.get("best_inner_theta") is not None:
            global_best = float(best_inner)
            global_payload = {
                "status": "SUCCESS",
                "best_inner_rmse": global_best,
                "outer": outer.as_dict(),
                "fixed_model": {
                    "alpha_deg": vg.fixed.alpha_deg,
                    "a_major": vg.fixed.a_major,
                    "a_minor": vg.fixed.a_minor,
                    "nugget_fit": vg.fixed.nugget_fit,
                    "sill_total": vg.fixed.sill_total,
                    "isotropic": vg.fixed.isotropic,
                    "variogram_model": vg.fixed.variogram_model,
                },
                "best_inner_theta": inner.get("best_inner_theta"),
                "inner_cv_details": inner.get("inner_cv_details"),
                "moi": {
                    "major_angle_deg": vg.moi.major_angle_deg,
                    "minor_angle_deg": vg.moi.minor_angle_deg,
                    "k_moi": vg.moi.k_moi,
                },
            }

        _flush()
        if inner.get("status") != "SUCCESS":
            raise optuna.TrialPruned("inner_failed")
        return float(best_inner)

    sampler = TPESampler(seed=seed, n_startup_trials=n_startup, n_ei_candidates=n_ei, multivariate=False)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_outer)
    _flush()

    return {
        "status": "SUCCESS" if global_payload is not None else "FAILED",
        "n_outer": n_outer,
        "n_inner": n_inner,
        "best_inner_rmse": None if global_payload is None else global_best,
        "best": global_payload,
        "runtime_seconds": time.perf_counter() - t0,
        "outer_trials_csv": str(out_dir / "outer_trials.csv"),
    }
