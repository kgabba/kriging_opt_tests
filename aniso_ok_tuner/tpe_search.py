"""Optuna TPE with per-trial logging and trials.csv (like aniso_idw_tuner)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .cv import CVResult, ModelFixed, Theta
from .search_space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_tpe(
    *,
    space: SearchSpace,
    fixed: ModelFixed,
    objective_fn: Callable[[Theta], CVResult],
    n_trials: int,
    seed: int,
    out_csv: Path | None = None,
    best_json: Path | None = None,
    n_startup_trials: int | None = None,
    n_ei_candidates: int = 64,
) -> dict[str, Any]:
    """Minimize delete-d RMSE; print each trial and flush trials.csv."""
    if n_startup_trials is None:
        n_startup_trials = max(10, int(0.3 * n_trials))

    rows: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_theta: Theta | None = None
    best_mae: float | None = None
    best_eval: int | None = None
    best_details: dict | None = None
    eval_counter = {"n": 0}
    t0 = time.perf_counter()

    def _snapshot() -> dict[str, Any]:
        return {
            "method": "tpe_ok_iso" if fixed.isotropic else "tpe_ok_aniso",
            "isotropic": fixed.isotropic,
            "seed": seed,
            "n_trials": n_trials,
            "n_startup_trials": n_startup_trials,
            "status": "SUCCESS" if best_theta is not None else "RUNNING",
            "best_delete_d_rmse": None if best_theta is None else best_rmse,
            "best_delete_d_mae": best_mae,
            "best_evaluation": best_eval,
            "best_theta": None if best_theta is None else best_theta.as_dict(),
            "fixed": {
                "alpha_deg": fixed.alpha_deg,
                "a_major": fixed.a_major,
                "a_minor": fixed.a_minor,
                "nugget_fit": fixed.nugget_fit,
                "sill_total": fixed.sill_total,
                "isotropic": fixed.isotropic,
            },
            "search_space": space.as_dict(),
            "cv_details": best_details,
            "completed_trials": eval_counter["n"],
            "total_runtime_seconds": time.perf_counter() - t0,
            "trials_csv": None if out_csv is None else str(out_csv),
        }

    def _flush() -> None:
        if out_csv is not None and rows:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        if best_json is not None and best_theta is not None:
            best_json.parent.mkdir(parents=True, exist_ok=True)
            best_json.write_text(
                json.dumps(_snapshot(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _objective(trial: optuna.Trial) -> float:
        eval_counter["n"] += 1
        ev = eval_counter["n"]

        if fixed.isotropic or space.isotropic:
            r = float(trial.suggest_float("R", space.r_major_min, space.r_major_max))
            theta = Theta(
                r_major=r,
                r_minor=r,
                n_max=int(trial.suggest_int("N_max", space.nmax_min, space.nmax_max)),
                nugget=float(
                    trial.suggest_float("nugget", space.nugget_min, space.nugget_max)
                ),
                range_scale=float(
                    trial.suggest_float(
                        "range_scale", space.range_scale_min, space.range_scale_max
                    )
                ),
            )
        else:
            r_maj = float(
                trial.suggest_float("R_major", space.r_major_min, space.r_major_max)
            )
            r_min = float(
                trial.suggest_float("R_minor", space.r_minor_min, space.r_minor_max)
            )
            if r_min > r_maj:
                r_maj, r_min = r_min, r_maj
            theta = Theta(
                r_major=r_maj,
                r_minor=r_min,
                n_max=int(trial.suggest_int("N_max", space.nmax_min, space.nmax_max)),
                nugget=float(
                    trial.suggest_float("nugget", space.nugget_min, space.nugget_max)
                ),
                range_scale=float(
                    trial.suggest_float(
                        "range_scale", space.range_scale_min, space.range_scale_max
                    )
                ),
            )
        t_eval0 = time.perf_counter()
        res = objective_fn(theta)
        runtime = time.perf_counter() - t_eval0

        nonlocal best_rmse, best_theta, best_mae, best_eval, best_details
        if res.valid and res.rmse < best_rmse:
            best_rmse = res.rmse
            best_theta = theta
            best_mae = res.mae
            best_eval = ev
            best_details = dict(res.details)

        best_so_far = best_rmse if best_theta is not None else float("nan")
        row = {
            "evaluation": ev,
            "trial_number": trial.number,
            "R_major": theta.r_major,
            "R_minor": theta.r_minor,
            "N_max": theta.n_max,
            "nugget": theta.nugget,
            "range_scale": theta.range_scale,
            "RMSE_delete_d": res.rmse,
            "MAE_delete_d": res.mae,
            "valid": res.valid,
            "n_nan_predictions": res.n_nan_predictions,
            "runtime_seconds": runtime,
            "best_so_far_RMSE": best_so_far,
        }
        rows.append(row)

        phase = "random" if ev <= n_startup_trials else "tpe"
        if fixed.isotropic:
            print(
                f"  [{ev}/{n_trials}] phase={phase} "
                f"RMSE={res.rmse:.6g} best={best_so_far:.6g} "
                f"R={theta.r_major:.4g} N={theta.n_max} "
                f"nug={theta.nugget:.4g} scale={theta.range_scale:.3f}",
                flush=True,
            )
        else:
            print(
                f"  [{ev}/{n_trials}] phase={phase} "
                f"RMSE={res.rmse:.6g} best={best_so_far:.6g} "
                f"Rmaj={theta.r_major:.4g} Rmin={theta.r_minor:.4g} "
                f"N={theta.n_max} nug={theta.nugget:.4g} scale={theta.range_scale:.3f}",
                flush=True,
            )

        trial.set_user_attr("valid", bool(res.valid))
        if ev % 25 == 0 or ev == n_trials:
            _flush()
        if not res.valid:
            raise optuna.TrialPruned("invalid_ok_configuration")
        return float(res.rmse)

    sampler = TPESampler(
        seed=int(seed),
        n_startup_trials=int(n_startup_trials),
        n_ei_candidates=int(n_ei_candidates),
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=int(n_trials))
    print(flush=True)
    _flush()

    out = _snapshot()
    out["status"] = "SUCCESS" if best_theta is not None else "FAILED"
    return out
