"""Inner Optuna: neighbourhood only; returns best theta (no full CSV dump)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import optuna
from optuna.samplers import TPESampler

from .cv import CVResult, ModelFixed, Theta, delete_d_score
from .search_space import InnerSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_inner(
    *,
    xy,
    z,
    fixed: ModelFixed,
    space: InnerSpace,
    splits: Sequence,
    n_trials: int,
    seed: int,
    invalid_penalty: float = 1e6,
    n_startup_trials: int | None = None,
    n_ei_candidates: int = 64,
    progress_every: int = 50,
) -> dict[str, Any]:
    """Minimize delete-d RMSE; return best inner result only."""
    if n_startup_trials is None:
        n_startup_trials = max(5, int(0.3 * n_trials))

    best_rmse = float("inf")
    best_theta: Theta | None = None
    best_mae: float | None = None
    best_details: dict | None = None
    eval_n = 0

    def _objective(trial: optuna.Trial) -> float:
        nonlocal best_rmse, best_theta, best_mae, best_details, eval_n
        eval_n += 1
        if space.isotropic:
            r = float(trial.suggest_float("R", space.r_major_min, space.r_major_max))
            theta = Theta(
                r_major=r, r_minor=r,
                n_max=int(trial.suggest_int("N_max", space.nmax_min, space.nmax_max)),
                nugget=float(trial.suggest_float("nugget", space.nugget_min, space.nugget_max)),
                range_scale=float(
                    trial.suggest_float("range_scale", space.range_scale_min, space.range_scale_max)
                ),
            )
        else:
            r_maj = float(trial.suggest_float("R_major", space.r_major_min, space.r_major_max))
            r_min = float(trial.suggest_float("R_minor", space.r_minor_min, space.r_minor_max))
            if r_min > r_maj:
                r_maj, r_min = r_min, r_maj
            theta = Theta(
                r_major=r_maj, r_minor=r_min,
                n_max=int(trial.suggest_int("N_max", space.nmax_min, space.nmax_max)),
                nugget=float(trial.suggest_float("nugget", space.nugget_min, space.nugget_max)),
                range_scale=float(
                    trial.suggest_float("range_scale", space.range_scale_min, space.range_scale_max)
                ),
            )
        res = delete_d_score(xy, z, theta, fixed, splits, invalid_penalty=invalid_penalty)
        if res.valid and res.rmse < best_rmse:
            best_rmse = res.rmse
            best_theta = theta
            best_mae = res.mae
            best_details = dict(res.details)
        if progress_every and eval_n % progress_every == 0:
            print(f"      inner [{eval_n}/{n_trials}] best={best_rmse:.6g}", flush=True)
        if not res.valid:
            raise optuna.TrialPruned("invalid")
        return float(res.rmse)

    sampler = TPESampler(
        seed=int(seed),
        n_startup_trials=int(n_startup_trials),
        n_ei_candidates=int(n_ei_candidates),
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=int(n_trials))

    return {
        "status": "SUCCESS" if best_theta is not None else "FAILED",
        "best_inner_rmse": None if best_theta is None else best_rmse,
        "best_inner_mae": best_mae,
        "best_inner_theta": None if best_theta is None else best_theta.as_dict(),
        "inner_cv_details": best_details,
        "inner_completed_trials": eval_n,
        "inner_search_space": space.as_dict(),
    }
