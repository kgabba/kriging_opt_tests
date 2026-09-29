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
from .search_space import SearchSpace, locked_r_minor

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
    outer_score_fn: Callable[[Theta], CVResult] | None = None,
    outer_patience: int | None = None,
) -> dict[str, Any]:
    """Minimize CV RMSE; optionally track outer holdout + early stop.

    Optuna always optimizes ``objective_fn`` (CV). If ``outer_score_fn`` is set,
    the reported ``best_theta`` is the trial with best **outer** RMSE, and the
    study stops after ``outer_patience`` trials without outer improvement.
    """
    if n_startup_trials is None:
        n_startup_trials = max(10, int(0.3 * n_trials))
    use_outer = outer_score_fn is not None
    patience = int(outer_patience) if outer_patience is not None else 0

    rows: list[dict[str, Any]] = []
    best_cv_rmse = float("inf")
    best_theta: Theta | None = None
    best_mae: float | None = None
    best_eval: int | None = None
    best_details: dict | None = None
    best_outer_rmse = float("inf")
    best_outer_mae: float | None = None
    best_outer_eval: int | None = None
    best_cv_at_outer: float | None = None
    early_stopped = False
    stop_reason: str | None = None
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
            # Primary reported score: outer when enabled, else CV
            "best_delete_d_rmse": (
                None
                if best_theta is None
                else (best_outer_rmse if use_outer else best_cv_rmse)
            ),
            "best_delete_d_mae": best_mae if not use_outer else best_outer_mae,
            "best_evaluation": best_eval,
            "best_theta": None if best_theta is None else best_theta.as_dict(),
            "best_cv_rmse": None if best_theta is None else (
                best_cv_at_outer if use_outer else best_cv_rmse
            ),
            "best_outer_rmse": None if not use_outer or best_theta is None else best_outer_rmse,
            "best_outer_mae": best_outer_mae,
            "best_outer_evaluation": best_outer_eval,
            "selection_metric": "outer_holdout" if use_outer else "cv",
            "outer_patience": patience if use_outer else None,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
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
        nonlocal best_cv_rmse, best_theta, best_mae, best_eval, best_details
        nonlocal best_outer_rmse, best_outer_mae, best_outer_eval, best_cv_at_outer
        nonlocal early_stopped, stop_reason

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
            r_min = locked_r_minor(r_maj, fixed.a_major, fixed.a_minor)
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
        outer_res: CVResult | None = None
        if use_outer and res.valid:
            outer_res = outer_score_fn(theta)  # type: ignore[misc]
        runtime = time.perf_counter() - t_eval0

        if res.valid and res.rmse < best_cv_rmse:
            best_cv_rmse = res.rmse
            if not use_outer:
                best_theta = theta
                best_mae = res.mae
                best_eval = ev
                best_details = dict(res.details)

        outer_rmse = float("nan")
        outer_mae = float("nan")
        if outer_res is not None and outer_res.valid:
            outer_rmse = outer_res.rmse
            outer_mae = outer_res.mae
            trial.set_user_attr("outer_rmse", float(outer_rmse))
            if outer_rmse < best_outer_rmse:
                best_outer_rmse = outer_rmse
                best_outer_mae = outer_mae
                best_outer_eval = ev
                best_theta = theta
                best_mae = outer_mae
                best_eval = ev
                best_details = dict(res.details)
                best_cv_at_outer = res.rmse

        best_so_far = (
            best_outer_rmse
            if use_outer and best_theta is not None
            else (best_cv_rmse if best_theta is not None else float("nan"))
        )
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
            "RMSE_outer": outer_rmse,
            "MAE_outer": outer_mae,
            "valid": res.valid,
            "n_nan_predictions": res.n_nan_predictions,
            "runtime_seconds": runtime,
            "best_so_far_RMSE": best_so_far,
        }
        rows.append(row)

        phase = "random" if ev <= n_startup_trials else "tpe"
        outer_txt = (
            f" outer={outer_rmse:.6g} best_outer={best_so_far:.6g}"
            if use_outer
            else f" best={best_so_far:.6g}"
        )
        if fixed.isotropic:
            print(
                f"  [{ev}/{n_trials}] phase={phase} "
                f"CV={res.rmse:.6g}{outer_txt} "
                f"R={theta.r_major:.4g} N={theta.n_max} "
                f"nug={theta.nugget:.4g} scale={theta.range_scale:.3f}",
                flush=True,
            )
        else:
            print(
                f"  [{ev}/{n_trials}] phase={phase} "
                f"CV={res.rmse:.6g}{outer_txt} "
                f"Rmaj={theta.r_major:.4g} Rmin={theta.r_minor:.4g}(locked) "
                f"N={theta.n_max} nug={theta.nugget:.4g} scale={theta.range_scale:.3f}",
                flush=True,
            )

        trial.set_user_attr("valid", bool(res.valid))
        if ev % 25 == 0 or ev == n_trials:
            _flush()

        if (
            use_outer
            and patience > 0
            and best_outer_eval is not None
            and (ev - best_outer_eval) >= patience
        ):
            early_stopped = True
            stop_reason = (
                f"outer_holdout no improve for {patience} trials "
                f"(best_outer_eval={best_outer_eval})"
            )
            trial.study.stop()

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
    if early_stopped:
        print(f"  early stop: {stop_reason}", flush=True)
    _flush()

    out = _snapshot()
    out["status"] = "SUCCESS" if best_theta is not None else "FAILED"
    return out
