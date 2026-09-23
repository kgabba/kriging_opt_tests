"""Delete-d CV containers for nested OK tuner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .kriging import ordinary_kriging_elliptical, ordinary_kriging_isotropic


@dataclass(frozen=True)
class Theta:
    r_major: float
    r_minor: float
    n_max: int
    nugget: float
    range_scale: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "R_major": float(self.r_major),
            "R_minor": float(self.r_minor),
            "R": float(self.r_major),
            "N_max": int(self.n_max),
            "nugget": float(self.nugget),
            "range_scale": float(self.range_scale),
        }


@dataclass(frozen=True)
class ModelFixed:
    alpha_deg: float
    a_major: float
    a_minor: float
    sill_total: float
    nugget_fit: float
    isotropic: bool
    variogram_model: str


@dataclass
class CVResult:
    rmse: float
    mae: float
    valid: bool
    n_nan_predictions: int
    details: dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def make_delete_d_splits(
    n_sample: int, *, holdout_size: int, n_repeats: int, seed: int
) -> list[np.ndarray]:
    if holdout_size < 1 or holdout_size >= n_sample:
        raise ValueError("holdout_size must be in [1, n_sample)")
    rng = np.random.default_rng(seed)
    return [rng.choice(n_sample, size=holdout_size, replace=False) for _ in range(n_repeats)]


def default_holdout_size(n_sample: int) -> int:
    return int(min(max(5, n_sample // 10), n_sample - 2))


def delete_d_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    splits: Sequence[np.ndarray],
    *,
    invalid_penalty: float = 1e6,
) -> CVResult:
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = len(z)
    all_idx = np.arange(n)
    nug = float(np.clip(theta.nugget, 0.0, 0.95 * fixed.sill_total))
    sil = max(fixed.sill_total - nug, 1e-12)
    model = fixed.variogram_model

    rmses: list[float] = []
    maes: list[float] = []
    total_nan = 0

    for hold_idx in splits:
        hold_idx = np.asarray(hold_idx, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[hold_idx] = False
        train_idx = all_idx[mask]
        if fixed.isotropic:
            pred = ordinary_kriging_isotropic(
                xy[train_idx], z[train_idx], xy[hold_idx],
                range_=fixed.a_major, nugget=nug, sill=sil,
                range_scale=theta.range_scale, radius=theta.r_major,
                n_max=theta.n_max, model=model,
            )
        else:
            pred = ordinary_kriging_elliptical(
                xy[train_idx], z[train_idx], xy[hold_idx],
                alpha_deg=fixed.alpha_deg, a_major=fixed.a_major, a_minor=fixed.a_minor,
                nugget=nug, sill=sil, range_scale=theta.range_scale,
                r_major=theta.r_major, r_minor=theta.r_minor,
                n_max=theta.n_max, model=model,
            )
        n_nan = int(np.isnan(pred).sum())
        total_nan += n_nan
        if n_nan > 0:
            return CVResult(
                rmse=float(invalid_penalty), mae=float(invalid_penalty),
                valid=False, n_nan_predictions=total_nan,
                details={"reason": "NaN predictions"},
            )
        rmses.append(rmse(z[hold_idx], pred))
        maes.append(mae(z[hold_idx], pred))

    return CVResult(
        rmse=float(np.mean(rmses)), mae=float(np.mean(maes)),
        valid=True, n_nan_predictions=0,
        details={
            "n_repeats": len(splits),
            "holdout_size": int(len(splits[0])) if splits else 0,
            "rmse_std": float(np.std(rmses)),
            "nugget_used": nug,
            "sill_partial_used": sil,
        },
    )
