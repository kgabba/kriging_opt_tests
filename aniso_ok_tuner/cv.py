"""CV scorers and trial parameter containers for OK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .kriging import ordinary_kriging_elliptical, ordinary_kriging_isotropic


@dataclass(frozen=True)
class Theta:
    """Optuna-tunable OK neighbourhood / model knobs (alpha fixed outside).

    For isotropic mode ``r_minor`` is ignored / kept equal to ``r_major`` (= R).
    """

    r_major: float
    r_minor: float
    n_max: int
    nugget: float
    range_scale: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "R_major": float(self.r_major),
            "R_minor": float(self.r_minor),
            "R": float(self.r_major),  # alias for isotropic circle
            "N_max": int(self.n_max),
            "nugget": float(self.nugget),
            "range_scale": float(self.range_scale),
        }


@dataclass(frozen=True)
class ModelFixed:
    """Quantities fixed from MOI + directional fit for one domain."""

    alpha_deg: float
    a_major: float
    a_minor: float
    sill_partial_base: float  # C from fit (before nugget retune)
    sill_total: float         # C0_fit + C  (or var), kept constant
    nugget_fit: float
    isotropic: bool = False


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


def _nugget_sill(theta: Theta, fixed: ModelFixed) -> tuple[float, float]:
    nug = float(np.clip(theta.nugget, 0.0, 0.95 * fixed.sill_total))
    sil = max(fixed.sill_total - nug, 1e-12)
    return nug, sil


def _ok_predict(
    xy: np.ndarray,
    z: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    *,
    nug: float,
    sil: float,
) -> np.ndarray:
    """OK predict test points from train indices."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        return np.full(test_idx.size, np.nan)
    if fixed.isotropic:
        return ordinary_kriging_isotropic(
            xy[train_idx],
            z[train_idx],
            xy[test_idx],
            range_=fixed.a_major,
            nugget=nug,
            sill=sil,
            range_scale=theta.range_scale,
            radius=theta.r_major,
            n_max=theta.n_max,
        )
    return ordinary_kriging_elliptical(
        xy[train_idx],
        z[train_idx],
        xy[test_idx],
        alpha_deg=fixed.alpha_deg,
        a_major=fixed.a_major,
        a_minor=fixed.a_minor,
        nugget=nug,
        sill=sil,
        range_scale=theta.range_scale,
        r_major=theta.r_major,
        r_minor=theta.r_minor,
        n_max=theta.n_max,
    )


def _score_holdout_folds(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    method: str,
    invalid_penalty: float,
    extra_details: dict | None = None,
) -> CVResult:
    """Score a list of (train_idx, test_idx) folds; mean RMSE/MAE over folds."""
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    nug, sil = _nugget_sill(theta, fixed)
    rmses: list[float] = []
    maes: list[float] = []
    total_nan = 0

    for train_idx, test_idx in folds:
        pred = _ok_predict(xy, z, train_idx, test_idx, theta, fixed, nug=nug, sil=sil)
        n_nan = int(np.isnan(pred).sum())
        total_nan += n_nan
        if n_nan > 0:
            return CVResult(
                rmse=float(invalid_penalty),
                mae=float(invalid_penalty),
                valid=False,
                n_nan_predictions=total_nan,
                details={"method": method, "reason": "NaN predictions"},
            )
        rmses.append(rmse(z[test_idx], pred))
        maes.append(mae(z[test_idx], pred))

    details = {
        "method": method,
        "n_folds": len(folds),
        "rmse_std": float(np.std(rmses)) if rmses else float("nan"),
        "mae_std": float(np.std(maes)) if maes else float("nan"),
        "nugget_used": nug,
        "sill_partial_used": sil,
    }
    if extra_details:
        details.update(extra_details)
    return CVResult(
        rmse=float(np.mean(rmses)) if rmses else float(invalid_penalty),
        mae=float(np.mean(maes)) if maes else float(invalid_penalty),
        valid=bool(rmses),
        n_nan_predictions=0,
        details=details,
    )


def make_delete_d_splits(
    n_sample: int,
    *,
    holdout_size: int,
    n_repeats: int,
    seed: int,
) -> list[np.ndarray]:
    if holdout_size < 1 or holdout_size >= n_sample:
        raise ValueError("holdout_size must be in [1, n_sample)")
    rng = np.random.default_rng(seed)
    return [rng.choice(n_sample, size=holdout_size, replace=False) for _ in range(n_repeats)]


def default_holdout_size(n_sample: int) -> int:
    return int(min(max(5, n_sample // 10), n_sample - 2))


def make_loo_indices(n_sample: int) -> list[int]:
    return list(range(int(n_sample)))


def make_kfold_splits(n_sample: int, *, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return list of test-index arrays for k random folds (disjoint cover)."""
    n_folds = int(n_folds)
    if n_folds < 2 or n_folds > n_sample:
        raise ValueError("n_folds must be in [2, n_sample]")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sample)
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(perm, n_folds) if len(chunk) > 0]


def make_spatial_block_splits(
    xy: np.ndarray,
    *,
    grid_nx: int = 3,
    grid_ny: int = 3,
) -> list[np.ndarray]:
    """Leave-one-cell-out test indices on an axis-aligned grid over bbox."""
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    grid_nx, grid_ny = int(grid_nx), int(grid_ny)
    if n == 0:
        return []
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    # tiny pad so max point falls in last cell
    eps = 1e-12
    dx = max((xmax - xmin) / grid_nx, eps)
    dy = max((ymax - ymin) / grid_ny, eps)
    ix = np.clip(((xy[:, 0] - xmin) / dx).astype(int), 0, grid_nx - 1)
    iy = np.clip(((xy[:, 1] - ymin) / dy).astype(int), 0, grid_ny - 1)
    cell = ix + iy * grid_nx
    folds: list[np.ndarray] = []
    for c in range(grid_nx * grid_ny):
        test = np.where(cell == c)[0]
        if test.size == 0:
            continue
        if test.size >= n:
            continue  # would leave empty train
        folds.append(test.astype(int))
    return folds


def delete_d_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    splits: Sequence[np.ndarray],
    *,
    invalid_penalty: float = 1e6,
) -> CVResult:
    """Mean delete-d RMSE for one OK hyperparameter set."""
    n = len(np.asarray(z).reshape(-1))
    all_idx = np.arange(n)
    folds = []
    for hold_idx in splits:
        hold_idx = np.asarray(hold_idx, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[hold_idx] = False
        folds.append((all_idx[mask], hold_idx))
    res = _score_holdout_folds(
        xy, z, theta, fixed, folds,
        method="delete_d",
        invalid_penalty=invalid_penalty,
        extra_details={
            "n_repeats": len(splits),
            "holdout_size": int(len(splits[0])) if splits else 0,
        },
    )
    return res


def loo_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    *,
    invalid_penalty: float = 1e6,
) -> CVResult:
    """Leave-one-out RMSE (mean over all points)."""
    n = len(np.asarray(z).reshape(-1))
    all_idx = np.arange(n)
    folds = [(np.delete(all_idx, i), np.array([i], dtype=int)) for i in range(n)]
    return _score_holdout_folds(
        xy, z, theta, fixed, folds,
        method="loo",
        invalid_penalty=invalid_penalty,
    )


def kfold_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    test_folds: Sequence[np.ndarray],
    *,
    invalid_penalty: float = 1e6,
) -> CVResult:
    """k-fold CV with precomputed test folds."""
    n = len(np.asarray(z).reshape(-1))
    all_idx = np.arange(n)
    folds = []
    for test_idx in test_folds:
        test_idx = np.asarray(test_idx, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask]
        if train_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return _score_holdout_folds(
        xy, z, theta, fixed, folds,
        method="kfold5" if len(test_folds) == 5 else "kfold",
        invalid_penalty=invalid_penalty,
    )


def spatial_block_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    test_folds: Sequence[np.ndarray],
    *,
    invalid_penalty: float = 1e6,
    grid_nx: int = 3,
    grid_ny: int = 3,
) -> CVResult:
    """Leave-one-spatial-block-out CV."""
    n = len(np.asarray(z).reshape(-1))
    all_idx = np.arange(n)
    folds = []
    for test_idx in test_folds:
        test_idx = np.asarray(test_idx, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask]
        if train_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return _score_holdout_folds(
        xy, z, theta, fixed, folds,
        method="spatial_block",
        invalid_penalty=invalid_penalty,
        extra_details={"grid_nx": int(grid_nx), "grid_ny": int(grid_ny)},
    )


def buffer_loo_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    *,
    buffer_radius: float = 0.4,
    invalid_penalty: float = 1e6,
) -> CVResult:
    """LOO with spatial buffer: exclude train points within ``buffer_radius`` of left-out."""
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = len(z)
    r = float(buffer_radius)
    r2 = r * r
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0
    for i in range(n):
        d2 = np.sum((xy - xy[i]) ** 2, axis=1)
        # exclude left-out and any neighbor within buffer
        train_mask = d2 > r2
        train_idx = np.where(train_mask)[0]
        if train_idx.size == 0:
            skipped += 1
            continue
        folds.append((train_idx, np.array([i], dtype=int)))
    if not folds:
        return CVResult(
            rmse=float(invalid_penalty),
            mae=float(invalid_penalty),
            valid=False,
            n_nan_predictions=0,
            details={"method": "buffer", "reason": "all folds empty train", "skipped": skipped},
        )
    return _score_holdout_folds(
        xy, z, theta, fixed, folds,
        method="buffer",
        invalid_penalty=invalid_penalty,
        extra_details={"buffer_radius": r, "skipped_empty_train": skipped},
    )


def make_buffered_delete_d_folds(
    xy: np.ndarray,
    *,
    holdout_size: int,
    n_repeats: int,
    buffer_radius: float,
    seed: int,
    alpha_deg: float | None = None,
    a_major: float | None = None,
    a_minor: float | None = None,
    anisotropic: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Delete-d holdouts; train excludes points within ``buffer_radius`` of holdout.

    If ``anisotropic``, distance is the OK anisotropy metric (rotate by ``alpha_deg``,
    compress minor by ``a_minor/a_major``); ``buffer_radius`` is then in major-equivalent
    length units (typically ``buffer_scale * a_major``).
    """
    from .kriging import anisotropic_metric_distance, rotate_to_major

    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if holdout_size < 1 or holdout_size >= n:
        raise ValueError("holdout_size must be in [1, n)")
    r = float(buffer_radius)
    rng = np.random.default_rng(seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    if anisotropic:
        if a_major is None or a_minor is None or alpha_deg is None:
            raise ValueError("anisotropic buffered folds need alpha_deg, a_major, a_minor")
        xy_m = rotate_to_major(xy, float(alpha_deg))
        a_maj, a_min = float(a_major), float(a_minor)
    else:
        xy_m = xy
        a_maj = a_min = None

    for _ in range(int(n_repeats)):
        hold_idx = rng.choice(n, size=int(holdout_size), replace=False)
        if anisotropic:
            dxy = xy_m[:, None, :] - xy_m[hold_idx][None, :, :]
            d_min = np.min(
                anisotropic_metric_distance(dxy[:, :, 0], dxy[:, :, 1], a_maj, a_min),
                axis=1,
            )
            train_mask = d_min > r
        else:
            d2 = np.min(
                np.sum((xy[:, None, :] - xy[hold_idx][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            train_mask = d2 > (r * r)
        train_idx = np.where(train_mask)[0]
        if train_idx.size == 0:
            continue
        folds.append((train_idx, np.asarray(hold_idx, dtype=int)))
    return folds


def buffered_delete_d_score(
    xy: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    buffer_radius: float,
    holdout_size: int,
    invalid_penalty: float = 1e6,
    buffer_metric: str = "euclidean",
    buffer_scale: float | None = None,
) -> CVResult:
    """Mean RMSE over buffered delete-d folds (leave-d + min distance)."""
    if not folds:
        return CVResult(
            rmse=float(invalid_penalty),
            mae=float(invalid_penalty),
            valid=False,
            n_nan_predictions=0,
            details={
                "method": "buffered_delete_d",
                "reason": "no valid folds",
                "buffer_radius": float(buffer_radius),
                "holdout_size": int(holdout_size),
                "buffer_metric": buffer_metric,
                "buffer_scale": buffer_scale,
            },
        )
    return _score_holdout_folds(
        xy, z, theta, fixed, folds,
        method="buffered_delete_d",
        invalid_penalty=invalid_penalty,
        extra_details={
            "buffer_radius": float(buffer_radius),
            "holdout_size": int(holdout_size),
            "n_repeats_used": len(folds),
            "buffer_metric": buffer_metric,
            "buffer_scale": buffer_scale,
        },
    )
