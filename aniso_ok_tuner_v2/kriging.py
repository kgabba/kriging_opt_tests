"""Iso / aniso Ordinary Kriging with pluggable variogram model."""

from __future__ import annotations

import numpy as np

from .models import get_gamma_fn


def rotate_to_major(xy: np.ndarray, alpha_deg: float) -> np.ndarray:
    a = np.deg2rad(float(alpha_deg))
    cos_a, sin_a = np.cos(a), np.sin(a)
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=float)
    return np.asarray(xy, dtype=float) @ rot.T


def anisotropic_metric_distance(
    dx: np.ndarray, dy: np.ndarray, a_major: float, a_minor: float
) -> np.ndarray:
    k = float(np.clip(float(a_minor) / max(float(a_major), 1e-12), 1e-6, 1.0))
    return np.sqrt(dx * dx + (dy / k) ** 2)


def elliptical_mask(dx: np.ndarray, dy: np.ndarray, r_major: float, r_minor: float) -> np.ndarray:
    rm, rn = max(float(r_major), 1e-12), max(float(r_minor), 1e-12)
    return (dx / rm) ** 2 + (dy / rn) ** 2 <= 1.0


def ordinary_kriging_elliptical(
    known_xy: np.ndarray,
    known_z: np.ndarray,
    pred_xy: np.ndarray,
    *,
    alpha_deg: float,
    a_major: float,
    a_minor: float,
    nugget: float,
    sill: float,
    range_scale: float,
    r_major: float,
    r_minor: float,
    n_max: int,
    model: str = "spherical",
) -> np.ndarray:
    """Anisotropic OK; ``model`` selects theoretical γ."""
    gamma_fn = get_gamma_fn(model)
    known_xy = np.asarray(known_xy, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xy = np.asarray(pred_xy, dtype=float)
    n_known, m = known_xy.shape[0], pred_xy.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    a_maj = max(float(a_major) * float(range_scale), 1e-8)
    k_base_maj = max(float(a_major), 1e-8)
    k_base_min = max(float(a_minor), 1e-8)
    known_r = rotate_to_major(known_xy, alpha_deg)
    pred_r = rotate_to_major(pred_xy, alpha_deg)
    preds = np.empty(m, dtype=float)
    n_max_i = max(int(n_max), 1)
    nug, sil = max(float(nugget), 0.0), max(float(sill), 1e-12)

    for i in range(m):
        dxy = known_r - pred_r[i]
        dx, dy = dxy[:, 0], dxy[:, 1]
        inside = elliptical_mask(dx, dy, r_major, r_minor)
        if np.any(inside):
            idx = np.where(inside)[0]
        else:
            d_all = anisotropic_metric_distance(dx, dy, k_base_maj, k_base_min)
            k = min(n_max_i, n_known)
            idx = np.argpartition(d_all, k - 1)[:k]
        if idx.size > n_max_i:
            d_met = anisotropic_metric_distance(dx[idx], dy[idx], k_base_maj, k_base_min)
            keep = np.argpartition(d_met, n_max_i - 1)[:n_max_i]
            idx = idx[keep]
        c, zz = known_r[idx], known_z[idx]
        n = len(idx)
        if n == 0:
            preds[i] = np.nan
            continue
        dx_ij = c[:, None, 0] - c[None, :, 0]
        dy_ij = c[:, None, 1] - c[None, :, 1]
        D = anisotropic_metric_distance(dx_ij, dy_ij, k_base_maj, k_base_min)
        d0 = anisotropic_metric_distance(c[:, 0] - pred_r[i, 0], c[:, 1] - pred_r[i, 1], k_base_maj, k_base_min)
        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = gamma_fn(D, nug, sil, a_maj)
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = gamma_fn(d0, nug, sil, a_maj)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))
    return preds


def ordinary_kriging_isotropic(
    known_xy: np.ndarray,
    known_z: np.ndarray,
    pred_xy: np.ndarray,
    *,
    range_: float,
    nugget: float,
    sill: float,
    range_scale: float,
    radius: float,
    n_max: int,
    model: str = "spherical",
) -> np.ndarray:
    """Isotropic OK with circular search."""
    gamma_fn = get_gamma_fn(model)
    known_xy = np.asarray(known_xy, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xy = np.asarray(pred_xy, dtype=float)
    n_known, m = known_xy.shape[0], pred_xy.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    a = max(float(range_) * float(range_scale), 1e-8)
    r = max(float(radius), 1e-12)
    n_max_i = max(int(n_max), 1)
    nug, sil = max(float(nugget), 0.0), max(float(sill), 1e-12)
    preds = np.empty(m, dtype=float)

    for i in range(m):
        d0_all = np.linalg.norm(known_xy - pred_xy[i], axis=1)
        inside = d0_all <= r
        if np.any(inside):
            idx = np.where(inside)[0]
        else:
            k = min(n_max_i, n_known)
            idx = np.argpartition(d0_all, k - 1)[:k]
        if idx.size > n_max_i:
            keep = np.argpartition(d0_all[idx], n_max_i - 1)[:n_max_i]
            idx = idx[keep]
        c, zz = known_xy[idx], known_z[idx]
        n = len(idx)
        if n == 0:
            preds[i] = np.nan
            continue
        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        d0 = np.linalg.norm(c - pred_xy[i], axis=1)
        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = gamma_fn(D, nug, sil, a)
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = gamma_fn(d0, nug, sil, a)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))
    return preds
