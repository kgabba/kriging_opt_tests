"""Elliptical anisotropic Ordinary Kriging (2D)."""

from __future__ import annotations

import numpy as np

# Soft rescue when search neighbourhood is empty: take this many nearest
# (same metric as the search). Avoids the old jump empty → N_max.
DEFAULT_N_FALLBACK = 4


def rotate_to_major(xy: np.ndarray, alpha_deg: float) -> np.ndarray:
    """Rotate so major axis aligns with +x' (math CCW from +X)."""
    a = np.deg2rad(float(alpha_deg))
    cos_a, sin_a = np.cos(a), np.sin(a)
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=float)
    return np.asarray(xy, dtype=float) @ rot.T


def spherical_gamma(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Spherical semivariogram γ(h) with parameters nugget, partial sill, range.

    Notes
    -----
    For 0 < h < a:  γ = C0 + C*(1.5*(h/a) - 0.5*(h/a)^3)
    For h >= a:     γ = C0 + C
    For h == 0:     γ = 0
    """
    h = np.asarray(h, dtype=float)
    out = np.full_like(h, nugget + sill, dtype=float)
    m = (h > 0) & (h < range_)
    hr = h[m] / max(float(range_), 1e-12)
    out[m] = nugget + sill * (1.5 * hr - 0.5 * hr**3)
    out[h == 0] = 0.0
    return out


def anisotropic_metric_distance(
    dx: np.ndarray,
    dy: np.ndarray,
    a_major: float,
    a_minor: float,
) -> np.ndarray:
    """Geometric-anisotropy distance with K = a_minor/a_major.

    h' = sqrt(dx^2 + (dy/K)^2); isotropic spherical range then uses a_major.
    """
    k = float(a_minor) / max(float(a_major), 1e-12)
    k = float(np.clip(k, 1e-6, 1.0))
    return np.sqrt(dx * dx + (dy / k) ** 2)


def elliptical_mask(dx: np.ndarray, dy: np.ndarray, r_major: float, r_minor: float) -> np.ndarray:
    """Neighbors inside search ellipse (dx/Rmaj)^2 + (dy/Rmin)^2 <= 1."""
    rm = max(float(r_major), 1e-12)
    rn = max(float(r_minor), 1e-12)
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
    n_fallback: int = DEFAULT_N_FALLBACK,
) -> np.ndarray:
    """Point OK with anisotropic metric + elliptical neighbourhood.

    Neighbour selection
    -------------------
    Prefer samples inside the search ellipse (capped at ``n_max``).
    If the ellipse is empty, take ``n_fallback`` nearest by the anisotropy
    metric (soft rescue — not a full ``n_max`` grab from anywhere).

    Parameters
    ----------
    known_xy, known_z, pred_xy :
        Training / prediction locations and grades.
    alpha_deg :
        Fixed MOI major angle (degrees CCW from +X).
    a_major, a_minor :
        Base directional ranges from variogram fit (before scale).
    nugget, sill :
        Spherical C0 and partial sill C (total sill = C0+C).
    range_scale :
        Multiplier applied to a_major (isotropic range in aniso metric).
    r_major, r_minor :
        Search ellipse semi-axes.
    n_max :
        Cap on neighbours inside the ellipse.
    n_fallback :
        Soft kNN size when the ellipse is empty (default 4).

    Returns
    -------
    np.ndarray
        Predictions; NaN only if no known samples at all.
    """
    known_xy = np.asarray(known_xy, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xy = np.asarray(pred_xy, dtype=float)
    n_known = known_xy.shape[0]
    m = pred_xy.shape[0]
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
    n_fb = max(int(n_fallback), 1)
    nug = max(float(nugget), 0.0)
    sil = max(float(sill), 1e-12)

    for i in range(m):
        dxy = known_r - pred_r[i]
        dx, dy = dxy[:, 0], dxy[:, 1]
        inside = elliptical_mask(dx, dy, r_major, r_minor)
        if np.any(inside):
            idx = np.where(inside)[0]
            if idx.size > n_max_i:
                d_met = anisotropic_metric_distance(
                    dx[idx], dy[idx], k_base_maj, k_base_min
                )
                keep = np.argpartition(d_met, n_max_i - 1)[:n_max_i]
                idx = idx[keep]
        else:
            d_all = anisotropic_metric_distance(dx, dy, k_base_maj, k_base_min)
            k = min(n_fb, n_known)
            idx = np.argpartition(d_all, k - 1)[:k]

        c = known_r[idx]
        zz = known_z[idx]
        n = len(idx)
        if n == 0:
            preds[i] = np.nan
            continue

        dx_ij = c[:, None, 0] - c[None, :, 0]
        dy_ij = c[:, None, 1] - c[None, :, 1]
        D = anisotropic_metric_distance(dx_ij, dy_ij, k_base_maj, k_base_min)

        dx0 = c[:, 0] - pred_r[i, 0]
        dy0 = c[:, 1] - pred_r[i, 1]
        d0 = anisotropic_metric_distance(dx0, dy0, k_base_maj, k_base_min)

        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = spherical_gamma(D, nug, sil, a_maj)
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = spherical_gamma(d0, nug, sil, a_maj)

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
    n_fallback: int = DEFAULT_N_FALLBACK,
) -> np.ndarray:
    """Point OK with Euclidean distance and circular neighbourhood.

    Prefer samples inside ``radius`` (capped at ``n_max``). If none, take
    ``n_fallback`` nearest Euclidean neighbours (soft rescue).

    Parameters
    ----------
    range_ :
        Base spherical range; effective model range = ``range_ * range_scale``.
    radius :
        Circular search radius.
    nugget, sill, n_max :
        As in anisotropic OK.
    n_fallback :
        Soft kNN size when the circle is empty (default 4).
    """
    known_xy = np.asarray(known_xy, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xy = np.asarray(pred_xy, dtype=float)
    n_known = known_xy.shape[0]
    m = pred_xy.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    a = max(float(range_) * float(range_scale), 1e-8)
    r = max(float(radius), 1e-12)
    n_max_i = max(int(n_max), 1)
    n_fb = max(int(n_fallback), 1)
    nug = max(float(nugget), 0.0)
    sil = max(float(sill), 1e-12)

    preds = np.empty(m, dtype=float)
    for i in range(m):
        d0_all = np.linalg.norm(known_xy - pred_xy[i], axis=1)
        inside = d0_all <= r
        if np.any(inside):
            idx = np.where(inside)[0]
            if idx.size > n_max_i:
                keep = np.argpartition(d0_all[idx], n_max_i - 1)[:n_max_i]
                idx = idx[keep]
        else:
            k = min(n_fb, n_known)
            idx = np.argpartition(d0_all, k - 1)[:k]

        c = known_xy[idx]
        zz = known_z[idx]
        n = len(idx)
        if n == 0:
            preds[i] = np.nan
            continue

        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        d0 = np.linalg.norm(c - pred_xy[i], axis=1)

        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = spherical_gamma(D, nug, sil, a)
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = spherical_gamma(d0, nug, sil, a)

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))

    return preds
