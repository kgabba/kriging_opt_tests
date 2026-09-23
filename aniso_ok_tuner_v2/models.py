"""Theoretical variogram models for OK and directional fitting."""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np

MODEL_NAMES = ("spherical", "exponential", "gaussian")


def spherical_gamma(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Spherical γ(h): C0 + C*(1.5 r - 0.5 r^3) for h < a."""
    h = np.asarray(h, dtype=float)
    out = np.full_like(h, nugget + sill, dtype=float)
    m = (h > 0) & (h < range_)
    hr = h[m] / max(float(range_), 1e-12)
    out[m] = nugget + sill * (1.5 * hr - 0.5 * hr**3)
    out[h == 0] = 0.0
    return out


def exponential_gamma(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Exponential γ(h): C0 + C*(1 - exp(-3h/a)); practical range ≈ a."""
    h = np.asarray(h, dtype=float)
    a = max(float(range_), 1e-12)
    out = nugget + sill * (1.0 - np.exp(-3.0 * h / a))
    out = np.asarray(out, dtype=float)
    out[h == 0] = 0.0
    return out


def gaussian_gamma(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Gaussian γ(h): C0 + C*(1 - exp(-3 (h/a)^2)); practical range ≈ a."""
    h = np.asarray(h, dtype=float)
    a = max(float(range_), 1e-12)
    out = nugget + sill * (1.0 - np.exp(-3.0 * (h / a) ** 2))
    out = np.asarray(out, dtype=float)
    out[h == 0] = 0.0
    return out


def get_gamma_fn(model: str) -> Callable[[np.ndarray, float, float, float], np.ndarray]:
    """Return γ(h; nugget, sill, range) for a named model."""
    key = str(model).lower()
    if key == "spherical":
        return spherical_gamma
    if key == "exponential":
        return exponential_gamma
    if key == "gaussian":
        return gaussian_gamma
    raise ValueError(f"unknown variogram model: {model}")


def fit_model(
    lag: np.ndarray,
    gamma: np.ndarray,
    model: str,
) -> tuple[float, float, float]:
    """Grid-search fit of nugget, partial sill, range for the chosen model."""
    lag = np.asarray(lag, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    fn = get_gamma_fn(model)
    if lag.size < 2:
        g0 = float(np.mean(gamma)) if gamma.size else 1.0
        r0 = float(lag.max()) if lag.size else 1.0
        return 0.0, max(g0, 1e-6), max(r0, 1e-6)

    best_sse = np.inf
    best = (0.0, float(np.max(gamma)), float(np.max(lag)))
    g_max = float(np.max(gamma))
    for nugget, sill, range_ in product(
        np.linspace(0.0, max(np.percentile(gamma, 25), 1e-9), 8),
        np.linspace(max(np.percentile(gamma, 40), 1e-9), g_max * 1.3 + 1e-9, 12),
        np.linspace(max(float(lag.min()), 1e-6), float(lag.max()), 15),
    ):
        pred = fn(lag, nugget, sill, range_)
        sse = float(np.sum((gamma - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (float(nugget), float(sill), float(range_))
    return best
