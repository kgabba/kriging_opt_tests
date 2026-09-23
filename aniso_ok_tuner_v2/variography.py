"""Variography rebuild for one outer trial (MVP pairs/MOI + multi-model fit)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MVP = Path(__file__).resolve().parents[2] / "aniso_idw_mvp"
if _MVP.exists() and str(_MVP) not in sys.path:
    sys.path.insert(0, str(_MVP))

from aniso_idw_mvp.directional import experimental_directional  # noqa: E402
from aniso_idw_mvp.moi import MOIResult, estimate_moi  # noqa: E402
from aniso_idw_mvp.pairs import PairCloud, compute_pairs  # noqa: E402
from aniso_idw_mvp.variogram_map import build_variogram_map  # noqa: E402

from .cv import ModelFixed
from .models import fit_model


@dataclass(frozen=True)
class OuterParams:
    geometry: str
    variogram_model: str
    n_lags: int
    max_dist_percentile: float
    tolerance_deg: float
    radius_prior_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry,
            "variogram_model": self.variogram_model,
            "n_lags": int(self.n_lags),
            "max_dist_percentile": float(self.max_dist_percentile),
            "tolerance_deg": float(self.tolerance_deg),
            "radius_prior_scale": float(self.radius_prior_scale),
        }


@dataclass(frozen=True)
class VariographyResult:
    fixed: ModelFixed
    moi: MOIResult
    pairs: PairCloud
    radius_prior_scale: float


def _fit_axis(pairs: PairCloud, angle: float, model: str, **kw) -> tuple[float, float, float]:
    lag, g, _ = experimental_directional(pairs, angle, **kw)
    return fit_model(lag, g, model)


def rebuild_variography(
    xy: np.ndarray,
    z: np.ndarray,
    outer: OuterParams,
    *,
    min_pairs: int = 8,
    map_n_lags: int | None = None,
) -> VariographyResult:
    """Rebuild map/MOI/ranges for one outer configuration."""
    pairs = compute_pairs(xy, z)
    n_map = int(map_n_lags) if map_n_lags is not None else max(int(outer.n_lags), 8)
    vmap = build_variogram_map(
        pairs,
        n_lags=n_map,
        max_dist_percentile=float(outer.max_dist_percentile),
    )
    moi = estimate_moi(vmap)

    dir_kw = dict(
        tolerance_deg=float(outer.tolerance_deg),
        n_lags=int(outer.n_lags),
        max_dist_percentile=float(outer.max_dist_percentile),
        min_pairs=int(min_pairs),
    )
    model = outer.variogram_model
    iso = outer.geometry == "isotropic"

    if iso:
        # omnidirectional: use wide tolerance on major axis as proxy
        n0, s0, a0 = _fit_axis(pairs, moi.major_angle_deg, model, **{**dir_kw, "tolerance_deg": 90.0})
        a_maj, a_min = float(a0), float(a0)
        nugget_fit, sill_partial = float(n0), float(s0)
        alpha = 0.0
    else:
        n_maj, s_maj, a_maj = _fit_axis(pairs, moi.major_angle_deg, model, **dir_kw)
        n_min, s_min, a_min = _fit_axis(pairs, moi.minor_angle_deg, model, **dir_kw)
        a_maj, a_min = float(a_maj), float(a_min)
        alpha = float(moi.major_angle_deg)
        if a_maj < a_min:
            a_maj, a_min = a_min, a_maj
            alpha = float(moi.minor_angle_deg)
            n_maj, s_maj = n_min, s_min
        nugget_fit, sill_partial = float(n_maj), float(s_maj)

    sill_total = max(nugget_fit + sill_partial, float(np.var(z)), 1e-6)
    fixed = ModelFixed(
        alpha_deg=alpha,
        a_major=max(a_maj, 1e-6),
        a_minor=max(a_min, 1e-6),
        sill_total=sill_total,
        nugget_fit=nugget_fit,
        isotropic=iso,
        variogram_model=model,
    )
    return VariographyResult(
        fixed=fixed, moi=moi, pairs=pairs, radius_prior_scale=float(outer.radius_prior_scale)
    )
