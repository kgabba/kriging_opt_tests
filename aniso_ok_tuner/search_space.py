"""Data-driven Optuna bounds for OK tuner."""

from __future__ import annotations

from dataclasses import dataclass

from .cv import ModelFixed


def anisotropy_ratio(a_major: float, a_minor: float) -> float:
    """K = a_minor / a_major clipped to (0, 1]."""
    k = float(a_minor) / max(float(a_major), 1e-12)
    return float(min(max(k, 1e-6), 1.0))


def locked_r_minor(r_major: float, a_major: float, a_minor: float) -> float:
    """Search minor axis locked to variogram anisotropy: R_min = R_maj * K."""
    return float(r_major) * anisotropy_ratio(a_major, a_minor)


@dataclass(frozen=True)
class SearchSpace:
    r_major_min: float
    r_major_max: float
    r_minor_min: float
    r_minor_max: float
    nugget_min: float
    nugget_max: float
    range_scale_min: float
    range_scale_max: float
    nmax_min: int
    nmax_max: int
    isotropic: bool = False
    # Aniso: R_minor is not Optuna-tuned; always R_major * (a_minor/a_major).
    search_ratio_locked: bool = True

    def as_dict(self) -> dict:
        d = {
            "isotropic": self.isotropic,
            "nugget": [self.nugget_min, self.nugget_max],
            "range_scale": [self.range_scale_min, self.range_scale_max],
            "N_max": [self.nmax_min, self.nmax_max],
        }
        if self.isotropic:
            d["R"] = [self.r_major_min, self.r_major_max]
        else:
            d["R_major"] = [self.r_major_min, self.r_major_max]
            d["search_ratio_locked"] = bool(self.search_ratio_locked)
            d["R_minor"] = (
                "locked = R_major * (a_minor/a_major)"
                if self.search_ratio_locked
                else [self.r_minor_min, self.r_minor_max]
            )
        return d


def build_search_space(
    fixed: ModelFixed,
    *,
    radius_margin: float = 0.60,
    nugget_margin: float = 0.60,
    range_scale_min: float = 0.40,
    range_scale_max: float = 1.60,
    nmax_min: int = 5,
    nmax_max: int = 80,
    radius_prior_scale: float = 1.0,
) -> SearchSpace:
    """±margin windows around ranges and fitted nugget; scale box from config.

    For isotropic mode, search radius prior centre is
    ``a_major * radius_prior_scale`` (Exp8 used ~1.5×range).

    For anisotropic mode, ``radius_prior_scale`` is **ignored** (centre =
    ``a_major``). Only ``R_major`` is searched; ``R_minor`` is locked to
    ``R_major * (a_minor/a_major)`` so search ellipse matches variogram anisotropy.
    """
    m = float(radius_margin)
    c0 = max(float(fixed.nugget_fit), 0.0)
    nm = float(nugget_margin)
    nug_lo = max(0.0, c0 * (1.0 - nm) if c0 > 1e-12 else 0.0)
    nug_hi = (
        min(0.95 * fixed.sill_total, c0 * (1.0 + nm))
        if c0 > 1e-12
        else min(0.95 * fixed.sill_total, 0.05 * fixed.sill_total)
    )
    if nug_hi < nug_lo:
        nug_hi = nug_lo + 1e-9

    if fixed.isotropic:
        r0 = max(float(fixed.a_major) * float(radius_prior_scale), 1e-6)
        return SearchSpace(
            r_major_min=r0 * (1.0 - m),
            r_major_max=r0 * (1.0 + m),
            r_minor_min=r0 * (1.0 - m),
            r_minor_max=r0 * (1.0 + m),
            nugget_min=float(nug_lo),
            nugget_max=float(nug_hi),
            range_scale_min=float(range_scale_min),
            range_scale_max=float(range_scale_max),
            nmax_min=int(nmax_min),
            nmax_max=int(nmax_max),
            isotropic=True,
            search_ratio_locked=True,
        )

    rmj = max(float(fixed.a_major), 1e-6)
    k = anisotropy_ratio(fixed.a_major, fixed.a_minor)
    r_maj_lo = rmj * (1.0 - m)
    r_maj_hi = rmj * (1.0 + m)
    return SearchSpace(
        r_major_min=r_maj_lo,
        r_major_max=r_maj_hi,
        # Derived bounds (documentation / legacy); not Optuna-tuned.
        r_minor_min=r_maj_lo * k,
        r_minor_max=r_maj_hi * k,
        nugget_min=float(nug_lo),
        nugget_max=float(nug_hi),
        range_scale_min=float(range_scale_min),
        range_scale_max=float(range_scale_max),
        nmax_min=int(nmax_min),
        nmax_max=int(nmax_max),
        isotropic=False,
        search_ratio_locked=True,
    )
