"""Data-driven Optuna bounds for OK tuner."""

from __future__ import annotations

from dataclasses import dataclass

from .cv import ModelFixed


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
            d["R_minor"] = [self.r_minor_min, self.r_minor_max]
        return d


def build_search_space(
    fixed: ModelFixed,
    *,
    radius_margin: float = 0.20,
    nugget_margin: float = 0.20,
    range_scale_min: float = 0.80,
    range_scale_max: float = 1.20,
    nmax_min: int = 5,
    nmax_max: int = 40,
    radius_prior_scale: float = 1.0,
) -> SearchSpace:
    """±margin windows around ranges and fitted nugget; scale box from config.

    For isotropic mode, search radius prior centre is
    ``a_major * radius_prior_scale`` (Exp8 used ~1.5×range).
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
        )

    rmj = max(float(fixed.a_major), 1e-6)
    rmn = max(float(fixed.a_minor), 1e-6)
    return SearchSpace(
        r_major_min=rmj * (1.0 - m),
        r_major_max=rmj * (1.0 + m),
        r_minor_min=rmn * (1.0 - m),
        r_minor_max=rmn * (1.0 + m),
        nugget_min=float(nug_lo),
        nugget_max=float(nug_hi),
        range_scale_min=float(range_scale_min),
        range_scale_max=float(range_scale_max),
        nmax_min=int(nmax_min),
        nmax_max=int(nmax_max),
        isotropic=False,
    )
