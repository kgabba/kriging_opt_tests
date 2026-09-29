"""Unit tests for locked aniso search ratio (R_minor = R_major * a_min/a_maj)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aniso_ok_tuner.cv import ModelFixed
from aniso_ok_tuner.search_space import (
    anisotropy_ratio,
    build_search_space,
    locked_r_minor,
)


def test_locked_r_minor():
    assert abs(locked_r_minor(100.0, a_major=50.0, a_minor=25.0) - 50.0) < 1e-12
    assert abs(anisotropy_ratio(50.0, 25.0) - 0.5) < 1e-12


def test_aniso_search_space_locks_ratio():
    fixed = ModelFixed(
        alpha_deg=30.0,
        a_major=50.0,
        a_minor=25.0,
        sill_partial_base=10.0,
        sill_total=12.0,
        nugget_fit=2.0,
        isotropic=False,
    )
    space = build_search_space(fixed, radius_margin=0.2)
    d = space.as_dict()
    assert d["search_ratio_locked"] is True
    assert "R_major" in d
    assert isinstance(d["R_minor"], str) and "locked" in d["R_minor"]
    # derived minor bounds follow K=0.5
    assert abs(space.r_minor_min / space.r_major_min - 0.5) < 1e-12
    assert abs(space.r_minor_max / space.r_major_max - 0.5) < 1e-12


def test_iso_search_space_unchanged():
    fixed = ModelFixed(
        alpha_deg=0.0,
        a_major=1.0,
        a_minor=1.0,
        sill_partial_base=1.0,
        sill_total=1.2,
        nugget_fit=0.2,
        isotropic=True,
    )
    d = build_search_space(fixed).as_dict()
    assert "R" in d
    assert "search_ratio_locked" not in d


if __name__ == "__main__":
    test_locked_r_minor()
    test_aniso_search_space_locks_ratio()
    test_iso_search_space_unchanged()
    print("ok")
