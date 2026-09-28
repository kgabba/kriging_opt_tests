"""Unit tests for iso/aniso auto selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aniso_ok_tuner.isotropy import resolve_isotropy_mode, select_isotropy


def test_resolve_mode_explicit_and_legacy():
    assert resolve_isotropy_mode({"isotropy_mode": "auto"}) == "auto"
    assert resolve_isotropy_mode({"isotropy_mode": "force_iso"}) == "force_iso"
    assert resolve_isotropy_mode({"isotropy_mode": "aniso"}) == "force_aniso"
    assert resolve_isotropy_mode({"isotropic": True}) == "force_iso"
    assert resolve_isotropy_mode({"isotropic": False}) == "force_aniso"
    assert resolve_isotropy_mode({}) == "auto"
    assert (
        resolve_isotropy_mode({"isotropy_mode": "auto", "isotropic": True}) == "auto"
    )


def test_few_points():
    iso, reason, m = select_isotropy(
        a_major=50.0,
        a_minor=20.0,
        nugget=1.0,
        sill_total=100.0,
        n_points=53,
        aniso_min_points=80,
        n_lags_major=8,
        n_lags_minor=8,
    )
    assert iso and reason == "few_points"
    assert m["n_points"] == 53


def test_sparse_directional():
    iso, reason, _ = select_isotropy(
        a_major=50.0,
        a_minor=20.0,
        nugget=1.0,
        sill_total=100.0,
        n_points=200,
        n_lags_major=8,
        n_lags_minor=2,
        min_directional_lags=4,
    )
    assert iso and reason == "sparse_directional"


def test_inconsistent_sills():
    iso, reason, m = select_isotropy(
        a_major=50.0,
        a_minor=20.0,
        nugget=1.0,
        sill_total=100.0,
        n_points=200,
        n_lags_major=8,
        n_lags_minor=8,
        sill_major=10.0,
        sill_minor=2.0,
        sill_ratio_max=2.5,
    )
    assert iso and reason == "inconsistent_directional_sills"
    assert m["sill_ratio"] == 5.0


def test_select_high_nugget():
    iso, reason, m = select_isotropy(
        a_major=50.0,
        a_minor=20.0,
        nugget=50.0,
        sill_total=100.0,
        n_points=200,
        n_lags_major=8,
        n_lags_minor=8,
        anisotropy_ratio_min=1.3,
        nugget_frac_iso=0.45,
    )
    assert iso and reason == "high_nugget_frac"
    assert m["nugget_frac_used"] == 0.5


def test_directional_nugget_triggers_noise():
    iso, reason, m = select_isotropy(
        a_major=50.0,
        a_minor=20.0,
        nugget=1.0,
        sill_total=100.0,
        n_points=200,
        n_lags_major=8,
        n_lags_minor=8,
        nugget_major=6.0,
        sill_major=4.0,  # frac 0.6
        nugget_frac_iso=0.45,
    )
    assert iso and reason == "high_nugget_frac"
    assert m["nugget_frac_used"] == 0.6


def test_select_weak_anisotropy():
    iso, reason, m = select_isotropy(
        a_major=33.0,
        a_minor=30.0,
        nugget=5.0,
        sill_total=100.0,
        n_points=200,
        n_lags_major=8,
        n_lags_minor=8,
        sill_major=5.0,
        sill_minor=5.0,
        anisotropy_ratio_min=1.3,
        nugget_frac_iso=0.45,
    )
    assert iso and reason == "weak_anisotropy"


def test_select_clear_anisotropy():
    iso, reason, m = select_isotropy(
        a_major=58.6,
        a_minor=24.0,
        nugget=10.0,
        sill_total=100.0,
        n_points=200,
        n_lags_major=8,
        n_lags_minor=8,
        sill_major=5.0,
        sill_minor=4.0,
        anisotropy_ratio_min=1.3,
        nugget_frac_iso=0.45,
    )
    assert not iso and reason == "clear_anisotropy"


if __name__ == "__main__":
    test_resolve_mode_explicit_and_legacy()
    test_few_points()
    test_sparse_directional()
    test_inconsistent_sills()
    test_select_high_nugget()
    test_directional_nugget_triggers_noise()
    test_select_weak_anisotropy()
    test_select_clear_anisotropy()
    print("ok")
