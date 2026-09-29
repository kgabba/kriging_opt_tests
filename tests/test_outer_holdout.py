"""Tests for outer holdout split + scoring helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aniso_ok_tuner.cv import ModelFixed, Theta, make_outer_holdout_split, score_outer_holdout


def test_outer_split_sizes():
    split = make_outer_holdout_split(100, fraction=0.15, min_points=10, seed=0)
    assert split is not None
    tr, ou = split
    assert len(ou) == 15
    assert len(tr) == 85
    assert len(np.intersect1d(tr, ou)) == 0


def test_outer_split_too_small():
    assert make_outer_holdout_split(25, min_points=10, min_train=20) is None


def test_score_outer_holdout_runs():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(40, 2))
    z = rng.normal(size=40)
    tr, ou = make_outer_holdout_split(40, fraction=0.2, min_points=8, min_train=20, seed=1)
    assert tr is not None
    fixed = ModelFixed(0.0, 2.0, 2.0, 1.0, 1.2, 0.2, isotropic=True)
    theta = Theta(2.0, 2.0, 10, 0.2, 1.0)
    res = score_outer_holdout(xy, z, tr, ou, theta, fixed, invalid_penalty=1e6)
    assert res.valid
    assert np.isfinite(res.rmse)


if __name__ == "__main__":
    test_outer_split_sizes()
    test_outer_split_too_small()
    test_score_outer_holdout_runs()
    print("ok")
