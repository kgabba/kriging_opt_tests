"""Unit tests for CV method allowlist."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aniso_ok_tuner.pipeline import _resolve_cv_method


def test_default_and_allowed():
    assert _resolve_cv_method({}) == "spatial_block"
    assert _resolve_cv_method({"cv": {"method": "spatial_block"}}) == "spatial_block"
    assert (
        _resolve_cv_method({"cv": {"method": "buffered_delete_d"}}) == "buffered_delete_d"
    )


def test_legacy_requires_flag():
    for m in ("loo", "kfold5", "buffer", "delete_d"):
        try:
            _resolve_cv_method({"cv": {"method": m}})
            raise AssertionError(f"expected ValueError for {m}")
        except ValueError:
            pass
        assert _resolve_cv_method({"cv": {"method": m, "allow_legacy": True}}) == m


if __name__ == "__main__":
    test_default_and_allowed()
    test_legacy_requires_flag()
    print("ok")
