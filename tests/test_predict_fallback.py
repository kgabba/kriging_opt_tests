"""Unit tests for predict_val auto domain-mean reporting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "predict_val", ROOT / "scripts" / "predict_val.py"
)
assert _SPEC and _SPEC.loader
_pv = importlib.util.module_from_spec(_SPEC)
sys.modules["predict_val"] = _pv
_SPEC.loader.exec_module(_pv)

_PREDICT_SECTION = _pv._PREDICT_SECTION
_append_predict_section = _pv._append_predict_section
_predict_footer_md = _pv._predict_footer_md
predict_domain = _pv.predict_domain


def test_skipped_counts_as_auto_mean(tmp_path: Path | None = None):
    train = pd.DataFrame(
        {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Grade": [10.0, 20.0]}
    )
    pred_xy = np.array([[0.5, 0.5], [2.0, 2.0]])
    payload = {"optimization": {"status": "SKIPPED", "reason": "test"}}
    pred, st = predict_domain(train, pred_xy, payload)
    assert st["n_auto_mean"] == 2
    assert st["n_ok"] == 0
    assert st["source"] == "skipped_domain_mean"
    assert np.allclose(pred, 15.0)


def test_summary_footer_replace(tmp_path: Path):
    md = tmp_path / "summary.md"
    md.write_text("# OK tune summary\n\n| Domain | n |\n", encoding="utf-8")
    domains = [
        {
            "domain": "Argovian",
            "n_pred": 40,
            "n_ok": 38,
            "n_auto_mean": 2,
            "source": "ok_with_auto_mean",
        },
        {
            "domain": "Other",
            "n_pred": 60,
            "n_ok": 60,
            "n_auto_mean": 0,
            "source": "ok_with_auto_mean",
        },
    ]
    footer = _predict_footer_md(domains, Path("out.csv"))
    _append_predict_section(md, footer)
    text = md.read_text(encoding="utf-8")
    assert _PREDICT_SECTION in text
    assert "auto domain-mean fills: 2 / 100" in text
    # second append replaces, does not duplicate
    _append_predict_section(md, footer)
    assert md.read_text(encoding="utf-8").count(_PREDICT_SECTION) == 1


if __name__ == "__main__":
    from pathlib import Path as P
    import tempfile

    test_skipped_counts_as_auto_mean()
    with tempfile.TemporaryDirectory() as d:
        test_summary_footer_replace(P(d))
    print("ok")
