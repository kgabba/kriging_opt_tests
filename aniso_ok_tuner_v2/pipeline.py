"""Per-domain nested OK pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_MVP = Path(__file__).resolve().parents[2] / "aniso_idw_mvp"
if _MVP.exists() and str(_MVP) not in sys.path:
    sys.path.insert(0, str(_MVP))

from aniso_idw_mvp.domain import iter_domains, safe_dirname  # noqa: E402
from aniso_idw_mvp.io import load_points  # noqa: E402

from .cv import default_holdout_size, make_delete_d_splits
from .outer_search import run_outer


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_domain(domain: str, df: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    xy = df[["X", "Y"]].to_numpy(float)
    z = df["Grade"].to_numpy(float)
    n = len(df)
    n_min = int(cfg.get("n_min_points", 15))
    if n < n_min:
        payload = {"domain": domain, "n_points": n, "status": "SKIPPED", "reason": f"n<{n_min}"}
        (out_dir / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    dd = cfg.get("delete_d", {})
    holdout = dd.get("holdout_size")
    holdout = default_holdout_size(n) if holdout is None else int(holdout)
    holdout = min(holdout, n - 2)
    splits = make_delete_d_splits(
        n,
        holdout_size=holdout,
        n_repeats=int(dd.get("n_repeats", 25)),
        seed=int(dd.get("seed", 42)),
    )
    print(
        f"[{domain}] nested OK outer={cfg.get('outer', {}).get('n_trials')} "
        f"inner={cfg.get('inner', {}).get('n_trials')} n={n} d={holdout} repeats={len(splits)}",
        flush=True,
    )
    result = run_outer(xy=xy, z=z, splits=splits, out_dir=out_dir, cfg=cfg)
    payload = {"domain": domain, "n_points": n, **result}
    (out_dir / "best_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return payload


def run_pipeline(data_path: str | Path, out_dir: str | Path, config_path: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    df = load_points(data_path)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for domain, sub in iter_domains(df):
        print(f"=== Domain {domain} (n={len(sub)}) ===", flush=True)
        summaries.append(run_domain(domain, sub, root / safe_dirname(domain), cfg))
    summary = {"csv": str(data_path), "domains": summaries}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
