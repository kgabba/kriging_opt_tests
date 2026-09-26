"""Per-domain orchestration: MOI variography → OK Optuna with trial logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# sibling MVP for variography / reporting figures
_MVP_ROOT = Path(__file__).resolve().parents[2] / "aniso_idw_mvp"
if _MVP_ROOT.exists() and str(_MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVP_ROOT))

from aniso_idw_mvp.directional import fit_major_minor  # noqa: E402
from aniso_idw_mvp.directional_rose import build_variogram_rose  # noqa: E402
from aniso_idw_mvp.domain import iter_domains, safe_dirname  # noqa: E402
from aniso_idw_mvp.io import load_points  # noqa: E402
from aniso_idw_mvp.moi import MOIResult, estimate_moi  # noqa: E402
from aniso_idw_mvp.pairs import compute_pairs  # noqa: E402
from aniso_idw_mvp.report import (  # noqa: E402
    save_directional_variogram,
    save_variogram_map,
    save_variogram_rose,
)
from aniso_idw_mvp.variogram_map import build_variogram_map  # noqa: E402

from .cv import (
    ModelFixed,
    buffer_loo_score,
    buffered_delete_d_score,
    default_holdout_size,
    delete_d_score,
    kfold_score,
    loo_score,
    make_buffered_delete_d_folds,
    make_delete_d_splits,
    make_kfold_splits,
    make_spatial_block_splits,
    spatial_block_score,
)
from .search_space import build_search_space
from .tpe_search import run_tpe


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg(cfg: dict, *keys, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_domain(
    domain: str,
    df_domain: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Variography + Optuna OK tune for one Domain (isolated)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xy = df_domain[["X", "Y"]].to_numpy(dtype=float)
    z = df_domain["Grade"].to_numpy(dtype=float)
    n = len(df_domain)
    n_min = int(_cfg(cfg, "n_min_points", default=15))

    pairs = compute_pairs(xy, z)
    vmap = build_variogram_map(
        pairs,
        n_lags=int(_cfg(cfg, "variogram_map", "n_lags", default=16)),
        max_dist_percentile=float(
            _cfg(cfg, "variogram_map", "max_dist_percentile", default=50.0)
        ),
    )
    save_variogram_map(vmap, out_dir / "variogram_map.png", title=f"{domain} map")

    moi = estimate_moi(
        vmap,
        max_lag_fraction=float(_cfg(cfg, "moi", "max_lag_fraction", default=0.8)),
        eps=float(_cfg(cfg, "moi", "eps", default=1e-8)),
    )
    rose = build_variogram_rose(
        pairs,
        step_deg=float(_cfg(cfg, "rose", "step_deg", default=15.0)),
        n_lags=int(_cfg(cfg, "rose", "n_lags", default=10)),
        max_dist_percentile=float(
            _cfg(cfg, "rose", "max_dist_percentile", default=50.0)
        ),
        min_pairs=int(_cfg(cfg, "rose", "min_pairs", default=10)),
    )
    save_variogram_rose(
        rose,
        out_dir / "variogram_rose.png",
        major_angle_deg=moi.major_angle_deg,
        title=f"{domain} rose",
    )

    dir_kw = dict(
        tolerance_deg=float(_cfg(cfg, "directional", "tolerance_deg", default=20.0)),
        n_lags=int(_cfg(cfg, "directional", "n_lags", default=12)),
        max_dist_percentile=float(
            _cfg(cfg, "directional", "max_dist_percentile", default=50.0)
        ),
        min_pairs=int(_cfg(cfg, "directional", "min_pairs", default=8)),
    )
    major_vg, minor_vg = fit_major_minor(
        pairs, moi.major_angle_deg, moi.minor_angle_deg, **dir_kw
    )
    if major_vg.range_ < minor_vg.range_:
        major_vg, minor_vg = minor_vg, major_vg
        moi = MOIResult(
            major_angle_deg=moi.minor_angle_deg,
            minor_angle_deg=moi.major_angle_deg,
            eigenvalue_ratio=moi.eigenvalue_ratio,
            k_moi=moi.k_moi,
            tensor=moi.tensor,
        )

    save_directional_variogram(major_vg, out_dir / "variogram_major.png", f"{domain} major")
    save_directional_variogram(minor_vg, out_dir / "variogram_minor.png", f"{domain} minor")

    # sill: prefer fit total, floor by sample variance
    nugget_fit = float(max(major_vg.nugget, 0.0))
    sill_partial = float(max(major_vg.sill, 1e-12))
    sill_total = max(nugget_fit + sill_partial, float(np.var(z)), 1e-6)

    fixed = ModelFixed(
        alpha_deg=float(moi.major_angle_deg),
        a_major=float(major_vg.range_),
        a_minor=float(minor_vg.range_),
        sill_partial_base=sill_partial,
        sill_total=sill_total,
        nugget_fit=nugget_fit,
        isotropic=bool(_cfg(cfg, "isotropic", default=False)),
    )

    payload_base = {
        "domain": domain,
        "n_points": n,
        "moi": {
            "major_angle_deg": moi.major_angle_deg,
            "minor_angle_deg": moi.minor_angle_deg,
            "k_moi": moi.k_moi,
        },
        "directional": {
            "major": {
                "range": major_vg.range_,
                "nugget": major_vg.nugget,
                "sill": major_vg.sill,
            },
            "minor": {
                "range": minor_vg.range_,
                "nugget": minor_vg.nugget,
                "sill": minor_vg.sill,
            },
        },
        "fixed_model": {
            "alpha_deg": fixed.alpha_deg,
            "a_major": fixed.a_major,
            "a_minor": fixed.a_minor,
            "nugget_fit": fixed.nugget_fit,
            "sill_total": fixed.sill_total,
        },
    }

    if n < n_min:
        payload_base["optimization"] = {
            "status": "SKIPPED",
            "reason": f"n_points={n} < n_min={n_min}",
        }
        (out_dir / "best_params.json").write_text(
            json.dumps(payload_base, indent=2), encoding="utf-8"
        )
        return payload_base

    space = build_search_space(
        fixed,
        radius_margin=float(_cfg(cfg, "search_space", "radius_margin", default=0.20)),
        nugget_margin=float(_cfg(cfg, "search_space", "nugget_margin", default=0.20)),
        range_scale_min=float(_cfg(cfg, "search_space", "range_scale_min", default=0.8)),
        range_scale_max=float(_cfg(cfg, "search_space", "range_scale_max", default=1.2)),
        nmax_min=int(_cfg(cfg, "search_space", "nmax_min", default=5)),
        nmax_max=int(_cfg(cfg, "search_space", "nmax_max", default=40)),
        radius_prior_scale=float(
            _cfg(cfg, "search_space", "radius_prior_scale", default=1.0)
        ),
    )

    penalty = float(_cfg(cfg, "optuna", "invalid_penalty", default=1e6))
    cv_method = str(_cfg(cfg, "cv", "method", default="delete_d")).lower()
    cv_seed = int(_cfg(cfg, "cv", "seed", default=_cfg(cfg, "delete_d", "seed", default=42)))

    if cv_method == "loo":
        def objective_fn(theta):
            return loo_score(xy, z, theta, fixed, invalid_penalty=penalty)

        cv_label = f"loo n={n}"
    elif cv_method == "kfold5":
        n_folds = int(_cfg(cfg, "cv", "n_folds", default=5))
        test_folds = make_kfold_splits(n, n_folds=n_folds, seed=cv_seed)

        def objective_fn(theta):
            return kfold_score(xy, z, theta, fixed, test_folds, invalid_penalty=penalty)

        cv_label = f"kfold{n_folds} n={n} folds={len(test_folds)}"
    elif cv_method == "spatial_block":
        grid_nx = int(_cfg(cfg, "cv", "grid_nx", default=3))
        grid_ny = int(_cfg(cfg, "cv", "grid_ny", default=3))
        test_folds = make_spatial_block_splits(xy, grid_nx=grid_nx, grid_ny=grid_ny)
        if not test_folds:
            raise RuntimeError(f"spatial_block produced no folds for domain={domain}")

        def objective_fn(theta):
            return spatial_block_score(
                xy, z, theta, fixed, test_folds,
                invalid_penalty=penalty, grid_nx=grid_nx, grid_ny=grid_ny,
            )

        cv_label = f"spatial_block {grid_nx}x{grid_ny} n={n} folds={len(test_folds)}"
    elif cv_method == "buffer":
        buffer_radius = float(_cfg(cfg, "cv", "buffer_radius", default=0.4))

        def objective_fn(theta):
            return buffer_loo_score(
                xy, z, theta, fixed,
                buffer_radius=buffer_radius, invalid_penalty=penalty,
            )

        cv_label = f"buffer_loo r={buffer_radius} n={n}"
    elif cv_method in ("buffered_delete_d", "buffer_delete_d"):
        holdout = _cfg(cfg, "cv", "holdout_size", default=None)
        if holdout is None:
            holdout = _cfg(cfg, "delete_d", "holdout_size", default=None)
        holdout = default_holdout_size(n) if holdout is None else int(holdout)
        holdout = min(holdout, n - 2)
        n_repeats = int(
            _cfg(
                cfg,
                "cv",
                "n_repeats",
                default=_cfg(cfg, "delete_d", "n_repeats", default=25),
            )
        )
        # Prefer scale×a_major (aniso-aware); else absolute buffer_radius (legacy / iso)
        buffer_scale = _cfg(cfg, "cv", "buffer_scale", default=None)
        use_aniso_metric = bool(
            _cfg(cfg, "cv", "buffer_anisotropic", default=not fixed.isotropic)
        ) and (not fixed.isotropic)
        if buffer_scale is not None:
            buffer_radius = float(buffer_scale) * float(fixed.a_major)
            buffer_metric = "aniso_ok" if use_aniso_metric else "euclidean"
        else:
            buffer_radius = float(_cfg(cfg, "cv", "buffer_radius", default=0.4))
            buffer_metric = "aniso_ok" if use_aniso_metric else "euclidean"
            buffer_scale = None

        folds = make_buffered_delete_d_folds(
            xy,
            holdout_size=holdout,
            n_repeats=n_repeats,
            buffer_radius=buffer_radius,
            seed=cv_seed,
            anisotropic=use_aniso_metric,
            alpha_deg=fixed.alpha_deg if use_aniso_metric else None,
            a_major=fixed.a_major if use_aniso_metric else None,
            a_minor=fixed.a_minor if use_aniso_metric else None,
        )
        if not folds:
            raise RuntimeError(
                f"buffered_delete_d produced no folds for domain={domain} "
                f"(d={holdout}, r={buffer_radius}, metric={buffer_metric})"
            )

        def objective_fn(theta):
            return buffered_delete_d_score(
                xy, z, theta, fixed, folds,
                buffer_radius=buffer_radius,
                holdout_size=holdout,
                invalid_penalty=penalty,
                buffer_metric=buffer_metric,
                buffer_scale=float(buffer_scale) if buffer_scale is not None else None,
            )

        cv_label = (
            f"buffered_delete_d n={n} d={holdout} r={buffer_radius:.4g} "
            f"metric={buffer_metric} scale={buffer_scale} "
            f"repeats={len(folds)}/{n_repeats}"
        )
    else:
        # default: delete-d (legacy)
        holdout = _cfg(cfg, "delete_d", "holdout_size", default=None)
        holdout = default_holdout_size(n) if holdout is None else int(holdout)
        holdout = min(holdout, n - 2)
        splits = make_delete_d_splits(
            n,
            holdout_size=holdout,
            n_repeats=int(_cfg(cfg, "delete_d", "n_repeats", default=25)),
            seed=int(_cfg(cfg, "delete_d", "seed", default=cv_seed)),
        )

        def objective_fn(theta):
            return delete_d_score(xy, z, theta, fixed, splits, invalid_penalty=penalty)

        cv_label = f"delete-d n={n} d={holdout} repeats={len(splits)}"

    print(
        f"[{domain}] OK Optuna cv={cv_method} {cv_label} "
        f"iso={fixed.isotropic} "
        f"alpha={fixed.alpha_deg:.2f}° "
        f"a_maj/min={fixed.a_major:.4g}/{fixed.a_minor:.4g}",
        flush=True,
    )
    opt = run_tpe(
        space=space,
        fixed=fixed,
        objective_fn=objective_fn,
        n_trials=int(_cfg(cfg, "optuna", "n_trials", default=120)),
        seed=int(_cfg(cfg, "optuna", "seed", default=42)),
        out_csv=out_dir / "trials.csv",
        best_json=out_dir / "best_params_optuna.json",
        n_startup_trials=_cfg(cfg, "optuna", "n_startup_trials", default=None),
        n_ei_candidates=int(_cfg(cfg, "optuna", "n_ei_candidates", default=64)),
    )
    payload_base["optimization"] = opt
    (out_dir / "best_params.json").write_text(
        json.dumps(payload_base, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload_base


def run_pipeline(
    data_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Tune OK for every Domain in CSV; write root summary.json."""
    cfg = load_config(config_path)
    df = load_points(data_path)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for domain, sub in iter_domains(df):
        print(f"=== Domain {domain} (n={len(sub)}) ===", flush=True)
        summaries.append(run_domain(domain, sub, root / safe_dirname(domain), cfg))

    summary = {"csv": str(data_path), "domains": summaries}
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
