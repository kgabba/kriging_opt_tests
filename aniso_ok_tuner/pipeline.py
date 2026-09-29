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

from aniso_idw_mvp.directional import (  # noqa: E402
    fit_major_minor,
    fit_omnidirectional_variogram,
)
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
    make_outer_holdout_split,
    make_spatial_block_splits,
    score_outer_holdout,
    spatial_block_score,
)
from .isotropy import explain_isotropy_selection, resolve_isotropy_mode, select_isotropy
from .search_space import build_search_space
from .tpe_search import run_tpe

# Production CV (default). Legacy methods need cv.allow_legacy: true.
ALLOWED_CV_METHODS = frozenset({"spatial_block", "buffered_delete_d"})
LEGACY_CV_METHODS = frozenset(
    {"loo", "kfold5", "kfold", "buffer", "delete_d"}
)


def _resolve_cv_method(cfg: dict[str, Any]) -> str:
    """Return normalized CV method; reject legacy unless allow_legacy."""
    raw = _cfg(cfg, "cv", "method", default="spatial_block")
    method = str(raw).strip().lower().replace("-", "_")
    if method == "buffer_delete_d":
        method = "buffered_delete_d"
    if method == "delete_d":
        method = "delete_d"

    allow_legacy = bool(_cfg(cfg, "cv", "allow_legacy", default=False))
    if method in ("spatial_block", "buffered_delete_d"):
        return method
    if method in LEGACY_CV_METHODS:
        if not allow_legacy:
            raise ValueError(
                f"cv.method={raw!r} is legacy. Allowed by default: "
                f"spatial_block, buffered_delete_d. "
                f"Set cv.allow_legacy: true only when explicitly requested."
            )
        return method
    raise ValueError(
        f"Unknown cv.method={raw!r}. Allowed: spatial_block, buffered_delete_d "
        f"(or legacy with allow_legacy: true: {sorted(LEGACY_CV_METHODS)})"
    )


def _dir_kw(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(
        tolerance_deg=float(_cfg(cfg, "directional", "tolerance_deg", default=20.0)),
        n_lags=int(_cfg(cfg, "directional", "n_lags", default=12)),
        max_dist_percentile=float(
            _cfg(cfg, "directional", "max_dist_percentile", default=50.0)
        ),
        min_pairs=int(_cfg(cfg, "directional", "min_pairs", default=8)),
    )


def _fit_omni(pairs, domain: str, out_dir: Path, cfg: dict[str, Any], z: np.ndarray):
    omni_vg = fit_omnidirectional_variogram(pairs, **_dir_kw(cfg))
    save_directional_variogram(
        omni_vg, out_dir / "variogram_omni.png", f"{domain} omni"
    )
    nugget_fit = float(max(omni_vg.nugget, 0.0))
    sill_partial = float(max(omni_vg.sill, 1e-12))
    sill_total = max(nugget_fit + sill_partial, float(np.var(z)), 1e-6)
    a_iso = float(omni_vg.range_)
    fixed = ModelFixed(
        alpha_deg=0.0,
        a_major=a_iso,
        a_minor=a_iso,
        sill_partial_base=sill_partial,
        sill_total=sill_total,
        nugget_fit=nugget_fit,
        isotropic=True,
    )
    directional = {
        "omnidirectional": {
            "range": omni_vg.range_,
            "nugget": omni_vg.nugget,
            "sill": omni_vg.sill,
        }
    }
    return fixed, directional, omni_vg


def _fit_aniso(pairs, domain: str, out_dir: Path, cfg: dict[str, Any], z: np.ndarray):
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

    major_vg, minor_vg = fit_major_minor(
        pairs, moi.major_angle_deg, moi.minor_angle_deg, **_dir_kw(cfg)
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

    save_directional_variogram(
        major_vg, out_dir / "variogram_major.png", f"{domain} major"
    )
    save_directional_variogram(
        minor_vg, out_dir / "variogram_minor.png", f"{domain} minor"
    )

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
        isotropic=False,
    )
    moi_payload = {
        "major_angle_deg": moi.major_angle_deg,
        "minor_angle_deg": moi.minor_angle_deg,
        "k_moi": moi.k_moi,
        "eigenvalue_ratio": moi.eigenvalue_ratio,
    }
    directional = {
        "major": {
            "range": major_vg.range_,
            "nugget": major_vg.nugget,
            "sill": major_vg.sill,
            "n_lags": int(major_vg.lag.size),
            "n_pairs_total": int(np.sum(major_vg.n_pairs)) if major_vg.n_pairs.size else 0,
        },
        "minor": {
            "range": minor_vg.range_,
            "nugget": minor_vg.nugget,
            "sill": minor_vg.sill,
            "n_lags": int(minor_vg.lag.size),
            "n_pairs_total": int(np.sum(minor_vg.n_pairs)) if minor_vg.n_pairs.size else 0,
        },
    }
    return fixed, moi_payload, directional


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
    mode = resolve_isotropy_mode(cfg)
    ratio_min = float(_cfg(cfg, "auto_isotropy", "anisotropy_ratio_min", default=1.3))
    nugget_frac_iso = float(_cfg(cfg, "auto_isotropy", "nugget_frac_iso", default=0.45))
    aniso_min_points = int(_cfg(cfg, "auto_isotropy", "aniso_min_points", default=80))
    min_directional_lags = int(
        _cfg(cfg, "auto_isotropy", "min_directional_lags", default=4)
    )
    sill_ratio_max = float(_cfg(cfg, "auto_isotropy", "sill_ratio_max", default=2.5))

    moi_payload: dict[str, Any] | None = None
    directional: dict[str, Any] = {}
    selection: dict[str, Any]

    if mode == "force_iso":
        fixed, directional, _ = _fit_omni(pairs, domain, out_dir, cfg, z)
        selection = {
            "mode": mode,
            "isotropic": True,
            "reason": "force_iso",
            "metrics": None,
            "explanation": explain_isotropy_selection(
                isotropic=True, reason="force_iso", mode=mode
            ),
        }
    elif mode == "force_aniso":
        fixed, moi_payload, directional = _fit_aniso(pairs, domain, out_dir, cfg, z)
        selection = {
            "mode": mode,
            "isotropic": False,
            "reason": "force_aniso",
            "metrics": None,
            "explanation": explain_isotropy_selection(
                isotropic=False, reason="force_aniso", mode=mode
            ),
        }
    else:
        # auto: omni for noise; MOI+directional for structure; then pick
        fixed_iso, dir_omni, _ = _fit_omni(pairs, domain, out_dir, cfg, z)
        fixed_aniso, moi_payload, dir_aniso = _fit_aniso(
            pairs, domain, out_dir, cfg, z
        )
        maj = dir_aniso.get("major") or {}
        mn = dir_aniso.get("minor") or {}
        use_iso, reason, metrics = select_isotropy(
            a_major=fixed_aniso.a_major,
            a_minor=fixed_aniso.a_minor,
            nugget=fixed_iso.nugget_fit,
            sill_total=fixed_iso.sill_total,
            n_points=n,
            n_lags_major=maj.get("n_lags"),
            n_lags_minor=mn.get("n_lags"),
            sill_major=maj.get("sill"),
            sill_minor=mn.get("sill"),
            nugget_major=maj.get("nugget"),
            nugget_minor=mn.get("nugget"),
            anisotropy_ratio_min=ratio_min,
            nugget_frac_iso=nugget_frac_iso,
            aniso_min_points=aniso_min_points,
            min_directional_lags=min_directional_lags,
            sill_ratio_max=sill_ratio_max,
        )
        # keep both VG diagnostics in the report
        directional = {**dir_omni, **dir_aniso}
        if use_iso:
            fixed = fixed_iso
        else:
            fixed = fixed_aniso
        explanation = explain_isotropy_selection(
            isotropic=bool(use_iso),
            reason=reason,
            metrics=metrics,
            mode="auto",
        )
        selection = {
            "mode": "auto",
            "isotropic": bool(use_iso),
            "reason": reason,
            "metrics": metrics,
            "thresholds": {
                "anisotropy_ratio_min": ratio_min,
                "nugget_frac_iso": nugget_frac_iso,
                "aniso_min_points": aniso_min_points,
                "min_directional_lags": min_directional_lags,
                "sill_ratio_max": sill_ratio_max,
            },
            "explanation": explanation,
        }
        print(
            f"[{domain}] auto isotropy → "
            f"{'ISO' if use_iso else 'ANISO'} ({reason}); {explanation}",
            flush=True,
        )

    fixed_model: dict[str, Any] = {
        "alpha_deg": fixed.alpha_deg,
        "a_major": fixed.a_major,
        "a_minor": fixed.a_minor,
        "nugget_fit": fixed.nugget_fit,
        "sill_total": fixed.sill_total,
        "isotropic": fixed.isotropic,
    }
    if fixed.isotropic:
        fixed_model["a_iso"] = fixed.a_major

    payload_base = {
        "domain": domain,
        "n_points": n,
        "isotropy_selection": selection,
        "moi": moi_payload,
        "directional": directional,
        "fixed_model": fixed_model,
    }

    if n < n_min:
        payload_base["optimization"] = {
            "status": "SKIPPED",
            "reason": f"n_points={n} < n_min={n_min}",
        }
        (out_dir / "best_params.json").write_text(
            json.dumps(payload_base, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_domain_summary_md(out_dir, payload_base)
        return payload_base

    space = build_search_space(
        fixed,
        radius_margin=float(_cfg(cfg, "search_space", "radius_margin", default=0.90)),
        nugget_margin=float(_cfg(cfg, "search_space", "nugget_margin", default=0.60)),
        range_scale_min=float(_cfg(cfg, "search_space", "range_scale_min", default=0.10)),
        range_scale_max=float(_cfg(cfg, "search_space", "range_scale_max", default=1.90)),
        nmax_min=int(_cfg(cfg, "search_space", "nmax_min", default=5)),
        nmax_max=int(_cfg(cfg, "search_space", "nmax_max", default=80)),
        radius_prior_scale=float(
            _cfg(cfg, "search_space", "radius_prior_scale", default=1.0)
        ),
    )

    penalty = float(_cfg(cfg, "optuna", "invalid_penalty", default=1e6))
    cv_method = _resolve_cv_method(cfg)
    cv_seed = int(_cfg(cfg, "cv", "seed", default=_cfg(cfg, "delete_d", "seed", default=42)))

    # Outer holdout for overfit monitor / early stopping (Optuna CV runs on remainder).
    outer_enabled = bool(_cfg(cfg, "outer_holdout", "enabled", default=True))
    outer_meta: dict[str, Any] | None = None
    train_idx: np.ndarray | None = None
    outer_idx: np.ndarray | None = None
    outer_score_fn = None
    outer_patience: int | None = None
    xy_cv, z_cv, n_cv = xy, z, n
    if outer_enabled:
        split = make_outer_holdout_split(
            n,
            fraction=float(_cfg(cfg, "outer_holdout", "fraction", default=0.15)),
            min_points=int(_cfg(cfg, "outer_holdout", "min_points", default=10)),
            max_fraction=float(_cfg(cfg, "outer_holdout", "max_fraction", default=0.20)),
            min_train=int(_cfg(cfg, "outer_holdout", "min_train", default=20)),
            seed=int(_cfg(cfg, "outer_holdout", "seed", default=43)),
        )
        if split is None:
            print(
                f"[{domain}] outer_holdout skipped (n={n} too small)",
                flush=True,
            )
        else:
            train_idx, outer_idx = split
            xy_cv = xy[train_idx]
            z_cv = z[train_idx]
            n_cv = int(len(train_idx))
            outer_patience = int(_cfg(cfg, "outer_holdout", "patience", default=300))
            outer_meta = {
                "enabled": True,
                "n_full": n,
                "n_train": n_cv,
                "n_outer": int(len(outer_idx)),
                "fraction": float(len(outer_idx)) / float(n),
                "patience": outer_patience,
                "seed": int(_cfg(cfg, "outer_holdout", "seed", default=43)),
                "note": "VG/MOI on full domain; Optuna CV on train; best θ by outer RMSE",
            }
            _tr, _ou = train_idx, outer_idx

            def outer_score_fn(theta, _tr=_tr, _ou=_ou):  # noqa: F811
                return score_outer_holdout(
                    xy, z, _tr, _ou, theta, fixed, invalid_penalty=penalty
                )

            print(
                f"[{domain}] outer_holdout n_outer={len(outer_idx)} "
                f"n_train={n_cv} patience={outer_patience}",
                flush=True,
            )

    if cv_method == "loo":
        def objective_fn(theta):
            return loo_score(xy_cv, z_cv, theta, fixed, invalid_penalty=penalty)

        cv_label = f"loo n={n_cv}"
    elif cv_method == "kfold5":
        n_folds = int(_cfg(cfg, "cv", "n_folds", default=5))
        test_folds = make_kfold_splits(n_cv, n_folds=n_folds, seed=cv_seed)

        def objective_fn(theta):
            return kfold_score(
                xy_cv, z_cv, theta, fixed, test_folds, invalid_penalty=penalty
            )

        cv_label = f"kfold{n_folds} n={n_cv} folds={len(test_folds)}"
    elif cv_method == "spatial_block":
        grid_nx = int(_cfg(cfg, "cv", "grid_nx", default=3))
        grid_ny = int(_cfg(cfg, "cv", "grid_ny", default=3))
        test_folds = make_spatial_block_splits(xy_cv, grid_nx=grid_nx, grid_ny=grid_ny)
        if not test_folds:
            raise RuntimeError(f"spatial_block produced no folds for domain={domain}")

        def objective_fn(theta):
            return spatial_block_score(
                xy_cv, z_cv, theta, fixed, test_folds,
                invalid_penalty=penalty, grid_nx=grid_nx, grid_ny=grid_ny,
            )

        cv_label = f"spatial_block {grid_nx}x{grid_ny} n={n_cv} folds={len(test_folds)}"
    elif cv_method == "buffer":
        buffer_radius = float(_cfg(cfg, "cv", "buffer_radius", default=0.4))

        def objective_fn(theta):
            return buffer_loo_score(
                xy_cv, z_cv, theta, fixed,
                buffer_radius=buffer_radius, invalid_penalty=penalty,
            )

        cv_label = f"buffer_loo r={buffer_radius} n={n_cv}"
    elif cv_method in ("buffered_delete_d", "buffer_delete_d"):
        holdout = _cfg(cfg, "cv", "holdout_size", default=None)
        if holdout is None:
            holdout = _cfg(cfg, "delete_d", "holdout_size", default=None)
        holdout = default_holdout_size(n_cv) if holdout is None else int(holdout)
        holdout = min(holdout, n_cv - 2)
        n_repeats = int(
            _cfg(
                cfg,
                "cv",
                "n_repeats",
                default=_cfg(cfg, "delete_d", "n_repeats", default=25),
            )
        )
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
            xy_cv,
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
                xy_cv, z_cv, theta, fixed, folds,
                buffer_radius=buffer_radius,
                holdout_size=holdout,
                invalid_penalty=penalty,
                buffer_metric=buffer_metric,
                buffer_scale=float(buffer_scale) if buffer_scale is not None else None,
            )

        cv_label = (
            f"buffered_delete_d n={n_cv} d={holdout} r={buffer_radius:.4g} "
            f"metric={buffer_metric} scale={buffer_scale} "
            f"repeats={len(folds)}/{n_repeats}"
        )
    else:
        holdout = _cfg(cfg, "delete_d", "holdout_size", default=None)
        holdout = default_holdout_size(n_cv) if holdout is None else int(holdout)
        holdout = min(holdout, n_cv - 2)
        splits = make_delete_d_splits(
            n_cv,
            holdout_size=holdout,
            n_repeats=int(_cfg(cfg, "delete_d", "n_repeats", default=25)),
            seed=int(_cfg(cfg, "delete_d", "seed", default=cv_seed)),
        )

        def objective_fn(theta):
            return delete_d_score(
                xy_cv, z_cv, theta, fixed, splits, invalid_penalty=penalty
            )

        cv_label = f"delete-d n={n_cv} d={holdout} repeats={len(splits)}"

    print(
        f"[{domain}] OK Optuna cv={cv_method} {cv_label} "
        f"iso={fixed.isotropic} "
        f"sel={selection.get('reason')} "
        f"alpha={fixed.alpha_deg:.2f}° "
        f"a_maj/min={fixed.a_major:.4g}/{fixed.a_minor:.4g}"
        + (f" (omni a_iso={fixed.a_major:.4g})" if fixed.isotropic else ""),
        flush=True,
    )
    opt = run_tpe(
        space=space,
        fixed=fixed,
        objective_fn=objective_fn,
        n_trials=int(_cfg(cfg, "optuna", "n_trials", default=1200)),
        seed=int(_cfg(cfg, "optuna", "seed", default=42)),
        out_csv=out_dir / "trials.csv",
        best_json=out_dir / "best_params_optuna.json",
        n_startup_trials=_cfg(cfg, "optuna", "n_startup_trials", default=None),
        n_ei_candidates=int(_cfg(cfg, "optuna", "n_ei_candidates", default=64)),
        outer_score_fn=outer_score_fn,
        outer_patience=outer_patience,
    )
    if outer_meta is not None:
        opt["outer_holdout"] = outer_meta
    payload_base["optimization"] = opt
    (out_dir / "best_params.json").write_text(
        json.dumps(payload_base, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_domain_summary_md(out_dir, payload_base)
    return payload_base


def _write_domain_summary_md(out_dir: Path, payload: dict[str, Any]) -> None:
    """Short human-readable domain report (isotropy choice + CV best)."""
    sel = payload.get("isotropy_selection") or {}
    fx = payload.get("fixed_model") or {}
    opt = payload.get("optimization") or {}
    lines = [
        f"# Domain `{payload.get('domain')}`",
        "",
        f"- n_points: **{payload.get('n_points')}**",
        f"- geometry: **{'ISO' if fx.get('isotropic') else 'ANISO'}**",
    ]
    expl = sel.get("explanation")
    if expl:
        lines += ["", f"**Почему:** {expl}"]
    elif sel.get("reason"):
        lines += ["", f"**Почему:** `{sel.get('reason')}`"]
    if opt.get("status") == "SKIPPED":
        lines += ["", f"**Skipped:** {opt.get('reason')}"]
    elif opt:
        th = opt.get("best_theta") or {}
        sel_m = opt.get("selection_metric") or "cv"
        lines += [
            "",
            f"- best by: **{sel_m}**",
            f"- CV / outer best RMSE: **{opt.get('best_delete_d_rmse')}** "
            f"(eval {opt.get('best_evaluation')})",
        ]
        if opt.get("best_outer_rmse") is not None:
            lines.append(
                f"- outer RMSE: **{opt.get('best_outer_rmse')}** "
                f"(eval {opt.get('best_outer_evaluation')}); "
                f"CV at that θ: {opt.get('best_cv_rmse')}"
            )
        oh = opt.get("outer_holdout") or {}
        if oh:
            lines.append(
                f"- outer holdout: n={oh.get('n_outer')}/{oh.get('n_full')} "
                f"patience={oh.get('patience')}"
            )
        if opt.get("early_stopped"):
            lines.append(f"- early stop: {opt.get('stop_reason')}")
        lines.append(f"- best θ: `{th}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    data_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Tune OK for every Domain in CSV; write root summary.json + summary.md."""
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

    md_lines = [
        "# OK tune summary",
        "",
        f"- train: `{data_path}`",
        "",
        "| Domain | n | Geom | Почему | CV RMSE |",
        "|--------|---|------|--------|---------|",
    ]
    for d in summaries:
        sel = d.get("isotropy_selection") or {}
        fx = d.get("fixed_model") or {}
        opt = d.get("optimization") or {}
        geom = "ISO" if fx.get("isotropic") else "ANISO"
        why = (sel.get("explanation") or sel.get("reason") or "—").replace("|", "/")
        cv = opt.get("best_delete_d_rmse")
        cv_s = f"{cv:.4f}" if isinstance(cv, (int, float)) else (opt.get("status") or "—")
        md_lines.append(
            f"| {d.get('domain')} | {d.get('n_points')} | {geom} | {why} | {cv_s} |"
        )
    (root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary
