#!/usr/bin/env python3
"""Predict jura_val_loc with best OK params from a tune out-dir → V1,V2 CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MVP = ROOT.parent / "aniso_idw_mvp"
if MVP.exists() and str(MVP) not in sys.path:
    sys.path.insert(0, str(MVP))

from aniso_idw_mvp.io import load_points  # noqa: E402
from aniso_ok_tuner.kriging import (  # noqa: E402
    ordinary_kriging_elliptical,
    ordinary_kriging_isotropic,
)

# Marker for replaceable footer in results-dir summary.md
_PREDICT_SECTION = "## Predict fallback (auto domain-mean)"


def _load_val(path: Path) -> pd.DataFrame:
    val = pd.read_csv(path)
    val = val.rename(columns={c: str(c).strip('"') for c in val.columns})
    if "seq(1:100)" in val.columns:
        val = val.rename(columns={"seq(1:100)": "id"})
    elif val.columns[0] not in ("id", "V1"):
        val = val.rename(columns={val.columns[0]: "id"})
    return val


def _domain_payload(results_dir: Path, domain: str) -> dict:
    p = results_dir / domain / "best_params.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def predict_domain(
    train_df: pd.DataFrame,
    pred_xy: np.ndarray,
    payload: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    """OK predict; NaN → domain-mean (auto). Returns (preds, stats)."""
    n = int(pred_xy.shape[0])
    opt = payload.get("optimization") or {}
    if opt.get("status") == "SKIPPED":
        z = train_df["Grade"].to_numpy(float)
        mean_z = float(np.mean(z))
        return np.full(n, mean_z), {
            "n_pred": n,
            "n_ok": 0,
            "n_auto_mean": n,
            "source": "skipped_domain_mean",
        }

    fixed = opt.get("fixed") or payload.get("fixed_model") or {}
    theta = opt.get("best_theta") or {}
    if not theta:
        raise ValueError(f"missing best_theta in domain={payload.get('domain')}")

    sill_total = float(fixed["sill_total"])
    nug = float(np.clip(float(theta["nugget"]), 0.0, 0.95 * sill_total))
    sil = max(sill_total - nug, 1e-12)
    xy = train_df[["X", "Y"]].to_numpy(float)
    z = train_df["Grade"].to_numpy(float)
    isotropic = bool(fixed.get("isotropic", opt.get("isotropic", False)))

    if isotropic:
        pred = ordinary_kriging_isotropic(
            xy, z, pred_xy,
            range_=float(fixed["a_major"]),
            nugget=nug,
            sill=sil,
            range_scale=float(theta["range_scale"]),
            radius=float(theta.get("R", theta["R_major"])),
            n_max=int(theta["N_max"]),
        )
    else:
        pred = ordinary_kriging_elliptical(
            xy, z, pred_xy,
            alpha_deg=float(fixed["alpha_deg"]),
            a_major=float(fixed["a_major"]),
            a_minor=float(fixed["a_minor"]),
            nugget=nug,
            sill=sil,
            range_scale=float(theta["range_scale"]),
            r_major=float(theta["R_major"]),
            r_minor=float(theta["R_minor"]),
            n_max=int(theta["N_max"]),
        )
    ok = np.isfinite(pred)
    n_ok = int(ok.sum())
    n_auto = n - n_ok
    mean_z = float(np.mean(z))
    filled = np.where(ok, pred, mean_z)
    return filled, {
        "n_pred": n,
        "n_ok": n_ok,
        "n_auto_mean": n_auto,
        "source": "ok_with_auto_mean",
    }


def _predict_footer_md(domains: list[dict[str, Any]], out_csv: Path) -> str:
    n_pred = sum(int(d["n_pred"]) for d in domains)
    n_auto = sum(int(d["n_auto_mean"]) for d in domains)
    n_ok = sum(int(d["n_ok"]) for d in domains)
    lines = [
        _PREDICT_SECTION,
        "",
        "Если OK вернул NaN (или domain SKIPPED) — **авто** подстановка "
        "среднего Grade домена (для submission). Не отдельная модель.",
        "",
        f"- predict CSV: `{out_csv}`",
        f"- **auto domain-mean fills: {n_auto} / {n_pred}** (OK: {n_ok})",
        "",
        "| Domain | n_pred | n_ok | n_auto_mean | source |",
        "|--------|--------|------|-------------|--------|",
    ]
    for d in domains:
        lines.append(
            f"| {d['domain']} | {d['n_pred']} | {d['n_ok']} | "
            f"{d['n_auto_mean']} | `{d['source']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _append_predict_section(summary_md: Path, footer: str) -> None:
    """Append or replace the Predict fallback section at end of summary.md."""
    if summary_md.exists():
        text = summary_md.read_text(encoding="utf-8")
        if _PREDICT_SECTION in text:
            text = text.split(_PREDICT_SECTION)[0].rstrip() + "\n\n"
        else:
            text = text.rstrip() + "\n\n"
    else:
        text = "# OK tune summary\n\n"
    summary_md.write_text(text + footer + "\n", encoding="utf-8")


def run_predict(
    *,
    train_csv: Path,
    val_csv: Path,
    results_dir: Path,
    out_csv: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = load_points(train_csv)
    val = _load_val(val_csv)
    preds = np.empty(len(val), dtype=float)
    is_arg = val["Rock"].astype(int) == 1
    domain_stats: list[dict[str, Any]] = []

    for domain, mask in [("Argovian", is_arg), ("Other", ~is_arg)]:
        payload = _domain_payload(results_dir, domain)
        sub = train.loc[train["Domain"] == domain]
        xy_pred = val.loc[mask, ["Xloc", "Yloc"]].to_numpy(float)
        if xy_pred.size == 0:
            continue
        filled, st = predict_domain(sub, xy_pred, payload)
        preds[mask.to_numpy()] = filled
        domain_stats.append({"domain": domain, **st})

    out = pd.DataFrame({"V1": val["id"].astype(int), "V2": preds})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    report = {
        "out_csv": str(out_csv),
        "n_pred": int(len(out)),
        "n_ok": sum(int(d["n_ok"]) for d in domain_stats),
        "n_auto_mean": sum(int(d["n_auto_mean"]) for d in domain_stats),
        "domains": domain_stats,
        "note": "auto domain-mean: NaN from OK (or SKIPPED) → mean Grade of domain",
    }
    (results_dir / "predict_fallback.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    footer = _predict_footer_md(domain_stats, out_csv)
    _append_predict_section(results_dir / "summary.md", footer)
    return out, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OK predict val → V1,V2 CSV")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    out, report = run_predict(
        train_csv=args.train,
        val_csv=args.val,
        results_dir=args.results_dir,
        out_csv=args.out,
    )
    print(
        f"Wrote {args.out} n={len(out)} mean={out.V2.mean():.4f} "
        f"nan={int(out.V2.isna().sum())} "
        f"auto_mean={report['n_auto_mean']}/{report['n_pred']}",
        flush=True,
    )
    print(
        f"Predict fallback (auto domain-mean): "
        f"{report['n_auto_mean']} / {report['n_pred']} "
        f"(see {args.results_dir / 'summary.md'})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
