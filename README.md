# Anisotropic Ordinary Kriging Tuner

Per-**Domain** Ordinary Kriging → **Optuna TPE** on neighbourhood / nugget /
`range_scale`, with **per-trial console + `trials.csv`**.

Variography comes from sibling [`aniso_idw_mvp`](https://github.com/kgabba/spatial-idw-opt).

## Iso vs aniso

| Config | Behaviour |
|--------|-----------|
| `isotropy_mode: auto` | Heuristic (default in `configs/default.yaml`) |
| `isotropy_mode: force_iso` | Always omni circle OK |
| `isotropy_mode: force_aniso` | Always MOI ellipse OK |
| legacy `isotropic: true/false` | Same as force_iso / force_aniso |

**Auto rules** (first match; thresholds under `auto_isotropy`):

1. **Few points** — `n < aniso_min_points` (default **80**) → ISO  
2. **Hard noise** — max(omni, directional) `nugget/sill ≥ nugget_frac_iso` (default 0.45) → ISO  
3. **Sparse directional VG** — major/minor lag bins `< min_directional_lags` (default 4) → ISO  
4. **Inconsistent sills** — major/minor sill ratio `> sill_ratio_max` (default 2.5) → ISO  
5. **Weak elongation** — `a_maj / a_min < anisotropy_ratio_min` (default 1.3) → ISO  
6. Else → aniso  

Selection is written to `best_params.json` → `isotropy_selection`
(`reason`, `metrics`, **`explanation`** — краткая фраза «почему ISO/ANISO»).
Also `summary.md` per domain and root `summary.md` with a Почему column.

- **isotropic path:** omnidirectional VG → `a_iso`  
- **anisotropic path:** MOI → major/minor directional ranges  

## What Optuna tunes

| Param | Window |
|-------|--------|
| `R` (iso) or `R_major` (aniso) | ±`radius_margin` of fitted ranges (default **±90%**; `a_iso` or directional major) |
| `N_max` | config box (default **5–80**) |
| `nugget` | ±`nugget_margin` of fitted C₀ (default **±60%**) |
| `range_scale` | config box (default **0.10–1.90**) |

**`radius_prior_scale` (config):** used **only in ISO** — centre of the `R` box is
`a_iso × radius_prior_scale` (often 1.5). **ANISO ignores it:** `R_major` box is
centred on `a_major` (scale 1.0), then `R_minor` locked to variogram ratio.
A `1.5` in an aniso/auto YAML does nothing for ANISO domains.

**Aniso search shape (locked):** `R_minor = R_major × (a_minor/a_major)` — same
anisotropy as the variogram. Optuna does **not** tune `R_minor` separately
(avoids mismatch vs `N_max` ranking / γ metric).

**Fixed:** iso → `a_iso` + sill; aniso → `alpha`, `a_major`/`a_minor`, sill.

**Neighbourhood:** prefer samples inside search `R` / ellipse (cap `N_max`).
If empty → soft rescue: **4** nearest (same metric), not a full `N_max` grab.

## CV methods (`cv.method`)

**Allowed by default** (production):

| Method | Idea |
|--------|------|
| `spatial_block` | leave-one-block-out on XY grid (`grid_nx` × `grid_ny`) |
| `buffered_delete_d` | random delete-d + exclude train within buffer (Euclidean or aniso OK metric; `buffer_scale × a_major`) |

**Legacy** (`loo`, `kfold5`, `buffer`, `delete_d`): require `cv.allow_legacy: true` in the YAML.
Do not use them unless the user explicitly asks. Default configs use `spatial_block`.

**Outer holdout / early stop** (`outer_holdout` in YAML, on by default): ~15% of
domain points held out of Optuna CV; each trial also scores that outer set.
Best θ is chosen by **outer RMSE**; stop if no outer improve for `patience`
(default **300**). Budget default **1200** trials.

Ready-made: `auto_cv_spatial_block`, Walker aniso spatial/buffered; legacy iso CV suite marked `allow_legacy`.

## Install

```bash
# sibling MVP for variography
pip install -e ../aniso_idw_mvp   # or clone spatial-idw-opt and point at its MVP path
pip install -e .
```

## Run

```bash
python tune.py --data path/to/points.csv --config configs/default.yaml --out-dir results/run
python tune.py --data path/to/points.csv --config configs/auto_cv_spatial_block.yaml \
  --out-dir results/walker_auto --trials 800
python tune.py --data path/to/points.csv --config configs/walker_aniso_spatial_block.yaml \
  --out-dir results/walker --trials 800
```

CSV columns: `X,Y,Grade` + optional `Domain` (missing → `DEFAULT`; each domain tuned independently).

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/predict_val.py` | predict val CSV from `best_params.json` |
| `scripts/run_cv_batch.py` | batch iso CV tune + predict |
| `scripts/score_walker_exhaustive.py` | Walker exhaustive RMSE (holdout excludes train XY by default) |

## Outputs per domain

`out/<domain>/trials.csv`, `best_params.json`, `best_params_optuna.json`,
variogram PNGs, root `summary.json`.

Experiment logs in this monorepo follow [`docs/experiment_report_format.md`](../docs/experiment_report_format.md)
(primary score = holdout / LB, not CV-only; `!` on Optuna bound hits).
