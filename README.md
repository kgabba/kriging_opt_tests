# Anisotropic Ordinary Kriging Tuner

Per-**Domain** Ordinary Kriging with MOI → directional ranges → **Optuna TPE** on
neighbourhood / nugget / `range_scale`. Console line + `trials.csv` every trial.

Variography (pairs, MOI, map/rose, directional spherical fit) comes from sibling
package [`aniso_idw_mvp`](https://github.com/kgabba/spatial-idw-opt) (or a local
`../aniso_idw_mvp` checkout).

## What Optuna tunes

| Param | Window |
|-------|--------|
| `R_major`, `R_minor` (aniso) or `R` (iso) | ±`radius_margin` of directional ranges |
| `N_max` | config box |
| `nugget` | ±`nugget_margin` of fitted C₀ |
| `range_scale` | config box (default 0.8–1.2) |

**Fixed** from MOI/variography: `alpha`, base `a_major`/`a_minor`, sill total.

## CV methods (`cv.method`)

| Method | Idea |
|--------|------|
| `spatial_block` | leave-one-block-out on XY grid (`grid_nx` × `grid_ny`) |
| `buffered_delete_d` | random delete-d + exclude train within buffer (Euclidean or aniso OK metric; `buffer_scale × a_major`) |
| `loo` / `kfold5` / `buffer` / `delete_d` | also supported |

Ready-made configs under `configs/` (Jura iso CV suite, Walker aniso spatial / buffered).

## Install

```bash
# sibling MVP for variography
pip install -e ../aniso_idw_mvp   # or clone spatial-idw-opt and point at its MVP path
pip install -e .
```

## Run

```bash
python tune.py --data path/to/points.csv --config configs/default.yaml --out-dir results/run
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
