# Experiment report format

Canonical template for geostat / OK / tuner experiment logs in this repo
(`experiments.ipynb`, `experiments_walker.ipynb`, `experiments_datasets.ipynb`, and future logs).

## Principles

1. **One summary table at the top** — all experiments, comparable scores side by side.
2. **One section per experiment** — params, search bounds, scores, short notes.
3. **Comparable scoring protocol** — same train / truth / holdout rules within a log.
4. **Mark search boundaries** — `!` on tuned params that hit the Optuna box edge.
5. **Ablations explicit** — if you freeze/change one knob (e.g. `nugget=0`), new Exp #.

## Scoring (default for Walker-style exhaustive truth)

| Name | Definition |
|------|------------|
| **holdout** | Score on truth grid **excluding** known train sample locations (preferred for ranking). |
| **all grid** | Score on **all** truth points (optional, for reference). |

Always state in the log header which protocol is primary.

For Kaggle LB logs (Jura): use Public / Private instead of holdout / all grid.

## Boundary marker `!`

- Applies only to **Optuna-tuned** (or otherwise searched) parameters.
- `!` means value is at **min or max** of the search box.
- Floats: within **2% of interval width** from either edge counts as boundary.
- Integers (`N_max`): exact equality to min or max.
- Manual one-off params (no search box): do **not** use `!`; say «ручные».
- If none hit the wall: write explicitly «ни один тюнимый param не на границе → без `!`».

## Top-of-log checklist

```markdown
# <Dataset> — experiments log

- **Train:** `path`
- **Truth / LB:** `path` or competition name
- **Primary score:** holdout RMSE/MAE (or Public/Private)
- **`!`:** Optuna bound hit (see docs/experiment_report_format.md)

| # | Method | Key params | Score… | Notes |
|---|--------|------------|--------|-------|
| 1 | … | … | … | … |
```

In **Key params**, append `!` to values on the bound, e.g. `N_max=5!`, `range_scale=1.20!`.

## Per-experiment section template

```markdown
---

## Exp N — <short title>

**Дата:** YYYY-MM-DD  
**Results / code:** `path/` or notebook cells  
**Best CV trial:** evaluation **K/N**, CV RMSE=**…** (and MAE if available; name the CV method, e.g. spatial_block)

### Pipeline
1. …
2. …

### Tuned params vs search space

| Param | Value | Search box | Boundary? |
|-------|-------|------------|-----------|
| `R_major` | 62.446 | [46.879, 70.319] | |
| `N_max` | 5 | [5, 40] | **!** (min) |
| `range_scale` | 1.20 | [0.80, 1.20] | **!** (max) |

<!-- If no bounds hit: -->
Ни один тюнимый параметр не на границе → **без `!`**.

**Fixed** (not tuned, no `!`): α=…, a_maj/min=…, sill=…

### Scores

| Metric | holdout | all grid |
|--------|---------|----------|
| RMSE | **…** | … |
| MAE | … | … |

<!-- Or for Kaggle: -->
| Board | Score |
|-------|-------|
| Public | … |
| Private | … |

### Notes
- …
```

## After each new Exp

1. Append row to the **summary table** (keep best score bold).
2. Add full section with bounds table when Optuna was used.
3. Keep holdout (or LB) as the number used for «лучше/хуже» claims.
4. After Optuna: always log **Best CV trial** as `evaluation K/N` **and** the best CV score value(s) (RMSE/MAE + CV method name). Do not leave only the trial index.
5. Store machine-readable artifacts under `*/results/...` when applicable
   (`best_params.json`, `exhaustive_score.json`, trials CSV).
