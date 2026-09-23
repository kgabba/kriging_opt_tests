# Nested OK Tuner v2

Outer Optuna (variography / geometry / model) × inner Optuna (OK neighbourhood).
Log = **best-per-outer** only (`outer_trials.csv`).

Requires sibling [`aniso_idw_mvp`](../aniso_idw_mvp) for pairs / MOI / map.

## Budget (default)

- Outer: 30 trials  
- Inner: 200 trials per outer  
- Per Domain independently  

## Outer params

`geometry` ∈ {isotropic, anisotropic}, `variogram_model` ∈ {spherical, exponential, gaussian},
`n_lags` 10–14, `max_dist_percentile` 40–60, `tolerance_deg` 16–24, `radius_prior_scale` 1.0–1.5

## Inner params

`R` or `R_major`/`R_minor` (±20%), `N_max`, `nugget` (±20% C₀), `range_scale` 0.8–1.2

## Run

```bash
cd aniso_ok_tuner_v2
pip install -e ../aniso_idw_mvp -e .
python tune.py --data ../data/data_algo.csv --config configs/smoke.yaml --out-dir results/smoke
python tune.py --data ../data/data_algo.csv --config configs/default.yaml --out-dir results/jura
```
