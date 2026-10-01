# SIC2004 — prepared for `aniso_ok_tuner`

Raw (original):
- `SIC2004_input.csv` — train (ID,X,Y,value), n=200
- `SIC2004_out.csv` — predict locations (ID,X,Y), n=808
- `1st_file_true_values.csv` — truth for out (ID,X,Y,value), n=808
- `2nd_file_true_values.csv` — alternate truth (differs from 1st; not used by default)

Prepared (no header issues; `Grade` = target):
| File | Role |
|------|------|
| `sic2004_algo.csv` | train → tune (`X,Y,Grade,Domain`) |
| `sic2004_pred_loc.csv` | locations to predict (`ID,X,Y`) |
| `sic2004_truth.csv` | truth from 1st file (`ID,X,Y,Grade`) |
| `sic2004_truth_aligned.csv` | out order + Grade (`ID,X,Y,Grade`) for scoring |

Protocol: tune on `sic2004_algo.csv` → predict `sic2004_pred_loc.csv` → RMSE/MAE vs `sic2004_truth_aligned.csv` (all 808; train XY disjoint from out).
