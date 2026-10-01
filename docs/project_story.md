# Проект: авто-тюнинг Ordinary Kriging

## Зачем

Автоподбор параметров OK (iso/aniso, R, N_max, nugget, range_scale) по train → проверка на независимом holdout. Без ручного кручения вариограммы на каждом датасете.

## Что сделали

- `aniso_ok_tuner`: VG/MOI → auto ISO|ANISO → Optuna (bufdel / spatial_block) → θ по CV.
- Долго тюнили протокол на **Jura**, **Walker** (боксы, lock ratio, outer — отбросили).
- Рабочий `main`: полный sample для VG, box ~±60%, без outer-as-objective.
- Логи: `experiments.ipynb`, `experiments_walker.ipynb`, `experiments_datasets.ipynb`.

## Инсайт

Outer / early stop и широкий бокс → лучше CV, хуже truth. Скромный box + полный train для VG.

## Результаты на SIC-соревнованиях

### SIC97 — 100 → 367 withheld

| Метод | RMSE |
|-------|-----:|
| **наш auto ANISO OK** | **51.74** |
| лучший участник (multiquadric + aniso) | 53.1 |
| automap | 51.4 |

С первой попытки — лучше лучшего участника, рядом с automap. Модель проще (OK), параметры — авто.

### SIC2004 — 200 → 808 (truth = 1st file)

| Метод | RMSE |
|-------|-----:|
| **наш auto ISO OK** | **12.43** |
| Fournier — Robust Kriging | **12.43** |
| Furrer — Robust Kriging | 12.44 |

**Уровень лучшего RMSE соревнования.** Участники, скорее всего, долго крутили вариограмму / robust-настройки вручную; мы — тот же пайплайн, без ручного анализа, Optuna за минуты, результат как у топа.

Holdout-точки нигде не участвовали в VG / CV / выборе trial.

## Дальше

Новые датасеты с blind holdout → тот же протокол → строка в `experiments_datasets.ipynb`.
