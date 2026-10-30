# MetaRecom implementation notes

## What changed

- Preserved the existing Flask app structure, routes, templates, CSS, and visual theme.
- Updated `core/model_evaluator.py` to tune candidate model hyperparameters with a dataset-aware search budget, reuse the tuning CV results rather than unnecessarily running another CV pass, and report the chosen parameters and backend.
- Updated `core/pipeline.py` to keep meta-learning as the candidate-selection mechanism, include XGBoost as an additional candidate when installed, and return execution diagnostics.
- Updated `core/meta_learner.py` to include optional XGBoost candidates while retaining the existing meta-learning/repository approach.
- Wired `compute_mode` and `tune_iter` through the upload and API routes.
- Added form controls for Auto / Prefer GPU / CPU-only mode and a maximum tuning-iteration setting. Existing preprocessing, candidate-model, CV-fold and tuning controls remain.
- Added a results panel showing actual backend, GPU probe/fallback status, effective CV folds, tuning iterations, and selected hyperparameters.
- Added `xgboost>=2.0.0` as an optional GPU-capable estimator dependency. GPU use still depends on a compatible local XGBoost/CUDA setup; sklearn models remain CPU-based.

## Performance behavior

Tuning effort is reduced automatically for larger datasets. Current caps are 2 iterations / 2 folds above 100,000 rows; 3 / 3 above 20,000; 5 / 3 above 5,000; and 8 / 4 above 1,000. Smaller datasets can use the requested settings. The effective settings are shown on the results page.

## Install and run

From the project folder, run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Back up your current folder before replacing it. The project was syntax-checked after modification; GPU behavior must be verified in the target Windows environment.
