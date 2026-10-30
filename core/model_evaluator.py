"""
MetaRecom - Model Training & Evaluation Engine
Dataset-aware hyperparameter tuning with optional XGBoost GPU acceleration.
CPU remains the reliable baseline; GPU use is reported per model and falls back
to CPU if the CUDA-enabled estimator cannot initialize or train.
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# Optional GPU-capable gradient-boosted tree implementation.
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = XGBRegressor = None
    XGBOOST_AVAILABLE = False


CLF_REGISTRY = {
    "LogisticRegression": LogisticRegression(max_iter=1500, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "RidgeClassifier": RidgeClassifier(),
}

REG_REGISTRY = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=3000),
    "SVR": SVR(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
}

HP_SPACES_CLF = {
    "LogisticRegression": {"model__C": uniform(0.01, 10)},
    "SVC": {"model__C": uniform(0.1, 10), "model__gamma": ["scale", "auto"]},
    "RandomForestClassifier": {
        "model__n_estimators": randint(60, 220), "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": randint(2, 10), "model__min_samples_leaf": randint(1, 5),
    },
    "GradientBoostingClassifier": {
        "model__n_estimators": randint(50, 160), "model__learning_rate": uniform(0.02, 0.25),
        "model__max_depth": randint(2, 5),
    },
    "KNeighborsClassifier": {"model__n_neighbors": randint(3, 15), "model__weights": ["uniform", "distance"]},
    "DecisionTreeClassifier": {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": randint(2, 10)},
    "ExtraTreesClassifier": {
        "model__n_estimators": randint(60, 220), "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": randint(1, 5),
    },
    "AdaBoostClassifier": {"model__n_estimators": randint(40, 160), "model__learning_rate": uniform(0.02, 0.8)},
    "GaussianNB": {"model__var_smoothing": uniform(1e-10, 1e-8)},
    "RidgeClassifier": {"model__alpha": uniform(0.01, 10)},
}
HP_SPACES_REG = {
    "LinearRegression": {},
    "Ridge": {"model__alpha": uniform(0.01, 10)},
    "Lasso": {"model__alpha": uniform(0.001, 1.0)},
    "SVR": {"model__C": uniform(0.1, 10), "model__gamma": ["scale", "auto"]},
    "RandomForestRegressor": {
        "model__n_estimators": randint(60, 220), "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": randint(1, 5),
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": randint(50, 160), "model__learning_rate": uniform(0.02, 0.25),
        "model__max_depth": randint(2, 5),
    },
    "KNeighborsRegressor": {"model__n_neighbors": randint(3, 15)},
    "DecisionTreeRegressor": {"model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": randint(1, 5)},
    "ExtraTreesRegressor": {
        "model__n_estimators": randint(60, 220), "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": randint(1, 5),
    },
    "AdaBoostRegressor": {"model__n_estimators": randint(40, 160), "model__learning_rate": uniform(0.02, 0.8)},
}
if XGBOOST_AVAILABLE:
    _xgb_space = {
        "model__n_estimators": randint(80, 220),
        "model__max_depth": randint(3, 9),
        "model__learning_rate": uniform(0.02, 0.25),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    }
    HP_SPACES_CLF["XGBClassifier"] = dict(_xgb_space)
    HP_SPACES_REG["XGBRegressor"] = dict(_xgb_space)


def available_model_names(task_type):
    """Return model names supported by the installed environment."""
    registry = CLF_REGISTRY if task_type == "classification" else REG_REGISTRY
    names = list(registry)
    if XGBOOST_AVAILABLE:
        names.append("XGBClassifier" if task_type == "classification" else "XGBRegressor")
    return names


class Preprocessor:
    """Handles categorical encoding, imputation, and scaling."""

    def __init__(self):
        self.label_encoders_ = {}
        self.imputer_ = SimpleImputer(strategy="median")
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        self.fitted_ = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for col in X.select_dtypes(exclude=[np.number]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
            self.label_encoders_[col] = le
        X_arr = self.imputer_.fit_transform(X.values.astype(float))
        X_arr = self.scaler_.fit_transform(X_arr)
        self.fitted_ = True
        return X_arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for col, le in self.label_encoders_.items():
            if col in X.columns:
                known = set(le.classes_)
                values = X[col].fillna("MISSING").astype(str)
                values = values.map(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(values)
        return self.scaler_.transform(self.imputer_.transform(X.values.astype(float)))


class ModelEvaluator:
    """
    Evaluate candidates using CV. Tuning budgets adapt to row count.
    compute_mode is 'auto', 'cpu', or 'gpu'. GPU acceleration currently applies
    to XGBoost when it is installed and the CUDA device can actually train.
    """

    def __init__(
        self, task_type: str, cv: int = 5, tune: bool = True,
        tune_iter: int = 15, random_state: int = 42, compute_mode: str = "auto"
    ):
        self.task_type = task_type
        self.cv = max(2, int(cv))
        self.tune = bool(tune)
        self.tune_iter = max(1, int(tune_iter))
        self.random_state = random_state
        self.compute_mode = str(compute_mode or "auto").lower()
        if self.compute_mode not in {"auto", "cpu", "gpu"}:
            self.compute_mode = "auto"
        self.results_ = []
        self.gpu_available_ = False
        self.gpu_status_ = "Not checked"
        self.effective_cv_ = self.cv
        self.effective_tune_iter_ = self.tune_iter

    def _xgb_estimator(self, task_type, device="cpu", n_classes=None):
        common = dict(
            n_estimators=120, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", device=device, random_state=self.random_state,
            n_jobs=1, verbosity=0,
        )
        if task_type == "classification":
            if n_classes and n_classes > 2:
                return XGBClassifier(**common, objective="multi:softprob", eval_metric="mlogloss")
            return XGBClassifier(**common, objective="binary:logistic", eval_metric="logloss")
        return XGBRegressor(**common, objective="reg:squarederror")

    def _probe_gpu(self):
        if not XGBOOST_AVAILABLE:
            self.gpu_status_ = "XGBoost is not installed; GPU acceleration unavailable."
            return False
        try:
            # A tiny real fit checks more than the presence of a driver.
            from sklearn.datasets import make_classification
            X_probe, y_probe = make_classification(
                n_samples=24, n_features=4, n_informative=3,
                n_redundant=0, random_state=1
            )
            probe = XGBClassifier(
                n_estimators=2, max_depth=2, tree_method="hist",
                device="cuda", n_jobs=1, eval_metric="logloss",
                verbosity=0, random_state=1
            )
            probe.fit(X_probe, y_probe)
            self.gpu_status_ = "GPU training probe succeeded."
            return True
        except Exception as exc:
            self.gpu_status_ = f"GPU probe failed; using CPU fallback ({type(exc).__name__})."
            return False

    def _effective_cv(self, y):
        folds = self.cv
        n = len(y)
        if self.task_type == "classification":
            _, counts = np.unique(y, return_counts=True)
            if len(counts):
                folds = min(folds, int(counts.min()))
        # Avoid invalid CV on tiny data.
        folds = min(folds, n)
        if folds < 2:
            raise ValueError("At least two usable rows per CV split are required.")
        return folds

    def _budget(self, n_rows):
        # Conservative budgets reduce runaway searches on large uploaded CSVs.
        if n_rows > 100000:
            return min(self.tune_iter, 2), min(self.cv, 2)
        if n_rows > 20000:
            return min(self.tune_iter, 3), min(self.cv, 3)
        if n_rows > 5000:
            return min(self.tune_iter, 5), min(self.cv, 3)
        if n_rows > 1000:
            return min(self.tune_iter, 8), min(self.cv, 4)
        return self.tune_iter, self.cv

    def _registry_for(self, task_type, n_classes, backend):
        registry = dict(CLF_REGISTRY if task_type == "classification" else REG_REGISTRY)
        if XGBOOST_AVAILABLE:
            name = "XGBClassifier" if task_type == "classification" else "XGBRegressor"
            registry[name] = self._xgb_estimator(task_type, device=("cuda" if backend == "GPU" else "cpu"), n_classes=n_classes)
        return registry

    def evaluate_candidates(self, X, y, candidate_models, progress_callback=None) -> pd.DataFrame:
        y = np.asarray(y)
        X = np.asarray(X)
        n_rows = len(y)
        if n_rows != len(X):
            raise ValueError("X and y have different numbers of rows.")
        if n_rows < 4:
            raise ValueError("Dataset must contain at least 4 rows for evaluation.")

        requested_iter, requested_cv = self._budget(n_rows)
        self.effective_tune_iter_ = requested_iter if self.tune else 0
        self.effective_cv_ = min(requested_cv, self._effective_cv(y))
        cv = self.effective_cv_

        if self.compute_mode == "cpu":
            self.gpu_available_ = False
            self.gpu_status_ = "CPU explicitly selected."
        else:
            self.gpu_available_ = self._probe_gpu()
            if self.compute_mode == "gpu" and not self.gpu_available_:
                self.gpu_status_ = "GPU requested but unavailable; safely fell back to CPU. " + self.gpu_status_

        registry_cpu = self._registry_for(self.task_type, len(np.unique(y)), "CPU")
        hp_spaces = HP_SPACES_CLF if self.task_type == "classification" else HP_SPACES_REG
        primary_metric = "accuracy" if self.task_type == "classification" else "r2"
        if self.task_type == "classification":
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        results = []
        candidates = list(dict.fromkeys(candidate_models or []))
        total = len(candidates)

        for step, model_name in enumerate(candidates):
            if progress_callback:
                try:
                    progress_callback(step, total, model_name)
                except TypeError:
                    progress_callback("model", model_name)
            if model_name not in registry_cpu:
                continue

            is_xgb = model_name.startswith("XGB") and XGBOOST_AVAILABLE
            backend = "GPU" if (is_xgb and self.gpu_available_) else "CPU"
            started = time.time()
            try:
                row = self._evaluate_one(
                    model_name, registry_cpu[model_name], hp_spaces.get(model_name, {}),
                    X, y, splitter, primary_metric, backend, cv
                )
            except Exception as exc:
                # Retry a GPU model once on CPU. Other model failures are recorded.
                if backend == "GPU":
                    try:
                        cpu_registry = self._registry_for(self.task_type, len(np.unique(y)), "CPU")
                        row = self._evaluate_one(
                            model_name, cpu_registry[model_name], hp_spaces.get(model_name, {}),
                            X, y, splitter, primary_metric, "CPU (GPU fallback)", cv
                        )
                        row["fallback_reason"] = f"GPU training failed: {type(exc).__name__}"
                    except Exception as cpu_exc:
                        row = self._error_row(model_name, cpu_exc, started)
                else:
                    row = self._error_row(model_name, exc, started)
            results.append(row)

        self.results_ = results
        df = pd.DataFrame(results)
        score_col = primary_metric
        if score_col in df.columns:
            df = df.sort_values(score_col, ascending=False, na_position="last").reset_index(drop=True)
        return df

    def _evaluate_one(self, name, estimator, hp_space, X, y, splitter, metric, backend, cv):
        started = time.time()
        # Each model gets a fresh estimator instance; registry objects are never reused fitted.
        estimator = clone(estimator)
        # Use one worker for GPU to avoid concurrent GPU jobs competing for VRAM.
        search_jobs = 1 if backend.startswith("GPU") else -1
        best_params = {}
        cv_scores = None

        if self.tune and hp_space:
            search = RandomizedSearchCV(
                Pipeline([("model", estimator)]),
                param_distributions=hp_space,
                n_iter=self.effective_tune_iter_,
                cv=splitter,
                scoring=metric,
                n_jobs=search_jobs,
                random_state=self.random_state,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X, y)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_mean = float(search.best_score_)
            cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        else:
            best_model = Pipeline([("model", estimator)])
            cv_scores = cross_val_score(
                best_model, X, y, cv=splitter, scoring=metric,
                n_jobs=search_jobs, error_score="raise"
            )
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            best_model.fit(X, y)

        # Retain the existing metric keys for the results page, but clearly label
        # secondary classification/regression metrics as in-sample diagnostics.
        best_model.fit(X, y)
        y_pred = best_model.predict(X)
        base = {
            "model": name,
            metric: round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "train_time_s": round(time.time() - started, 2),
            "best_params": best_params,
            "primary_metric": metric,
            "backend": backend,
            "metric_note": "Primary score is cross-validated; secondary metrics are calculated on training data.",
            "cv_folds_used": cv,
            "tuning_iterations_used": self.effective_tune_iter_ if self.tune and hp_space else 0,
        }
        if self.task_type == "classification":
            average = "binary" if len(np.unique(y)) == 2 else "macro"
            try:
                if hasattr(best_model, "predict_proba"):
                    proba = best_model.predict_proba(X)
                    if proba.shape[1] == 2:
                        auc = roc_auc_score(y, proba[:, 1])
                    else:
                        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
                else:
                    auc = None
            except Exception:
                auc = None
            base.update({
                "f1": round(float(f1_score(y, y_pred, average=average, zero_division=0)), 4),
                "precision": round(float(precision_score(y, y_pred, average=average, zero_division=0)), 4),
                "recall": round(float(recall_score(y, y_pred, average=average, zero_division=0)), 4),
                "roc_auc": round(float(auc), 4) if auc is not None else None,
            })
        else:
            base.update({
                "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y, y_pred)), 4),
            })
        return base

    @staticmethod
    def _error_row(name, exc, started):
        return {
            "model": name,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "accuracy": -999,
            "r2": -999,
            "train_time_s": round(time.time() - started, 2),
            "backend": "FAILED",
        }

    def get_best(self):
        if not self.results_:
            return {}
        score_col = "accuracy" if self.task_type == "classification" else "r2"
        valid = [
            row for row in self.results_
            if score_col in row and row[score_col] != -999 and pd.notna(row[score_col])
        ]
        return max(valid, key=lambda row: row[score_col]) if valid else {}
