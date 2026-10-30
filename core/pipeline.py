"""
MetaRecom - Main Pipeline Orchestrator
Dataset -> meta-features -> meta-learning -> candidate selection -> tuning/evaluation.
"""

import numpy as np
import pandas as pd

from core.meta_features import MetaFeatureExtractor
from core.meta_learner import ExperimentRepository, MetaLearner
from core.model_evaluator import (
    ModelEvaluator, Preprocessor, available_model_names, XGBOOST_AVAILABLE
)
from core.explainability import ExplainabilityEngine


class MetaRecomPipeline:
    def __init__(
        self,
        repo_path="data/experiment_repo.json",
        top_meta_candidates=5,
        cv_folds=5,
        tune_hyperparams=True,
        tune_iter=15,
        task_type="auto",
        progress_callback=None,
        compute_mode="auto",
    ):
        self.repo_path = repo_path
        self.top_meta_candidates = max(1, int(top_meta_candidates))
        self.cv_folds = max(2, int(cv_folds))
        self.tune_hyperparams = bool(tune_hyperparams)
        self.tune_iter = max(1, int(tune_iter))
        self.task_type = task_type
        self.progress_callback = progress_callback
        self.compute_mode = str(compute_mode or "auto").lower()
        if self.compute_mode not in {"auto", "cpu", "gpu"}:
            self.compute_mode = "auto"

        self.extractor = MetaFeatureExtractor()
        self.repo = ExperimentRepository(repo_path=repo_path)
        self.explainer = ExplainabilityEngine()
        self.result_ = {}

    def run(self, X: pd.DataFrame, y: pd.Series, dataset_name="dataset") -> dict:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if len(X) != len(y):
            raise ValueError("Features and target have different numbers of rows.")
        if len(X) < 4:
            raise ValueError("Dataset must contain at least 4 rows.")
        if X.shape[1] < 1:
            raise ValueError("Dataset must contain at least one feature column.")

        self._log("step", "Extracting meta-features...")
        task_type = self.task_type
        if task_type == "auto":
            task_type = self.extractor._infer_task(y)
        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task type: {task_type}")

        meta_features = self.extractor.extract(X, y, task_type=task_type)
        self._log("meta_features", meta_features)

        self._log("step", "Running meta-learner to select candidates...")
        learner = MetaLearner(self.repo, k=5)
        learner.fit()
        meta_recommendations = learner.recommend(
            meta_features, task_type, top_n=self.top_meta_candidates
        )
        candidate_names = [r["model"] for r in meta_recommendations]

        # Ensure the optional GPU-capable estimator is actually considered.
        # The meta-learner remains the source of the primary candidate ranking.
        xgb_name = "XGBClassifier" if task_type == "classification" else "XGBRegressor"
        if XGBOOST_AVAILABLE and xgb_name in available_model_names(task_type):
            if xgb_name not in candidate_names:
                candidate_names.append(xgb_name)

        self._log("meta_recommendations", meta_recommendations)
        self._log("candidates", candidate_names)

        self._log("step", "Preprocessing dataset...")
        preprocessor = Preprocessor()
        X_proc = preprocessor.fit_transform(X)

        if task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            y_proc = LabelEncoder().fit_transform(y.fillna("MISSING").astype(str))
        else:
            numeric_y = pd.to_numeric(y, errors="coerce")
            fill_value = numeric_y.median()
            if pd.isna(fill_value):
                fill_value = 0.0
            y_proc = numeric_y.fillna(fill_value).to_numpy(dtype=float)

        self._log("step", f"Training/evaluating {len(candidate_names)} candidate models...")
        evaluator = ModelEvaluator(
            task_type=task_type,
            cv=self.cv_folds,
            tune=self.tune_hyperparams,
            tune_iter=self.tune_iter,
            compute_mode=self.compute_mode,
        )
        results_df = evaluator.evaluate_candidates(
            X_proc, y_proc, candidate_names,
            progress_callback=self.progress_callback,
        )

        best = evaluator.get_best()
        score_col = "accuracy" if task_type == "classification" else "r2"
        if not best:
            errors = []
            if "error" in results_df.columns:
                errors = results_df["error"].dropna().astype(str).tolist()
            details = "; ".join(errors[:3]) or "No candidate produced a valid score."
            raise RuntimeError(f"All candidate models failed. {details}")

        best_model_name = best["model"]
        best_score = float(best.get(score_col, 0.0))
        self._log("step", f"Best model: {best_model_name} ({score_col}={best_score:.4f})")

        feature_importance_df = None
        try:
            if best_model_name not in {"XGBClassifier", "XGBRegressor"}:
                from core.model_evaluator import CLF_REGISTRY, REG_REGISTRY
                registry = CLF_REGISTRY if task_type == "classification" else REG_REGISTRY
                if best_model_name in registry:
                    from sklearn.base import clone
                    final_model = clone(registry[best_model_name])
                    final_model.fit(X_proc, y_proc)
                    feature_importance_df = self.explainer.compute_feature_importance(
                        final_model, X_proc, list(X.columns), task_type
                    )
        except Exception:
            feature_importance_df = None

        self._log("step", "Generating explanation...")
        explanation = self.explainer.explain_recommendation(
            best_model_name, best_score, results_df,
            meta_features, task_type, meta_recommendations
        )

        for _, row in results_df.iterrows():
            score = row.get(score_col, np.nan)
            if pd.isna(score) or float(score) == -999:
                continue
            self.repo.add(
                meta_features=meta_features,
                model_name=row["model"],
                score=float(score),
                task_type=task_type,
                dataset_name=dataset_name,
            )

        result = {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "meta_features": meta_features,
            "meta_recommendations": meta_recommendations,
            "candidate_models": candidate_names,
            "all_results": results_df,
            "recommendation": {
                "model": best_model_name,
                "score": best_score,
                "score_metric": score_col,
                "best_params": best.get("best_params", {}),
                "backend": best.get("backend", "CPU"),
            },
            "execution": {
                "requested_mode": self.compute_mode,
                "gpu_available": bool(evaluator.gpu_available_),
                "gpu_status": evaluator.gpu_status_,
                "backend_used_for_recommendation": best.get("backend", "CPU"),
                "cv_folds_used": evaluator.effective_cv_,
                "tuning_iterations_used": evaluator.effective_tune_iter_,
                "xgboost_available": bool(XGBOOST_AVAILABLE),
            },
            "explanation": explanation,
            "feature_importance": feature_importance_df,
            "n_experiments_in_repo": len(self.repo.records),
        }
        self.result_ = result
        return result

    def _log(self, key, value):
        if self.progress_callback:
            self.progress_callback(key, value)

    def print_summary(self, result=None):
        r = result or self.result_
        if not r:
            print("No result yet. Call .run() first.")
            return
        rec = r["recommendation"]
        exp = r["explanation"]
        print("\n" + "=" * 60)
        print("  MetaRecom — Recommendation Summary")
        print("=" * 60)
        print(f"  Dataset      : {r['dataset_name']}")
        print(f"  Task Type    : {r['task_type'].upper()}")
        print(f"  Best Model   : {rec['model']}")
        print(f"  {rec['score_metric'].upper():<12} : {rec['score']:.4f}")
        print(f"  Backend      : {rec.get('backend', 'CPU')}")
        print(f"  Confidence   : {exp['confidence_label']} ({exp['confidence_score']:.0%})")
        print("=" * 60)
        print("\n── Meta-Feature Insights ──")
        for insight in exp.get("meta_feature_insights", []):
            print(f"  • {insight}")
        print("\n── Top Model Comparison ──")
        df = r["all_results"]
        cols = [c for c in ["model", rec["score_metric"], "cv_std", "train_time_s", "backend"] if c in df.columns]
        print(df[cols].head(5).to_string(index=False))
        print("\n── Execution ──")
        print(r.get("execution", {}))
        print("\n── Narrative ──")
        print(exp["narrative"])
        print("=" * 60)
