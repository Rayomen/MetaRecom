"""
MetaRecom - Meta-Learning Engine
Stores past experiment results and uses similarity-based meta-learning
to suggest candidate models for new datasets without brute-force training.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# ─── Canonical model registry ───────────────────────────────────────────────
CLASSIFICATION_MODELS = [
    'LogisticRegression',
    'SVC',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'KNeighborsClassifier',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier',
    'AdaBoostClassifier',
    'GaussianNB',
    'RidgeClassifier',
]

REGRESSION_MODELS = [
    'LinearRegression',
    'Ridge',
    'Lasso',
    'SVR',
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'KNeighborsRegressor',
    'DecisionTreeRegressor',
    'ExtraTreesRegressor',
    'AdaBoostRegressor',
]



# Optional GPU-capable gradient boosting models; CPU fallback remains available.
try:
    import xgboost  # noqa: F401
    if "XGBClassifier" not in CLASSIFICATION_MODELS:
        CLASSIFICATION_MODELS.append("XGBClassifier")
    if "XGBRegressor" not in REGRESSION_MODELS:
        REGRESSION_MODELS.append("XGBRegressor")
except Exception:
    pass


class ExperimentRepository:
    """
    Persistent store of past meta-feature → model performance pairs.
    Acts as the long-term memory of the meta-learning system.
    """

    def __init__(self, repo_path: str = 'data/experiment_repo.json'):
        self.repo_path = repo_path
        self.records = []
        self._load()

    def _load(self):
        if os.path.exists(self.repo_path):
            with open(self.repo_path, 'r') as f:
                self.records = json.load(f)
        else:
            self.records = self._get_seed_experiments()
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.repo_path) if os.path.dirname(self.repo_path) else '.', exist_ok=True)
        with open(self.repo_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def add(self, meta_features: dict, model_name: str, score: float,
            task_type: str, dataset_name: str = 'unknown'):
        record = {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'task_type': task_type,
            'model_name': model_name,
            'score': float(score),
            'meta_features': {k: float(v) for k, v in meta_features.items()},
        }
        self.records.append(record)
        self._save()

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def _get_seed_experiments(self) -> list:
        """
        Pre-seeded experiments covering a variety of dataset profiles.
        In production, this would be populated from real benchmark runs.
        These represent distilled knowledge from datasets like OpenML-CC18.
        """
        seed = []

        # --- Small binary classification, balanced ---
        base_clf_small = {
            'n_samples': 500, 'n_features': 10, 'log_n_samples': 6.2,
            'log_n_features': 2.4, 'samples_per_feature': 50.0,
            'ratio_numeric': 0.8, 'ratio_categorical': 0.2,
            'n_numeric_features': 8, 'n_cat_features': 2,
            'missing_ratio_overall': 0.0, 'missing_ratio_max_col': 0.0,
            'n_cols_with_missing': 0, 'mean_mean': 0.5, 'mean_std': 0.3,
            'mean_skewness': 0.2, 'mean_kurtosis': 0.1, 'max_skewness': 0.8,
            'high_skew_ratio': 0.1, 'feature_corr_mean': 0.2,
            'n_classes': 2, 'class_entropy': 0.69, 'class_imbalance': 1.1,
            'minority_class_ratio': 0.48, 'is_binary': 1,
            'target_mean': 0.0, 'target_std': 0.0,
            'pca_95_components': 7, 'pca_ratio': 0.7,
            'top1_pca_variance': 0.25, 'outlier_ratio': 0.01,
            'task_is_classification': 1
        }
        for model, score in [('LogisticRegression', 0.87), ('RandomForestClassifier', 0.89),
                               ('SVC', 0.86), ('GradientBoostingClassifier', 0.90),
                               ('KNeighborsClassifier', 0.82), ('GaussianNB', 0.78),
                               ('AdaBoostClassifier', 0.85), ('DecisionTreeClassifier', 0.80),
                               ('ExtraTreesClassifier', 0.88), ('RidgeClassifier', 0.85)]:
            seed.append({'timestamp': '2024-01-01T00:00:00', 'dataset_name': 'breast_cancer',
                         'task_type': 'classification', 'model_name': model,
                         'score': score, 'meta_features': base_clf_small.copy()})

        # --- Large high-dimensional classification ---
        base_clf_large = {**base_clf_small,
            'n_samples': 10000, 'n_features': 50, 'log_n_samples': 9.2,
            'log_n_features': 3.9, 'samples_per_feature': 200.0,
            'feature_corr_mean': 0.15, 'pca_95_components': 30, 'pca_ratio': 0.6}
        for model, score in [('RandomForestClassifier', 0.93), ('GradientBoostingClassifier', 0.94),
                               ('ExtraTreesClassifier', 0.92), ('SVC', 0.88),
                               ('LogisticRegression', 0.85), ('KNeighborsClassifier', 0.80),
                               ('AdaBoostClassifier', 0.88), ('GaussianNB', 0.72),
                               ('DecisionTreeClassifier', 0.83), ('RidgeClassifier', 0.84)]:
            seed.append({'timestamp': '2024-01-01T00:00:00', 'dataset_name': 'large_synthetic',
                         'task_type': 'classification', 'model_name': model,
                         'score': score, 'meta_features': base_clf_large.copy()})

        # --- Imbalanced multiclass ---
        base_clf_imbal = {**base_clf_small,
            'n_classes': 5, 'class_entropy': 1.2, 'class_imbalance': 8.0,
            'minority_class_ratio': 0.05, 'is_binary': 0, 'task_is_classification': 1}
        for model, score in [('GradientBoostingClassifier', 0.82), ('RandomForestClassifier', 0.80),
                               ('ExtraTreesClassifier', 0.79), ('AdaBoostClassifier', 0.76),
                               ('SVC', 0.74), ('LogisticRegression', 0.68),
                               ('KNeighborsClassifier', 0.65), ('GaussianNB', 0.60),
                               ('DecisionTreeClassifier', 0.70), ('RidgeClassifier', 0.65)]:
            seed.append({'timestamp': '2024-01-01T00:00:00', 'dataset_name': 'imbalanced_multi',
                         'task_type': 'classification', 'model_name': model,
                         'score': score, 'meta_features': base_clf_imbal.copy()})

        # --- Small regression ---
        base_reg_small = {**base_clf_small,
            'n_classes': 0, 'class_entropy': 0.0, 'class_imbalance': 0.0,
            'minority_class_ratio': 0.0, 'is_binary': 0,
            'target_mean': 50.0, 'target_std': 20.0, 'task_is_classification': 0}
        for model, score in [('RandomForestRegressor', 0.88), ('GradientBoostingRegressor', 0.90),
                               ('Ridge', 0.82), ('Lasso', 0.80), ('SVR', 0.84),
                               ('LinearRegression', 0.79), ('ExtraTreesRegressor', 0.87),
                               ('KNeighborsRegressor', 0.75), ('AdaBoostRegressor', 0.83),
                               ('DecisionTreeRegressor', 0.78)]:
            seed.append({'timestamp': '2024-01-01T00:00:00', 'dataset_name': 'boston_housing',
                         'task_type': 'regression', 'model_name': model,
                         'score': score, 'meta_features': base_reg_small.copy()})

        # --- High-cardinality regression with skew ---
        base_reg_skew = {**base_reg_small,
            'n_samples': 5000, 'mean_skewness': 1.8, 'max_skewness': 3.2,
            'high_skew_ratio': 0.5, 'outlier_ratio': 0.05}
        for model, score in [('GradientBoostingRegressor', 0.87), ('RandomForestRegressor', 0.85),
                               ('ExtraTreesRegressor', 0.84), ('Lasso', 0.72),
                               ('Ridge', 0.74), ('SVR', 0.78), ('LinearRegression', 0.70),
                               ('KNeighborsRegressor', 0.73), ('AdaBoostRegressor', 0.80),
                               ('DecisionTreeRegressor', 0.76)]:
            seed.append({'timestamp': '2024-01-01T00:00:00', 'dataset_name': 'skewed_regression',
                         'task_type': 'regression', 'model_name': model,
                         'score': score, 'meta_features': base_reg_skew.copy()})

        return seed


class MetaLearner:
    """
    Similarity-based meta-learner using k-NN over meta-feature space.
    Given a new dataset's meta-features, finds the most similar past
    datasets and aggregates their model performance to rank candidates.
    """

    def __init__(self, repo: ExperimentRepository, k: int = 5):
        self.repo = repo
        self.k = k
        self.scaler = StandardScaler()
        self.knn = None
        self._mf_keys = None
        self._fitted = False

    def fit(self):
        """Fit the k-NN meta-learner on the experiment repository."""
        df = self.repo.get_dataframe()
        if len(df) < self.k:
            self._fitted = False
            return self

        mf_df = pd.json_normalize(df['meta_features'])
        self._mf_keys = sorted(mf_df.columns.tolist())
        X_meta = mf_df[self._mf_keys].fillna(0).values

        self.scaler.fit(X_meta)
        X_scaled = self.scaler.transform(X_meta)

        n_neighbors = min(self.k, len(X_scaled))
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.knn.fit(X_scaled)

        self._df = df
        self._mf_matrix = mf_df[self._mf_keys].fillna(0).values
        self._fitted = True
        return self

    def recommend(self, meta_features: dict, task_type: str, top_n: int = 5) -> list:
        """
        Recommend top-N models for a new dataset.

        Returns
        -------
        list of dicts: [{'model': name, 'predicted_score': float, 'confidence': float}]
        """
        if not self._fitted:
            # Cold start: return defaults based on task type
            return self._cold_start_defaults(task_type, top_n)

        # Align meta-feature vector to training keys
        vec = np.array([float(meta_features.get(k, 0.0)) for k in self._mf_keys]).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)

        distances, indices = self.knn.kneighbors(vec_scaled)
        distances = distances[0]
        indices = indices[0]

        # Weight by inverse distance (closer = more relevant)
        weights = 1.0 / (distances + 1e-6)
        weights /= weights.sum()

        # Aggregate weighted scores per model
        model_scores = {}
        model_weights = {}
        for idx, weight in zip(indices, weights):
            record = self._df.iloc[idx]
            if record['task_type'] != task_type:
                continue
            model = record['model_name']
            score = record['score']
            model_scores[model]   = model_scores.get(model, 0.0) + weight * score
            model_weights[model]  = model_weights.get(model, 0.0) + weight

        # Normalize
        for model in model_scores:
            model_scores[model] /= model_weights[model]

        # If no same-task records found, fall back
        if not model_scores:
            return self._cold_start_defaults(task_type, top_n)

        # Sort and return top_n
        ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = ranked[0][1] if ranked else 1.0

        results = []
        for model, pred_score in ranked[:top_n]:
            confidence = float(np.clip(pred_score / max_score, 0, 1))
            results.append({
                'model': model,
                'predicted_score': round(float(pred_score), 4),
                'confidence': round(confidence, 4),
            })
        return results

    def _cold_start_defaults(self, task_type: str, top_n: int) -> list:
        models = CLASSIFICATION_MODELS if task_type == 'classification' else REGRESSION_MODELS
        return [{'model': m, 'predicted_score': None, 'confidence': 0.0}
                for m in models[:top_n]]
