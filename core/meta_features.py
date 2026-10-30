"""
MetaRecom - Meta-Feature Extraction Engine
Extracts 25+ statistical, information-theoretic, and structural
meta-features from any tabular dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class MetaFeatureExtractor:
    """
    Extracts dataset-level meta-features that characterize a dataset's
    statistical and structural properties. These are used by the
    meta-learning engine to recommend candidate models.
    """

    def __init__(self):
        self.meta_features_ = {}

    def extract(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'auto') -> dict:
        """
        Full pipeline: extract all meta-features from dataset.

        Parameters
        ----------
        X : pd.DataFrame - Feature matrix
        y : pd.Series    - Target variable
        task_type : str  - 'classification', 'regression', or 'auto'

        Returns
        -------
        dict of meta-features (all numeric, ML-ready)
        """
        X = X.copy()
        y = y.copy()

        if task_type == 'auto':
            task_type = self._infer_task(y)

        mf = {}

        # ── 1. BASIC STATISTICS ──────────────────────────────────────────
        mf['n_samples']       = int(X.shape[0])
        mf['n_features']      = int(X.shape[1])
        mf['log_n_samples']   = float(np.log1p(X.shape[0]))
        mf['log_n_features']  = float(np.log1p(X.shape[1]))
        mf['samples_per_feature'] = float(X.shape[0] / max(X.shape[1], 1))

        # ── 2. FEATURE TYPE RATIOS ───────────────────────────────────────
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        mf['ratio_numeric']      = float(len(num_cols) / max(X.shape[1], 1))
        mf['ratio_categorical']  = float(len(cat_cols) / max(X.shape[1], 1))
        mf['n_numeric_features'] = int(len(num_cols))
        mf['n_cat_features']     = int(len(cat_cols))

        # ── 3. MISSING VALUE STATS ───────────────────────────────────────
        missing_per_col = X.isnull().mean()
        mf['missing_ratio_overall'] = float(X.isnull().mean().mean())
        mf['missing_ratio_max_col'] = float(missing_per_col.max())
        mf['n_cols_with_missing']   = int((missing_per_col > 0).sum())

        # ── 4. NUMERIC FEATURE STATISTICS ───────────────────────────────
        if len(num_cols) > 0:
            X_num = X[num_cols].fillna(X[num_cols].median())
            col_means   = X_num.mean()
            col_stds    = X_num.std().fillna(0)
            col_skews   = X_num.skew().fillna(0)
            col_kurt    = X_num.kurtosis().fillna(0)

            mf['mean_mean']          = float(col_means.mean())
            mf['mean_std']           = float(col_stds.mean())
            mf['mean_skewness']      = float(col_skews.mean())
            mf['mean_kurtosis']      = float(col_kurt.mean())
            mf['max_skewness']       = float(col_skews.abs().max())
            mf['high_skew_ratio']    = float((col_skews.abs() > 1).mean())
            mf['feature_corr_mean']  = float(X_num.corr().abs().where(
                lambda x: ~np.eye(x.shape[0], dtype=bool)).stack().mean()
                if len(num_cols) > 1 else 0.0)
        else:
            for k in ['mean_mean','mean_std','mean_skewness','mean_kurtosis',
                      'max_skewness','high_skew_ratio','feature_corr_mean']:
                mf[k] = 0.0

        # ── 5. CLASS / TARGET STATS ──────────────────────────────────────
        if task_type == 'classification':
            le = LabelEncoder()
            y_enc = le.fit_transform(y.fillna('MISSING').astype(str))
            classes, counts = np.unique(y_enc, return_counts=True)
            probs = counts / counts.sum()
            mf['n_classes']          = int(len(classes))
            mf['class_entropy']      = float(stats.entropy(probs))
            mf['class_imbalance']    = float(counts.max() / counts.min() if counts.min() > 0 else 999)
            mf['minority_class_ratio'] = float(counts.min() / counts.sum())
            mf['is_binary']          = int(len(classes) == 2)
            mf['target_mean']        = 0.0
            mf['target_std']         = 0.0
        else:
            y_num = pd.to_numeric(y, errors='coerce').fillna(y.median() if pd.api.types.is_numeric_dtype(y) else 0)
            mf['n_classes']          = 0
            mf['class_entropy']      = 0.0
            mf['class_imbalance']    = 0.0
            mf['minority_class_ratio'] = 0.0
            mf['is_binary']          = 0
            mf['target_mean']        = float(y_num.mean())
            mf['target_std']         = float(y_num.std())

        # ── 6. DIMENSIONALITY & COMPLEXITY ──────────────────────────────
        if len(num_cols) >= 2:
            X_pca = X[num_cols].fillna(X[num_cols].median())
            X_std = (X_pca - X_pca.mean()) / (X_pca.std() + 1e-8)
            pca = PCA()
            pca.fit(X_std)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n95 = int(np.searchsorted(cum_var, 0.95)) + 1
            mf['pca_95_components']  = int(n95)
            mf['pca_ratio']          = float(n95 / len(num_cols))
            mf['top1_pca_variance']  = float(pca.explained_variance_ratio_[0])
        else:
            mf['pca_95_components']  = 1
            mf['pca_ratio']          = 1.0
            mf['top1_pca_variance']  = 1.0

        # ── 7. OUTLIER INDICATOR ────────────────────────────────────────
        if len(num_cols) > 0:
            X_num = X[num_cols].fillna(X[num_cols].median())
            z = np.abs(stats.zscore(X_num, nan_policy='omit'))
            mf['outlier_ratio'] = float((z > 3).mean().mean())
        else:
            mf['outlier_ratio'] = 0.0

        # ── 8. TASK TYPE FLAG ────────────────────────────────────────────
        mf['task_is_classification'] = int(task_type == 'classification')

        self.meta_features_ = mf
        return mf

    def _infer_task(self, y: pd.Series) -> str:
        """Infer task type from target column."""
        if not pd.api.types.is_numeric_dtype(y):
            return 'classification'
        n_unique = y.nunique()
        if n_unique <= 20 or (n_unique / len(y) < 0.05):
            return 'classification'
        return 'regression'

    def to_vector(self, mf: dict = None) -> np.ndarray:
        """Convert meta-features dict to a fixed-length numeric vector."""
        if mf is None:
            mf = self.meta_features_
        keys = sorted(mf.keys())
        return np.array([float(mf[k]) for k in keys])

    def get_feature_names(self) -> list:
        keys = sorted(self.meta_features_.keys())
        return keys
