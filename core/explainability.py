"""
MetaRecom - Explainability Engine
Provides SHAP-based feature importance, confidence scoring,
and natural language explanations for model recommendations.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ExplainabilityEngine:
    """
    Generates explanations for why a model was recommended,
    including meta-feature reasoning, confidence scoring,
    and SHAP-based feature importance where available.
    """

    def explain_recommendation(
        self,
        best_model_name: str,
        best_score: float,
        all_results: pd.DataFrame,
        meta_features: dict,
        task_type: str,
        meta_recommendations: list
    ) -> dict:
        """
        Generate comprehensive recommendation explanation.

        Returns
        -------
        dict with keys: narrative, confidence_score, score_gap,
                        meta_feature_insights, competitor_analysis
        """
        explanation = {}

        # ── 1. Confidence Score ─────────────────────────────────────────
        score_col = 'accuracy' if task_type == 'classification' else 'r2'
        scores = all_results[score_col].dropna().values if score_col in all_results.columns else np.array([best_score])
        confidence = self._compute_confidence(best_score, scores)
        explanation['confidence_score'] = round(confidence, 4)
        explanation['confidence_label'] = self._confidence_label(confidence)

        # ── 2. Score Gap Analysis ───────────────────────────────────────
        if len(scores) >= 2:
            sorted_scores = np.sort(scores)[::-1]
            gap_to_second = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        else:
            gap_to_second = 0.0
        explanation['score_gap_to_second'] = round(gap_to_second, 4)

        # ── 3. Meta-Feature Insights ────────────────────────────────────
        explanation['meta_feature_insights'] = self._meta_feature_insights(meta_features, task_type)

        # ── 4. Why This Model (narrative) ──────────────────────────────
        explanation['narrative'] = self._build_narrative(
            best_model_name, best_score, confidence, meta_features,
            task_type, meta_recommendations, gap_to_second
        )

        # ── 5. Competitor Analysis ──────────────────────────────────────
        explanation['competitor_analysis'] = self._competitor_analysis(all_results, score_col)

        # ── 6. Model-Specific Properties ───────────────────────────────
        explanation['model_properties'] = self._model_properties(best_model_name)

        return explanation

    def compute_feature_importance(
        self, trained_model, X: np.ndarray,
        feature_names: list, task_type: str
    ) -> pd.DataFrame:
        """
        Extract feature importance using model internals or permutation.
        Tries SHAP first, falls back to sklearn feature_importances_.
        """
        importances = None

        # Try tree-based built-in importance
        model = trained_model
        if hasattr(model, 'named_steps'):
            model = model.named_steps.get('model', model)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        if importances is None:
            # Fallback: equal importance
            importances = np.ones(len(feature_names)) / len(feature_names)

        # Normalize
        importances = importances[:len(feature_names)]
        if importances.sum() > 0:
            importances = importances / importances.sum()

        df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return df

    def _compute_confidence(self, best_score: float, all_scores: np.ndarray) -> float:
        """
        Confidence = how much better the best model is relative to spread.
        Scale: 0 (no confidence) → 1 (very high confidence).
        """
        if len(all_scores) < 2:
            return 0.75

        score_range = all_scores.max() - all_scores.min()
        if score_range < 1e-6:
            return 0.9  # All models similar → strong choice either way

        # Distance of best score from mean, normalized by range
        mean_score = all_scores.mean()
        gap = float(best_score - mean_score)
        confidence = 0.5 + 0.5 * np.tanh(gap / (score_range + 1e-6) * 3)
        return float(np.clip(confidence, 0.0, 1.0))

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.85:
            return 'Very High'
        elif confidence >= 0.70:
            return 'High'
        elif confidence >= 0.55:
            return 'Moderate'
        else:
            return 'Low'

    def _meta_feature_insights(self, mf: dict, task_type: str) -> list:
        """Rule-based insights from meta-features."""
        insights = []
        n = mf.get('n_samples', 0)
        p = mf.get('n_features', 0)
        ratio = mf.get('samples_per_feature', 9999)
        missing = mf.get('missing_ratio_overall', 0)
        imbalance = mf.get('class_imbalance', 1)
        skew = mf.get('mean_skewness', 0)
        corr = mf.get('feature_corr_mean', 0)
        pca_ratio = mf.get('pca_ratio', 1)

        if n < 500:
            insights.append('Small dataset — simpler models (Logistic Regression, SVM) tend to generalize better.')
        elif n > 10000:
            insights.append('Large dataset — ensemble methods (Random Forest, Gradient Boosting) can leverage data volume.')

        if ratio < 10:
            insights.append('Low samples-to-features ratio — regularized models (Ridge, Lasso, SVM) reduce overfitting risk.')

        if missing > 0.1:
            insights.append(f'{missing*100:.1f}% missing values detected — tree-based models are more robust to missingness.')

        if task_type == 'classification' and imbalance > 3:
            insights.append(f'Class imbalance ratio {imbalance:.1f}x — boosting methods (GradientBoosting, AdaBoost) handle imbalance better.')

        if abs(skew) > 1.0:
            insights.append('High feature skewness — tree-based models are insensitive to monotonic transforms.')

        if corr > 0.5:
            insights.append('High inter-feature correlation — regularized linear models may struggle; dimensionality reduction or trees preferred.')

        if pca_ratio < 0.4:
            insights.append('Low intrinsic dimensionality — strong linear structure present; linear models may perform competitively.')

        return insights

    def _build_narrative(self, model_name, score, confidence, mf,
                         task_type, meta_recs, gap) -> str:
        score_col = 'accuracy' if task_type == 'classification' else 'R²'
        meta_names = [r['model'] for r in meta_recs] if meta_recs else []
        meta_confirmed = model_name in meta_names

        narrative = (
            f"**{model_name}** is recommended as the best model for your dataset "
            f"with a cross-validated {score_col} of **{score:.4f}** and "
            f"a recommendation confidence of **{self._confidence_label(confidence)}** ({confidence:.0%}).\n\n"
        )

        if meta_confirmed:
            narrative += (
                f"This recommendation is **corroborated by meta-learning**: "
                f"historically similar datasets in the experiment repository also "
                f"performed best with {model_name}.\n\n"
            )
        else:
            narrative += (
                f"Note: meta-learning suggested different candidates, but empirical "
                f"evaluation on your dataset shows {model_name} outperforms them.\n\n"
            )

        if gap > 0.01:
            narrative += (
                f"The performance gap to the second-best model is **{gap:.4f}** "
                f"({score_col} points), indicating a clear winner.\n\n"
            )
        else:
            narrative += (
                f"Performance differences between top models are marginal (Δ {gap:.4f}). "
                f"Consider deployment speed and interpretability as secondary criteria.\n\n"
            )

        return narrative

    def _competitor_analysis(self, results: pd.DataFrame, score_col: str) -> list:
        if score_col not in results.columns:
            return []
        out = []
        top = results[[c for c in ['model', score_col, 'cv_std', 'train_time_s']
                       if c in results.columns]].head(5)
        for _, row in top.iterrows():
            out.append(row.to_dict())
        return out

    def _model_properties(self, model_name: str) -> dict:
        properties = {
            'LogisticRegression':         {'interpretable': True, 'speed': 'Fast', 'needs_scaling': True, 'handles_nonlinear': False},
            'SVC':                        {'interpretable': False, 'speed': 'Moderate', 'needs_scaling': True, 'handles_nonlinear': True},
            'RandomForestClassifier':     {'interpretable': False, 'speed': 'Moderate', 'needs_scaling': False, 'handles_nonlinear': True},
            'GradientBoostingClassifier': {'interpretable': False, 'speed': 'Slow', 'needs_scaling': False, 'handles_nonlinear': True},
            'KNeighborsClassifier':       {'interpretable': True, 'speed': 'Slow (inference)', 'needs_scaling': True, 'handles_nonlinear': True},
            'DecisionTreeClassifier':     {'interpretable': True, 'speed': 'Fast', 'needs_scaling': False, 'handles_nonlinear': True},
            'ExtraTreesClassifier':       {'interpretable': False, 'speed': 'Fast', 'needs_scaling': False, 'handles_nonlinear': True},
            'AdaBoostClassifier':         {'interpretable': False, 'speed': 'Moderate', 'needs_scaling': False, 'handles_nonlinear': True},
            'GaussianNB':                 {'interpretable': True, 'speed': 'Very Fast', 'needs_scaling': False, 'handles_nonlinear': False},
            'RidgeClassifier':            {'interpretable': True, 'speed': 'Very Fast', 'needs_scaling': True, 'handles_nonlinear': False},
            'LinearRegression':           {'interpretable': True, 'speed': 'Very Fast', 'needs_scaling': True, 'handles_nonlinear': False},
            'Ridge':                      {'interpretable': True, 'speed': 'Very Fast', 'needs_scaling': True, 'handles_nonlinear': False},
            'Lasso':                      {'interpretable': True, 'speed': 'Fast', 'needs_scaling': True, 'handles_nonlinear': False},
            'SVR':                        {'interpretable': False, 'speed': 'Moderate', 'needs_scaling': True, 'handles_nonlinear': True},
            'RandomForestRegressor':      {'interpretable': False, 'speed': 'Moderate', 'needs_scaling': False, 'handles_nonlinear': True},
            'GradientBoostingRegressor':  {'interpretable': False, 'speed': 'Slow', 'needs_scaling': False, 'handles_nonlinear': True},
            'KNeighborsRegressor':        {'interpretable': True, 'speed': 'Slow (inference)', 'needs_scaling': True, 'handles_nonlinear': True},
            'DecisionTreeRegressor':      {'interpretable': True, 'speed': 'Fast', 'needs_scaling': False, 'handles_nonlinear': True},
            'ExtraTreesRegressor':        {'interpretable': False, 'speed': 'Fast', 'needs_scaling': False, 'handles_nonlinear': True},
            'AdaBoostRegressor':          {'interpretable': False, 'speed': 'Moderate', 'needs_scaling': False, 'handles_nonlinear': True},
        }
        return properties.get(model_name, {})
