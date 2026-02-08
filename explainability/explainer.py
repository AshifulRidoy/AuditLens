"""
Explainability engine providing SHAP analysis, reason codes, and counterfactuals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import shap
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generates explanations for credit risk model decisions.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_descriptions: Dict[str, str],
        immutable_features: List[str],
        config: Dict,
    ):
        """
        Initialize the explainability engine.

        Args:
            model: Trained model
            feature_names: List of feature names
            feature_descriptions: Human-readable feature descriptions
            immutable_features: List of features that cannot be changed
            config: Explainability configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.immutable_features = immutable_features
        self.config = config
        self.explainer = None
        self.base_value = None

    def initialize_shap_explainer(self, X_background: pd.DataFrame) -> None:
        """
        Initialize SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP (typically training data sample)
        """
        logger.info("Initializing SHAP explainer...")

        # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
        try:
            # Try TreeExplainer first (faster for tree models)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using SHAP TreeExplainer")
        except:
            # Fall back to KernelExplainer for other model types
            def model_predict(X):
                if isinstance(X, pd.DataFrame):
                    return self.model.predict_proba(X)[:, 1]
                else:
                    X_df = pd.DataFrame(X, columns=self.feature_names)
                    return self.model.predict_proba(X_df)[:, 1]

            # Sample background data if too large
            if len(X_background) > 100:
                X_background = X_background.sample(n=100, random_state=42)

            self.explainer = shap.KernelExplainer(model_predict, X_background)
            logger.info("Using SHAP KernelExplainer")

        # Calculate base value (expected model output)
        self.base_value = self.explainer.expected_value
        if isinstance(self.base_value, np.ndarray):
            self.base_value = self.base_value[0]

    def explain_global(self, X: pd.DataFrame, max_display: int = 20) -> Dict[str, Any]:
        """
        Generate global explanations showing overall model behavior.

        Args:
            X: Dataset to analyze
            max_display: Maximum number of features to display

        Returns:
            Dictionary with global explanation data
        """
        logger.info("Generating global SHAP explanations...")

        if self.explainer is None:
            raise ValueError(
                "SHAP explainer not initialized. Call initialize_shap_explainer first."
            )

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification

        # Calculate global feature importance (mean absolute SHAP)
        global_importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": global_importance,
                "description": [
                    self.feature_descriptions.get(f, f) for f in self.feature_names
                ],
            }
        ).sort_values("importance", ascending=False)

        return {
            "shap_values": shap_values,
            "feature_importance": importance_df.head(max_display),
            "base_value": self.base_value,
            "feature_names": self.feature_names,
        }

    def explain_local(self, X_instance: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance.

        Args:
            X_instance: Single instance to explain (as DataFrame)
            top_n: Number of top features to include in explanation

        Returns:
            Dictionary with local explanation data
        """
        if self.explainer is None:
            raise ValueError(
                "SHAP explainer not initialized. Call initialize_shap_explainer first."
            )

        # Calculate SHAP values for this instance
        shap_values = self.explainer.shap_values(X_instance)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]  # Use positive class
        else:
            shap_values = shap_values[0]

        # Get prediction - handle both sklearn and LightGBM models
        if hasattr(self.model, "predict_proba"):
            prediction_proba = self.model.predict_proba(X_instance)[0, 1]
        else:
            # LightGBM Booster object
            prediction_proba = self.model.predict(X_instance)[0]

        # Create feature contributions DataFrame
        contributions = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": X_instance.iloc[0].values,
                "shap_value": shap_values,
                "abs_shap_value": np.abs(shap_values),
                "description": [
                    self.feature_descriptions.get(f, f) for f in self.feature_names
                ],
            }
        )

        contributions = contributions.sort_values("abs_shap_value", ascending=False)

        return {
            "prediction_proba": prediction_proba,
            "base_value": self.base_value,
            "shap_values": shap_values,
            "contributions": contributions.head(top_n),
            "all_contributions": contributions,
        }

    def generate_reason_codes(
        self, local_explanation: Dict[str, Any], threshold: float = 0.05
    ) -> List[str]:
        """
        Generate human-readable reason codes from SHAP values.

        Args:
            local_explanation: Local explanation dictionary
            threshold: Minimum absolute SHAP value to include

        Returns:
            List of reason code strings
        """
        contributions = local_explanation["all_contributions"]
        prediction_proba = local_explanation["prediction_proba"]

        # Filter contributions by threshold
        significant_contributions = contributions[
            contributions["abs_shap_value"] >= threshold
        ]

        reason_codes = []

        # Add decision outcome
        decision = "APPROVED" if prediction_proba < 0.5 else "REJECTED"
        reason_codes.append(
            f"Application {decision} (Default Risk: {prediction_proba:.1%})"
        )

        # Add top contributing factors
        for _, row in significant_contributions.head(5).iterrows():
            direction = "increased" if row["shap_value"] > 0 else "decreased"
            reason_codes.append(
                f"{row['description']}: {direction} risk by {abs(row['shap_value']):.3f}"
            )

        return reason_codes

    def generate_counterfactual(
        self,
        X_instance: pd.DataFrame,
        current_prediction: float,
        target_prediction: float = 0.49,
        max_iterations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation showing minimal changes for different outcome.

        Args:
            X_instance: Instance to generate counterfactual for
            current_prediction: Current prediction probability
            target_prediction: Target prediction probability
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary with counterfactual explanation
        """
        logger.info("Generating counterfactual explanation...")

        X_original = X_instance.copy()
        feature_values = X_instance.iloc[0].values

        # Identify mutable features (indices)
        mutable_indices = [
            i
            for i, f in enumerate(self.feature_names)
            if f not in self.immutable_features
        ]

        if len(mutable_indices) == 0:
            return {
                "success": False,
                "message": "No mutable features available for counterfactual",
            }

        # Define objective function: minimize distance while achieving target
        def objective(x_new):
            # Reconstruct full feature vector
            x_full = feature_values.copy()
            x_full[mutable_indices] = x_new

            # Calculate prediction - handle both sklearn and LightGBM models
            X_new = pd.DataFrame([x_full], columns=self.feature_names)
            if hasattr(self.model, "predict_proba"):
                pred = self.model.predict_proba(X_new)[0, 1]
            else:
                # LightGBM Booster
                pred = self.model.predict(X_new)[0]

            # Distance from original
            distance = euclidean(feature_values[mutable_indices], x_new)

            # Penalty for not reaching target
            target_penalty = 1000 * max(0, pred - target_prediction) ** 2

            return distance + target_penalty

        # Initial guess: original mutable features
        x0 = feature_values[mutable_indices]

        # Define bounds (realistic feature ranges)
        bounds = []
        for idx in mutable_indices:
            # Use reasonable bounds based on feature
            feature_name = self.feature_names[idx]
            original_value = feature_values[idx]

            # Set bounds as +/- 50% of original value, or reasonable defaults
            if original_value > 0:
                lower = max(0, original_value * 0.3)
                upper = original_value * 2.0
            else:
                lower = -10
                upper = 10

            bounds.append((lower, upper))

        # Optimize
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )

        if not result.success:
            return {
                "success": False,
                "message": "Optimization failed to find counterfactual",
            }

        # Construct counterfactual instance
        x_counterfactual = feature_values.copy()
        x_counterfactual[mutable_indices] = result.x

        X_counterfactual = pd.DataFrame([x_counterfactual], columns=self.feature_names)
        if hasattr(self.model, "predict_proba"):
            cf_prediction = self.model.predict_proba(X_counterfactual)[0, 1]
        else:
            # LightGBM Booster
            cf_prediction = self.model.predict(X_counterfactual)[0]

        # Calculate changes
        changes = []
        for i, idx in enumerate(mutable_indices):
            original = feature_values[idx]
            new_value = result.x[i]

            if abs(new_value - original) > 1e-6:  # Significant change
                changes.append(
                    {
                        "feature": self.feature_names[idx],
                        "description": self.feature_descriptions.get(
                            self.feature_names[idx], self.feature_names[idx]
                        ),
                        "original_value": original,
                        "new_value": new_value,
                        "change": new_value - original,
                        "percent_change": ((new_value - original) / (original + 1e-10))
                        * 100,
                    }
                )

        changes_df = pd.DataFrame(changes).sort_values(
            "change", key=abs, ascending=False
        )

        return {
            "success": True,
            "original_prediction": current_prediction,
            "counterfactual_prediction": cf_prediction,
            "changes": changes_df,
            "n_features_changed": len(changes),
            "counterfactual_instance": X_counterfactual,
        }


class FairnessAnalyzer:
    """
    Analyzes model fairness across protected attributes.
    """

    def __init__(
        self, sensitive_features: List[str], privileged_groups: Dict[str, List]
    ):
        """
        Initialize fairness analyzer.

        Args:
            sensitive_features: List of sensitive/protected feature names
            privileged_groups: Dictionary mapping features to privileged group values
        """
        self.sensitive_features = sensitive_features
        self.privileged_groups = privileged_groups

    def analyze_fairness(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze fairness metrics across sensitive attributes.

        Args:
            X: Feature data
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary with fairness analysis results
        """
        fairness_results = {}

        for feature in self.sensitive_features:
            if feature not in X.columns:
                continue

            # Get privileged and unprivileged groups
            privileged_values = self.privileged_groups.get(feature, [])

            privileged_mask = X[feature].isin(privileged_values)
            unprivileged_mask = ~privileged_mask

            # Calculate approval rates
            privileged_approval_rate = y_pred[privileged_mask].mean()
            unprivileged_approval_rate = y_pred[unprivileged_mask].mean()

            # Statistical parity difference
            spd = privileged_approval_rate - unprivileged_approval_rate

            # Disparate impact (ratio of approval rates)
            disparate_impact = unprivileged_approval_rate / (
                privileged_approval_rate + 1e-10
            )

            # Equal opportunity (TPR difference)
            privileged_tpr = (
                y_pred[privileged_mask & (y_true == 1)].mean()
                if (privileged_mask & (y_true == 1)).sum() > 0
                else 0
            )
            unprivileged_tpr = (
                y_pred[unprivileged_mask & (y_true == 1)].mean()
                if (unprivileged_mask & (y_true == 1)).sum() > 0
                else 0
            )
            eod = privileged_tpr - unprivileged_tpr

            fairness_results[feature] = {
                "privileged_approval_rate": privileged_approval_rate,
                "unprivileged_approval_rate": unprivileged_approval_rate,
                "statistical_parity_difference": spd,
                "disparate_impact": disparate_impact,
                "equal_opportunity_difference": eod,
                "privileged_count": privileged_mask.sum(),
                "unprivileged_count": unprivileged_mask.sum(),
            }

        return fairness_results
