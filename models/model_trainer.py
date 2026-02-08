"""
Model training module for baseline and advanced credit risk models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Base class for credit risk models.
    """

    def __init__(self, model_type: str, config: Dict):
        """
        Initialize the credit risk model.

        Args:
            model_type: Type of model ('baseline' or 'advanced')
            config: Model configuration dictionary
        """
        self.model_type = model_type
        self.config = config
        self.model = None
        self.feature_names = None
        self.performance_metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features

        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features
            threshold: Classification threshold

        Returns:
            Array of predicted labels
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of performance metrics
        """
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            "auc": roc_auc_score(y, y_pred_proba),
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }

        return metrics

    def save(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "config": self.config,
                "model_type": self.model_type,
                "performance_metrics": self.performance_metrics,
            },
            filepath,
        )
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the model from disk.

        Args:
            filepath: Path to load the model from
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.config = data["config"]
        self.model_type = data["model_type"]
        self.performance_metrics = data.get("performance_metrics", {})
        logger.info(f"Model loaded from {filepath}")


class BaselineModel(CreditRiskModel):
    """
    Interpretable baseline model using Logistic Regression.
    """

    def __init__(self, config: Dict):
        """
        Initialize the baseline model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__("baseline", config)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Training baseline logistic regression model...")

        self.feature_names = X_train.columns.tolist()

        # Initialize and train logistic regression
        self.model = LogisticRegression(
            solver=self.config.get("solver", "lbfgs"),
            max_iter=self.config.get("max_iter", 1000),
            class_weight=self.config.get("class_weight", "balanced"),
            random_state=self.config.get("random_state", 42),
        )

        self.model.fit(X_train, y_train)

        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        logger.info(f"Training AUC: {train_metrics['auc']:.4f}")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation AUC: {val_metrics['auc']:.4f}")
            self.performance_metrics["validation"] = val_metrics

        self.performance_metrics["training"] = train_metrics

    def get_feature_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients as a DataFrame.

        Returns:
            DataFrame with feature names, coefficients, and odds ratios
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        coef_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.model.coef_[0],
                "odds_ratio": np.exp(self.model.coef_[0]),
                "abs_coefficient": np.abs(self.model.coef_[0]),
            }
        )

        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
        return coef_df


class AdvancedModel(CreditRiskModel):
    """
    Advanced ML model using LightGBM.
    """

    def __init__(self, config: Dict):
        """
        Initialize the advanced model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__("advanced", config)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Training advanced LightGBM model...")

        self.feature_names = X_train.columns.tolist()

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": self.config.get("num_leaves", 31),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "feature_fraction": self.config.get("colsample_bytree", 0.8),
            "bagging_fraction": self.config.get("subsample", 0.8),
            "bagging_freq": 5,
            "max_depth": self.config.get("max_depth", 6),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "verbose": -1,
            "random_state": self.config.get("random_state", 42),
        }

        # Handle class imbalance
        if self.config.get("class_weight") == "balanced":
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            params["scale_pos_weight"] = scale_pos_weight

        # Train with early stopping if validation set provided
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.get("n_estimators", 200),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Evaluate performance
        train_metrics = self.evaluate(X_train, y_train)
        logger.info(f"Training AUC: {train_metrics['auc']:.4f}")

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation AUC: {val_metrics['auc']:.4f}")
            self.performance_metrics["validation"] = val_metrics

        self.performance_metrics["training"] = train_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for LightGBM model.

        Args:
            X: Features

        Returns:
            Array of predicted probabilities [negative_class, positive_class]
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # LightGBM Booster returns single column of probabilities for binary classification
        pos_proba = self.model.predict(X, num_iteration=self.model.best_iteration)

        # Handle both single predictions and batch predictions
        if isinstance(pos_proba, (int, float)):
            pos_proba = np.array([pos_proba])

        neg_proba = 1 - pos_proba

        return np.vstack([neg_proba, pos_proba]).T

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance from the LightGBM model.

        Args:
            importance_type: Type of importance ('gain', 'split', or 'weight')

        Returns:
            DataFrame with feature names and importance values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance = self.model.feature_importance(importance_type=importance_type)

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        )

        importance_df = importance_df.sort_values("importance", ascending=False)
        return importance_df


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    """

    @staticmethod
    def evaluate_calibration(
        y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate probability calibration.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration metrics
        """
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy="quantile"
        )

        # Calculate calibration slope (ideal = 1.0)
        from scipy.stats import linregress

        slope, intercept, r_value, _, _ = linregress(prob_pred, prob_true)

        return {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibration_r2": r_value**2,
        }

    @staticmethod
    def compare_models(
        models: Dict[str, CreditRiskModel], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models on test data.

        Args:
            models: Dictionary of model name to model instance
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame comparing model performances
        """
        results = []

        for name, model in models.items():
            metrics = model.evaluate(X_test, y_test)
            metrics["model"] = name
            results.append(metrics)

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[
            ["model", "auc", "accuracy", "precision", "recall", "f1"]
        ]

        return comparison_df
