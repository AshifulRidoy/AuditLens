"""
Configuration settings for the Credit Risk Explainer system.
"""

# Model Configuration
MODEL_CONFIG = {
    "baseline": {
        "algorithm": "logistic_regression",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    },
    "advanced": {
        "algorithm": "lightgbm",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced"
    }
}

# Data Configuration
DATA_CONFIG = {
    "datasets": {
        "home_credit": {
            "url": "https://www.kaggle.com/c/home-credit-default-risk/data",
            "target_column": "TARGET",
            "application_id": "SK_ID_CURR"
        },
        "german_credit": {
            "url": "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)",
            "target_column": "target_default",
            "application_id": "application_id"
        }
    },
    "train_test_split": 0.2,
    "validation_split": 0.2,
    "random_state": 42
}

# Feature Configuration
FEATURE_CONFIG = {
    "immutable_features": [
        "age",
        "past_defaults",
        "years_since_first_credit",
        "total_past_due_events"
    ],
    "mutable_features": [
        "income",
        "debt_to_income_ratio",
        "employment_length",
        "credit_utilization",
        "num_open_accounts"
    ],
    "sensitive_features": [
        "gender",
        "age_group",
        "region"
    ]
}

# Explainability Configuration
EXPLAINABILITY_CONFIG = {
    "shap": {
        "algorithm": "tree",  # or "kernel" for model-agnostic
        "check_additivity": False,
        "feature_perturbation": "tree_path_dependent"
    },
    "lime": {
        "num_features": 10,
        "num_samples": 5000
    },
    "reason_codes": {
        "top_n_features": 5,
        "threshold": 0.05  # Minimum absolute SHAP value to include
    },
    "counterfactual": {
        "max_iterations": 1000,
        "distance_metric": "euclidean",
        "max_features_to_change": 3,
        "tolerance": 0.01
    }
}

# Fairness Configuration
FAIRNESS_CONFIG = {
    "protected_attributes": ["gender", "age_group", "region"],
    "privileged_groups": {
        "gender": [1],  # Example: Male = 1
        "age_group": [2, 3]  # Example: Middle-aged groups
    },
    "metrics": [
        "statistical_parity_difference",
        "equal_opportunity_difference",
        "disparate_impact"
    ],
    "alert_thresholds": {
        "statistical_parity_difference": 0.1,
        "equal_opportunity_difference": 0.1,
        "disparate_impact": 0.8
    }
}

# Model Monitoring Configuration
MONITORING_CONFIG = {
    "drift_detection": {
        "method": "ks_test",
        "threshold": 0.05,
        "window_size": 1000
    },
    "performance_monitoring": {
        "metrics": ["auc", "accuracy", "precision", "recall", "f1"],
        "alert_thresholds": {
            "auc_drop": 0.05,
            "accuracy_drop": 0.05
        }
    }
}

# UI Configuration
UI_CONFIG = {
    "framework": "streamlit",
    "title": "Why Was This Loan Approved?",
    "subtitle": "Explainable Credit Risk Decision System",
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730"
    },
    "max_file_upload_mb": 200
}

# Performance Requirements
PERFORMANCE_CONFIG = {
    "inference_timeout_ms": 500,
    "shap_timeout_ms": 2000,
    "counterfactual_timeout_ms": 5000,
    "ui_response_timeout_ms": 3000
}

# Deployment Configuration
DEPLOYMENT_CONFIG = {
    "platform": "streamlit_cloud",  # or "heroku", "aws_free_tier"
    "port": 8501,
    "max_memory_mb": 512,
    "storage": "local"  # or "s3", "gcs"
}

# Acceptance Criteria Thresholds
ACCEPTANCE_CRITERIA = {
    "baseline_model_auc": 0.70,
    "advanced_model_auc": 0.75,
    "auc_std_dev": 0.03,
    "calibration_slope_target": 1.0,
    "calibration_slope_tolerance": 0.1,
    "counterfactual_success_rate": 1.0,
    "max_features_changed": 3
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/credit_risk_explainer.log"
}
