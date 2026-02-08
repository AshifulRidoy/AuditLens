"""
Data processing utilities for the Credit Risk Explainer system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering for credit risk models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataProcessor.
        
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def load_german_credit_data(self) -> pd.DataFrame:
        """
        Load and preprocess the German Credit dataset.
        Creates a synthetic version for demonstration purposes.
        
        Returns:
            DataFrame with preprocessed German Credit data
        """
        logger.info("Loading German Credit dataset...")
        
        # Create synthetic German Credit-like data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'application_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 75, n_samples),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'employment_length': np.random.randint(0, 40, n_samples),
            'debt_to_income_ratio': np.random.uniform(0, 0.8, n_samples),
            'credit_utilization': np.random.uniform(0, 1, n_samples),
            'num_open_accounts': np.random.randint(0, 15, n_samples),
            'past_defaults': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'years_since_first_credit': np.random.randint(0, 50, n_samples),
            'total_past_due_events': np.random.poisson(0.5, n_samples),
            'num_credit_inquiries': np.random.poisson(2, n_samples),
            'has_mortgage': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'has_dependents': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'education_level': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'gender': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'region': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.25, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on risk factors
        risk_score = (
            -0.3 * (df['income'] / df['income'].max()) +
            0.4 * df['debt_to_income_ratio'] +
            0.3 * df['credit_utilization'] +
            0.5 * df['past_defaults'] +
            0.2 * (df['num_credit_inquiries'] / 10) +
            -0.2 * (df['employment_length'] / df['employment_length'].max()) +
            np.random.normal(0, 0.2, n_samples)
        )
        
        df['target_default'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        logger.info(f"Loaded {len(df)} samples with {df['target_default'].mean():.2%} default rate")
        return df
    
    def classify_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Classify features into immutable, mutable, and sensitive categories.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature classifications
        """
        immutable_features = [
            'age', 'past_defaults', 'years_since_first_credit', 
            'total_past_due_events'
        ]
        
        sensitive_features = ['gender', 'region']
        
        # All numeric features except immutable and sensitive
        all_features = [col for col in df.columns 
                       if col not in ['application_id', 'target_default']]
        
        mutable_features = [f for f in all_features 
                           if f not in immutable_features + sensitive_features]
        
        return {
            'immutable': immutable_features,
            'mutable': mutable_features,
            'sensitive': sensitive_features,
            'all': all_features
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Engineering additional features...")
        
        df = df.copy()
        
        # Age groups for fairness analysis
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 25, 35, 50, 100], 
                                 labels=[1, 2, 3, 4])
        df['age_group'] = df['age_group'].astype(int)
        
        # Income-to-debt ratio (inverse of debt-to-income)
        df['income_to_debt_ratio'] = 1 / (df['debt_to_income_ratio'] + 0.01)
        
        # Credit history length
        df['credit_history_length'] = df['years_since_first_credit']
        
        # Account utilization intensity
        df['account_utilization_intensity'] = (
            df['credit_utilization'] * df['num_open_accounts']
        )
        
        # Employment stability indicator
        df['employment_stability'] = (df['employment_length'] > 5).astype(int)
        
        # Risk interaction features
        df['high_util_high_debt'] = (
            (df['credit_utilization'] > 0.7) & 
            (df['debt_to_income_ratio'] > 0.5)
        ).astype(int)
        
        return df
    
    def prepare_for_modeling(
        self, 
        df: pd.DataFrame,
        target_col: str = 'target_default',
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for modeling with train/val/test splits.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing data for modeling...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['application_id', target_col]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store feature names
        self.feature_names = feature_cols
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical variables
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_val, X_test)
        """
        logger.info("Scaling numerical features...")
        
        # Fit scaler on training data only
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        # Scale only numerical features
        num_features = [f for f in self.numerical_features if f in X_train.columns]
        
        X_train_scaled[num_features] = self.scaler.fit_transform(X_train[num_features])
        X_val_scaled[num_features] = self.scaler.transform(X_val[num_features])
        X_test_scaled[num_features] = self.scaler.transform(X_test[num_features])
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def get_feature_info(self) -> Dict:
        """
        Get information about features.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'all_features': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'n_features': len(self.feature_names)
        }


def create_feature_descriptions() -> Dict[str, str]:
    """
    Create human-readable descriptions for features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'age': 'Applicant age in years',
        'income': 'Annual income in currency units',
        'employment_length': 'Years in current employment',
        'debt_to_income_ratio': 'Total debt divided by annual income',
        'credit_utilization': 'Credit card balance divided by credit limit',
        'num_open_accounts': 'Number of currently open credit accounts',
        'past_defaults': 'Number of previous loan defaults',
        'years_since_first_credit': 'Years since first credit account opened',
        'total_past_due_events': 'Total number of past-due payment events',
        'num_credit_inquiries': 'Number of recent credit inquiries',
        'has_mortgage': 'Has an active mortgage (1=Yes, 0=No)',
        'has_dependents': 'Has financial dependents (1=Yes, 0=No)',
        'education_level': 'Education level (1=High School, 2=Associate, 3=Bachelor, 4=Graduate)',
        'gender': 'Gender (0=Female, 1=Male)',
        'region': 'Geographic region code',
        'age_group': 'Age category (1=18-25, 2=26-35, 3=36-50, 4=51+)',
        'income_to_debt_ratio': 'Inverse of debt-to-income ratio',
        'credit_history_length': 'Length of credit history in years',
        'account_utilization_intensity': 'Product of credit utilization and number of accounts',
        'employment_stability': 'Employment tenure > 5 years (1=Yes, 0=No)',
        'high_util_high_debt': 'High credit utilization and high debt (1=Yes, 0=No)'
    }
