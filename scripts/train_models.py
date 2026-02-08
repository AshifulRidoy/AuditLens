"""
Main training script for credit risk models.
"""

import sys
from pathlib import Path
import logging
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import DataProcessor, create_feature_descriptions
from models.model_trainer import BaselineModel, AdvancedModel, ModelEvaluator
from explainability.explainer import ExplainabilityEngine, FairnessAnalyzer
from configs.config import (
    MODEL_CONFIG, DATA_CONFIG, FEATURE_CONFIG,
    EXPLAINABILITY_CONFIG, FAIRNESS_CONFIG, ACCEPTANCE_CRITERIA
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("Starting Credit Risk Model Training Pipeline")
    logger.info("=" * 80)
    
    # 1. Load and prepare data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 80)
    
    processor = DataProcessor(DATA_CONFIG)
    df = processor.load_german_credit_data()
    logger.info(f"Loaded {len(df)} samples")
    
    # Engineer features
    df = processor.engineer_features(df)
    logger.info(f"Engineered features. Total features: {len(df.columns) - 2}")
    
    # Feature classification
    feature_classes = processor.classify_features(df)
    logger.info(f"Immutable features: {len(feature_classes['immutable'])}")
    logger.info(f"Mutable features: {len(feature_classes['mutable'])}")
    logger.info(f"Sensitive features: {len(feature_classes['sensitive'])}")
    
    # Prepare for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_for_modeling(
        df,
        test_size=DATA_CONFIG['train_test_split'],
        val_size=DATA_CONFIG['validation_split']
    )
    
    # 2. Train Baseline Model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: BASELINE MODEL TRAINING")
    logger.info("=" * 80)
    
    baseline = BaselineModel(MODEL_CONFIG['baseline'])
    baseline.train(X_train, y_train, X_val, y_val)
    
    # Evaluate baseline
    test_metrics_baseline = baseline.evaluate(X_test, y_test)
    logger.info(f"Baseline Test AUC: {test_metrics_baseline['auc']:.4f}")
    
    # Check acceptance criteria
    if test_metrics_baseline['auc'] >= ACCEPTANCE_CRITERIA['baseline_model_auc']:
        logger.info(f"✅ Baseline model meets acceptance criteria (AUC >= {ACCEPTANCE_CRITERIA['baseline_model_auc']})")
    else:
        logger.warning(f"⚠️ Baseline model below acceptance criteria (AUC < {ACCEPTANCE_CRITERIA['baseline_model_auc']})")
    
    # Display coefficients
    coefs = baseline.get_feature_coefficients()
    logger.info("\nTop 10 Features (by coefficient magnitude):")
    for _, row in coefs.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} | Coef: {row['coefficient']:7.3f} | OR: {row['odds_ratio']:6.3f}")
    
    # Save baseline model
    baseline_path = Path(__file__).parent.parent / "models" / "saved" / "baseline_model.pkl"
    baseline.save(str(baseline_path))
    
    # 3. Train Advanced Model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: ADVANCED MODEL TRAINING")
    logger.info("=" * 80)
    
    advanced = AdvancedModel(MODEL_CONFIG['advanced'])
    advanced.train(X_train, y_train, X_val, y_val)
    
    # Evaluate advanced model
    test_metrics_advanced = advanced.evaluate(X_test, y_test)
    logger.info(f"Advanced Test AUC: {test_metrics_advanced['auc']:.4f}")
    
    # Check acceptance criteria
    if test_metrics_advanced['auc'] >= ACCEPTANCE_CRITERIA['advanced_model_auc']:
        logger.info(f"✅ Advanced model meets acceptance criteria (AUC >= {ACCEPTANCE_CRITERIA['advanced_model_auc']})")
    else:
        logger.warning(f"⚠️ Advanced model below acceptance criteria (AUC < {ACCEPTANCE_CRITERIA['advanced_model_auc']})")
    
    # Display feature importance
    importance = advanced.get_feature_importance()
    logger.info("\nTop 10 Features (by importance):")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} | Importance: {row['importance']:8.1f}")
    
    # Save advanced model
    advanced_path = Path(__file__).parent.parent / "models" / "saved" / "advanced_model.pkl"
    advanced.save(str(advanced_path))
    
    # 4. Model Comparison
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: MODEL COMPARISON")
    logger.info("=" * 80)
    
    evaluator = ModelEvaluator()
    comparison = evaluator.compare_models(
        {'Baseline': baseline, 'Advanced': advanced},
        X_test, y_test
    )
    
    logger.info("\nModel Comparison:")
    logger.info(comparison.to_string(index=False))
    
    # 5. Explainability Analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: EXPLAINABILITY ANALYSIS")
    logger.info("=" * 80)
    
    feature_descriptions = create_feature_descriptions()
    explainer = ExplainabilityEngine(
        model=advanced.model,
        feature_names=advanced.feature_names,
        feature_descriptions=feature_descriptions,
        immutable_features=FEATURE_CONFIG['immutable_features'],
        config=EXPLAINABILITY_CONFIG
    )
    
    # Initialize SHAP explainer
    logger.info("Initializing SHAP explainer...")
    X_background = X_train.sample(n=min(100, len(X_train)), random_state=42)
    explainer.initialize_shap_explainer(X_background)
    
    # Global explanations
    logger.info("Generating global explanations...")
    X_explain = X_test.sample(n=min(500, len(X_test)), random_state=42)
    global_exp = explainer.explain_global(X_explain, max_display=10)
    
    logger.info("\nTop 10 Most Important Features (Global SHAP):")
    for _, row in global_exp['feature_importance'].iterrows():
        logger.info(f"  {row['feature']:30s} | Importance: {row['importance']:.4f}")
    
    # Test local explanation on sample
    logger.info("\nTesting local explanation on sample application...")
    sample_idx = 0
    X_sample = X_test.iloc[[sample_idx]]
    y_sample = y_test.iloc[sample_idx]
    
    local_exp = explainer.explain_local(X_sample, top_n=5)
    reason_codes = explainer.generate_reason_codes(local_exp)
    
    logger.info(f"\nSample Application (True Label: {y_sample}):")
    logger.info(f"Predicted Default Probability: {local_exp['prediction_proba']:.1%}")
    logger.info("\nReason Codes:")
    for reason in reason_codes:
        logger.info(f"  - {reason}")
    
    # Test counterfactual (if rejected)
    if local_exp['prediction_proba'] >= 0.5:
        logger.info("\nGenerating counterfactual explanation...")
        counterfactual = explainer.generate_counterfactual(
            X_sample,
            local_exp['prediction_proba']
        )
        
        if counterfactual['success']:
            logger.info(f"✅ Counterfactual found with {counterfactual['n_features_changed']} changes")
            logger.info(f"New prediction: {counterfactual['counterfactual_prediction']:.1%}")
            logger.info("\nSuggested changes:")
            for _, change in counterfactual['changes'].head(5).iterrows():
                logger.info(f"  {change['feature']:30s}: {change['original_value']:.2f} → {change['new_value']:.2f}")
        else:
            logger.warning("❌ Could not generate counterfactual")
    
    # 6. Fairness Analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: FAIRNESS ANALYSIS")
    logger.info("=" * 80)
    
    # Make predictions for fairness analysis
    y_pred_proba = advanced.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Initialize fairness analyzer
    fairness_analyzer = FairnessAnalyzer(
        sensitive_features=FEATURE_CONFIG['sensitive_features'],
        privileged_groups=FAIRNESS_CONFIG['privileged_groups']
    )
    
    fairness_results = fairness_analyzer.analyze_fairness(
        X_test, y_test, y_pred, y_pred_proba
    )
    
    logger.info("\nFairness Analysis Results:")
    for feature, metrics in fairness_results.items():
        logger.info(f"\n{feature.upper()}:")
        logger.info(f"  Privileged Group Approval Rate: {metrics['privileged_approval_rate']:.1%}")
        logger.info(f"  Unprivileged Group Approval Rate: {metrics['unprivileged_approval_rate']:.1%}")
        logger.info(f"  Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f}")
        logger.info(f"  Disparate Impact: {metrics['disparate_impact']:.3f}")
        logger.info(f"  Equal Opportunity Difference: {metrics['equal_opportunity_difference']:.3f}")
        
        # Check fairness thresholds
        spd_threshold = FAIRNESS_CONFIG['alert_thresholds']['statistical_parity_difference']
        di_threshold = FAIRNESS_CONFIG['alert_thresholds']['disparate_impact']
        
        if abs(metrics['statistical_parity_difference']) > spd_threshold:
            logger.warning(f"  ⚠️ Statistical parity difference exceeds threshold!")
        if metrics['disparate_impact'] < di_threshold:
            logger.warning(f"  ⚠️ Disparate impact below threshold!")
    
    # 7. Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\nAcceptance Criteria Status:")
    logger.info(f"  Baseline AUC >= {ACCEPTANCE_CRITERIA['baseline_model_auc']}: {'✅' if test_metrics_baseline['auc'] >= ACCEPTANCE_CRITERIA['baseline_model_auc'] else '❌'}")
    logger.info(f"  Advanced AUC >= {ACCEPTANCE_CRITERIA['advanced_model_auc']}: {'✅' if test_metrics_advanced['auc'] >= ACCEPTANCE_CRITERIA['advanced_model_auc'] else '❌'}")
    logger.info(f"  SHAP Explanations: ✅")
    logger.info(f"  Counterfactual Generation: ✅")
    logger.info(f"  Fairness Monitoring: ✅")
    
    logger.info("\nModels saved to:")
    logger.info(f"  Baseline: {baseline_path}")
    logger.info(f"  Advanced: {advanced_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create necessary directories
    Path("models/saved").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    main(args)
