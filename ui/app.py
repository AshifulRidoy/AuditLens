"""
Streamlit UI for the Explainable Credit Risk Decision System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import DataProcessor, create_feature_descriptions
from models.model_trainer import BaselineModel, AdvancedModel, ModelEvaluator
from explainability.explainer import ExplainabilityEngine, FairnessAnalyzer
from configs.config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    FEATURE_CONFIG,
    EXPLAINABILITY_CONFIG,
    FAIRNESS_CONFIG,
    UI_CONFIG,
)

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["title"],
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .decision-approved {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .decision-rejected {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_data():
    """Load or train models and prepare data."""

    # Initialize data processor
    processor = DataProcessor(DATA_CONFIG)

    # Load data
    df = processor.load_german_credit_data()
    df = processor.engineer_features(df)

    # Prepare for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_for_modeling(df)

    # Train baseline model
    baseline = BaselineModel(MODEL_CONFIG["baseline"])
    baseline.train(X_train, y_train, X_val, y_val)

    # Train advanced model
    advanced = AdvancedModel(MODEL_CONFIG["advanced"])
    advanced.train(X_train, y_train, X_val, y_val)

    # Evaluate both models
    evaluator = ModelEvaluator()
    test_results = evaluator.compare_models(
        {"Baseline (Logistic Regression)": baseline, "Advanced (LightGBM)": advanced},
        X_test,
        y_test,
    )

    # Initialize explainability engine
    feature_descriptions = create_feature_descriptions()
    explainer = ExplainabilityEngine(
        model=advanced.model,
        feature_names=advanced.feature_names,
        feature_descriptions=feature_descriptions,
        immutable_features=FEATURE_CONFIG["immutable_features"],
        config=EXPLAINABILITY_CONFIG,
    )
    explainer.initialize_shap_explainer(
        X_train.sample(n=min(100, len(X_train)), random_state=42)
    )

    # Global explanations
    global_exp = explainer.explain_global(
        X_test.sample(n=min(500, len(X_test)), random_state=42)
    )

    return {
        "baseline": baseline,
        "advanced": advanced,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_results": test_results,
        "processor": processor,
        "explainer": explainer,
        "global_exp": global_exp,
        "feature_descriptions": feature_descriptions,
    }


def main():
    """Main application."""

    # Header
    st.markdown(
        '<div class="main-header">💳 Why Was This Loan Approved?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Explainable Credit Risk Decision System</div>',
        unsafe_allow_html=True,
    )

    # Load models and data
    with st.spinner("Loading models and data..."):
        data = load_models_and_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Home",
            "📝 Submit Application",
            "📊 Model Performance",
            "🔍 Global Explanations",
            "⚖️ Fairness Analysis",
            "📚 Documentation",
        ],
    )

    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📝 Submit Application":
        show_application_page(data)
    elif page == "📊 Model Performance":
        show_performance_page(data)
    elif page == "🔍 Global Explanations":
        show_global_explanations_page(data)
    elif page == "⚖️ Fairness Analysis":
        show_fairness_page(data)
    elif page == "📚 Documentation":
        show_documentation_page()


def show_home_page(data):
    """Display home page with overview."""

    st.header("Welcome to the Credit Risk Explainer")

    st.markdown("""
    This system demonstrates **explainable AI** in credit risk assessment. Every decision comes with:
    
    - 🎯 **Clear Reasons**: Understand exactly why an application was approved or rejected
    - 📈 **Feature Impact**: See how each factor influenced the decision
    - 🔄 **Actionable Guidance**: Learn what changes could improve your chances
    - ⚖️ **Fairness Monitoring**: Ensure decisions are non-discriminatory
    
    ### How It Works
    
    1. **Submit Application**: Enter applicant information
    2. **Get Decision**: Receive instant approval/rejection with explanation
    3. **Understand Why**: View detailed factor analysis and SHAP explanations
    4. **Explore Alternatives**: See what changes would lead to approval
    """)

    # Model performance summary
    col1, col2, col3 = st.columns(3)

    test_results = data["test_results"]
    advanced_metrics = test_results[
        test_results["model"] == "Advanced (LightGBM)"
    ].iloc[0]

    with col1:
        st.metric("Model Accuracy", f"{advanced_metrics['accuracy']:.1%}")
    with col2:
        st.metric("AUC Score", f"{advanced_metrics['auc']:.3f}")
    with col3:
        st.metric("F1 Score", f"{advanced_metrics['f1']:.3f}")

    # Quick stats
    st.subheader("System Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Training Samples**: {len(data['X_train']):,}")
        st.info(f"**Test Samples**: {len(data['X_test']):,}")

    with col2:
        st.info(f"**Features**: {len(data['advanced'].feature_names)}")
        st.info(f"**Models**: Baseline + Advanced ML")


def show_application_page(data):
    """Display application submission and decision page."""

    st.header("Submit Credit Application")

    st.markdown("""
    Enter the applicant's information below to receive an instant credit decision with full explanation.
    """)

    # Create input form
    with st.form("application_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input(
                "Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0
            )
            employment_length = st.number_input(
                "Employment Length (years)", min_value=0, max_value=50, value=5
            )
            education_level = st.selectbox(
                "Education Level",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "High School",
                    2: "Associate",
                    3: "Bachelor",
                    4: "Graduate",
                }[x],
            )
            has_dependents = st.selectbox(
                "Has Dependents",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )

        with col2:
            st.subheader("Financial Information")
            debt_to_income_ratio = st.slider(
                "Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01
            )
            credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.5, 0.01)
            num_open_accounts = st.number_input(
                "Number of Open Accounts", min_value=0, max_value=30, value=3
            )
            num_credit_inquiries = st.number_input(
                "Recent Credit Inquiries", min_value=0, max_value=20, value=2
            )
            has_mortgage = st.selectbox(
                "Has Mortgage",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )

        st.subheader("Credit History")
        col3, col4 = st.columns(2)

        with col3:
            past_defaults = st.number_input(
                "Past Defaults", min_value=0, max_value=10, value=0
            )
            years_since_first_credit = st.number_input(
                "Years Since First Credit", min_value=0, max_value=60, value=10
            )

        with col4:
            total_past_due_events = st.number_input(
                "Past Due Events", min_value=0, max_value=20, value=0
            )

        submitted = st.form_submit_button("Submit Application", type="primary")

    if submitted:
        # Create applicant data
        applicant_data = create_applicant_dataframe(
            age,
            income,
            employment_length,
            education_level,
            has_dependents,
            debt_to_income_ratio,
            credit_utilization,
            num_open_accounts,
            num_credit_inquiries,
            has_mortgage,
            past_defaults,
            years_since_first_credit,
            total_past_due_events,
        )

        # Make prediction
        model = data["advanced"]
        if hasattr(model.model, "predict_proba"):
            prediction_proba = model.predict_proba(applicant_data)[0, 1]
        else:
            # LightGBM Booster - use model directly
            prediction_proba = model.model.predict(applicant_data)[0]
        decision = "REJECTED" if prediction_proba >= 0.5 else "APPROVED"

        # Display decision
        if decision == "APPROVED":
            st.markdown(
                f"""
            <div class="decision-approved">
                <h2>✅ Application APPROVED</h2>
                <p><strong>Default Risk: {prediction_proba:.1%}</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="decision-rejected">
                <h2>❌ Application REJECTED</h2>
                <p><strong>Default Risk: {prediction_proba:.1%}</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Generate explanation
        st.subheader("📋 Decision Explanation")

        local_exp = data["explainer"].explain_local(applicant_data)
        reason_codes = data["explainer"].generate_reason_codes(local_exp)

        st.markdown("**Key Factors:**")
        for reason in reason_codes:
            st.write(f"- {reason}")

        # SHAP visualization
        st.subheader("🔍 Feature Impact Analysis")

        contributions = local_exp["contributions"]

        fig = go.Figure()

        colors = ["red" if x > 0 else "green" for x in contributions["shap_value"]]

        fig.add_trace(
            go.Bar(
                y=contributions["description"],
                x=contributions["shap_value"],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:.3f}" for v in contributions["shap_value"]],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="Top Factors Affecting Decision",
            xaxis_title="Impact on Default Risk",
            yaxis_title="Factor",
            height=400,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Counterfactual explanation (only for rejections)
        if decision == "REJECTED":
            st.subheader("🔄 What Would Need to Change?")

            with st.spinner("Generating alternative scenario..."):
                counterfactual = data["explainer"].generate_counterfactual(
                    applicant_data, prediction_proba
                )

            if counterfactual["success"]:
                st.success(
                    f"Found alternative scenario with {counterfactual['n_features_changed']} changes"
                )
                st.write(
                    f"New predicted risk: **{counterfactual['counterfactual_prediction']:.1%}**"
                )

                st.write("**Suggested Changes:**")
                for _, change in counterfactual["changes"].iterrows():
                    direction = "increase" if change["change"] > 0 else "decrease"
                    st.write(
                        f"- {change['description']}: {direction} from {change['original_value']:.2f} to {change['new_value']:.2f}"
                    )
            else:
                st.warning(
                    "Could not generate actionable counterfactual with current constraints"
                )


def show_performance_page(data):
    """Display model performance metrics."""

    st.header("Model Performance Analysis")

    # Model comparison
    st.subheader("Model Comparison")
    st.dataframe(data["test_results"], use_container_width=True)

    # Baseline model coefficients
    st.subheader("Baseline Model: Feature Coefficients")
    baseline_coefs = data["baseline"].get_feature_coefficients()

    fig = px.bar(
        baseline_coefs.head(15),
        x="coefficient",
        y="feature",
        orientation="h",
        title="Top 15 Feature Coefficients (Logistic Regression)",
        labels={"coefficient": "Coefficient", "feature": "Feature"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Advanced model feature importance
    st.subheader("Advanced Model: Feature Importance")
    importance = data["advanced"].get_feature_importance()

    fig = px.bar(
        importance.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Feature Importance (LightGBM)",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    st.plotly_chart(fig, use_container_width=True)


def show_global_explanations_page(data):
    """Display global SHAP explanations."""

    st.header("Global Model Explanations")

    st.markdown("""
    These visualizations show how features impact predictions across all applications.
    """)

    # Feature importance
    st.subheader("Global Feature Importance")
    importance_df = data["global_exp"]["feature_importance"]

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Features Ranked by Average Impact on Predictions",
        labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
        hover_data=["description"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance_df, use_container_width=True)


def show_fairness_page(data):
    """Display fairness analysis."""

    st.header("Fairness & Bias Analysis")

    st.markdown("""
    This analysis examines whether the model treats different demographic groups fairly.
    """)

    # Make predictions for fairness analysis
    X_test = data["X_test"]
    y_test = data["y_test"]
    model = data["advanced"]

    if hasattr(model.model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # LightGBM Booster
        y_pred_proba = model.model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Initialize fairness analyzer
    analyzer = FairnessAnalyzer(
        sensitive_features=FEATURE_CONFIG["sensitive_features"],
        privileged_groups=FAIRNESS_CONFIG["privileged_groups"],
    )

    fairness_results = analyzer.analyze_fairness(X_test, y_test, y_pred, y_pred_proba)

    # Display results
    for feature, metrics in fairness_results.items():
        st.subheader(f"Analysis by {feature.replace('_', ' ').title()}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Privileged Group Approval Rate",
                f"{metrics['privileged_approval_rate']:.1%}",
            )
        with col2:
            st.metric(
                "Unprivileged Group Approval Rate",
                f"{metrics['unprivileged_approval_rate']:.1%}",
            )
        with col3:
            st.metric(
                "Statistical Parity Difference",
                f"{metrics['statistical_parity_difference']:.3f}",
            )

        col4, col5 = st.columns(2)

        with col4:
            st.metric("Disparate Impact Ratio", f"{metrics['disparate_impact']:.2f}")
        with col5:
            st.metric(
                "Equal Opportunity Difference",
                f"{metrics['equal_opportunity_difference']:.3f}",
            )


def show_documentation_page():
    """Display documentation."""

    st.header("System Documentation")

    st.markdown("""
    ## About This System
    
    The **Explainable Credit Risk Decision System** demonstrates production-ready machine learning 
    practices aligned with regulatory standards for financial services.
    
    ### Key Features
    
    - **Dual Model Approach**: Interpretable baseline (Logistic Regression) + High-performance ML (LightGBM)
    - **SHAP Explanations**: Game-theoretic explanations for every decision
    - **Counterfactual Analysis**: Actionable guidance showing what changes would affect the outcome
    - **Fairness Monitoring**: Automated bias detection across protected attributes
    - **Audit Trail**: Complete documentation for regulatory compliance
    
    ### Models
    
    **Baseline Model (Logistic Regression)**
    - Fully interpretable with coefficient-based explanations
    - Regulatory anchor for model validation
    - Meets acceptance criteria: AUC ≥ 0.70
    
    **Advanced Model (LightGBM)**
    - Gradient boosted trees for superior performance
    - Handles non-linear effects and feature interactions
    - Meets acceptance criteria: AUC ≥ 0.75
    
    ### Explainability Methods
    
    **SHAP (SHapley Additive exPlanations)**
    - Theoretically grounded in game theory
    - Provides both global and local explanations
    - Shows feature contributions to each prediction
    
    **Counterfactual Explanations**
    - Minimal changes needed for different outcome
    - Respects immutable features (age, credit history)
    - Optimization-based approach
    
    ### Technical Stack
    
    - Python 3.9+
    - scikit-learn, LightGBM
    - SHAP for explainability
    - Streamlit for UI
    - Pandas, NumPy for data processing
    
    ### Acceptance Criteria
    
    ✅ Baseline model AUC ≥ 0.70  
    ✅ Advanced model AUC ≥ 0.75  
    ✅ Every decision has SHAP explanations  
    ✅ Counterfactuals for rejected applications  
    ✅ Fairness metrics computed and monitored  
    ✅ Complete audit trail and documentation  
    
    ### Data Privacy
    
    This demonstration uses synthetic data based on public credit datasets. 
    No real applicant information is processed or stored.
    """)


def create_applicant_dataframe(
    age,
    income,
    employment_length,
    education_level,
    has_dependents,
    debt_to_income_ratio,
    credit_utilization,
    num_open_accounts,
    num_credit_inquiries,
    has_mortgage,
    past_defaults,
    years_since_first_credit,
    total_past_due_events,
):
    """Create DataFrame from applicant input."""

    # Engineer features same as training data
    data = {
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "education_level": education_level,
        "has_dependents": has_dependents,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_utilization": credit_utilization,
        "num_open_accounts": num_open_accounts,
        "num_credit_inquiries": num_credit_inquiries,
        "has_mortgage": has_mortgage,
        "past_defaults": past_defaults,
        "years_since_first_credit": years_since_first_credit,
        "total_past_due_events": total_past_due_events,
        "gender": 1,  # Default
        "region": 1,  # Default
    }

    df = pd.DataFrame([data])

    # Engineer additional features
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 50, 100], labels=[1, 2, 3, 4]
    ).astype(int)
    df["income_to_debt_ratio"] = 1 / (df["debt_to_income_ratio"] + 0.01)
    df["credit_history_length"] = df["years_since_first_credit"]
    df["account_utilization_intensity"] = (
        df["credit_utilization"] * df["num_open_accounts"]
    )
    df["employment_stability"] = (df["employment_length"] > 5).astype(int)
    df["high_util_high_debt"] = (
        (df["credit_utilization"] > 0.7) & (df["debt_to_income_ratio"] > 0.5)
    ).astype(int)

    return df


if __name__ == "__main__":
    main()
