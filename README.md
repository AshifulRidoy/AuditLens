# Credit Risk Decision System

**Explainable, Regulator-Ready Credit Risk Decisioning**

> *"Why Was This Loan Approved?"* - A demonstration system showing how explainable AI can be embedded into production credit workflows.

---

## 🎯 Executive Summary

This system demonstrates a governance-first approach to credit risk decisioning using explainable AI. Unlike typical ML projects focused solely on predictive accuracy, this implementation prioritizes:

- **Transparency**: Every decision includes human-readable explanations
- **Accountability**: Complete audit trails for regulatory review
- **Fairness**: Ongoing bias detection and monitoring
- **Actionability**: Counterfactual guidance for declined applicants

**Key Innovation**: AI agents orchestrate governance workflows without making credit decisions themselves, automating analysis and reporting while keeping humans in control.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   CREDIT APPLICATION                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              DUAL MODEL ARCHITECTURE                     │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │  Baseline Model  │    │ Production Model │          │
│  │  (Logistic Reg)  │    │   (LightGBM)     │          │
│  │  Governance ✓    │    │  Performance ✓   │          │
│  └──────────────────┘    └──────────────────┘          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              EXPLAINABILITY LAYER                        │
│  • SHAP-based feature contributions                     │
│  • Reason code generation                               │
│  • Counterfactual scenarios                             │
│  • Global importance analysis                           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              DECISION ENGINE                             │
│  PD < 0.3:  APPROVE                                     │
│  0.3-0.7:   MANUAL REVIEW                               │
│  PD ≥ 0.7:  DECLINE                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              AI GOVERNANCE AGENTS                        │
│  🤖 Model Governance → Drift detection                  │
│  🤖 Explanation Agent → Plain-language translation       │
│  🤖 Monitoring Agent → Stability tracking               │
│  🤖 Reporting Agent → Auto-documentation                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUTS                               │
│  📊 Decision + Explanation                              │
│  📄 Adverse Action Notice                               │
│  📈 Fairness Metrics                                    │
│  📋 Governance Reports                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone or download the repository
cd credit_risk_system

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Run full pipeline with German Credit dataset
python main.py --dataset german

# Run with Home Credit dataset
python main.py --dataset home_credit
```

This will:
1. Load and prepare data
2. Train baseline and production models
3. Generate explanations for sample applications
4. Conduct fairness analysis
5. Generate governance reports
6. Save all outputs to `outputs/` directory

### Launch Interactive UI

```bash
streamlit run ui/app.py
```

Then open your browser to `http://localhost:8501`

---

## 📋 Features

### Core Decision Engine

✅ **Dual Model Architecture**
- Governance baseline: Logistic regression (interpretable, auditable)
- Production model: LightGBM (high performance)
- Automatic comparison for validation

✅ **Three-Tier Decision Framework**
- Approve: Low risk (PD < 0.3)
- Manual Review: Moderate risk (PD 0.3-0.7)
- Decline: High risk (PD ≥ 0.7)

### Explainability Framework

✅ **Local Explanations** (Per-Application)
- SHAP values for feature contributions
- Top 3-5 reason codes with impact percentages
- Directional indicators (increases/decreases risk)

✅ **Global Explanations** (Portfolio-Level)
- Feature importance rankings
- Partial dependence plots
- Sensitivity analysis

✅ **Counterfactual Scenarios**
- Minimal changes to flip decision
- Honors feature mutability constraints
- Actionable guidance for consumers

### AI Governance Agents

🤖 **Model Governance Agent**
- Compares baseline vs. production predictions
- Detects model drift
- Generates validation summaries

🤖 **Explanation Agent**
- Translates technical explanations
- Tailors tone for different audiences
- Drafts adverse action letters

🤖 **Monitoring Agent**
- Tracks explanation stability
- Detects feature importance shifts
- Monitors fairness metrics

🤖 **Reporting Agent**
- Auto-generates weekly reports
- Creates model cards
- Compiles audit summaries

### Fairness Monitoring

✅ **Demographic Parity Analysis**
- Approval rate comparison across groups
- Disparity ratio calculation
- Automated flagging of issues

✅ **Equalized Odds**
- False positive rate monitoring
- False negative rate monitoring
- Cross-group error rate comparison

✅ **Counterfactual Fairness**
- Feasibility gap analysis
- Path-to-approval equity

---

## 📁 Project Structure

```
credit_risk_system/
├── data/
│   └── ingestion.py          # Data loading and schema mapping
├── models/
│   └── training.py            # Model training (baseline + production)
├── explainability/
│   └── explanations.py        # SHAP, reason codes, counterfactuals
├── agents/
│   └── governance.py          # AI agents for governance workflows
├── utils/
│   └── fairness.py            # Fairness metrics and bias detection
├── ui/
│   └── app.py                 # Streamlit interactive dashboard
├── outputs/                   # Generated reports and artifacts
├── main.py                    # Main orchestration script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🎬 Usage Examples

### Example 1: Single Application Decision

```python
from data.ingestion import DataIngestion
from models.training import CreditRiskModels, DecisionEngine
from explainability.explanations import ExplanationGenerator

# Load data
ingestion = DataIngestion()
data, schema = ingestion.load_german_credit()
train_data, test_data = ingestion.prepare_train_test_split(data)

# Train models
models = CreditRiskModels(schema)
models.train_baseline_model(train_data)
models.train_production_model(train_data)

# Make decision
engine = DecisionEngine()
app = test_data.iloc[[0]]
pd_score = models.predict(app, 'production')[0]
decision = engine.make_decision(pd_score)

print(f"Decision: {decision} (PD: {pd_score:.3f})")
```

### Example 2: Generate Explanation

```python
# Initialize explainer
X_train = models.preprocess_features(train_data)
explainer = ExplanationGenerator(models.production_model, 'lightgbm', models.feature_names)
explainer.initialize_explainer(X_train[:100])

# Explain prediction
X_app = models.preprocess_features(app)
explanation = explainer.explain_prediction(X_app)

# Display top factors
for feature, shap_value in explanation['top_features'][:3]:
    print(f"{feature}: {shap_value:+.4f}")
```

### Example 3: Fairness Analysis

```python
from utils.fairness import FairnessMonitor

monitor = FairnessMonitor(['gender'])
test_pd = models.predict(test_data, 'production')
test_decisions = engine.classify_batch(test_pd)

parity = monitor.calculate_demographic_parity(test_data, test_decisions, 'gender')
print(f"Approval rates: {parity['approval_rates']}")
print(f"Parity achieved: {parity['parity_achieved']}")
```

---

## 📊 Interactive Dashboard

The Streamlit UI provides five views:

1. **📊 Dashboard**: Portfolio overview, decision distribution, PD score distribution
2. **🔍 Individual Application**: Detailed decision analysis with explanations and counterfactuals
3. **⚖️ Fairness Analysis**: Demographic parity and equalized odds monitoring
4. **📈 Model Performance**: AUC comparison, feature importance, model validation
5. **📄 Reports**: Weekly governance reports, model cards, audit documentation

---

## 🎯 Success Criteria

The system meets the PRD requirements:

✅ **Functional Completeness**: All FR-1 through FR-14 implemented
✅ **Explanation Quality**: Every decision includes auditable SHAP-based reason codes
✅ **Counterfactual Validity**: Realistic scenarios honoring mutability constraints
✅ **Governance Readiness**: Outputs mirror real credit risk documentation
✅ **Reproducibility**: Complete setup in requirements.txt, documented workflows
✅ **Agent Orchestration**: AI agents automate governance without making credit decisions

---

## ⚖️ Fairness and Ethics

### Fairness Monitoring
- Demographic parity tracked across all sensitive attributes
- Equalized odds analysis for error rate disparities
- Counterfactual feasibility gaps measured

### Known Limitations
- ⚠️ **Not regulatory-compliant**: Demonstration system only
- ⚠️ **Historical bias**: May reflect biases in training data
- ⚠️ **Requires validation**: All outputs need legal/compliance review for production use

### Ethical Safeguards
- Sensitive attributes excluded from model features
- Transparent explanations for all decisions
- Actionable counterfactuals provided to declined applicants
- Regular bias audits conducted

---

## 🔬 Technical Details

### Models

**Baseline Model**
- Algorithm: Logistic Regression with L2 regularization
- Purpose: Governance baseline, validation reference
- Interpretability: Fully transparent coefficients

**Production Model**
- Algorithm: LightGBM Gradient Boosting
- Purpose: Optimized predictions
- Explainability: SHAP TreeExplainer

### Explainability Methods

**SHAP (SHapley Additive exPlanations)**
- TreeExplainer for tree-based models
- Exact Shapley values
- Feature contribution decomposition

**Counterfactuals**
- Greedy search for minimal changes
- Mutability constraints enforced
- Feasibility validation

### Datasets

**German Credit (1,000 applications)**
- Purpose: Baseline governance demonstration
- Features: 12 credit risk indicators
- Default rate: ~30%

**Home Credit (5,000 applications)**
- Purpose: Production-scale complexity
- Features: 17+ risk indicators
- Default rate: ~8-12%

---

## 📈 Performance Metrics

| Metric | Baseline | Production | Lift |
|--------|----------|------------|------|
| **AUC-ROC** | >0.70 | >0.75 | +0.05 |
| **Interpretability** | Full | SHAP | Moderate |
| **Speed (single app)** | <1s | <5s | - |
| **Explanation Stability** | N/A | >0.75 | - |

---

## 🚨 Important Disclaimers

### NOT FOR PRODUCTION USE

This is a **demonstration system** showing methodology and best practices. It is:

- ❌ NOT certified for regulated lending
- ❌ NOT legally compliant without additional review
- ❌ NOT guaranteed to be bias-free
- ❌ NOT a replacement for human judgment

### Required for Production Deployment

Before using in real lending:
- ✅ Legal and compliance review
- ✅ Regulatory approval
- ✅ Fair lending testing
- ✅ Model risk management validation
- ✅ Consumer protection safeguards
- ✅ Ongoing monitoring and governance

---

## 📚 Documentation

- **README.md** (this file): Business narrative and setup
- **TECHNICAL_SPEC.md**: Detailed architecture and algorithms
- **outputs/model_card.txt**: Standardized model documentation
- **outputs/weekly_report.txt**: Sample governance report

---

## 🤝 Contributing

This is a demonstration project. For production use, consult with:
- Legal/compliance teams
- Model risk management
- Fair lending specialists
- Regulatory experts

---

## 📝 License

This is a demonstration system for educational purposes.

---

## 📧 Contact

For questions about credit risk modeling, explainable AI, or governance frameworks, consult your organization's:
- Credit Risk Analytics Team
- Model Validation Group
- Compliance Department

---

## 🙏 Acknowledgments

Built following industry best practices for:
- Explainable AI (SHAP, LIME)
- Model governance (SR 11-7, OCC 2011-12)
- Fair lending (ECOA, Regulation B)
- Consumer protection (FCRA)

**PRD Version**: 1.0  
**System Version**: 1.0  
**Last Updated**: February 2026

---

## ⚡ Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run pipeline
python main.py --dataset german

# Launch UI
streamlit run ui/app.py

# View outputs
ls outputs/
```

**That's it!** You now have a complete, explainable credit risk decision system. 🎉
