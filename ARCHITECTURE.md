# OpenClaw Credit Risk Agent — Architecture Document

## 1. Executive Summary

This document describes the architectural design of the **OpenClaw-based Agentic Layer** wrapped around the existing Lending Club Credit Risk / Expected Loss framework.

The architecture strictly adheres to a **zero-modification policy** on the existing parent project:
- All parent credit risk model files, trained model artifacts, CSV scorecards, notebooks, and reference guides remain **read-only source material**.
- The new agentic layer resides entirely inside `openclaw_credit_risk_agent/` with bundled standalone model artifacts in `data/`.
- All numerical and credit predictions (PD, LGD, EAD, EL, SHAP, PSI, Stress Testing) are executed by deterministic Python functions. **The LLM never calculates or overrides numerical model values.**

---

## 2. Existing Credit Risk Model Architecture

The parent repository implements an Internal Ratings-Based (IRB) / Basel III and IFRS 9 compliant credit risk calculation pipeline based on over 115,000 Lending Club loan records (2007–2018).

### 2.1 Expected Loss Decomposition
$$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$

| Component | Definition | Parent Implementation | Key Output Metric |
| :--- | :--- | :--- | :--- |
| **PD** (Probability of Default) | Likelihood of default within a 12-month horizon | Logistic Regression Scorecard with Weight of Evidence (WoE) binning & Information Value (IV) selection | Score (300–850), PD ($0.0–1.0$) |
| **LGD** (Loss Given Default) | Economic loss percentage after recovery | Two-stage hurdle model: Stage 1 (Logistic classifier for recovery probability) $\times$ Stage 2 (Linear regression for recovery rate) | Recovery Rate ($0.0–1.0$), LGD = $1 - \text{Recovery Rate}$ |
| **EAD** (Exposure at Default) | Dollar exposure at the moment of default | Credit Conversion Factor (CCF) Linear Regression bounded in $[0, 1]$ | CCF ($0.0–1.0$), EAD = $\text{CCF} \times \text{Funded Amount}$ |

### 2.2 Challenger ML Models & Explainability
The challenger pipeline in `ml_challenger_lending_club.py` and serialized artifacts in `data/ml_output/` (or `ml_challenger_output/`) provide:
- **XGBoost Classifier** (`xgboost.joblib`): Optimized gradient boosting model capturing non-linear interactions.
- **Random Forest Classifier** (`random_forest.joblib`): Non-linear ensemble benchmark.
- **Challenger Logistic Regression** (`logistic_regression.joblib`): Standard scaled continuous feature baseline with `scaler.joblib`.
- **SHAP Explainability** (`shap.TreeExplainer`): Game-theoretic feature attributions providing global feature importance and individual waterfall explanations.
- **Population Stability Index (PSI)**: Quantifies drift between development baseline and live applicant populations.
- **Macroeconomic Stress Testing**: Simulates adverse macro shocks (e.g. $-2\%$ GDP, $+5\%$ unemployment) on baseline default probabilities.

---

## 3. Existing Model Entry Points & Artifacts

The agent layer interfaces with the following model artifacts (bundled in `data/` for standalone operation, with fallback to parent workspace):

```text
openclaw_credit_risk_agent/
│
├── data/
│   ├── df_scorecard.csv                <-- WoE Scorecard coefficients & score point table (103 bins)
│   └── ml_output/
│       ├── xgboost.joblib              <-- Serialized XGBoost model (32 origination features)
│       ├── random_forest.joblib        <-- Serialized Random Forest model
│       ├── logistic_regression.joblib  <-- Serialized standard Logistic Regression model
│       ├── scaler.joblib               <-- Fitted StandardScaler for continuous features
│       ├── model_comparison.csv        <-- Pre-computed benchmark metrics
│       ├── threshold_analysis.csv      <-- Precision/Recall/F1 at operating thresholds
│       └── psi_per_feature.csv         <-- Baseline PSI distribution data
```

### Untouched Files
All parent files are treated as immutable read-only assets:
- `df_scorecard.csv`
- `ml_challenger_lending_club.py`
- `ml_challenger_output/*`
- `*.ipynb` (all Jupyter notebooks)
- `credit_risk_doc.*`, `credit_risk_interview_study_guide.md`, `walkthrough.md`

---

## 4. Target OpenClaw Multi-Agent Architecture

```text
                           ┌─────────────────────────────────────────┐
                           │              Client Layer               │
                           │  • CLI Demo (`app.py`)                  │
                           │  • FastAPI Service (`POST /assess`)     │
                           └────────────────────┬────────────────────┘
                                                │
                                                ▼
                           ┌─────────────────────────────────────────┐
                           │      OpenClaw Orchestrator Agent        │
                           │  (Groq LLM Reasoning + Agent Router)    │
                           └────────────────────┬────────────────────┘
                                                │
       ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
       │                  │                     │                     │                  │
       ▼                  ▼                     ▼                     ▼                  ▼
 ┌────────────┐    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐    ┌────────────┐
 │ Data Agent │    │ Risk Model  │       │ Model Comp. │       │ Explain.    │    │ Policy /   │
 │ (Validate  │    │ Agent (PD,  │       │ Agent (LR / │       │ Agent       │    │ Stress /   │
 │ & Binning) │    │ LGD,EAD,EL) │       │ RF / XGB)   │       │ (SHAP)      │    │ Monitoring │
 └─────┬──────┘    └──────┬──────┘       └──────┬──────┘       └──────┬──────┘    └─────┬──────┘
       │                  │                     │                     │                 │
       └──────────────────┴─────────────────────┼─────────────────────┴─────────────────┘
                                                │
                                                ▼
                          ┌───────────────────────────────────────────┐
                          │   OpenClaw Python Tool Adapter Layer      │
                          │   (`openclaw_credit_risk_agent/tools/`)   │
                          │   • predict_pd()                          │
                          │   • predict_lgd()                         │
                          │   • predict_ead()                         │
                          │   • calculate_expected_loss()             │
                          │   • run_full_credit_assessment()          │
                          │   • explain_prediction()                  │
                          │   • compare_models()                      │
                          │   • run_stress_test()                     │
                          │   • calculate_psi()                       │
                          └─────────────────────┬─────────────────────┘
                                                │ Direct read / import
                                                ▼
                          ┌───────────────────────────────────────────┐
                          │    Existing Deterministic Credit Engine   │
                          │    • df_scorecard.csv                     │
                          │    • ml_challenger_output/*.joblib        │
                          │    • LGD 2-stage & EAD CCF models         │
                          └───────────────────────────────────────────┘
```

---

## 5. Logical Agent Responsibilities

1. **Orchestrator Agent**:
   - Manages the lifecycle of a credit risk request.
   - Coordinates sub-agents and compiles a unified assessment memorandum with executive summary, model breakdown, policy decision, adverse action reasons, and audit tags.
2. **Data Agent**:
   - Validates applicant schemas, checks numerical bounds (income, loan amount, DTI), performs term/emp_length parsing, and maps raw inputs to scorecard bins and challenger feature vectors.
3. **Risk Model Agent**:
   - Executes deterministic calls for PD, LGD, EAD, Credit Score, Risk Grade, and Expected Loss.
4. **Model Comparison Agent**:
   - Runs side-by-side evaluations comparing the regulatory Logistic Scorecard against the XGBoost, Random Forest, and Challenger Logistic models.
5. **Explainability Agent**:
   - Computes SHAP force/waterfall values and generates human-readable top risk drivers compliant with adverse action regulations.
6. **Stress Testing Agent**:
   - Evaluates portfolio/applicant performance under macroeconomic stress scenarios (Baseline, Mild Downturn, Severe Recession).
7. **Monitoring Agent**:
   - Evaluates population stability (PSI) for applicant cohorts against development baselines.
8. **Policy Agent**:
   - Evaluates underwriting business rules (Auto-Approve, Manual Review, Auto-Reject) independently of model predictions.

---

## 6. Guardrails & Audit Trail

- **Determinism**: All quantitative metrics are computed by Python code.
- **Privacy & Security**: Zero secret leakage. Environment variable `GROQ_API_KEY` is never printed or logged.
- **Auditability**: Every assessment is recorded with a unique ID, timestamp, input feature snapshot, exact model predictions, policy evaluation, and tool calls in `openclaw_credit_risk_agent/audit_logs/`.
