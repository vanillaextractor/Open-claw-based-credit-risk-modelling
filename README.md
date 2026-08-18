# OpenClaw Credit Risk Agent

Enterprise Multi-Agent Credit Risk Assessment Layer built around the **Lending Club Expected Loss & ML Challenger Modeling Engine**.

Strictly compliant with **Basel III Internal Ratings-Based (IRB)** and **IFRS 9 / ECL forward-looking guidelines**.

---

## 🌟 Overview

The **OpenClaw Credit Risk Agent** wraps a multi-agent orchestration layer around the deterministic credit risk models in the parent project without modifying any existing parent files.

```text
                               ┌────────────────────────────────┐
                               │       User / Client            │
                               │  (CLI `app.py` / FastAPI)      │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   OpenClaw Orchestrator Agent  │
                               │  (Groq LLM Reasoning Engine)   │
                               └───────────────┬────────────────┘
                                               │
      ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
      │                  │                     │                     │                  │
      ▼                  ▼                     ▼                     ▼                  ▼
┌────────────┐    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐    ┌────────────┐
│ Data Agent │    │ Risk Model  │       │ Model Comp. │       │ Explain.    │    │ Policy /   │
│ (Validate  │    │ Agent (PD,  │       │ Agent (LR/  │       │ Agent       │    │ Stress /   │
│ & Binning) │    │ LGD,EAD,EL) │       │ RF/XGBoost) │       │ (SHAP)      │    │ Monitoring │
└─────┬──────┘    └──────┬──────┘       └──────┬──────┘       └──────┬──────┘    └─────┬──────┘
      │                  │                     │                     │                 │
      └──────────────────┼─────────────────────┴─────────────────────┴─────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────────────┐
        │       OpenClaw Python Tool Adapter Layer               │
        │  `openclaw_credit_risk_agent/tools/`                   │
        │  • predict_pd()             • predict_lgd()            │
        │  • predict_ead()            • calculate_expected_loss()│
        │  • run_full_credit_assessment()                        │
        │  • explain_prediction()     • compare_models()         │
        │  • run_stress_test()        • calculate_psi()          │
        └────────────────────────┬───────────────────────────────┘
                                 │ imports & references (READ-ONLY)
                                 ▼
        ┌────────────────────────────────────────────────────────┐
        │       EXISTING DETERMINISTIC PARENT MODELS             │
        │  • `df_scorecard.csv` (Basel Scorecard Coefs & Points) │
        │  • `ml_challenger_output/*.joblib` (XGB, RF, LR, Scaler│
        │  • 2-Stage LGD & CCF EAD Mathematical Models           │
        │  • SHAP TreeExplainer & PSI Engine                     │
        └────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features

1. **Source of Truth Guarantee**: The LLM *never* computes or modifies numerical risk parameters. All values ($\text{PD}, \text{LGD}, \text{EAD}, \text{EL}$, Credit Score, SHAP) are produced by deterministic Python algorithms.
2. **Expected Loss Engine**:
   $$\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$$
   - **PD**: Basel III WoE Logistic Scorecard ($300-850$ credit score points).
   - **LGD**: Two-stage hurdle recovery rate model.
   - **EAD**: Credit Conversion Factor (CCF) model.
3. **ML Challenger Benchmarking**: Side-by-side comparison against XGBoost ($\text{AUC}=0.723, \text{KS}=0.326$), Random Forest, and Standard Logistic Regression.
4. **SHAP Explainability**: Game-theoretic feature attributions providing positive and negative risk drivers for ECOA/FCRA adverse action reporting.
5. **Macroeconomic Stress Testing**: Forward-looking IFRS 9 scenario analysis under $-2\%$ GDP and $+5\%$ unemployment shocks.
6. **Population Stability Index (PSI)**: Quantifies feature distribution drift against development baselines.
7. **Separate Underwriting Policy Engine**: Decouples statistical risk prediction from business lending rules (`AUTO_APPROVE`, `MANUAL_REVIEW`, `AUTO_REJECT`).
8. **Structured Audit Trail**: Immutable JSON logs recorded in `audit_logs/` with zero secret leakage.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Installed packages: `fastapi`, `uvicorn`, `groq`, `pydantic`, `scikit-learn`, `xgboost`, `shap`, `joblib`, `pytest`

### 2. Environment Configuration
Copy the sample environment file:
```bash
cp openclaw_credit_risk_agent/.env.example openclaw_credit_risk_agent/.env
```

Edit `openclaw_credit_risk_agent/.env`:
```ini
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=deepseek-r1-distill-llama-70b
GROQ_TEMPERATURE=0.1
PORT=8000
```

> **Note**: The system functions completely in deterministic mode even if `GROQ_API_KEY` is not provided.

### 3. OpenClaw Runtime Setup (Optional)
If running OpenClaw gateway / CLI:
```bash
npm install -g openclaw
```
The tool definitions are pre-configured in [`openclaw_credit_risk_agent/config/openclaw_config.json`](file:///Users/pulkitchauhan/Desktop/cv/Credit-Risk-Modeling-main/openclaw_credit_risk_agent/config/openclaw_config.json).

---

## 💻 Running the Application

### 1. Interactive CLI Demo
Run the CLI demo from the parent directory:
```bash
# Evaluate a prime low-risk borrower profile
python3 openclaw_credit_risk_agent/app.py --sample low_risk

# Evaluate a borderline borrower profile
python3 openclaw_credit_risk_agent/app.py --sample borderline

# Evaluate a high-risk subprime borrower profile
python3 openclaw_credit_risk_agent/app.py --sample high_risk

# Interactive mode
python3 openclaw_credit_risk_agent/app.py --interactive

# From JSON input
python3 openclaw_credit_risk_agent/app.py --json-input path/to/applicant.json
```

### 2. FastAPI Microservice
Start the REST API server:
```bash
python3 -m uvicorn openclaw_credit_risk_agent.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation available at:
`http://localhost:8000/docs`

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/credit-risk/assess` | Full End-to-End Multi-Agent Credit Assessment |
| `POST` | `/credit-risk/pd` | Calculate Probability of Default (Scorecard / Challenger) |
| `POST` | `/credit-risk/lgd` | Calculate Loss Given Default (2-stage hurdle) |
| `POST` | `/credit-risk/ead` | Calculate Exposure at Default (CCF model) |
| `POST` | `/credit-risk/expected-loss` | Calculate Expected Loss ($EL = PD \times LGD \times EAD$) |
| `POST` | `/credit-risk/compare` | Compare Scorecard vs XGBoost vs RF vs Logistic |
| `POST` | `/credit-risk/explain` | Extract SHAP risk drivers and feature impact waterfall |
| `POST` | `/credit-risk/stress-test` | Run macroeconomic stress testing under GDP/Unemp shocks |
| `GET` | `/credit-risk/psi` | Retrieve feature Population Stability Index metrics |
| `GET` | `/credit-risk/health` | Health check & model artifact verification |

### Example Request (`POST /credit-risk/assess`)
```bash
curl -X POST "http://localhost:8000/credit-risk/assess" \
     -H "Content-Type: application/json" \
     -d '{
       "application_id": "APP-2026-001",
       "loan_amnt": 15000.0,
       "int_rate": 7.5,
       "grade": "A",
       "sub_grade": "A2",
       "emp_length": "6 years",
       "home_ownership": "MORTGAGE",
       "annual_inc": 105000.0,
       "verification_status": "Source Verified",
       "purpose": "debt_consolidation",
       "addr_state": "CA",
       "dti": 12.5
     }'
```

### Example Structured Response
```json
{
  "assessment_id": "CR-7B2F10A4",
  "timestamp": "2026-08-18T00:16:26.006422Z",
  "application_id": "APP-2026-001",
  "pd": 0.0162,
  "lgd": 0.8292,
  "ead": 11017.50,
  "expected_loss": 148.00,
  "credit_score": 737,
  "risk_grade": "A",
  "model": "logistic_scorecard",
  "challenger_pd": 0.2841,
  "top_risk_drivers": [
    {
      "feature": "int_rate",
      "feature_label": "Interest Rate",
      "feature_value": 7.5,
      "impact": -1.8349,
      "direction": "DECREASES_RISK",
      "description": "Interest Rate of 7.50"
    }
  ],
  "policy_decision": "AUTO_APPROVE",
  "policy_reasons": [
    "Excellent credit profile: Score 737, PD 1.6%, DTI 12.5%"
  ],
  "stress_test_summary": {
    "Baseline": 148.00,
    "Mild Downturn": 229.77,
    "Severe Recession": 329.86
  },
  "explanation_summary": "Applicant evaluated under Basel Scorecard: Score 737 (Grade A), PD 1.62%, LGD 82.9%, EAD $11,017.50, Expected Loss $148.00. Policy Decision: AUTO_APPROVE."
}
```

---

## 🧪 Running Tests

Execute the comprehensive automated test suite:
```bash
python3 -m pytest openclaw_credit_risk_agent/tests/ -v
```

Test coverage includes:
- **`test_pd_scorecard.py`**: Scorecard WoE dummy mapping, score points, log-odds calculation, PD bounds.
- **`test_lgd_ead_el.py`**: Two-stage LGD, CCF EAD, and exact $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$ verification.
- **`test_ml_challenger.py`**: XGBoost, Random Forest, Scaler, and Challenger Logistic Regression scoring.
- **`test_shap_explainability.py`**: SHAP feature attributions and adverse action notice generation.
- **`test_stress_testing.py`**: Macroeconomic stress simulations (GDP $-2\%$, Unemployment $+5\%$).
- **`test_psi_monitoring.py`**: Population Stability Index equation and distribution drift detection.
- **`test_agents_tools.py`**: Agent isolation, schema sanitization, and tool adapter wrappers.
- **`test_e2e_assessment.py`**: End-to-end multi-agent orchestration and FastAPI HTTP endpoints.

---

## 🛡️ Guardrails & Governance

1. **Zero Math in LLM**: Statistical risk calculation logic is executed strictly in Python.
2. **Immutable Read-Only Parent**: The existing project files (`df_scorecard.csv`, `ml_challenger_output/*`, `*.ipynb`) are read-only and never modified.
3. **Secret Isolation**: `GROQ_API_KEY` is never logged or exposed in audit trails.
4. **Audit Logging**: Every credit assessment is logged as an immutable JSON file in `audit_logs/` recording timestamps, input snapshots, model versions, decisions, and tool executions.

---

## 📚 Project Layout

```text
openclaw_credit_risk_agent/
├── ARCHITECTURE.md                 # System architecture specification
├── README.md                       # Main documentation & user guide
├── .env.example                    # Environment configuration template
├── app.py                          # Interactive CLI demo application
├── api.py                          # FastAPI REST microservice
├── config/
│   ├── settings.py                 # Pydantic configuration loader
│   └── openclaw_config.json        # OpenClaw tool registry schema
├── agents/
│   ├── base_agent.py               # Groq LLM integration base class
│   ├── orchestrator_agent.py       # Chief Risk Officer multi-agent coordinator
│   ├── data_agent.py               # Ingestion & schema validation agent
│   ├── risk_model_agent.py         # Quantitative PD, LGD, EAD, EL agent
│   ├── model_comparison_agent.py   # Challenger benchmarking agent
│   ├── explainability_agent.py     # SHAP interpretability agent
│   ├── stress_testing_agent.py     # Macroeconomic stress testing agent
│   ├── monitoring_agent.py         # Population Stability Index agent
│   └── policy_agent.py             # Underwriting decision rules agent
├── tools/
│   ├── schemas.py                  # Pydantic domain models
│   ├── credit_model_adapter.py     # Deterministic model adapter
│   ├── audit_logger.py             # Privacy-safe audit trail logger
│   └── openclaw_tools.py           # Callable OpenClaw/MCP tool functions
├── docs/
│   └── agent-workflow.md           # Multi-agent decision flow & governance
├── audit_logs/                     # Automated JSON audit trail records
└── tests/                          # Automated Pytest suite (36 unit & e2e tests)
```
