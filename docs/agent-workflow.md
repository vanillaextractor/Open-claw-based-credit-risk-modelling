# OpenClaw Credit Risk Agent Workflow & Multi-Agent Architecture

This document describes the multi-agent decision flow, governance mechanisms, tool routing, and auditability standards of the **OpenClaw Credit Risk Agent**.

---

## 1. Multi-Agent Topology

The framework follows a hierarchical orchestrator-worker agent pattern:

```text
                               ┌────────────────────────┐
                               │  Client / Application  │
                               │  (CLI `app.py` / API)  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │   Orchestrator Agent   │
                               │   (Groq LLM Reasoning) │
                               └───────────┬────────────┘
                                           │
        ┌───────────────────┬──────────────┴───────┬───────────────────┐
        │                   │                      │                   │
        ▼                   ▼                      ▼                   ▼
┌──────────────┐    ┌───────────────┐      ┌───────────────┐    ┌──────────────┐
│  Data Agent  │    │  Risk Model   │      │ Model Comp.   │    │ Explain.     │
│  (Validation │    │  Agent (PD,   │      │ Agent (XGB /  │    │ Agent        │
│  & Dummies)  │    │  LGD, EAD, EL)│      │ RF / LogReg)  │    │ (SHAP Force) │
└───────┬──────┘    └───────┬───────┘      └───────┬───────┘    └──────┬───────┘
        │                   │                      │                   │
        └───────────────────┴──────────────┬───────┴───────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  Stress Testing Agent  │
                               │  (Macro Downturns)     │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │    Monitoring Agent    │
                               │    (PSI Drift Engine)  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │      Policy Agent      │
                               │   (Underwriting Rules) │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │   Structured Memo &    │
                               │   Audit Trail Logging  │
                               └────────────────────────┘
```

---

## 2. Step-by-Step Workflow Execution

### Step 1: Ingestion & Schema Sanitization (`DataAgent`)
- **Action**: Receives raw applicant JSON/dict.
- **Validation**: Ensures mandatory attributes (`loan_amnt`, `annual_inc`, `int_rate`, `dti`) fall within acceptable business bounds.
- **Normalization**: Maps employment strings (`"10+ years"` $\to 10.0$), terms (`"36 months"` $\to 36$), state abbreviations, and loan purposes.
- **Output**: Validated `ApplicantData` Pydantic model.

### Step 2: Quantitative Risk Engine Execution (`RiskModelAgent`)
- **Action**: Calls deterministic Python models from parent repository / bundled data:
  1. **PD Scorecard**: Maps applicant into 103 WoE dummy categories against [`df_scorecard.csv`](../data/df_scorecard.csv), computes log-odds $\sum \beta_i X_i$, Credit Score ($300-850$), and Probability of Default:
     $$\text{PD} = 1 - \frac{\exp(\beta_0 + \sum \beta_i X_i)}{1 + \exp(\beta_0 + \sum \beta_i X_i)}$$
  2. **Two-Stage Hurdle LGD**: Estimates recovery probability $\times$ conditional recovery rate. $\text{LGD} = 1 - \text{Recovery Rate}$.
  3. **Credit Conversion Factor (CCF) EAD**: Estimates exposure at default. $\text{EAD} = \text{CCF} \times \text{Funded Amount}$.
  4. **Expected Loss**:
     $$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$
- **Output**: `PDResult`, `LGDResult`, `EADResult`, `ExpectedLossResult`.

### Step 3: Challenger Model Benchmarking (`ModelComparisonAgent`)
- **Action**: Evaluates applicant feature vector against serialized ML challenger models:
  - XGBoost Classifier (`xgboost.joblib`)
  - Random Forest Classifier (`random_forest.joblib`)
  - Standard Logistic Regression (`logistic_regression.joblib`)
- **Evaluation**: Calculates model divergence, rank orders risk, and determines consensus risk level (`LOW_RISK`, `MEDIUM_RISK`, `HIGH_RISK`).

### Step 4: Game-Theoretic Interpretability (`ExplainabilityAgent`)
- **Action**: Computes SHAP attributions using `shap.TreeExplainer` on the gradient boosted tree.
- **Attribution**: Identifies the top 5 risk drivers and labels directional risk impact (`INCREASES_RISK` vs `DECREASES_RISK`).
- **Adverse Action**: Generates FCRA / ECOA compliant notice statements explaining reasons for score degradation if applicable.

### Step 5: Forward-Looking Macro Stress Testing (`StressTestingAgent`)
- **Action**: Simulates CCAR / IFRS 9 macroeconomic shocks:
  - **Baseline**: Current economic trajectory ($0\%$ GDP shock, $0\%$ unemployment shock)
  - **Mild Downturn**: $-1.0\%$ GDP, $+2.0\%$ unemployment surge ($\text{PD} \times 1.35$)
  - **Severe Recession**: $-2.0\%$ GDP, $+5.0\%$ unemployment surge ($\text{PD} \times 1.85$)
- **Capital Buffer**: Calculates incremental Expected Loss and capital buffer requirement under adverse scenarios.

### Step 6: Population Stability Monitoring (`MonitoringAgent`)
- **Action**: Evaluates Population Stability Index (PSI):
  $$\text{PSI} = \sum (\text{Actual}\% - \text{Expected}\%) \times \ln\left(\frac{\text{Actual}\%}{\text{Expected}\%}\right)$$
- **Classification**:
  - $\text{PSI} < 0.10$: Stable population
  - $0.10 \le \text{PSI} \le 0.25$: Moderate shift, monitor closely
  - $\text{PSI} > 0.25$: Significant population drift requiring model recalibration

### Step 7: Underwriting Policy Enforcement (`PolicyAgent`)
- **Action**: Evaluates business policy rules independently of model probabilities:
  - **AUTO_APPROVE**: $\text{PD} \le 5.0\%$, $\text{Score} \ge 580$, $\text{DTI} \le 45.0\%$, $\text{Loan} \le \$40,000$, no active delinquencies.
  - **MANUAL_REVIEW**: Borderline PD ($5\% - 25\%$), large loan amount ($> \$40,000$), elevated DTI ($> 45\%$), or borderline stress tolerance. Specifies required underwriter verification conditions.
  - **AUTO_REJECT**: $\text{PD} \ge 25.0\%$, $\text{Score} < 580$, or severe active delinquencies.

### Step 8: Multi-Agent Synthesis & Audit Logging (`OrchestratorAgent`)
- **Action**: Compiles complete assessment record.
- **LLM Reasoning**: Uses Groq LLM (e.g. `deepseek-r1-distill-llama-70b` or `llama-3.3-70b-versatile`) to generate a synthesized executive credit memorandum while preserving exact numerical outputs.
- **Audit Persistence**: Writes an immutable JSON audit log to `openclaw_credit_risk_agent/audit_logs/` (with all secrets masked).

---

## 3. Core Guardrails Matrix

| # | Guardrail Mandate | Enforcement Mechanism |
| :--- | :--- | :--- |
| **G1** | LLM cannot modify or override PD | PD computed strictly via deterministic Scorecard / XGBoost Python code |
| **G2** | LLM cannot modify LGD or EAD | Two-stage LGD and CCF EAD computed strictly via deterministic Python code |
| **G3** | Zero data fabrication | Missing input fields are rejected or imputed via documented fallback rules |
| **G4** | Policy decision remains separate from model | Model outputs risk ($\text{PD}, \text{EL}$); Policy Agent evaluates business rules |
| **G5** | Secret & API Key protection | `GROQ_API_KEY` is never printed, logged, or included in JSON audit trails |
| **G6** | Auditability | Every transaction receives a unique UUID, ISO timestamp, feature snapshot, and tool call list |
| **G7** | Immutability of parent repository | Parent project files remain 100% untouched and read-only |
