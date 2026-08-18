"""FastAPI Service for OpenClaw Credit Risk Assessment Layer.

Exposes RESTful endpoints for credit risk modeling, multi-agent orchestration,
SHAP explainability, and macroeconomic stress testing.
Reuses the exact same Python tool adapter layer.
"""

import sys
import types
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure openclaw_credit_risk_agent package namespace resolves whether run
# standalone (e.g. uvicorn api:app) or as part of a parent workspace.
_ROOT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _ROOT_DIR.parent

if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

if "openclaw_credit_risk_agent" not in sys.modules:
    _pkg = types.ModuleType("openclaw_credit_risk_agent")
    _pkg.__path__ = [str(_ROOT_DIR)]
    _pkg.__file__ = str(_ROOT_DIR / "__init__.py")
    sys.modules["openclaw_credit_risk_agent"] = _pkg

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openclaw_credit_risk_agent.config.settings import get_settings
from openclaw_credit_risk_agent.agents.orchestrator_agent import get_orchestrator
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter
from openclaw_credit_risk_agent.tools.schemas import (
    ApplicantData,
    PDResult,
    LGDResult,
    EADResult,
    ExpectedLossResult,
    SHAPExplanation,
    ModelComparisonResult,
    StressTestResult,
    PSIResult,
    FullCreditAssessment,
)

settings = get_settings()

app = FastAPI(
    title="OpenClaw Credit Risk Assessment API",
    description="Enterprise Multi-Agent Credit Risk Modeling & Expected Loss Calculation Service (Basel III / IFRS 9)",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
@app.get("/credit-risk/health", tags=["Health"])
def health_check():
    """Verify service and model artifact readiness."""
    adapter = get_credit_adapter()
    return {
        "status": "healthy",
        "service": "openclaw_credit_risk_agent",
        "version": "1.0.0",
        "artifacts_loaded": {
            "scorecard_csv": adapter.scorecard_df is not None,
            "xgboost_model": adapter.xgb_model is not None,
            "random_forest_model": adapter.rf_model is not None,
            "logistic_regression_model": adapter.lr_model is not None,
            "scaler": adapter.scaler is not None,
            "shap_explainer": adapter.explainer is not None,
            "psi_data": adapter.psi_df is not None,
        },
        "groq_configured": bool(settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here"),
        "groq_model": settings.groq_model,
    }


@app.post(
    "/credit-risk/assess",
    response_model=FullCreditAssessment,
    summary="Full End-to-End Multi-Agent Credit Risk Assessment",
    tags=["Assessment"]
)
def assess_credit_risk(applicant: ApplicantData):
    """Run full credit risk assessment returning PD, LGD, EAD, Expected Loss,
    Risk Grade, Model Comparison, SHAP Drivers, Stress Test, and Policy Decision.
    """
    try:
        orchestrator = get_orchestrator()
        assessment = orchestrator.assess_applicant(applicant)
        return assessment
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Credit risk assessment failed: {str(e)}"
        )


@app.post(
    "/credit-risk/pd",
    response_model=PDResult,
    summary="Predict Probability of Default (PD)",
    tags=["Models"]
)
def calculate_pd(applicant: ApplicantData, model_type: str = "scorecard"):
    """Compute PD using the regulatory scorecard or specified challenger model."""
    try:
        adapter = get_credit_adapter()
        return adapter.predict_pd(applicant, model_type=model_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PD calculation failed: {str(e)}"
        )


@app.post(
    "/credit-risk/lgd",
    response_model=LGDResult,
    summary="Predict Loss Given Default (LGD)",
    tags=["Models"]
)
def calculate_lgd(applicant: ApplicantData):
    """Compute LGD using the two-stage hurdle recovery rate model."""
    try:
        adapter = get_credit_adapter()
        return adapter.predict_lgd(applicant)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LGD calculation failed: {str(e)}"
        )


@app.post(
    "/credit-risk/ead",
    response_model=EADResult,
    summary="Predict Exposure at Default (EAD)",
    tags=["Models"]
)
def calculate_ead(applicant: ApplicantData):
    """Compute EAD using the Credit Conversion Factor (CCF) model."""
    try:
        adapter = get_credit_adapter()
        return adapter.predict_ead(applicant)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"EAD calculation failed: {str(e)}"
        )


class ExpectedLossInput(BaseModel):
    pd: float
    lgd: float
    ead: float
    loan_amount: Optional[float] = None


@app.post(
    "/credit-risk/expected-loss",
    response_model=ExpectedLossResult,
    summary="Calculate Expected Loss (EL = PD * LGD * EAD)",
    tags=["Models"]
)
def calculate_el(payload: ExpectedLossInput):
    """Compute expected monetary loss."""
    try:
        adapter = get_credit_adapter()
        return adapter.calculate_expected_loss(
            pd_val=payload.pd,
            lgd_val=payload.lgd,
            ead_val=payload.ead,
            loan_amount=payload.loan_amount
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expected Loss calculation failed: {str(e)}"
        )


@app.post(
    "/credit-risk/compare",
    response_model=ModelComparisonResult,
    summary="Compare Regulatory Scorecard with Challenger Models",
    tags=["Challenger Models"]
)
def compare_challengers(applicant: ApplicantData):
    """Compare Scorecard, XGBoost, Random Forest, and Standard Logistic Regression."""
    try:
        adapter = get_credit_adapter()
        return adapter.compare_models(applicant)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model comparison failed: {str(e)}"
        )


@app.post(
    "/credit-risk/explain",
    response_model=SHAPExplanation,
    summary="SHAP Explainability & Risk Drivers",
    tags=["Explainability"]
)
def explain_applicant(applicant: ApplicantData, top_n: int = 5):
    """Extract top positive and negative risk drivers using SHAP game theory."""
    try:
        adapter = get_credit_adapter()
        return adapter.explain_prediction(applicant, top_n=top_n)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability failed: {str(e)}"
        )


@app.post(
    "/credit-risk/stress-test",
    response_model=StressTestResult,
    summary="Macroeconomic Stress Testing",
    tags=["Stress Testing"]
)
def stress_test_applicant(applicant: ApplicantData, scenario: str = "all"):
    """Simulate macroeconomic shocks (GDP, Unemployment) on credit parameters."""
    try:
        adapter = get_credit_adapter()
        return adapter.run_stress_test(applicant, scenario=scenario)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress testing failed: {str(e)}"
        )


@app.get(
    "/credit-risk/psi",
    response_model=PSIResult,
    summary="Population Stability Index (PSI) Monitoring",
    tags=["Monitoring"]
)
def get_psi():
    """Retrieve population stability metrics across origination features."""
    try:
        adapter = get_credit_adapter()
        return adapter.get_psi_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PSI retrieval failed: {str(e)}"
        )
