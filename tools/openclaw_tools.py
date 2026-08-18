"""OpenClaw Tool Definitions for Credit Risk Modeling.

These functions serve as standard OpenClaw/MCP tools callable by agents or external callers.
"""

from typing import Dict, Any, Optional, Union
import numpy as np

from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter


def _to_applicant_data(app_input: Union[Dict[str, Any], ApplicantData]) -> ApplicantData:
    """Normalize input dict into validated ApplicantData instance."""
    if isinstance(app_input, ApplicantData):
        return app_input
    return ApplicantData(**app_input)


def predict_pd(application: Union[Dict[str, Any], ApplicantData], model_type: str = "scorecard") -> Dict[str, Any]:
    """Calculate Probability of Default using regulatory WoE scorecard or challenger models."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.predict_pd(app, model_type=model_type)
    return result.model_dump()


def predict_lgd(application: Union[Dict[str, Any], ApplicantData]) -> Dict[str, Any]:
    """Calculate Loss Given Default using the two-stage hurdle recovery rate model."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.predict_lgd(app)
    return result.model_dump()


def predict_ead(application: Union[Dict[str, Any], ApplicantData]) -> Dict[str, Any]:
    """Calculate Exposure at Default using the CCF model."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.predict_ead(app)
    return result.model_dump()


def calculate_expected_loss(pd: float, lgd: float, ead: float, loan_amount: Optional[float] = None) -> Dict[str, Any]:
    """Calculate Expected Loss (EL = PD * LGD * EAD)."""
    adapter = get_credit_adapter()
    result = adapter.calculate_expected_loss(pd, lgd, ead, loan_amount=loan_amount)
    return result.model_dump()


def explain_prediction(application: Union[Dict[str, Any], ApplicantData], top_n: int = 5) -> Dict[str, Any]:
    """Extract SHAP risk drivers and feature impact waterfall."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.explain_prediction(app, top_n=top_n)
    return result.model_dump()


def compare_models(application: Union[Dict[str, Any], ApplicantData]) -> Dict[str, Any]:
    """Compare Scorecard vs XGBoost vs Random Forest vs Logistic Regression."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.compare_models(app)
    return result.model_dump()


def run_stress_test(application: Union[Dict[str, Any], ApplicantData], scenario: str = "all") -> Dict[str, Any]:
    """Run macroeconomic stress testing under GDP and unemployment shocks."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.run_stress_test(app, scenario=scenario)
    return result.model_dump()


def calculate_psi(
    expected: Optional[list] = None,
    actual: Optional[list] = None,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Calculate Population Stability Index between two cohorts."""
    adapter = get_credit_adapter()
    if expected is not None and actual is not None:
        exp_arr = np.array(expected)
        act_arr = np.array(actual)
        psi_val = adapter.calculate_psi(exp_arr, act_arr, n_bins=n_bins)
        status = "Stable" if psi_val < 0.10 else ("Minor shift" if psi_val < 0.25 else "Major shift")
        return {"psi": psi_val, "status": status}
    else:
        # Return pre-computed PSI summary
        return adapter.get_psi_summary().model_dump()


def run_full_credit_assessment(application: Union[Dict[str, Any], ApplicantData]) -> Dict[str, Any]:
    """Run end-to-end full credit assessment returning structured output."""
    app = _to_applicant_data(application)
    adapter = get_credit_adapter()
    result = adapter.run_full_credit_assessment(app)
    return result.model_dump()
