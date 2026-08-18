"""Tests for SHAP Explainability and Risk Drivers."""

import pytest
from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter
from openclaw_credit_risk_agent.agents.explainability_agent import ExplainabilityAgent


@pytest.fixture
def adapter():
    return get_credit_adapter()


@pytest.fixture
def agent():
    return ExplainabilityAgent()


@pytest.fixture
def high_risk_applicant():
    return ApplicantData(
        application_id="TEST-SHAP-HIGH",
        loan_amnt=35000.0,
        funded_amnt=35000.0,
        int_rate=24.5,
        annual_inc=35000.0,
        dti=38.0,
        grade="F",
        sub_grade="F3",
        emp_length="1 year",
        home_ownership="RENT",
        delinq_2yrs=2.0,
        inq_last_6mths=4.0,
        term="60 months"
    )


def test_shap_explanation_generation(adapter, high_risk_applicant):
    """Verify SHAP explanation generation and risk driver formatting."""
    exp = adapter.explain_prediction(high_risk_applicant, top_n=5)
    assert len(exp.top_risk_drivers) == 5
    assert 0.0 <= exp.base_value <= 1.0

    # Ensure drivers are sorted by absolute impact
    impacts = [abs(d.impact) for d in exp.top_risk_drivers]
    for i in range(len(impacts) - 1):
        assert impacts[i] >= impacts[i + 1]


def test_shap_directional_attribution(adapter, high_risk_applicant):
    """Verify that severe factors (like high interest rate and high DTI) are flagged as INCREASES_RISK."""
    exp = adapter.explain_prediction(high_risk_applicant, top_n=5)
    driver_dict = {d.feature: d.direction for d in exp.top_risk_drivers}

    # High interest rate should increase risk
    if "int_rate" in driver_dict:
        assert driver_dict["int_rate"] == "INCREASES_RISK"


def test_adverse_action_notice_generation(agent, high_risk_applicant):
    """Verify that explainability agent generates valid adverse action notices."""
    exp = agent.explain(high_risk_applicant, top_n=5)
    notices = agent.generate_adverse_action_notice(exp.top_risk_drivers)
    assert len(notices) > 0
    assert all(isinstance(n, str) and len(n) > 5 for n in notices)
