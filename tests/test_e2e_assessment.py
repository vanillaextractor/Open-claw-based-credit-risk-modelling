"""End-to-End Tests for OpenClaw Credit Risk Assessment Layer and FastAPI."""

import pytest
from fastapi.testclient import TestClient
from openclaw_credit_risk_agent.api import app
from openclaw_credit_risk_agent.agents.orchestrator_agent import get_orchestrator
from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.openclaw_tools import run_full_credit_assessment


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def prime_applicant():
    return {
        "application_id": "E2E-PRIME-001",
        "loan_amnt": 15000.0,
        "funded_amnt": 15000.0,
        "int_rate": 7.5,
        "grade": "A",
        "sub_grade": "A2",
        "emp_length": "6 years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 105000.0,
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 12.5,
    }


def test_e2e_assessment_function(prime_applicant):
    """Verify run_full_credit_assessment returns complete structured dictionary."""
    res = run_full_credit_assessment(prime_applicant)
    assert "assessment_id" in res
    assert "pd" in res
    assert "lgd" in res
    assert "ead" in res
    assert "expected_loss" in res
    assert "credit_score" in res
    assert "risk_grade" in res
    assert "challenger_pd" in res
    assert "top_risk_drivers" in res
    assert "policy_decision" in res
    assert "explanation_summary" in res

    # Verify mathematical coherence: EL = PD * LGD * EAD
    calc_el = round(res["pd"] * res["lgd"] * res["ead"], 2)
    assert pytest.approx(res["expected_loss"], abs=1e-2) == calc_el


def test_orchestrator_agent_workflow(prime_applicant):
    """Verify Orchestrator agent compiles full assessment object."""
    orchestrator = get_orchestrator()
    app_data = ApplicantData(**prime_applicant)
    assessment = orchestrator.assess_applicant(app_data)
    assert assessment.credit_score >= 700
    assert assessment.policy_decision.value in ["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_REJECT"]
    assert len(assessment.top_risk_drivers) > 0


def test_fastapi_health_endpoint(client):
    """Verify GET /credit-risk/health."""
    response = client.get("/credit-risk/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"]["scorecard_csv"] is True
    assert data["artifacts_loaded"]["xgboost_model"] is True


def test_fastapi_assess_endpoint(client, prime_applicant):
    """Verify POST /credit-risk/assess."""
    response = client.post("/credit-risk/assess", json=prime_applicant)
    assert response.status_code == 200
    data = response.json()
    assert "assessment_id" in data
    assert 0.0 < data["pd"] < 0.10
    assert data["risk_grade"] in ["A", "B"]
    assert data["policy_decision"] == "AUTO_APPROVE"
