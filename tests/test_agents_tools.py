"""Tests for Individual Agents and OpenClaw Tool Wrappers."""

import pytest
from openclaw_credit_risk_agent.tools import openclaw_tools
from openclaw_credit_risk_agent.agents.data_agent import DataAgent
from openclaw_credit_risk_agent.agents.policy_agent import PolicyAgent
from openclaw_credit_risk_agent.tools.schemas import ApplicantData, PolicyDecisionType


@pytest.fixture
def sample_payload():
    return {
        "application_id": "TEST-TOOL-001",
        "loan_amnt": 16000.0,
        "funded_amnt": 16000.0,
        "int_rate": 8.5,
        "grade": "A",
        "sub_grade": "A3",
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 95000.0,
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 12.0,
    }


def test_openclaw_tools_predict_pd(sample_payload):
    """Test predict_pd standalone tool function."""
    res = openclaw_tools.predict_pd(sample_payload, model_type="scorecard")
    assert "pd" in res
    assert "credit_score" in res
    assert 0.0 < res["pd"] < 0.10


def test_openclaw_tools_predict_lgd(sample_payload):
    """Test predict_lgd standalone tool function."""
    res = openclaw_tools.predict_lgd(sample_payload)
    assert "lgd" in res
    assert 0.0 <= res["lgd"] <= 1.0


def test_openclaw_tools_predict_ead(sample_payload):
    """Test predict_ead standalone tool function."""
    res = openclaw_tools.predict_ead(sample_payload)
    assert "ead" in res
    assert res["ead"] > 0.0


def test_openclaw_tools_calculate_expected_loss():
    """Test calculate_expected_loss standalone tool function."""
    res = openclaw_tools.calculate_expected_loss(pd=0.03, lgd=0.40, ead=10000.0, loan_amount=10000.0)
    assert res["expected_loss"] == 120.0  # 0.03 * 0.40 * 10000 = 120.0
    assert res["el_rate_pct"] == 1.20


def test_openclaw_tools_explain_prediction(sample_payload):
    """Test explain_prediction standalone tool function."""
    res = openclaw_tools.explain_prediction(sample_payload, top_n=3)
    assert "top_risk_drivers" in res
    assert len(res["top_risk_drivers"]) == 3


def test_openclaw_tools_compare_models(sample_payload):
    """Test compare_models standalone tool function."""
    res = openclaw_tools.compare_models(sample_payload)
    assert "models" in res
    assert len(res["models"]) == 4


def test_openclaw_tools_run_stress_test(sample_payload):
    """Test run_stress_test standalone tool function."""
    res = openclaw_tools.run_stress_test(sample_payload)
    assert "scenarios" in res
    assert len(res["scenarios"]) == 3


def test_data_agent_imputation():
    """Verify that DataAgent sanitizes incomplete inputs."""
    agent = DataAgent()
    raw = {"loan_amnt": 12000.0, "annual_inc": 55000.0, "dti": 15.0, "int_rate": 10.0}
    app, summary = agent.validate_and_prepare(raw)
    assert app.grade == "B"  # default
    assert app.home_ownership == "RENT"  # default
    assert summary["is_valid"] is True


def test_policy_agent_rejection():
    """Verify that PolicyAgent auto-rejects subprime credit profiles."""
    agent = PolicyAgent()
    app = ApplicantData(loan_amnt=30000.0, annual_inc=20000.0, int_rate=25.0, dti=48.0, grade="G")
    policy_res = agent.evaluate(app, pd_val=0.35, score=480, el_val=8500.0)
    assert policy_res.decision == PolicyDecisionType.AUTO_REJECT
    assert len(policy_res.reasons) > 0
