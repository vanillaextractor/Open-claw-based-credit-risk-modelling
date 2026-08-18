"""Tests for Macroeconomic Stress Testing Engine."""

import pytest
from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter


@pytest.fixture
def adapter():
    return get_credit_adapter()


@pytest.fixture
def applicant():
    return ApplicantData(
        loan_amnt=20000.0,
        funded_amnt=20000.0,
        int_rate=12.0,
        annual_inc=75000.0,
        dti=18.0,
        grade="B",
        home_ownership="MORTGAGE"
    )


def test_stress_test_all_scenarios(adapter, applicant):
    """Verify that all standard macro scenarios are evaluated."""
    res = adapter.run_stress_test(applicant, scenario="all")
    assert len(res.scenarios) == 3

    scenario_names = [s.scenario_name for s in res.scenarios]
    assert "Baseline" in scenario_names
    assert "Mild Downturn" in scenario_names
    assert "Severe Recession" in scenario_names


def test_stress_monotonicity(adapter, applicant):
    """Verify that stressed PD and Expected Loss increase with scenario severity."""
    res = adapter.run_stress_test(applicant, scenario="all")
    sc_map = {s.scenario_name: s for s in res.scenarios}

    base = sc_map["Baseline"]
    mild = sc_map["Mild Downturn"]
    severe = sc_map["Severe Recession"]

    assert base.stressed_pd <= mild.stressed_pd <= severe.stressed_pd
    assert base.stressed_el <= mild.stressed_el <= severe.stressed_el
    assert severe.capital_buffer_needed > 0.0
