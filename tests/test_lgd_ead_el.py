"""Tests for LGD, EAD, and Expected Loss (EL = PD * LGD * EAD)."""

import pytest
from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter


@pytest.fixture
def adapter():
    return get_credit_adapter()


@pytest.fixture
def sample_applicant():
    return ApplicantData(
        loan_amnt=20000.0,
        funded_amnt=20000.0,
        int_rate=12.0,
        annual_inc=75000.0,
        dti=18.0,
        grade="B",
        home_ownership="MORTGAGE",
        term="36 months"
    )


def test_lgd_bounds(adapter, sample_applicant):
    """Verify that Loss Given Default and Recovery Rate are bounded in [0, 1]."""
    lgd_res = adapter.predict_lgd(sample_applicant)
    assert 0.0 <= lgd_res.lgd <= 1.0
    assert 0.0 <= lgd_res.recovery_rate <= 1.0
    assert pytest.approx(lgd_res.lgd + lgd_res.recovery_rate, abs=1e-4) == 1.0


def test_ead_bounds(adapter, sample_applicant):
    """Verify that EAD is proportional to funded amount and CCF is bounded."""
    ead_res = adapter.predict_ead(sample_applicant)
    assert 0.0 <= ead_res.ccf <= 1.0
    assert ead_res.ead <= sample_applicant.funded_amnt
    assert ead_res.ead > 0.0
    assert pytest.approx(ead_res.ead, abs=1e-2) == pytest.approx(ead_res.ccf * sample_applicant.funded_amnt, abs=1e-2)


def test_expected_loss_formula_exact(adapter):
    """Verify the exact mathematical equation: EL = PD * LGD * EAD."""
    pd_val = 0.05
    lgd_val = 0.45
    ead_val = 15000.0

    el_res = adapter.calculate_expected_loss(pd_val, lgd_val, ead_val, loan_amount=15000.0)
    expected = round(pd_val * lgd_val * ead_val, 2)  # 0.05 * 0.45 * 15000 = 337.50

    assert el_res.expected_loss == expected
    assert pytest.approx(el_res.expected_loss, abs=1e-2) == 337.50
    assert pytest.approx(el_res.el_rate_pct, abs=1e-2) == 2.25  # 337.50 / 15000 * 100 = 2.25%


def test_expected_loss_edge_cases(adapter):
    """Verify Expected Loss behavior at zero and maximum values."""
    # Zero PD -> Zero EL
    res_zero = adapter.calculate_expected_loss(0.0, 0.5, 10000.0)
    assert res_zero.expected_loss == 0.0

    # 100% Default & Loss -> EL equals EAD
    res_max = adapter.calculate_expected_loss(1.0, 1.0, 10000.0)
    assert res_max.expected_loss == 10000.0
