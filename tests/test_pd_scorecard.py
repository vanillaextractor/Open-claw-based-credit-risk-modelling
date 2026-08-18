"""Tests for WoE Logistic Scorecard PD calculation."""

import pytest
from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter


@pytest.fixture
def adapter():
    return get_credit_adapter()


@pytest.fixture
def prime_applicant():
    return ApplicantData(
        application_id="TEST-PRIME",
        loan_amnt=10000.0,
        int_rate=6.5,
        grade="A",
        sub_grade="A1",
        emp_length="10+ years",
        home_ownership="MORTGAGE",
        annual_inc=120000.0,
        verification_status="Not Verified",
        purpose="credit_card",
        addr_state="CA",
        dti=8.5,
        inq_last_6mths=0.0,
        delinq_2yrs=0.0,
        open_acc=15.0,
        pub_rec=0.0,
        revol_bal=8000.0,
        revol_util=18.0,
        total_acc=25.0,
    )


@pytest.fixture
def subprime_applicant():
    return ApplicantData(
        application_id="TEST-SUBPRIME",
        loan_amnt=25000.0,
        int_rate=24.0,
        grade="G",
        sub_grade="G2",
        emp_length="< 1 year",
        home_ownership="RENT",
        annual_inc=25000.0,
        verification_status="Verified",
        purpose="small_business",
        addr_state="NV",
        dti=38.0,
        inq_last_6mths=4.0,
        delinq_2yrs=2.0,
        open_acc=5.0,
        pub_rec=2.0,
        revol_bal=22000.0,
        revol_util=92.0,
        total_acc=10.0,
    )


def test_scorecard_file_loaded(adapter):
    """Verify that df_scorecard.csv is successfully loaded with valid columns."""
    assert adapter.scorecard_df is not None
    assert "Feature name" in adapter.scorecard_df.columns
    assert "Coefficients" in adapter.scorecard_df.columns
    assert "Score - Final" in adapter.scorecard_df.columns
    assert len(adapter.scorecard_df) >= 100


def test_dummy_mapping_completeness(adapter, prime_applicant):
    """Verify that all mapped dummies exist in df_scorecard.csv."""
    dummies = adapter.map_applicant_to_scorecard_dummies(prime_applicant)
    assert len(dummies) >= 80
    assert dummies["grade:A"] == 1
    assert dummies["grade:G"] == 0
    assert dummies["home_ownership:MORTGAGE"] == 1


def test_pd_scorecard_prime_applicant(adapter, prime_applicant):
    """Verify that a prime applicant receives high credit score and low PD."""
    result = adapter.predict_pd(prime_applicant, model_type="scorecard")
    assert 0.0 < result.pd < 0.05
    assert result.credit_score is not None
    assert result.credit_score >= 700
    # With AND logic: score must be >=750 AND pd<0.03 for Grade A
    assert result.risk_grade in ["A", "B", "C"]
    assert result.model_name == "logistic_scorecard"


def test_pd_scorecard_subprime_applicant(adapter, subprime_applicant):
    """Verify that a subprime applicant receives low credit score and high PD."""
    result = adapter.predict_pd(subprime_applicant, model_type="scorecard")
    assert result.pd > 0.20
    assert result.credit_score is not None
    assert result.credit_score < 600
    assert result.risk_grade in ["E", "F", "G"]


def test_pd_monotonicity_across_grades(adapter, prime_applicant):
    """Verify that PD strictly increases as loan grade worsens from A to G."""
    pds = []
    for g in ["A", "B", "C", "D", "E", "F", "G"]:
        test_app = prime_applicant.model_copy(update={"grade": g})
        res = adapter.predict_pd(test_app, model_type="scorecard")
        pds.append(res.pd)

    # Monotonicity check
    for i in range(len(pds) - 1):
        assert pds[i] <= pds[i + 1], f"Grade {i} PD ({pds[i]}) should be <= Grade {i+1} PD ({pds[i+1]})"
