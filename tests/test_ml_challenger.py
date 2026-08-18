"""Tests for ML Challenger Models (XGBoost, Random Forest, Logistic Regression)."""

import pytest
from openclaw_credit_risk_agent.tools.schemas import ApplicantData
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter


@pytest.fixture
def adapter():
    return get_credit_adapter()


@pytest.fixture
def sample_applicant():
    return ApplicantData(
        application_id="TEST-CHALLENGER",
        loan_amnt=18000.0,
        funded_amnt=18000.0,
        int_rate=11.2,
        annual_inc=85000.0,
        dti=16.5,
        grade="B",
        sub_grade="B2",
        emp_length="6 years",
        home_ownership="MORTGAGE",
        purpose="debt_consolidation",
        addr_state="TX",
        term="36 months"
    )


def test_ml_artifacts_exist(adapter):
    """Verify that all four parent joblib model artifacts are present and loaded."""
    assert adapter.xgb_model is not None, "XGBoost model failed to load from ml_challenger_output"
    assert adapter.rf_model is not None, "Random Forest model failed to load from ml_challenger_output"
    assert adapter.lr_model is not None, "Logistic Regression model failed to load from ml_challenger_output"
    assert adapter.scaler is not None, "StandardScaler failed to load from ml_challenger_output"


def test_ml_feature_construction(adapter, sample_applicant):
    """Verify that 32 features are correctly engineered without missing columns."""
    X = adapter.map_applicant_to_ml_features(sample_applicant)
    assert X.shape == (1, 32)
    assert not X.isnull().any().any(), "Engineered feature matrix must not contain null values"
    assert "loan_amnt" in X.columns
    assert "dti" in X.columns
    assert "loan_to_income" in X.columns
    assert "installment_to_income" in X.columns


def test_xgboost_prediction(adapter, sample_applicant):
    """Verify XGBoost classification probability output."""
    res = adapter.predict_pd(sample_applicant, model_type="xgboost")
    assert res.model_name == "xgboost"
    assert 0.0 < res.pd < 1.0


def test_random_forest_prediction(adapter, sample_applicant):
    """Verify Random Forest classification probability output."""
    res = adapter.predict_pd(sample_applicant, model_type="random_forest")
    assert res.model_name == "random_forest"
    assert 0.0 < res.pd < 1.0


def test_challenger_logistic_regression_prediction(adapter, sample_applicant):
    """Verify Challenger Logistic Regression output."""
    res = adapter.predict_pd(sample_applicant, model_type="logistic_regression")
    assert res.model_name == "logistic_regression"
    assert 0.0 < res.pd < 1.0


def test_model_comparison_consensus(adapter, sample_applicant):
    """Verify multi-model comparison across all four models."""
    comp_res = adapter.compare_models(sample_applicant)
    assert len(comp_res.models) == 4
    model_names = [m.model_name for m in comp_res.models]
    assert any("Scorecard" in name for name in model_names)
    assert any("XGBoost" in name for name in model_names)
    assert comp_res.consensus_risk_level in ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
