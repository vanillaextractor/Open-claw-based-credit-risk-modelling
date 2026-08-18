"""Pydantic data schemas for OpenClaw Credit Risk Agent."""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PolicyDecisionType(str, Enum):
    """Underwriting Policy Decision Categories."""
    AUTO_APPROVE = "AUTO_APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    AUTO_REJECT = "AUTO_REJECT"


class RiskGrade(str, Enum):
    """Internal Risk Rating Grades."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class ApplicantData(BaseModel):
    """Normalized Applicant Profile Payload."""

    application_id: Optional[str] = Field(default=None, description="Unique loan application ID")
    loan_amnt: float = Field(..., ge=500, le=100000, description="Requested loan amount ($)")
    funded_amnt: Optional[float] = Field(default=None, description="Funded loan amount ($)")
    int_rate: float = Field(..., ge=1.0, le=40.0, description="Interest rate (%)")
    installment: Optional[float] = Field(default=None, description="Monthly installment amount ($)")
    grade: str = Field(default="B", description="Loan grade (A-G)")
    sub_grade: Optional[str] = Field(default=None, description="Loan sub-grade (A1-G5)")
    emp_length: Optional[str] = Field(default="5 years", description="Employment length string or numeric")
    home_ownership: str = Field(default="RENT", description="Home ownership: RENT, OWN, MORTGAGE, OTHER")
    annual_inc: float = Field(..., ge=1000, description="Annual income ($)")
    verification_status: str = Field(default="Not Verified", description="Income verification status")
    purpose: str = Field(default="debt_consolidation", description="Loan purpose")
    addr_state: Optional[str] = Field(default="CA", description="US State code")
    dti: float = Field(..., ge=0.0, le=100.0, description="Debt-to-Income ratio (%)")
    delinq_2yrs: Optional[float] = Field(default=0.0, description="Delinquencies in last 2 years")
    inq_last_6mths: Optional[float] = Field(default=0.0, description="Inquiries in last 6 months")
    mths_since_last_delinq: Optional[float] = Field(default=None, description="Months since last delinquency")
    mths_since_last_record: Optional[float] = Field(default=None, description="Months since last public record")
    open_acc: Optional[float] = Field(default=10.0, description="Number of open credit lines")
    pub_rec: Optional[float] = Field(default=0.0, description="Number of derogatory public records")
    revol_bal: Optional[float] = Field(default=15000.0, description="Total revolving balance ($)")
    revol_util: Optional[float] = Field(default=45.0, description="Revolving line utilization rate (%)")
    total_acc: Optional[float] = Field(default=20.0, description="Total number of credit lines")
    initial_list_status: Optional[str] = Field(default="f", description="Initial listing status (f, w)")
    term: Optional[str] = Field(default="36 months", description="Loan term (36 or 60 months)")
    collections_12_mths_ex_med: Optional[float] = Field(default=0.0, description="Collections in 12 months excluding medical")
    acc_now_delinq: Optional[float] = Field(default=0.0, description="Accounts currently delinquent")
    tot_coll_amt: Optional[float] = Field(default=0.0, description="Total collection amounts ever owed")
    tot_cur_bal: Optional[float] = Field(default=50000.0, description="Total current balance of all accounts")
    total_rev_hi_lim: Optional[float] = Field(default=30000.0, description="Total revolving high credit/limit")
    mths_since_earliest_cr_line: Optional[float] = Field(default=180.0, description="Months since earliest credit line")
    mths_since_issue_d: Optional[float] = Field(default=36.0, description="Months since issue date")


class PDResult(BaseModel):
    """Probability of Default calculation output."""
    model_name: str
    pd: float = Field(..., ge=0.0, le=1.0)
    credit_score: Optional[int] = Field(default=None, ge=300, le=850)
    risk_grade: str
    log_odds: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class LGDResult(BaseModel):
    """Loss Given Default calculation output."""
    model_name: str
    lgd: float = Field(..., ge=0.0, le=1.0)
    recovery_rate: float = Field(..., ge=0.0, le=1.0)
    stage_1_prob: Optional[float] = None
    stage_2_amount: Optional[float] = None


class EADResult(BaseModel):
    """Exposure at Default calculation output."""
    model_name: str
    ead: float = Field(..., ge=0.0)
    ccf: float = Field(..., ge=0.0, le=1.0)
    funded_amount: float


class ExpectedLossResult(BaseModel):
    """Expected Loss calculation output."""
    expected_loss: float = Field(..., ge=0.0)
    pd: float
    lgd: float
    ead: float
    el_rate_pct: float = Field(..., description="Expected loss as percentage of loan amount")


class RiskDriver(BaseModel):
    """Individual SHAP/Scorecard Risk Driver."""
    feature: str
    feature_label: str
    feature_value: Any
    impact: float
    direction: str = Field(..., description="INCREASES_RISK or DECREASES_RISK")
    description: str


class SHAPExplanation(BaseModel):
    """SHAP Interpretability Output."""
    model_name: str
    base_value: float
    prediction_value: float
    top_risk_drivers: List[RiskDriver]


class ModelComparisonEntry(BaseModel):
    """Comparison entry for a single model."""
    model_name: str
    model_family: str
    predicted_pd: float
    relative_risk_rank: int
    auc_roc: float
    ks_stat: float
    brier_score: float


class ModelComparisonResult(BaseModel):
    """Multi-model comparison output."""
    primary_model: str
    models: List[ModelComparisonEntry]
    consensus_risk_level: str


class StressScenarioImpact(BaseModel):
    """Impact of a specific macroeconomic scenario."""
    scenario_name: str
    gdp_shock_pct: float
    unemployment_shock_pct: float
    baseline_pd: float
    stressed_pd: float
    baseline_el: float
    stressed_el: float
    incremental_loss: float
    capital_buffer_needed: float


class StressTestResult(BaseModel):
    """Comprehensive Stress Testing Output."""
    baseline_el: float
    scenarios: List[StressScenarioImpact]
    worst_case_el: float
    resilience_rating: str


class PSIFeatureResult(BaseModel):
    """PSI for a single feature."""
    feature: str
    psi_value: float
    status: str


class PSIResult(BaseModel):
    """Population Stability Index Output."""
    overall_status: str
    stable_features: int
    minor_shift_features: int
    major_shift_features: int
    feature_details: List[PSIFeatureResult]


class PolicyEvaluationResult(BaseModel):
    """Underwriting Policy Decision."""
    policy_version: str = "v1.0"
    decision: PolicyDecisionType
    reasons: List[str]
    rule_flags: Dict[str, bool]
    recommended_conditions: List[str]


class FullCreditAssessment(BaseModel):
    """Complete End-to-End Structured Credit Assessment."""
    assessment_id: str
    timestamp: str
    application_id: Optional[str] = None
    pd: float
    lgd: float
    ead: float
    expected_loss: float
    credit_score: int
    risk_grade: str
    model: str = "logistic_scorecard"
    challenger_pd: float
    top_risk_drivers: List[RiskDriver]
    policy_decision: PolicyDecisionType
    policy_reasons: List[str]
    model_comparison: Optional[ModelComparisonResult] = None
    stress_test_summary: Optional[Dict[str, float]] = None
    explanation_summary: str
