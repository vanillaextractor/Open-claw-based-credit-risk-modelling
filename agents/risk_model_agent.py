"""Risk Model Agent for Quantitative Credit Risk & Expected Loss Calculations."""

from typing import Dict, Any
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import (
    ApplicantData,
    PDResult,
    LGDResult,
    EADResult,
    ExpectedLossResult
)
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter

ROLE_PROMPT = """You are the Risk Model Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to orchestrate the deterministic execution of the Probability of Default (PD),
Loss Given Default (LGD), Exposure at Default (EAD), and Expected Loss (EL = PD * LGD * EAD) models.
Provide quantitative commentary on capital risk without altering calculated numbers."""


class RiskModelAgent(BaseAgent):
    """Executes deterministic credit risk models."""

    def __init__(self):
        super().__init__(name="RiskModelAgent", role_prompt=ROLE_PROMPT)
        self.adapter = get_credit_adapter()

    def run_assessment(self, applicant: ApplicantData) -> Dict[str, Any]:
        """Execute PD, LGD, EAD, and Expected Loss models deterministically."""
        # 1. PD Scorecard
        pd_result: PDResult = self.adapter.predict_pd(applicant, model_type="scorecard")

        # 2. LGD Hurdle Model
        lgd_result: LGDResult = self.adapter.predict_lgd(applicant)

        # 3. EAD CCF Model
        ead_result: EADResult = self.adapter.predict_ead(applicant)

        # 4. Expected Loss
        el_result: ExpectedLossResult = self.adapter.calculate_expected_loss(
            pd_val=pd_result.pd,
            lgd_val=lgd_result.lgd,
            ead_val=ead_result.ead,
            loan_amount=applicant.loan_amnt
        )

        return {
            "pd": pd_result.pd,
            "credit_score": pd_result.credit_score,
            "risk_grade": pd_result.risk_grade,
            "lgd": lgd_result.lgd,
            "recovery_rate": lgd_result.recovery_rate,
            "ead": ead_result.ead,
            "ccf": ead_result.ccf,
            "expected_loss": el_result.expected_loss,
            "el_rate_pct": el_result.el_rate_pct,
            "model_metadata": {
                "pd_model": pd_result.model_name,
                "lgd_model": lgd_result.model_name,
                "ead_model": ead_result.model_name
            }
        }
