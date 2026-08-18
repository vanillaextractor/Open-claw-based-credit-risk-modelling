"""Policy Agent for Business Decisioning and Underwriting Rules."""

from typing import Dict, Any
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import ApplicantData, PolicyEvaluationResult
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter

ROLE_PROMPT = """You are the Policy Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to apply configurable underwriting policies and governance guardrails.
Enforce the strict boundary between statistical risk predictions and institution lending decisions."""


class PolicyAgent(BaseAgent):
    """Enforces credit risk underwriting rules and governance thresholds."""

    def __init__(self):
        super().__init__(name="PolicyAgent", role_prompt=ROLE_PROMPT)
        self.adapter = get_credit_adapter()

    def evaluate(self, applicant: ApplicantData, pd_val: float, score: int, el_val: float) -> PolicyEvaluationResult:
        """Apply underwriting rules to determine loan outcome."""
        return self.adapter.evaluate_policy(applicant, pd_val, score, el_val)
