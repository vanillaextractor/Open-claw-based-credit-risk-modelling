"""Stress Testing Agent for Macroeconomic Shock Scenarios."""

from typing import Dict, Any
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import ApplicantData, StressTestResult
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter

ROLE_PROMPT = """You are the Stress Testing Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to assess credit exposure resilience under adverse macroeconomic scenarios
(such as GDP contractions and unemployment spikes) following CCAR / EBA principles.
Quantify capital shortfall and provisioning buffer requirements."""


class StressTestingAgent(BaseAgent):
    """Evaluates portfolio and applicant resilience under macro shocks."""

    def __init__(self):
        super().__init__(name="StressTestingAgent", role_prompt=ROLE_PROMPT)
        self.adapter = get_credit_adapter()

    def stress_test(self, applicant: ApplicantData, scenario: str = "all") -> StressTestResult:
        """Run macro shock simulations on the applicant."""
        return self.adapter.run_stress_test(applicant, scenario=scenario)
