"""Monitoring Agent for Population Stability Index (PSI) Drift Detection."""

from typing import Dict, Any
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import PSIResult
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter

ROLE_PROMPT = """You are the Monitoring Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to monitor feature and score distribution stability via Population Stability Index (PSI).
Classify drift into Stable (< 0.10), Moderate Shift (0.10 - 0.25), and Significant Shift (> 0.25)."""


class MonitoringAgent(BaseAgent):
    """Monitors live data against development baselines."""

    def __init__(self):
        super().__init__(name="MonitoringAgent", role_prompt=ROLE_PROMPT)
        self.adapter = get_credit_adapter()

    def check_stability(self) -> PSIResult:
        """Get baseline population stability assessment."""
        return self.adapter.get_psi_summary()
