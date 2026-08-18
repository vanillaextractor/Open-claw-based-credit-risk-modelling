"""Model Comparison Agent for Challenger Model Benchmarking."""

from typing import Dict, Any
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import ApplicantData, ModelComparisonResult
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter

ROLE_PROMPT = """You are the Model Comparison Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to analyze predictions across multiple model architectures (Regulatory Scorecard,
XGBoost Gradient Boosting, Random Forest, and Standard Logistic Regression).
Highlight model consensus or divergences without modifying underlying outputs."""


class ModelComparisonAgent(BaseAgent):
    """Compares baseline regulatory scorecard with ML challenger models."""

    def __init__(self):
        super().__init__(name="ModelComparisonAgent", role_prompt=ROLE_PROMPT)
        self.adapter = get_credit_adapter()

    def compare(self, applicant: ApplicantData) -> ModelComparisonResult:
        """Run multi-model comparison on the applicant."""
        return self.adapter.compare_models(applicant)
