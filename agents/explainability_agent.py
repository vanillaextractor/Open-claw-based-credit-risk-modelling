"""Explainability Agent for SHAP Interpretability and Adverse Action Reporting."""

from typing import Dict, Any, List
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import ApplicantData, SHAPExplanation, RiskDriver
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter

ROLE_PROMPT = """You are the Explainability Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to translate quantitative SHAP (SHapley Additive exPlanations) values and scorecard points
into plain English adverse action reasons and underwriting justifications.
Adhere strictly to regulatory transparency standards (ECOA, FCRA, EU AI Act Article 22)."""


class ExplainabilityAgent(BaseAgent):
    """Interprets model decisions via SHAP and scorecard feature attributions."""

    def __init__(self):
        super().__init__(name="ExplainabilityAgent", role_prompt=ROLE_PROMPT)
        self.adapter = get_credit_adapter()

    def explain(self, applicant: ApplicantData, top_n: int = 5) -> SHAPExplanation:
        """Extract SHAP explanation for applicant."""
        return self.adapter.explain_prediction(applicant, top_n=top_n)

    def generate_adverse_action_notice(self, drivers: List[RiskDriver]) -> List[str]:
        """Convert negative risk drivers into regulatory adverse action disclosure statements."""
        statements = []
        for d in drivers:
            if d.direction == "INCREASES_RISK":
                if "dti" in d.feature.lower():
                    val = f"{d.feature_value}%" if isinstance(d.feature_value, (int, float)) else str(d.feature_value)
                    statements.append(f"High debt-to-income ratio ({val}) relative to income level.")
                elif "int_rate" in d.feature.lower():
                    val = f"{d.feature_value}%" if isinstance(d.feature_value, (int, float)) else str(d.feature_value)
                    statements.append(f"High risk premium on interest rate ({val}).")
                elif "delinq" in d.feature.lower():
                    val = f"{d.feature_value} occurrences" if isinstance(d.feature_value, (int, float)) else str(d.feature_value)
                    statements.append(f"Past delinquency record on credit history ({val}).")
                elif "inq" in d.feature.lower():
                    val = f"{d.feature_value} in last 6 months" if isinstance(d.feature_value, (int, float)) else str(d.feature_value)
                    statements.append(f"Recent credit inquiries ({val}).")
                elif "inc" in d.feature.lower():
                    if isinstance(d.feature_value, (int, float)):
                        statements.append(f"Annual income level (${d.feature_value:,.0f}) insufficient for debt service cap.")
                    else:
                        statements.append(f"Annual income level ({d.feature_value}) insufficient for debt service cap.")
                else:
                    statements.append(f"{d.feature_label} ({d.feature_value}) unfavorably affected the credit rating.")
        return statements if statements else ["No severe adverse action factors identified."]
